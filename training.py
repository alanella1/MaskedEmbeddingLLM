import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models import TransformerLM
from dataloader import get_dataloader
import torch._dynamo
import pytorch_warmup as warmup
import torch.nn.utils as utils

torch._dynamo.config.suppress_errors = True
# Set Forever
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 3
UNK_TOKEN_ID = 0
VOCAB_SIZE = 5000
NUM_SEQS = 30000

# TUNEABLES
run_name = '1e4_128_masked_rare_tokens_more_bias'
LEARNING_RATE = 1e-4
USE_SMALL_SET = False
USE_MASKED_EMBEDDING = True

# less likely to tune
EMBEDDING_DIM = 256
D_MODEL = 256
HIDDEN_DIM = 1024
WEIGHT_DECAY = 0.0
# ========================== #
#       HYPERPARAMETERS      #
# ========================== #
BATCH_SIZE = 16
BATCH_SIZE_NORM = 128
batch_ratio = BATCH_SIZE_NORM / BATCH_SIZE
N_EPOCHS = 100

TRAIN_DATA_FOLDER = "DATA/training_data/train_00001.msgpack.gz"
TEST_DATA_FOLDER = "DATA/validation_data/train_00001.msgpack.gz"
BASE_LOGGING_FOLDER = 'loggin_stuff'
ACCUMULATION_STEPS = int(batch_ratio)  # 256 simulated batch size

# LOG PROBS
with open("log_probs.json", "r", encoding="utf-8") as f:
    log_probs = json.load(f)
LOG_PROBS = {int(k): (-1 * v) for k, v in log_probs.items()}

# MODEL
device = torch.device("cuda")
model = TransformerLM(USE_MASKED_EMBEDDING,
                      LOG_PROBS,
                      embedding_dim=EMBEDDING_DIM,
                      d_model=D_MODEL,
                      num_heads=8,
                      num_layers=6,
                      hidden_dim=HIDDEN_DIM,
                      dropout=0.1).to(device)

# Speed
torch.backends.cudnn.benchmark = True

# DATA

# OPTIMIZER & LOSS
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# SCHEDULING
if USE_SMALL_SET:
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
else:
    steps_per_epoch = NUM_SEQS // BATCH_SIZE_NORM
    warmup_period = steps_per_epoch // 2

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5, threshold=0.01)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

# TENSORBOARD
writer = SummaryWriter(log_dir=os.path.join(BASE_LOGGING_FOLDER, run_name, "train"))


def validate(model):
    """Runs validation on the test set and logs the loss."""
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    n = 0
    test_loader = get_dataloader(TEST_DATA_FOLDER, BATCH_SIZE, small_set=USE_SMALL_SET)

    with torch.no_grad():  # No gradient calculation needed for validation
        for input_batch in test_loader:
            input_batch = input_batch.to(device, non_blocking=True)

            # Model inference
            logits = model(input_batch)  # (batch_size, seq_len, vocab_size)

            # Shift target tokens to align with predictions
            shifted_targets = input_batch[:, 1:].contiguous()
            shifted_logits = logits[:, :-1, :]

            # Compute loss
            loss = criterion(shifted_logits.reshape(-1, model.vocab_size), shifted_targets.reshape(-1))

            # Track total validation loss
            total_val_loss += loss.item()
            n += 1

    # Compute average validation loss per token
    avg_val_loss = total_val_loss / n
    return avg_val_loss


def train():
    global_step = 1
    total_loss = 0.0
    accum_loss = 0.0
    for epoch in range(N_EPOCHS):
        val_loss = validate(model)

        if epoch > 0:
            scheduler.step(val_loss)
            
        writer.add_scalar('validation_loss', val_loss, (global_step - 1) * BATCH_SIZE)
        model.train()
        train_loader = get_dataloader(TRAIN_DATA_FOLDER, BATCH_SIZE, small_set=USE_SMALL_SET)
        for batch_idx, (input_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device, non_blocking=True)

            logits = model(input_batch)  # Output shape: (batch_size, vocab_size)
            # Shift targets by 1 token to align with predictions
            shifted_targets = input_batch[:, 1:].contiguous()  # ✅ Fix: Call contiguous() correctly
            shifted_logits = logits[:, :-1, :]  # ✅ Trim logits to match target length
            loss = criterion(shifted_logits.reshape(-1, model.vocab_size), shifted_targets.reshape(-1)) / ACCUMULATION_STEPS

            # Backpropagation
            loss.backward()

            # Keep track of losses
            this_loss = loss.item()
            accum_loss += this_loss
            total_loss += this_loss

            if global_step % ACCUMULATION_STEPS == 0:
                # LOSS
                writer.add_scalar('loss', total_loss, global_step * BATCH_SIZE)
                total_loss = 0  # Reset loss tracker

                utils.clip_grad_norm_(model.parameters(), 1.0)
                with warmup_scheduler.dampening():
                    optimizer.step()

                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step * BATCH_SIZE)
                # zero grad
                for param in model.parameters():
                    param.grad = None

                accum_loss = 0.0

            global_step += 1

        # Done With Epoch
        print("epoch: " + str(epoch))
        checkpoint_path = f"checkpoints/{run_name}_step_{epoch}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model saved at step {global_step}: {checkpoint_path}")


if __name__ == "__main__":
    train()
    writer.close()
