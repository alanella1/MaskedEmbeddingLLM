import torch
import torch.nn as nn
import numpy as np

PAD_TOKEN_ID = torch.tensor(1, device='cuda')
UNK_TOKEN_ID = torch.tensor(0, device='cuda')
VOCAB_SIZE = 5000


class MaskedUnigramEmbedding(nn.Module):
    """
    Vocab dict should be dict of {token_id (int): (positive) log_prob (float)}
    """

    def __init__(self, mask_embedding, vocab_dict, embedding_dim):
        super().__init__()
        self.vocab_size = len(vocab_dict.keys())
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.special_token_ids = {0, 1}
        self.masking = mask_embedding
        if self.masking:
            self.register_buffer("masks", self.create_mask(vocab_dict, embedding_dim))

    def create_mask(self, vocab_dict, embedding_dim):
        """Generates a tensor mask where rare words use fewer dimensions."""
        masks = torch.zeros(VOCAB_SIZE, embedding_dim)
        a = np.log(16.39)  # min log prob
        b = np.log(3.28)  # max log prob

        for token_id, log_prob in vocab_dict.items():
            if token_id in self.special_token_ids:
                active_dims = embedding_dim
            else:
                log_p = np.log(log_prob)
                proportion = ((log_p - b) / (a - b))
                active_dims = max(8, min(embedding_dim, int(proportion * embedding_dim)))
            masks[token_id, :active_dims] = 1  # Enable only required dimensions
        return masks

    def forward(self, token_ids):
        return self.embedding(token_ids)  # Lookup embeddings

    def apply_mask(self, embeddings, token_ids):
        mask = self.masks[token_ids]
        embeddings.masked_fill_(mask == 0, 1e-6)
        return embeddings


class MyTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None,
                memory_is_causal=None):
        """
        Custom Transformer Decoder Layer without Cross-Attention.
        """
        x = tgt  # Input to the decoder

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)  # Self-Attention
            x = x + self._ff_block(self.norm2(x))  # Feed Forward
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x


class TransformerLM(nn.Module):
    def __init__(self, masked_embedding, vocab_dict, embedding_dim=128, d_model=64, num_heads=8, num_layers=6, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.vocab_size = len(vocab_dict)
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.using_masked_embedding = masked_embedding

        # **Masked Unigram Embedding Layer**
        self.embedding = MaskedUnigramEmbedding(masked_embedding, vocab_dict, embedding_dim)

        # **Positional Encoding**
        self.positional_encoding = self.create_positional_encoding(d_model, max_len=512)

        # Average out embeddings, some get scaled up, some scaled down
        self.embedding_embedding = nn.Linear(self.embedding_dim, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            bias=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # **Final Linear Layer (Logits for next-token prediction)**
        self.output_layer = (nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, self.vocab_size))
        )
        # self.output_layer = nn.Linear(d_model, self.vocab_size)

    def create_positional_encoding(self, embedding_dim, max_len=512):
        """Generates positional encodings"""
        pe = torch.zeros(max_len, embedding_dim, device='cuda')
        position = torch.arange(0, max_len, dtype=torch.float, device='cuda').unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float, device='cuda')
                             * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)

    def forward(self, token_ids):
        """
        token_ids: (batch_size, seq_len)
        """
        batch_size, seq_len = token_ids.shape

        # **Masked Embedding Lookup**
        embeddings = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        dummy_memory = torch.zeros(batch_size, 1, self.d_model, device=embeddings.device)  # Minimal size

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=token_ids.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))  # Use -inf for masking

        pad_mask = (token_ids == PAD_TOKEN_ID)

        if self.embedding_dim != self.d_model:
            embeddings = self.embedding_embedding(embeddings)
        # **Add Positional Encoding**
        embeddings = embeddings + self.positional_encoding
        embeddings = self.embedding.apply_mask(embeddings, token_ids)

        # **Pass Through Transformer**
        transformer_output = self.transformer(tgt=embeddings,
                                              tgt_mask=causal_mask,
                                              memory=dummy_memory,
                                              tgt_key_padding_mask=pad_mask)  # (batch_size, seq_len, embedding_dim)

        logits = self.output_layer(transformer_output)  # (batch_size, seq_len, vocab_size)

        return logits  # No softmax (use CrossEntropyLoss which applies it internally)
