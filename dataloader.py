import json
import os
import gzip
import msgpack
import torch
import threading
import queue
from collections import deque
from torch.utils.data import IterableDataset, DataLoader, Dataset
import random
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="bpe_tokenizer.model")
VOCAB_SIZE = sp.vocab_size()
PAD_ID = 1


class MultiThreadedDataset(IterableDataset):
    def __init__(self, data_file, max_input_len=512, num_workers=2):
        """
        Args:
            data_folder (str): Path to `.msgpack.gz` files.
            max_input_len (int): Max sequence length.
            num_workers (int): Number of background threads for preloading data.
        """
        super().__init__()
        self.data_file = data_file
        self.max_input_len = max_input_len
        self.file_index = 0  # Tracks which msgpack file we're processing
        self.token_sequences = deque()  # Stores token sequences in memory
        self.buffer = queue.Queue(maxsize=100000)  # Stores processed training examples
        self.num_workers = num_workers

        self._load_msgpack()
        self._start_workers()

    def _load_msgpack(self):
        """Loads token sequences from the `.msgpack.gz` file into memory."""
        try:
            with gzip.open(self.data_file, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)  # Load token lists
            random.shuffle(data)  # Shuffle token sequences to prevent order bias

            for token_list in data:
                self.token_sequences.append(token_list["token_ids"])  # Store sequences

            print(f"✅ Loaded {len(self.token_sequences)} token sequences from {self.data_file}")

        except Exception as e:
            print(f"⚠️ Error loading {self.data_file}: {e}")

    def _start_workers(self):
        """Starts background workers for dynamic data loading."""
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._load_examples, daemon=True)
            worker.start()

    def _load_examples(self):
        """Dynamically loads token sequences and processes them into training examples."""
        while len(self.token_sequences) > 0:
            if self.buffer.qsize() < 5000:

                try:
                    token_ids = self.token_sequences.popleft()
                except IndexError:
                    return

                # ✅ Process non-overlapping patches
                for i in range(0, len(token_ids), self.max_input_len):  # Step by `max_input_len`
                    patch = token_ids[i: i + self.max_input_len]

                    # ✅ Pad if necessary
                    if len(patch) < self.max_input_len:
                        patch += [PAD_ID] * (self.max_input_len - len(patch))

                    input_tensor = torch.tensor(patch, dtype=torch.long)

                    try:
                        self.buffer.put(input_tensor, timeout=1)  # Prevent deadlock
                    except queue.Full:
                        break  # Stop adding if queue is full

        # ✅ Signal completion by adding `None` to the queue
        self.buffer.put(None)

    def __iter__(self):
        """Returns an iterator that yields training examples."""
        while True:
            if self.buffer.empty():
                print("Waited")
            sample = self.buffer.get()  # Blocks if queue is empty (waiting for data)
            if sample is None:
                break
            yield sample


class DummyDataset(Dataset):
    def __init__(self, max_input_len=512):
        """
        Loads `dummy_file.txt`, tokenizes it, and processes it into training examples.
        """
        self.max_input_len = max_input_len
        self.tokenizer = sp
        self.token_ids = self._load_dummy_file()

        # ✅ Create non-overlapping patches
        self.token_patches = self._create_patches()

    def _load_dummy_file(self):
        """Loads and tokenizes `dummy_file.txt` into a single token sequence."""
        dummy_path = "DATA/dummy_file.txt"

        try:
            with open(dummy_path, "r", encoding="utf-8") as f:
                text = f.read().strip()  # Read the whole file

            # ✅ Use the tokenizer to get token IDs
            token_ids = self.tokenizer.EncodeAsIds(text)

            return token_ids

        except Exception as e:
            print(f"⚠️ Error loading dummy file: {e}")
            return []

    def _create_patches(self):
        """Splits tokenized text into non-overlapping patches of length `max_input_len`."""
        patches = []
        for i in range(0, len(self.token_ids), self.max_input_len):  # Step by `max_input_len`
            patch = self.token_ids[i: i + self.max_input_len]  # Get the patch
            if len(patch) < self.max_input_len:
                patch += [PAD_ID] * (self.max_input_len - len(patch))  # Pad if necessary
            patches.append(patch)
        return patches

    def __len__(self):
        """Returns the number of non-overlapping patches."""
        return len(self.token_patches)

    def __getitem__(self, idx):
        """Returns a single token sequence of length `max_input_len`."""
        return torch.tensor(self.token_patches[idx], dtype=torch.long)


def get_dataloader(data_folder, batch_size=32, num_workers=2, small_set=False):
    """
    Creates a PyTorch DataLoader for the multi-threaded dataset.
    """
    if small_set:
        dataset = DummyDataset(max_input_len=512)
    else:
        dataset = MultiThreadedDataset(data_folder, num_workers=num_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0)  # `num_workers=0` since dataset preloads
    return dataloader
