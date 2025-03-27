import gc
import gzip
import json
import os
import random
import re
from collections import Counter, deque
import mmap
from srsly import msgpack
from tqdm import tqdm
import sentencepiece as spm

MAX_EXAMPLES_PER_FILE = 50_000
TOKEN_FREQ_FILE = 'token_frequency.json'


def is_spammy(words, word_counts):
    return (max(word_counts.values()) / len(words)) > 0.2 if words else False


def is_too_repetitive(words, word_counts):
    top_5 = sum(count for word, count in word_counts.most_common(5))
    return (top_5 / len(words)) > 0.5 if words else False


def has_too_many_numbers(words):
    num_words = sum(1 for w in words if w.isdigit())
    return (num_words / len(words)) > 0.3 if words else False


def is_glossary_style(sentences):
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
    return avg_length < 5


def is_too_simple(sentences):
    short_sentences = sum(1 for s in sentences if len(s.split()) <= 5)
    return (short_sentences / len(sentences)) > 0.8 if sentences else False


def is_menu_style(text):
    lines = text.split("\n")
    short_lines = sum(1 for line in lines if len(line.strip().split()) <= 3)
    return (short_lines / len(lines)) > 0.8 if lines else False


def check_remove_text(text):
    text_split = text.split()
    if len(text_split) < 50 or len(text_split) > 3_000:
        return False

    if has_too_many_numbers(text_split):
        return False

    # Use regex to split sentences properly
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences

    if is_glossary_style(sentences):
        return False
    if is_too_simple(sentences):
        return False

    word_counts = Counter(text_split)
    if is_spammy(text_split, word_counts):
        return False
    if is_too_repetitive(text_split, word_counts):
        return False

    if is_menu_style(text):
        return False

    return True


def read_file_fast(path):
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            return mm.read().decode('utf-8', errors='ignore')


def make_examples(train):
    if train:
        output_folder = 'DATA/training_data'
        base_dir = "D:/Documents/HuffmanLLM/DATA/openwebtext/processed/0006"
    else:
        output_folder = 'DATA/validation_data'
        base_dir = "D:/Documents/HuffmanLLM/DATA/openwebtext/testext_processed/0000"

    sp = spm.SentencePieceProcessor(model_file="bpe_tokenizer.model")
    # Init folder
    os.makedirs(output_folder, exist_ok=True)

    examples = deque()
    file_count = 1
    example_count = 0
    removed_files = 0
    # FILE LOOPING
    filenames = os.listdir(base_dir)
    for filename in tqdm(filenames, desc="Scanning text files"):
        if not train and random.random() > 0.1:
            continue
        full_path = os.path.join(base_dir, filename)
        try:
            text = read_file_fast(full_path)
            if check_remove_text(text):
                tokens = sp.EncodeAsIds(text) + [sp.eos_id()]
                examples.append({"token_ids": tokens})
                example_count += 1
            else:
                removed_files += 1

        except Exception as e:
            print(f"⚠️ Error processing {full_path}: {e}")
            pass

    # FILE LOOPING

    if examples:
        output_file = os.path.join(output_folder, f"train_{file_count:05d}.msgpack.gz")
        with gzip.open(output_file, "wb") as f:
            f.write(msgpack.packb(list(examples), use_bin_type=True))

        print(f"✅ Final batch saved: {example_count} examples to {output_file}")


make_examples(True)
