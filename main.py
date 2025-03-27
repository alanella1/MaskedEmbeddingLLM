import json
import math
import os
from collections import Counter
import numpy as np
from tqdm import tqdm
import sentencepiece as spm

DATA_FOLDER = "D:/Documents/HuffmanLLM/DATA/openwebtext/processed/0006"


def create_unigram_tokenizer():
    base_dir = "D:/Documents/HuffmanLLM/DATA/openwebtext/processed/0006"
    corpus_file = "D:/Documents/HuffmanLLM/DATA/unigram_corpus.txt"
    min_freq = 10

    token_freqs = Counter()
    for filename in tqdm(os.listdir(base_dir), desc="Scanning for Tokens"):
        full_path = os.path.join(base_dir, filename)
        try:
            with open(full_path, "rb") as in_file:
                tokens = in_file.read().decode("utf-8", errors='ignore').split()
                token_freqs.update(tokens)
        except Exception as e:
            print(f" Skipping file {filename} due to error: {e}")

    with open(corpus_file, "w", encoding="utf-8") as out_file:
        for token, freq in token_freqs.items():
            if freq >= min_freq:  # Only include frequent tokens
                scaled_freq = min(1000, freq // min_freq)  # Cap frequency to prevent overrepresentation
                out_file.write(" ".join([token] * scaled_freq) + "\n")  # Repeat tokens

    print("Token frequencies saved to disk.")

    # Train SentencePiece on this expanded corpus
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix="bpe_tokenizer",
        vocab_size=5000,
        model_type="bpe",
        character_coverage=1.0,
        shuffle_input_sentence=True,
        normalization_rule_name="nmt_nfkc",
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]"
    )

    print("Unigram LM tokenizer trained and saved.")


def average_token_len():
    sp = spm.SentencePieceProcessor(model_file="bpe_tokenizer.model")
    total_size = 0
    num = 0
    for filename in tqdm(os.listdir(DATA_FOLDER), desc="Average Token Length"):
        full_path = os.path.join(DATA_FOLDER, filename)
        with open(full_path, "rb") as in_file:
            txt = in_file.read().decode("utf-8", errors='ignore').strip()
            tokens = sp.encode_as_pieces(txt)
            size = len(tokens)
            total_size += size
            num += 1

    average = total_size / num
    print(f"Average Token Length: {average}")


def create_log_prob():
    """Creates log probabilities for BPE tokens by scanning all files in a directory."""

    # Load the trained BPE tokenizer
    sp = spm.SentencePieceProcessor(model_file='bpe_tokenizer.model')

    # Initialize counter for token frequencies
    token_counts = Counter()
    total_tokens = 0

    # Loop through all files in the directory
    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)

        # Read file and count token occurrences
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = sp.encode_as_ids(line.strip())  # Tokenize line into BPE token IDs
                token_counts.update(tokens)
                total_tokens += len(tokens)  # Track total number of tokens

    # Convert frequencies to log probabilities
    log_probs = {}
    for token_id, count in token_counts.items():
        prob = count / total_tokens  # Probability of token occurrence
        log_prob = math.log(prob) if prob > 0 else float('-inf')  # Convert to log-prob
        log_probs[token_id] = log_prob

    # Save log probabilities to a JSON file
    output_file = 'log_probs.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_probs, f, ensure_ascii=False, indent=4)

    print(f"✅ log_probs.json created successfully. Tokens processed: {len(token_counts)}")


average_token_len()
