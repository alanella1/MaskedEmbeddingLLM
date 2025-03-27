# MaskedEmbeddingLLM

**MaskedEmbeddingLLM** is a project that explores the effects of masking token embeddings during the training of a straightforward decoder-only Language Model (LLM). By selectively masking embeddings, the project aims to understand how such interventions impact the learning dynamics and performance of LLMs.

## Introduction

In natural language processing, token embeddings play a crucial role in capturing semantic information. This project investigates how masking these embeddings during training affects the behavior and efficacy of decoder-only LLMs. The insights gained can contribute to a deeper understanding of model robustness and generalization.

## Project Structure

The repository consists of the following files:

- `BuildDatabase.py`: Script for constructing the dataset used in training.
- `dataloader.py`: Handles data loading and preprocessing tasks.
- `main.py`: The main script to initiate model training and evaluation.
- `models.py`: Contains the architecture definition of the decoder-only LLM.
- `preprocess_corpus.py`: Preprocessing routines for the text corpus.
- `training.py`: Implements the training loop and masking strategies.
- `bpe_tokenizer.model` & `bpe_tokenizer.vocab`: Byte Pair Encoding (BPE) tokenizer model and vocabulary files.
- `log_probs.json`: Stores log probabilities from model evaluations.
