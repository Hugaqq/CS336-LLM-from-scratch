# CS336 Assignment 1: Transformer Language Model from Scratch

  This repository contains my implementation of Stanford CS336 Assignment 1. The project
  builds a Transformer language model from scratch with PyTorch, with an emphasis on
  understanding the architecture, training process, and inference behavior of modern language
  models.

  Rather than relying on high-level PyTorch implementations of Transformer components, the
  project implements the main building blocks directly, including linear layers, embeddings,
  RMSNorm, Rotary Position Embeddings, causal self-attention, SwiGLU feed-forward networks,
  and the complete Transformer language model.

  The current codebase serves as a baseline for future experiments with more efficient
  optimization, inference, and architectural techniques.

  ## Implemented Components

  ### Tokenization

  - Byte Pair Encoding training
  - Vocabulary and merge-rule serialization
  - Special-token handling
  - Text encoding and decoding
  - Parallel dataset tokenization
  - Memory-mapped token datasets

  ### Transformer Architecture

  - Linear and embedding layers
  - RMSNorm
  - Rotary Position Embeddings
  - Scaled dot-product attention
  - Causal multi-head self-attention
  - SwiGLU feed-forward networks
  - Pre-normalization Transformer blocks
  - Transformer language-model head

  ### Training

  - CUDA training
  - BF16 automatic mixed precision
  - TF32 matrix multiplication
  - AdamW optimization
  - Learning-rate warmup and cosine decay
  - Global gradient clipping
  - Training and validation loss evaluation
  - Weights & Biases logging
  - Compiled model execution with `torch.compile`

  ### Checkpointing and Generation

  - Model and optimizer checkpoint serialization
  - Training resumption from checkpoints
  - Autoregressive text generation
  - Temperature sampling
  - Nucleus (`top-p`) sampling
  - Context-window truncation
  - End-of-sequence termination

  ## Repository Structure

  ```text
  .
  ├── configs/
  │   └── base.py
  │       Baseline model, training, data, and path configuration
  │
  ├── cs336_basics/
  │   ├── transformer_component.py
  │   │   Core Transformer architecture
  │   ├── optimizer.py
  │   │   AdamW, learning-rate scheduling, and gradient clipping
  │   ├── loss.py
  │   │   Cross-entropy loss
  │   ├── data_loader.py
  │   │   Language-model batch construction
  │   ├── checkpointing.py
  │   │   Model and optimizer checkpoint management
  │   ├── BPE/
  │   │   └── bpe.py
  │   │       BPE vocabulary training
  │   └── Tokenizer/
  │       └── tokenizer.py
  │           Tokenizer encoding and decoding
  │
  ├── scripts/
  │   ├── train_tokenizer.py
  │   │   Offline BPE tokenizer training
  │   └── tokenize_dataset.py
  │       Offline dataset tokenization
  │
  ├── train.py
  │   CUDA training entry point
  │
  ├── generation.py
  │   Autoregressive generation entry point
  │
  ├── tests/
  │   Assignment tests and reference snapshots
  │
  └── data/
      Raw datasets, tokenizer files, and encoded token arrays
  ```
  ## Project Workflow

  The project follows the pipeline below:

  Raw text
     ↓
  BPE tokenizer training
     ↓
  Vocabulary and merge rules
     ↓
  Dataset tokenization
     ↓
  Memory-mapped token arrays
     ↓
  Batch sampling
     ↓
  Transformer language model
     ↓
  Cross-entropy loss
     ↓
  Optimization and checkpointing
     ↓
  Autoregressive generation
  
  The baseline configuration in configs/base.py acts as the shared source of truth for model
  dimensions, training hyperparameters, dataset selection, and derived file paths.

  ## Setup

  This project uses uv to manage its Python environment and dependencies.

  Install uv if it is not already available:

  pip install uv

  Install the project dependencies:

  uv sync

  ## Testing

  The repository contains the original CS336 tests and reference snapshots.

  uv run pytest

  The assignment handout is available at
  ./cs336_assignment1_basics.pdf.

  ## Current Scope

  The current implementation is a CUDA-focused baseline intended for studying the complete
  language-model pipeline.

  The primary goal is to keep the relationship between model architecture, optimization, data
  flow, and inference behavior explicit. General-purpose framework abstractions and support
  for additional hardware backends are not current priorities.

  ## Future Work

  Planned extensions include:

  - Muon optimizer
  - KV cache for efficient autoregressive generation
  - Attention Residuals
  - Improved inference interfaces
  - Training and inference performance analysis
  - Controlled ablation experiments
  - Comparisons between baseline and optimized implementations

  These extensions will be introduced incrementally so that their effects on model
  correctness, optimization behavior, memory usage, and generation performance can be studied
  independently.
