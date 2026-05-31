# src — model and tokenizer code

The `src/` package contains all the model and tokenizer code as importable Python modules.

## Modules

- **`components.py`** — `RoPE`, `SwiGLU`. The small standalone neural-net building blocks.
- **`attention.py`** — `GroupedAttention`, `MultiAttention`, `Block`. The attention machinery and transformer block.
- **`model.py`** — `TransformerLM`. The top-level model class and the `generate` method.
- **`tokenizer.py`** — Utilities for training and loading a byte-level BPE tokenizer.
- **`__init__.py`** — Re-exports the main classes so you can `from src import TransformerLM`.

See the main project [README](../README.md) for architecture details and the comparison table.
