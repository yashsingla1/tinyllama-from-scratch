# tinyllama-from-scratch

A from-scratch Llama-style transformer trained on TinyStoriesInstruct. Implements RMSNorm, RoPE, SwiGLU, and Grouped-Query Attention. Generates short children's stories from structured prompts.

**🚀 Live demo:** [huggingface.co/spaces/Yasho1/story-generator](https://huggingface.co/spaces/Yasho1/story-generator)

---

## What this is

A small (~2M parameter) decoder-only transformer, written from scratch in PyTorch, trained on the [TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) dataset. The architecture matches Llama-3-style design choices: pre-normalization with RMSNorm, rotary positional embeddings, SwiGLU feed-forward blocks, and Grouped-Query Attention.

The model takes a structured prompt — words to include, a one-line summary, and stylistic features — and generates a children's story that matches.

## Why I built this

To understand modern transformer architectures from the inside out. Each component (RMSNorm vs LayerNorm, RoPE vs learned embeddings, SwiGLU vs ReLU MLP, GQA vs vanilla MHA) was added as a deliberate swap on top of a vanilla GPT baseline, with validation loss measured at each step.

## Architecture

| Component | Choice | Why |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | Drops mean-centering, faster, no quality loss vs LayerNorm |
| Position encoding | RoPE | Encodes relative position via rotation; no learned position table |
| Feed-forward | SwiGLU | Gated FFN with Swish activation; replaces standard ReLU/GELU MLP |
| Attention | Grouped-Query Attention | 8 query heads, 2 KV heads — 4× smaller KV-cache for inference |
| Tokenizer | Custom byte-level BPE | 8000 vocab, trained on TinyStories |

Hyperparameters used: `embed_size=128`, `num_heads=8`, `num_kv_heads=2`, `num_blocks=8`, `block_size=128`, `vocab_size=8000`.

## Component-by-component results

Each modern component was added on top of a vanilla GPT baseline and measured against the previous version on TinyStories character-level data.

| Architecture | Validation loss | Notes |
|---|---|---|
| Vanilla GPT (LayerNorm + learned pos + ReLU MLP + MHA) | baseline | — |
| + RMSNorm | ≈ baseline | No quality loss; simpler + faster |
| + RoPE | -5% | Relative position helps |
| + SwiGLU | -5% | Gated FFN consistently helps |
| + GQA | ≈ same | Quality-neutral, 4× KV-cache savings |

Switching to BPE tokenization later isn't loss-comparable to these — different tokenizer, different units. Generated samples improved dramatically.

## Example output

Prompt:

    Features: Dialogue
    Words: cat, fence, bird
    Summary: A cat watches a bird from a fence.

Generated story:

> A cat lived in a house. The cat liked to play with a ball. One day, the cat saw a bird with a hurt wing. The bird wanted help.
>
> "Can I help you, cat?" asked the cat.
> "Sure, I can," said the bird.
>
> The cat was happy. "Yes, please," said the cat. "I will put a bandage on your leg."
>
> The cat looked at the cat and smiled. The cat was happy. They played together all day long. The cat and the cat were good friends.

The model follows the prompt for the first few paragraphs, then begins to drift — a typical small-model limitation.

## Repository structure

    src/
    ├── components.py      # RoPE, SwiGLU
    ├── attention.py       # GroupedAttention, MultiAttention, Block
    ├── model.py           # TransformerLM (top-level model + generate)
    ├── tokenizer.py       # BPE training and loading utilities
    └── __init__.py        # Package exports
    notebooks/             # Training notebook
    samples/               # Generated story examples
    checkpoints/           # (gitignored — trained weights distributed separately)

## Honest scope

This project is intentionally small. The architecture is modern and the tokenizer is custom-trained, but the model is ~2M parameters and trained on a constrained children's-story corpus. The goal was depth of understanding (every component justified from first principles), not raw output quality. The samples it generates are coherent within the TinyStories domain — characters, simple plots, dialogue — but it's not competing with anything you'd actually deploy.

Limitations I'd address in a longer project:

- **KV-cache for fast inference.** GQA's memory advantage is theoretical without it; my generation is naïve full-recompute.
- **Natural-language prompts.** Currently requires structured prompts (Words/Summary/Features). A small "translator" model trained on free-form to structured pairs would unlock natural input.
- **Length and coherence.** Stories drift after ~150 tokens. Bigger model + longer training would help; this is the standard small-model trade.

## Tech stack

- **PyTorch** — model and training
- **HuggingFace `tokenizers`** — BPE training (GPT-2 byte-level scheme)
- **HuggingFace `datasets`** — TinyStoriesInstruct loading
- **Gradio + HuggingFace Spaces** — deployed demo

## Try it

[Live demo on Hugging Face Spaces.](https://huggingface.co/spaces/yashsingla1/yashsingla1) Type words, a summary, and stylistic features. The model generates a story.

## Contact

[GitHub](https://github.com/yashsingla1) · LinkedIn: *(https://www.linkedin.com/in/-yash-singla/)* · Email: *(yashsingla3385@gmail.com)*

