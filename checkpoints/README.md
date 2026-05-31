# Checkpoints

Trained model weights for `tinyllama-from-scratch`.

## Why this folder is mostly empty

The `.pt` checkpoint file is ~30 MB and is gitignored — git is not the right tool for large binary artifacts. You have two options to obtain the weights.

## Option 1 — download from the deployed demo

The trained model is already loaded into the [live demo on Hugging Face Spaces](https://huggingface.co/spaces/Yasho1/story-generator). You can clone that Space to get the `model.pt` file directly:

    git lfs install
    git clone https://huggingface.co/spaces/Yasho1/story-generator
    # model.pt is in the cloned folder

## Option 2 — train from scratch

If you'd rather train the model yourself:

1. Run the training notebook in `notebooks/`. It loads TinyStoriesInstruct, trains the model on a T4 GPU (or any CUDA-capable GPU), and saves the resulting checkpoint.
2. Expected training time on a free Kaggle T4: ~2–3 hours.
3. The output `.pt` file lands in the working directory of the notebook.

## Loading the checkpoint

Once you have the `.pt` file, load it with:

    import torch
    from src.model import TransformerLM

    checkpoint = torch.load("model.pt", map_location="cpu")
    model = TransformerLM(
        vocab_size=8000,
        embed_size=128,
        num_heads=8,
        num_kv_heads=2,
        num_blocks=8,
        max_seq_len=128,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

The tokenizer is at `tokenizer.json` (also in the Space repo) and loads via `src/tokenizer.py`'s `load_tokenizer()`.