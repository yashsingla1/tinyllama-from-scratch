"""
BPE tokenizer training and loading.

Uses byte-level BPE — GPT-2's scheme — implemented via the `tokenizers`
library. Byte-level means the alphabet is the 256 byte values, so any
input (any Unicode, code, emoji) can always be encoded. The vocabulary
is learned from data; the splitting/encoding scheme follows GPT-2.

The trained tokenizer file is a single JSON containing the vocabulary,
the learned merges, the pre-tokenizer config, and the decoder config —
everything needed to round-trip text <-> token IDs.
"""

from typing import Callable, Iterable

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def train_bpe(
    text_iterator: Iterable[list[str]],
    save_path: str,
    vocab_size: int = 8000,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Train a fresh byte-level BPE on text from an iterator.

    Args:
        text_iterator: yields batches of strings (typically dataset chunks).
        save_path: where to save the trained tokenizer JSON.
        vocab_size: target vocabulary size including special tokens and
            the 256 base byte tokens.
        special_tokens: list of atomic tokens to register (e.g. ["<UNK>", "<END>"]).
            Each gets a fixed ID and is never split or merged.

    Returns:
        The trained Tokenizer object.
    """
    if special_tokens is None:
        special_tokens = ["<UNK>", "<END>"]

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    tokenizer.save(save_path)

    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """
    Load a previously-saved tokenizer JSON.

    Re-attaches the ByteLevel decoder so .decode() produces clean text
    instead of the visible-character form with Ġ markers.
    """
    tokenizer = Tokenizer.from_file(path)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def get_encode_decode(tokenizer: Tokenizer) -> tuple[Callable, Callable]:
    """
    Return simple `encode(text) -> list[int]` and `decode(ids) -> text`
    functions that wrap the Tokenizer's verbose API.
    """

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids)

    return encode, decode