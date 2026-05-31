"""
Top-level transformer model.

Wires together token embeddings, the transformer block stack, a final
LayerNorm, and the output projection to vocabulary. Also implements
autoregressive generation with temperature and top-k sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Block


class TransformerLM(nn.Module):
    """
    Llama-style decoder-only language model.

    Architecture:
      - Token embeddings (no learned positional embeddings — RoPE handles position)
      - Stack of transformer blocks (pre-norm, GQA, SwiGLU, residuals)
      - Final RMSNorm
      - Linear output projection to vocabulary

    Position information enters via RoPE inside each attention layer,
    not as an added positional embedding at the input.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        num_blocks: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = Block(
            num_blocks=num_blocks,
            embed_size=embed_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.norm_final = nn.RMSNorm(embed_size)
        self.output_proj = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_emb = self.token_embedding(x)
        x = self.blocks(x_emb)
        x = self.norm_final(x)
        logits = self.output_proj(x)

        if target is None:
            return logits, None

        logits_flat = logits.view(-1, logits.shape[-1])
        target_flat = target.view(-1)
        loss = F.cross_entropy(logits_flat, target_flat)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_tokens: int,
        end_token_id: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: starting token IDs, shape (B, T).
            max_tokens: how many tokens to generate.
            end_token_id: stop generation when this ID is sampled.
            temperature: sampling temperature (lower = more focused).
            top_k: if set, restrict sampling to the top-k most likely tokens.
        """
        for _ in range(max_tokens):
            idx_c = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_c)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == end_token_id:
                break

            idx = torch.cat((idx, next_id), dim=1)

        return idx