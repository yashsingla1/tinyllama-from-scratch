"""
Attention machinery for the transformer.

  - GroupedAttention: Grouped-Query Attention (Llama-style).
  - MultiAttention: one transformer block (attention + feed-forward,
    pre-norm with residuals).
  - Block: a stack of MultiAttention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RoPE, SwiGLU


class GroupedAttention(nn.Module):
    """
    Grouped-Query Attention (Ainslie et al., 2023).

    Uses more query heads than key/value heads — query heads are grouped,
    and each group shares a single K/V head. This shrinks the KV-cache
    memory cost at inference (the main constraint for production LLMs)
    while keeping query-side expressivity.

    Used by Llama 2 70B, Llama 3, Mistral, and most modern open LLMs.
    """

    def __init__(
        self,
        embed_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
    ):
        super().__init__()
        assert embed_size % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = embed_size // num_q_heads
        self.n_rep = num_q_heads // num_kv_heads  # query heads per K/V head

        self.q_proj = nn.Linear(embed_size, num_q_heads * self.head_size, bias=False)
        self.k_proj = nn.Linear(embed_size, num_kv_heads * self.head_size, bias=False)
        self.v_proj = nn.Linear(embed_size, num_kv_heads * self.head_size, bias=False)
        self.rope = RoPE(self.head_size, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project to Q, K, V and split into heads
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_size)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_size)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_size)

        # Replicate K/V heads to match query heads (GQA "sharing")
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K (NOT V — position belongs on routing, not content)
        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention with causal mask
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Recombine heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return out


class MultiAttention(nn.Module):
    """
    One transformer block: pre-norm attention sublayer + pre-norm feed-forward
    sublayer, each wrapped in a residual connection.

    Both sublayers follow the pattern  x = x + dropout(sublayer(norm(x))) —
    the "residual stream" flows undisturbed; each sublayer reads a normalized
    copy and writes a correction back.
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.group_head = GroupedAttention(embed_size, num_heads, num_kv_heads, max_seq_len)
        self.nor