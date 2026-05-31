"""
Building-block components for the transformer:
  - RoPE: rotary positional embedding
  - SwiGLU: gated feed-forward with Swish activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (Su et al., 2021).

    Rotates query and key vectors by an angle proportional to position.
    The dot product q·k between two rotated vectors then depends only on
    the difference of their positions, giving the model relative-position
    information directly at the attention step.
    """

    def __init__(self, head_size: int, max_seq_len: int):
        super().__init__()
        freqs = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        positions = torch.arange(max_seq_len).unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        self.register_buffer("cos_a", torch.cos(angles))
        self.register_buffer("sin_a", torch.sin(angles))

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = x.shape[-2]
        cos_a = self.cos_a[start_pos : start_pos + T]
        sin_a = self.sin_a[start_pos : start_pos + T]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos_a - x2 * sin_a
        rot2 = x1 * sin_a + x2 * cos_a

        out = torch.empty_like(x)
        out[..., 0::2] = rot1
        out[..., 1::2] = rot2
        return out


class SwiGLU(nn.Module):
    """
    Gated feed-forward block from "GLU Variants Improve Transformer" (Shazeer, 2020).

    Two parallel linear projections produce a value branch and a gate branch.
    The gate is passed through Swish (SiLU); the two branches are multiplied
    element-wise; a final linear projects back to the embedding dimension.
    This replaces the standard ReLU/GELU MLP and is used by Llama, Mistral,
    and most modern open LLMs.
    """

    def __init__(self, embed_size: int):
        super().__init__()
        hidden = (8 * embed_size) // 3
        self.gate_proj = nn.Linear(embed_size, hidden)
        self.value_proj = nn.Linear(embed_size, hidden)
        self.down_proj = nn.Linear(hidden, embed_size)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.swish(self.gate_proj(x))
        value = self.value_proj(x)
        return self.down_proj(gate * value)