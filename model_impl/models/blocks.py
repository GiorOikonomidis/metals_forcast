"""Reusable architecture blocks."""

import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    """
    Simple cross-attention block with residual + layer norm and weight capture
    for later visualization/attribution.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln   = nn.LayerNorm(d_model)
        self.last_weights: torch.Tensor | None = None  # (batch, heads, Q, K)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_output, attn_weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=False)
        self.last_weights = attn_weights.detach().cpu()
        return self.ln(q + attn_output)
