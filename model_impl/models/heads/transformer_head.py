"""Transformer head: a small self-attention encoder over the fused sequence before the final projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerHead(nn.Module):
    """
    Runs the fused per-timestep sequence through a small TransformerEncoder
    (self-attention across the full context, unlike LinearHead/MLPHead which
    only see the last timestep) and maps its last-timestep output to all
    `pred_len` horizon days via one Linear layer.

    Parameters
    ----------
    d_model : int
        Per-stream hidden size (the fused sequence has width `d_model * 3`,
        used directly as the encoder's model dimension).
    pred_len : int
        Number of horizon days to predict.
    vocab : int
        Chronos token-bin count (classification target size per day).
    n_heads : int
        Attention heads in the encoder layer.
    n_layers : int
        Number of stacked encoder layers.
    d_ff : int
        Feed-forward width inside each encoder layer.
    dropout : float
        Dropout used inside the encoder and before the final projection.
    """

    def __init__(self, d_model: int, pred_len: int, vocab: int,
                 n_heads: int = 8, n_layers: int = 2, d_ff: int = 1024,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.vocab = vocab
        fused_dim = d_model * 3
        enc_layer = nn.TransformerEncoderLayer(fused_dim, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(fused_dim, pred_len * vocab)

    def forward(self, fused_seq: torch.Tensor, tok_eur: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_seq : torch.Tensor
            (batch, ctx_len, d_model*3) fused per-timestep representations,
            fed through the encoder in full.
        tok_eur : torch.Tensor
            Unused here — accepted for interface parity with other heads.
        y : torch.Tensor | None
            Unused here — accepted for interface parity with other heads.

        Returns
        -------
        torch.Tensor
            (batch, pred_len, vocab) logits.
        """
        h = self.encoder(fused_seq)
        last = self.drop(h[:, -1, :])
        return self.proj(last).reshape(-1, self.pred_len, self.vocab)

    def mc_targets(self) -> list[nn.Module]:
        """
        Returns
        -------
        list[nn.Module]
            Submodules to flip into train() mode for MC-Dropout.
        """
        return [self.encoder, self.drop, self.proj]
