"""MLP head: last-timestep slice + a 2-layer MLP (ReLU + Dropout) before the final projection."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    Slices the last timestep out of the fused per-timestep sequence and maps
    it to all `pred_len` horizon days through a hidden ReLU/Dropout layer
    instead of LinearHead's single Linear.

    Parameters
    ----------
    d_model : int
        Per-stream hidden size (the fused sequence has width `d_model * 3`).
    pred_len : int
        Number of horizon days to predict.
    vocab : int
        Chronos token-bin count (classification target size per day).
    hidden_dim : int
        Width of the hidden MLP layer.
    dropout : float
        Dropout probability between the two projections.
    """

    def __init__(self, d_model: int, pred_len: int, vocab: int,
                 hidden_dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.vocab = vocab
        self.fc1 = nn.Linear(d_model * 3, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, pred_len * vocab)

    def forward(self, fused_seq: torch.Tensor, tok_eur: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_seq : torch.Tensor
            (batch, ctx_len, d_model*3) fused per-timestep representations;
            only the last timestep is used.
        tok_eur : torch.Tensor
            Unused here — accepted for interface parity with other heads.
        y : torch.Tensor | None
            Unused here — accepted for interface parity with other heads.

        Returns
        -------
        torch.Tensor
            (batch, pred_len, vocab) logits.
        """
        fused = fused_seq[:, -1, :]
        h = self.drop(self.act(self.fc1(fused)))
        return self.fc2(h).reshape(-1, self.pred_len, self.vocab)

    def mc_targets(self) -> list[nn.Module]:
        """
        Returns
        -------
        list[nn.Module]
            Submodules to flip into train() mode for MC-Dropout.
        """
        return [self.fc1, self.drop, self.fc2]
