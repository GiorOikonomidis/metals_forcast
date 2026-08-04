"""Original head: last-timestep slice + a single Linear over all horizon days at once."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """
    Slices the last timestep out of the fused per-timestep sequence and maps
    it straight to all `pred_len` horizon days via one Linear layer.

    Parameters
    ----------
    d_model : int
        Per-stream hidden size (the fused sequence has width `d_model * 3`).
    pred_len : int
        Number of horizon days to predict.
    vocab : int
        Chronos token-bin count (classification target size per day).
    """

    def __init__(self, d_model: int, pred_len: int, vocab: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.vocab = vocab
        self.linear = nn.Linear(d_model * 3, pred_len * vocab)

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
        return self.linear(fused).reshape(-1, self.pred_len, self.vocab)

    def mc_targets(self) -> list[nn.Module]:
        """
        Returns
        -------
        list[nn.Module]
            Submodules to flip into train() mode for MC-Dropout.
        """
        return [self.linear]
