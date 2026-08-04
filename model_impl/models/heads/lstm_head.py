"""LSTM head: runs the fused per-timestep sequence through an LSTM and projects its final hidden state."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMHead(nn.Module):
    """
    Encodes the fused per-timestep sequence with an LSTM (unlike LinearHead/
    MLPHead, which only see the last timestep) and maps its final hidden
    state to all `pred_len` horizon days via one Linear layer.

    Parameters
    ----------
    d_model : int
        Per-stream hidden size (the fused sequence has width `d_model * 3`).
    pred_len : int
        Number of horizon days to predict.
    vocab : int
        Chronos token-bin count (classification target size per day).
    hidden_dim : int
        LSTM hidden size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (ignored when num_layers == 1) and
        before the final projection.
    """

    def __init__(self, d_model: int, pred_len: int, vocab: int,
                 hidden_dim: int = 512, num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.vocab = vocab
        self.lstm = nn.LSTM(d_model * 3, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, pred_len * vocab)

    def forward(self, fused_seq: torch.Tensor, tok_eur: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_seq : torch.Tensor
            (batch, ctx_len, d_model*3) fused per-timestep representations,
            fed through the LSTM in full.
        tok_eur : torch.Tensor
            Unused here — accepted for interface parity with other heads.
        y : torch.Tensor | None
            Unused here — accepted for interface parity with other heads.

        Returns
        -------
        torch.Tensor
            (batch, pred_len, vocab) logits.
        """
        _, (h_n, _) = self.lstm(fused_seq)
        last_layer = h_n[-1]  # (batch, hidden_dim)
        return self.proj(self.drop(last_layer)).reshape(-1, self.pred_len, self.vocab)

    def mc_targets(self) -> list[nn.Module]:
        """
        Returns
        -------
        list[nn.Module]
            Submodules to flip into train() mode for MC-Dropout.
        """
        return [self.lstm, self.drop, self.proj]
