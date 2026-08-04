"""
MC-Dropout predictive sampling.

`predict_distribution` has its own module because both the eval pipeline and
the faithfulness study call it — importing it from either of those would make
them circular, and metrics.py stays a pure arrays-in/floats-out leaf.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # avoids importing the model/chronos stack just for annotations
    from chronos import ChronosPipeline

    from model_impl.models.cross_chronos import MultiCrossChronos


def predict_distribution(model: "MultiCrossChronos", chrono: "ChronosPipeline",
                         scale_win: torch.Tensor, ctx_eur: torch.Tensor,
                         ctx_news: torch.Tensor, ctx_covariate: torch.Tensor,
                         mc_samples: int) -> np.ndarray:
    """
    Run MC-Dropout forward passes and decode token ids to predictions.

    Dropout is enabled on the trainable submodules only for the duration of the
    loop and restored afterwards; the frozen Chronos encoder stays deterministic.
    Decoding uses `scale_win`, so the output lands in whatever space the target
    was tokenized in — price levels for no_diff, diffs otherwise.

    Returns
    -------
    np.ndarray  (mc_samples, PRED_LEN) float32
    """
    model.mc_dropout(True)
    samples_tok = []
    for _ in range(mc_samples):
        with torch.no_grad():
            logits = model(ctx_eur, ctx_news, ctx_covariate, mc=True).cpu()
        samples_tok.append(logits.argmax(dim=-1).squeeze(0).numpy())
    model.mc_dropout(False)

    samples_tok = np.stack(samples_tok)  # (MC, PRED_LEN)
    preds = chrono.tokenizer.output_transform(
        torch.tensor(samples_tok, dtype=torch.long), scale_win
    ).numpy().squeeze().astype(np.float32)  # (MC, PRED_LEN)
    return np.nan_to_num(preds, nan=np.nanmean(preds))
