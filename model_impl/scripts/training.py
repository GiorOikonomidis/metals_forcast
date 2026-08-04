"""
Training stage: fits the model with early stopping. Per-epoch validation CE is
part of this loop (it drives early stopping and the LR plateau) — the full
validation metric suite is a separate stage in scripts/validation.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from model_impl.utils.logger_utils.logger import get_logger
from model_impl.utils.runtime_utils import DEVICE
from model_impl.utils.tracking_utils import mlflow_tracker

logger = get_logger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from model_impl.utils.schemas import EarlyStopperConfig, TrainingConfig
    from model_impl.models.cross_chronos import MultiCrossChronos
    from model_impl.schedulers.cold_start import ColdStartScheduler


def run(model: MultiCrossChronos, opt: torch.optim.Optimizer,
        tr_loader: DataLoader, va_loader: DataLoader, scheduler: ColdStartScheduler | None,
        training_cfg: TrainingConfig, early_stopper_cfg: EarlyStopperConfig, lr_metric: str,
        ) -> tuple[list[float], list[float], list[int], list[float], list[float]]:
    """
    Train with early stopping over DataLoaders. No shuffle — order preserved.

    `scheduler` is None when SchedulerConfig.use is False — the LR then stays
    fixed at whatever the optimizer was built with, and `.step()` is skipped.
    `lr_metric` is SchedulerConfig.metric ("val" or "train"); irrelevant when
    scheduler is None.

    Returns
    -------
    tr_losses    : list[float]  CE loss per train batch step
    va_losses    : list[float]  CE loss per val batch step
    epoch_starts : list[int]    global step index where each epoch begins
    ep_tr_means  : list[float]  mean train CE per epoch
    ep_va_means  : list[float]  mean val CE per epoch
    """
    ce      = nn.CrossEntropyLoss(label_smoothing=training_cfg.label_smoothing)  # train objective
    ce_eval = nn.CrossEntropyLoss(label_smoothing=0.0)              # val monitoring: true CE,
                                                                    # drives early-stop + LR plateau
    best = float('inf')
    wait = 0
    best_state = None

    tr_losses, va_losses, epoch_starts = [], [], []
    ep_tr_means, ep_va_means = [], []
    global_step = 0

    for ep in range(training_cfg.epochs):
        epoch_starts.append(global_step)
        ep_tr_losses = []

        model.train()
        for xe, xn, xc, y in tr_loader:
            opt.zero_grad()
            logits = model(xe.to(DEVICE), xn.to(DEVICE), xc.to(DEVICE))
            loss = ce(logits.reshape(-1, model.vocab), y.reshape(-1).to(DEVICE))
            loss.backward()
            if training_cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.grad_clip)
            opt.step()
            tr_losses.append(loss.item())
            ep_tr_losses.append(loss.item())
            global_step += 1

        model.eval()
        ep_va_losses = []
        with torch.no_grad():
            for xe, xn, xc, y in va_loader:
                vl = ce_eval(
                    model(xe.to(DEVICE), xn.to(DEVICE), xc.to(DEVICE)).reshape(-1, model.vocab),
                    y.reshape(-1).to(DEVICE)
                ).item()
                va_losses.append(vl)
                ep_va_losses.append(vl)

        ep_tr_mean = float(np.mean(ep_tr_losses))
        ep_val_mean = float(np.mean(ep_va_losses))
        ep_tr_means.append(ep_tr_mean)
        ep_va_means.append(ep_val_mean)
        current_lr = scheduler.lr if scheduler is not None else opt.param_groups[0]["lr"]
        logger.info("ep=%3d  train_ce=%.4f  val_ce=%.4f  lr=%.2e",
                    ep, ep_tr_mean, ep_val_mean, current_lr)
        mlflow_tracker.log_metrics(
            {"train_ce": ep_tr_mean, "val_ce": ep_val_mean, "lr": current_lr}, step=ep,
        )

        # LR scheduler — observe the configured metric (lower is better)
        if scheduler is not None:
            scheduler.step(ep_tr_mean if lr_metric == "train" else ep_val_mean)

        if ep_val_mean < best:
            best = ep_val_mean
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        elif early_stopper_cfg.use:
            wait += 1
            if wait >= early_stopper_cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return tr_losses, va_losses, epoch_starts, ep_tr_means, ep_va_means
