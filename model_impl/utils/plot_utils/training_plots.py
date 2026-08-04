"""Training curves: per-step and per-epoch train/val cross-entropy."""

from pathlib import Path

import matplotlib.pyplot as plt

from model_impl.artifacts_logs.writers import savefig


def plot_loss_step(outdir: Path, tr_losses: list[float], va_losses: list[float],
                   epoch_starts: list[int], n_tr: int, n_va: int) -> None:
    """Per-step train/val CE on a shared global-step axis, with epoch boundaries."""
    va_x = [ep + n_tr + b for ep in epoch_starts for b in range(n_va)][:len(va_losses)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr_losses, color="steelblue", linewidth=0.8, label="train CE")
    ax.plot(va_x, va_losses, color="darkorange", linewidth=0.8, label="val CE")
    for ep in epoch_starts:
        ax.axvline(ep, color="red", linewidth=0.5, alpha=0.5)
    ax.set(xlabel="step", ylabel="CE loss", title="Train / Val loss per step")
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(outdir, fig, "loss_curve_step")


def plot_loss_epoch(outdir: Path, ep_tr_means: list[float], ep_va_means: list[float]) -> None:
    """Per-epoch mean train/val CE."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_tr_means, color="steelblue", marker="o", label="train CE")
    ax.plot(ep_va_means, color="darkorange", marker="o", label="val CE")
    ax.set(xlabel="epoch", ylabel="mean CE loss", title="Train / Val loss per epoch")
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(outdir, fig, "loss_curve_epoch")
