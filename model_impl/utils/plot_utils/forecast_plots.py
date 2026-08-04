"""Forecast figures: per-horizon overviews, per-window panels, attention maps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model_impl.artifacts_logs.writers import ensure_dir, savefig
from model_impl.consts import HORIZONS_DIR


def plot_horizon_forecasts(out_dir: Path, fw_rows: list[dict], pred_len: int, index: str) -> None:
    """
    One plot per horizon step h (1..pred_len). Each shows, across all windows in
    date order, the h-step-ahead predicted (median) vs true value on a shared date axis.
    Saved as horizons/horizon_h{h}.png.
    """
    horizons_dir = out_dir / HORIZONS_DIR
    ensure_dir(horizons_dir)

    df = pd.DataFrame(fw_rows)
    df["t"] = pd.to_datetime(df["t"])

    for h in range(1, pred_len + 1):
        sub = df[df["horizon"] == h].sort_values("t")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(sub["t"], sub["truth"],  color="green", label="Actual")
        ax.plot(sub["t"], sub["median"], "--", color="darkorange", label="Predicted (median)")
        ax.set(xlabel="date", ylabel="price",
               title=f"{index} · {h}-step-ahead forecast vs actual (all windows)")
        ax.grid(True, alpha=0.3); ax.legend()
        savefig(horizons_dir, fig, f"horizon_h{h}")


def _draw_forecast_panel(ax, ctx_dates, ctx_vals, fc_dates,
                         truth, median, low80, high80, ylabel: str) -> None:
    """Draw one context+forecast panel (last context point prepended for continuity)."""
    x    = ctx_dates[-1:].append(fc_dates)
    tail = ctx_vals[-1:]
    ax.plot(ctx_dates, ctx_vals, color="steelblue", label="Context")
    ax.plot(x, np.concatenate([tail, truth]),  color="green", label="Actual")
    ax.plot(x, np.concatenate([tail, median]), "--", color="darkorange", label="Predicted (median)")
    ax.fill_between(x, np.concatenate([tail, low80]), np.concatenate([tail, high80]),
                    color="darkorange", alpha=0.20, label="80% CI")
    ax.axvline(ctx_dates[-1], color="red", linewidth=0.5, alpha=0.5)
    ax.set(xlabel="date", ylabel=ylabel)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)


def plot_forecast_window(forecasts_dir: Path, ctx_dates, ctx_prices,
                         fc_dates, truth, median, low80, high80,
                         title: str, diff_panel: dict | None = None) -> None:
    """
    Per-window forecast, saved as <fc_start>.png.

    Single price panel by default. If `diff_panel` is given (target was differenced),
    a top panel shows the diff-space view (what the model predicts) and the bottom
    panel the reconstructed price-space view. `diff_panel` keys:
    ctx, truth, median, low80, high80 (all diff-space arrays over the same dates).
    """
    if diff_panel is None:
        fig, ax = plt.subplots(figsize=(9, 4))
        _draw_forecast_panel(ax, ctx_dates, ctx_prices, fc_dates,
                             truth, median, low80, high80, ylabel="price")
        ax.set_title(title)
    else:
        fig, (ax_d, ax_p) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        _draw_forecast_panel(ax_d, ctx_dates, diff_panel["ctx"], fc_dates,
                             diff_panel["truth"], diff_panel["median"],
                             diff_panel["low80"], diff_panel["high80"], ylabel="diff")
        ax_d.set_title(title)
        _draw_forecast_panel(ax_p, ctx_dates, ctx_prices, fc_dates,
                             truth, median, low80, high80, ylabel="price")
    savefig(forecasts_dir, fig, fc_dates[0].strftime("%Y-%m-%d"),mlflow_ignore=True)


def plot_attention_maps(outdir: Path, weights_news: np.ndarray,
                        weights_covariate: np.ndarray, index: str) -> None:
    """Head-averaged cross-attention heatmaps for one window."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(weights_news, cmap="viridis", ax=ax)
    ax.set_title(f"Attention: News → {index} (Window 1)")
    savefig(outdir, fig, "attn_news_to_index_w1")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(weights_covariate, cmap="magma", ax=ax)
    ax.set_title(f"Attention: Covariates → {index} (Window 1)")
    savefig(outdir, fig, "attn_covariate_to_eurusd_w1")
