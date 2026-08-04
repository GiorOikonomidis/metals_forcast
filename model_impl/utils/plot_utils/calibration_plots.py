"""Calibration and skill figures: per-window metrics, coverage, reliability, PIT, pinball."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_impl.artifacts_logs.writers import savefig
from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)


def plot_metric_per_window(outdir: Path, values: list[float], label: str, name: str) -> None:
    """Line chart of one metric across evaluated windows."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(values) + 1), values, marker="o")
    ax.set_title(f"{label} per window"); ax.set_xlabel("Window"); ax.set_ylabel(label)
    ax.grid(True)
    savefig(outdir, fig, name)


def plot_coverage_summary(outdir: Path, cov50: float, cov80: float, cov90: float) -> None:
    """Mean empirical coverage at the three nominal levels."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1, 2], [cov50, cov80, cov90])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["50%", "80%", "90%"])
    ax.set_ylim(0, 1); ax.set_ylabel("Empirical coverage")
    ax.set_title("Average empirical coverage")
    savefig(outdir, fig, "coverage_summary")


def plot_reliability_curve(outdir: Path, forecasts_df: pd.DataFrame, ece_quantiles: np.ndarray) -> None:
    """
    Reliability (P-P) curve over the full nominal quantile grid, rebuilt from the
    per-window forecast table.

    `ece_quantiles` is the grid built by metrics.build_ece_quantiles from
    EvaluationConfig.ece_grid — the same grid the run's ECE score used.

    Only q10/q50/q90 are stored per window, so the intermediate nominal levels are
    approximated by piecewise-linear interpolation between them. For a curve that
    does not rely on that approximation, store the full quantile grid in the eval
    loop instead.

    Takes the DataFrame directly rather than re-reading it from disk — the
    caller already has it in memory, and TRACKING.LOCAL.use off means
    writers.write_csv may not have written anything to read back anyway.
    """
    try:
        emp_cov = []
        for q in ece_quantiles:
            qcol = None
            if abs(q - 0.1) < 1e-6: qcol = "q10"
            elif abs(q - 0.5) < 1e-6: qcol = "q50"
            elif abs(q - 0.9) < 1e-6: qcol = "q90"
            else:
                if q < 0.5:
                    qcol_low, qcol_high, t = "q10", "q50", (q - 0.1) / 0.4
                else:
                    qcol_low, qcol_high, t = "q50", "q90", (q - 0.5) / 0.4
                q_pred = (1 - t) * forecasts_df[qcol_low].values + t * forecasts_df[qcol_high].values
                emp_cov.append(np.mean(forecasts_df["truth"].values <= q_pred))
                continue
            q_pred = forecasts_df[qcol].values
            emp_cov.append(np.mean(forecasts_df["truth"].values <= q_pred))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", label="Ideal")
        ax.plot(ece_quantiles, emp_cov, marker="o", label="Empirical")
        ax.set_xlabel("Nominal quantile"); ax.set_ylabel("Empirical frequency")
        ax.set_title("Reliability (P-P) curve")
        ax.legend(); ax.grid(True)
        savefig(outdir, fig, "reliability_pp_curve")
    except Exception as e:
        logger.warning("reliability curve skipped: %s", e)


def plot_pit_histogram(outdir: Path, pit_all: list[float]) -> None:
    """PIT histogram — a calibrated ensemble gives a uniform one."""
    pit_all_np = np.clip(np.array(pit_all, dtype=np.float32), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit_all_np, bins=20, range=(0, 1), density=True, edgecolor="k")
    ax.set_title("PIT histogram (ideal = uniform)"); ax.set_xlabel("PIT"); ax.set_ylabel("Density")
    savefig(outdir, fig, "pit_histogram")


def plot_skill_vs_naive(outdir: Path, mse_skill: float, mae_skill: float) -> None:
    """Skill scores against the naïve random-walk baseline."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [mse_skill, mae_skill])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MSE skill", "MAE skill"])
    ax.set_ylim(0, 1); ax.set_title("Skill vs naïve RW (higher is better)")
    savefig(outdir, fig, "skill_vs_naive")


def plot_pinball_summary(outdir: Path, p10: list[float], p50: list[float], p90: list[float]) -> None:
    """Mean pinball loss at the three evaluated quantiles."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1, 2], [np.mean(p10), np.mean(p50), np.mean(p90)])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["q=0.1", "q=0.5", "q=0.9"])
    ax.set_ylabel("Pinball loss"); ax.set_title("Average pinball loss")
    savefig(outdir, fig, "pinball_loss_summary")
