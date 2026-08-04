"""Faithfulness figures: deletion/insertion curves, stability and placebo summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model_impl.artifacts_logs.writers import savefig


def plot_faith_curves_window(outdir: Path, curves_news: dict) -> None:
    """Deletion and insertion curves for a single window (paper illustration)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves_news["ks"], curves_news["del_curve"], marker="o", label="Deletion (top-k)")
    ax.plot(curves_news["ks"], curves_news["del_curve_rand"], "--", marker="x", label="Deletion (random)")
    ax.plot(curves_news["ks"], curves_news["del_curve_inv"], ":", marker="s", label="Deletion (least-k)")
    ax.set_title("Deletion curve · NEWS (ΔCRPS vs k)"); ax.set_xlabel("k steps"); ax.set_ylabel("ΔCRPS")
    ax.grid(True); ax.legend()
    savefig(outdir, fig, "faith_del_curve_news_w1")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curves_news["ks"], curves_news["ins_curve"], marker="o", label="Insertion (top-k)")
    ax.set_title("Insertion curve · NEWS (ΔCRPS improvement vs k)"); ax.set_xlabel("k steps"); ax.set_ylabel("ΔCRPS improvement")
    ax.grid(True); ax.legend()
    savefig(outdir, fig, "faith_ins_curve_news_w1")


def _plot_faith_mean_std(outdir: Path, ks: list[int], y: np.ndarray,
                         title: str, name: str, ylabel: str = "ΔCRPS") -> None:
    """Mean ±1σ band of one curve family across windows."""
    y = np.asarray(y)
    m, s = y.mean(0), y.std(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, m)
    ax.fill_between(ks, m - s, m + s, alpha=0.2)
    ax.set_title(title); ax.set_xlabel("k steps"); ax.set_ylabel(ylabel); ax.grid(True)
    savefig(outdir, fig, name)


def _plot_faith_hist(outdir: Path, values: list[float], title: str,
                     xlabel: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=15, edgecolor="k")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    savefig(outdir, fig, name)


def plot_faith_aggregates(outdir: Path, ks: list[int], del_mat: np.ndarray,
                          del_rand_mat: np.ndarray, del_inv_mat: np.ndarray,
                          ins_mat: np.ndarray, spearman_news: list[float],
                          spearman_covariate: list[float], jacc_news: list[float],
                          placebo_keys: list[str], placebo_mat: np.ndarray) -> None:
    """All cross-window faithfulness figures."""
    _plot_faith_mean_std(outdir, ks, del_mat,
                         "Deletion curve · NEWS (media ±1σ)", "faith_del_curve_news_mean")
    _plot_faith_mean_std(outdir, ks, del_rand_mat,
                         "Deletion random · NEWS (media ±1σ)", "faith_del_curve_news_rand_mean")
    _plot_faith_mean_std(outdir, ks, del_inv_mat,
                         "Deletion least-k · NEWS (media ±1σ)", "faith_del_curve_news_inv_mean")
    _plot_faith_mean_std(outdir, ks, ins_mat,
                         "Insertion curve · NEWS (media ±1σ)", "faith_ins_curve_news_mean",
                         ylabel="ΔCRPS improvement")

    _plot_faith_hist(outdir, spearman_news, "Spearman(saliency, ΔCRPS LOTO) · NEWS",
                     "ρ", "faith_spearman_hist_news")
    _plot_faith_hist(outdir, spearman_covariate, "Spearman(saliency, ΔCRPS LOTO) · COVARIATE",
                     "ρ", "faith_spearman_hist_covariate")
    _plot_faith_hist(outdir, jacc_news, "Saliency Stability (Jaccard@5) · NEWS",
                     "Jaccard", "faith_jaccard_hist_news")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(placebo_keys)), placebo_mat.mean(0))
    ax.set_xticks(range(len(placebo_keys))); ax.set_xticklabels(placebo_keys)
    ax.set_ylabel("ΔCRPS vs baseline"); ax.set_title("Placebos NEWS (mean per window)")
    savefig(outdir, fig, "faith_placebos_news")
