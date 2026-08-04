"""
Saliency faithfulness study (enabled by the faithfulness flag in main).

Asks whether the cross-attention weights the model puts on the news/covariate
streams actually explain its forecast: deletion/insertion curves, leave-one-
timestep-out CRPS deltas ranked against saliency, top-k stability under MC
dropout, and time-shift/shuffle placebos.

Scored in whatever space the target was tokenized in — for a differenced
type_of_diff that is diff space, not the price space the summary.json metrics
use, so the CRPS numbers here are not comparable with those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.stats import spearmanr

from model_impl.consts import FAITH_PER_WINDOW_FILE, FAITH_SUMMARY_FILE
from model_impl.utils.evaluation_utils.inference import predict_distribution
from model_impl.utils.evaluation_utils.metrics import loss_crps
from model_impl.utils.plot_utils.faithfulness_plots import (
    plot_faith_aggregates, plot_faith_curves_window,
)

if TYPE_CHECKING:  # type-only imports — the model/chronos stack stays a runtime argument
    from chronos import ChronosPipeline

    from model_impl.utils.schemas import FaithConfig
    from model_impl.models.cross_chronos import MultiCrossChronos


# ──────────────────────────────────────────────────────────────────────────────
# Saliency & masking primitives
# ──────────────────────────────────────────────────────────────────────────────
def last_step_temporal_saliency(attn_weights: torch.Tensor) -> np.ndarray:
    """
    Extract normalised temporal importance vector from cross-attention weights.

    Parameters
    ----------
    attn_weights : torch.Tensor  (1, heads, Q, K)  from CrossBlock.last_weights (CPU)

    Returns
    -------
    np.ndarray  (K,) float32 — importance of each key timestep for the last query step
    """
    w = attn_weights.squeeze(0).mean(0)  # (Q, K)
    s = w[-1]                            # (K,) — last EUR query step
    s = s / (s.sum() + 1e-9)
    return s.numpy().astype(np.float32)


def mask_timesteps(X: torch.Tensor, idxs: list[int], strategy: str,
                   cached_rep: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace selected timestep positions in X with a baseline value.

    Parameters
    ----------
    X          : torch.FloatTensor  (1, ctx, D)  news or covariate stream for one window
    idxs       : list[int]          temporal positions to mask
    strategy   : str                "mean" (replace with sequence mean) or "zero"
    cached_rep : torch.Tensor | None  precomputed replacement tensor (1, ctx, D)

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]  (masked X, replacement tensor used)
    """
    X2 = X.clone()
    if cached_rep is None:
        if strategy == "mean":
            rep = X.mean(dim=1, keepdim=True).expand(-1, X.shape[1], -1)
        elif strategy == "zero":
            rep = torch.zeros_like(X)
        else:
            raise ValueError("strategy must be 'mean' or 'zero'")
    else:
        rep = cached_rep
    if len(idxs) > 0:
        X2[:, idxs, :] = rep[:, idxs, :]
    return X2, rep


def jaccard_topk(sets: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """
    Pairwise Jaccard similarity over multiple top-k index sets.

    Parameters
    ----------
    sets : list of arrays, each containing the top-k timestep indices from one MC run

    Returns
    -------
    tuple[np.ndarray, float]  (J matrix (m, m), mean of upper-triangle entries)
    """
    m = len(sets)
    J = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            a, b = set(sets[i]), set(sets[j])
            inter = len(a & b); uni = len(a | b)
            J[i, j] = inter / max(uni, 1)
    tri = J[np.triu_indices(m, 1)]
    return J, float(tri.mean()) if len(tri) else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Curves
# ──────────────────────────────────────────────────────────────────────────────
def deletion_insertion_curves(model: MultiCrossChronos, chrono: ChronosPipeline,
                              scale_win: torch.Tensor, ctx_eur: torch.Tensor,
                              ctx_news: torch.Tensor, ctx_covariate: torch.Tensor,
                              truth: np.ndarray, saliency_vec: np.ndarray,
                              ks: list[int], strategy: str, mc_samples: int,
                              rng_seed: int, stream: str = "news",
                              device: str = "cpu") -> dict:
    """
    Compute deletion and insertion curves for faithfulness evaluation.

    Masks the top-k, random-k, and least-k salient timesteps and measures
    the change in CRPS relative to the unmasked baseline.

    Parameters
    ----------
    stream       : "news" or "covariate"
    ks           : list of absolute step counts to mask/unmask
    saliency_vec : (ctx,) float32 importance weights from last_step_temporal_saliency

    Returns
    -------
    dict with keys: base_crps, ks, del_curve, del_curve_rand, del_curve_inv, ins_curve, order
    """
    base_preds = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate, mc_samples)
    base_crps = loss_crps(base_preds, truth)

    order = np.argsort(-saliency_vec)             # top → bottom
    order_inv = np.argsort(saliency_vec)          # bottom → top
    rng = np.random.default_rng(rng_seed)

    del_curve, del_curve_rand, del_curve_inv = [], [], []
    ins_curve = []

    # Precalcular reemplazos
    cached_rep_news = cached_rep_covariate = None
    if stream == "news":
        _, cached_rep_news = mask_timesteps(ctx_news, [], strategy)
    else:
        _, cached_rep_covariate = mask_timesteps(ctx_covariate, [], strategy)

    # Deletion: ir sumando top-k enmascarados
    for k in ks:
        idxs_topk = order[:k].tolist()
        idxs_rand = rng.choice(len(saliency_vec), size=k, replace=False).tolist()
        idxs_invk = order_inv[:k].tolist()

        if stream == "news":
            Xn_topk, _ = mask_timesteps(ctx_news, idxs_topk, strategy, cached_rep_news)
            Xn_rand, _ = mask_timesteps(ctx_news, idxs_rand, strategy, cached_rep_news)
            Xn_invk, _ = mask_timesteps(ctx_news, idxs_invk, strategy, cached_rep_news)
            preds_topk = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_topk, ctx_covariate, mc_samples)
            preds_rand = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_rand, ctx_covariate, mc_samples)
            preds_invk = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_invk, ctx_covariate, mc_samples)
        else:
            Xc_topk, _ = mask_timesteps(ctx_covariate, idxs_topk, strategy, cached_rep_covariate)
            Xc_rand, _ = mask_timesteps(ctx_covariate, idxs_rand, strategy, cached_rep_covariate)
            Xc_invk, _ = mask_timesteps(ctx_covariate, idxs_invk, strategy, cached_rep_covariate)
            preds_topk = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_topk, mc_samples)
            preds_rand = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_rand, mc_samples)
            preds_invk = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_invk, mc_samples)

        del_curve.append(loss_crps(preds_topk, truth) - base_crps)
        del_curve_rand.append(loss_crps(preds_rand, truth) - base_crps)
        del_curve_inv.append(loss_crps(preds_invk, truth) - base_crps)

    # Insertion: partir de TODO enmascarado y "devolver" top-k
    if stream == "news":
        Xn_all_mask, _ = mask_timesteps(ctx_news, list(range(len(saliency_vec))), strategy, cached_rep_news)
        base_ins = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_all_mask, ctx_covariate, mc_samples)
        base_ins_crps = loss_crps(base_ins, truth)
        for k in ks:
            idxs_keep = order[:k].tolist()
            Xn_k = Xn_all_mask.clone()
            Xn_k[:, idxs_keep, :] = ctx_news[:, idxs_keep, :]
            preds_k = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_k, ctx_covariate, mc_samples)
            ins_curve.append(base_ins_crps - loss_crps(preds_k, truth))  # mejora respecto a "todo enmascarado"
    else:
        Xc_all_mask, _ = mask_timesteps(ctx_covariate, list(range(len(saliency_vec))), strategy, cached_rep_covariate)
        base_ins = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_all_mask, mc_samples)
        base_ins_crps = loss_crps(base_ins, truth)
        for k in ks:
            idxs_keep = order[:k].tolist()
            Xc_k = Xc_all_mask.clone()
            Xc_k[:, idxs_keep, :] = ctx_covariate[:, idxs_keep, :]
            preds_k = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_k, mc_samples)
            ins_curve.append(base_ins_crps - loss_crps(preds_k, truth))

    return {
        "base_crps": base_crps,
        "ks": ks,
        "del_curve": del_curve,
        "del_curve_rand": del_curve_rand,
        "del_curve_inv": del_curve_inv,
        "ins_curve": ins_curve,
        "order": order,
    }


def loto_deltas(model: MultiCrossChronos, chrono: ChronosPipeline,
                scale_win: torch.Tensor, ctx_eur: torch.Tensor,
                ctx_news: torch.Tensor, ctx_covariate: torch.Tensor,
                truth: np.ndarray, strategy: str, mc_samples: int,
                stream: str = "news") -> tuple[np.ndarray, float]:
    """
    Leave-One-Timestep-Out CRPS deltas.

    Masks each context timestep individually and records the change in CRPS
    relative to the full-context baseline.

    Returns
    -------
    tuple[np.ndarray, float]  (deltas of shape (ctx,), base_crps)
    """
    base_preds = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate, mc_samples)
    base_crps = loss_crps(base_preds, truth)

    ctx = ctx_news.shape[1] if stream == "news" else ctx_covariate.shape[1]
    deltas = np.zeros(ctx, dtype=np.float32)
    for t in range(ctx):
        if stream == "news":
            Xn_t, _ = mask_timesteps(ctx_news, [t], strategy)
            preds_t = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_t, ctx_covariate, mc_samples)
        else:
            Xc_t, _ = mask_timesteps(ctx_covariate, [t], strategy)
            preds_t = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, Xc_t, mc_samples)
        deltas[t] = loss_crps(preds_t, truth) - base_crps
    return deltas, base_crps


# ──────────────────────────────────────────────────────────────────────────────
# Per-window driver
# ──────────────────────────────────────────────────────────────────────────────
def run_window(model: MultiCrossChronos, chrono: ChronosPipeline, outdir: Path,
               window: int, scale_win: torch.Tensor, ctx_eur: torch.Tensor,
               ctx_news: torch.Tensor, ctx_covariate: torch.Tensor,
               truth: np.ndarray, faith_cfg: FaithConfig, mc_samples: int) -> None:
    """
    Run the full study for one evaluation window and append a record to
    faithfulness_per_window.jsonl. Requires that a forward pass has already
    populated the model's CrossBlock.last_weights for this window.

    `window` is 1-based to match the saliency_*_w{n}.npy filenames.
    `mc_samples` is EvaluationConfig.mc_samples (the main eval loop's MC count);
    `faith_cfg.mc_samples` is capped against it since this study runs many more
    forward passes per window than the main loop.
    """
    mc = min(faith_cfg.mc_samples, mc_samples)
    strategy = faith_cfg.mask_strategy

    # Temporal saliency (last EUR step → news/covariate keys)
    s_news = last_step_temporal_saliency(model.eur_news_q.last_weights)   # (CTX_LEN,)
    s_covariate = last_step_temporal_saliency(model.eur_covariate_q.last_weights)   # (CTX_LEN,)
    np.save(outdir / f"saliency_news_w{window}.npy", s_news)
    np.save(outdir / f"saliency_covariate_w{window}.npy", s_covariate)

    # Deletion/Insertion curves (CRPS)
    curves_news = deletion_insertion_curves(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate,
                                            truth, s_news, faith_cfg.ks, strategy, mc,
                                            faith_cfg.rng_seed, stream="news")
    curves_covariate = deletion_insertion_curves(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate,
                                            truth, s_covariate, faith_cfg.ks, strategy, mc,
                                            faith_cfg.rng_seed, stream="covariate")

    # LOTO and saliency-vs-ΔCRPS rank correlation
    deltas_news, base_crps = loto_deltas(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate,
                                         truth, strategy, mc, stream="news")
    deltas_covariate, _         = loto_deltas(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate,
                                         truth, strategy, mc, stream="covariate")
    rho_news, _ = spearmanr(s_news, deltas_news)
    rho_covariate, _ = spearmanr(s_covariate, deltas_covariate)

    # Saliency stability (Jaccard@k) under MC-dropout on the attention weights
    sal_sets = []
    model.eval()
    model.mc_dropout(True)   # dropout ON for trainable layers; frozen encoder stays eval
    for _mc in range(faith_cfg.stability_runs):
        _ = model(ctx_eur, ctx_news, ctx_covariate, mc=True)
        s_news_mc = last_step_temporal_saliency(model.eur_news_q.last_weights)
        sal_sets.append(np.argsort(-s_news_mc)[:faith_cfg.topk])
    model.mc_dropout(False)  # restore full eval
    _, jaccard_mean = jaccard_topk(sal_sets)

    # Placebos: time-shift and shuffle on the news stream
    placebo = {}
    base_preds = predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, ctx_covariate, mc)
    base_crps_placebo = loss_crps(base_preds, truth)
    for d in faith_cfg.placebo_shifts:
        Xn_shift = torch.roll(ctx_news, shifts=d, dims=1)
        preds_s  = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_shift, ctx_covariate, mc)
        placebo[f"shift_{d}"] = loss_crps(preds_s, truth) - base_crps_placebo
    perm = torch.randperm(ctx_news.shape[1])
    Xn_shuf = ctx_news[:, perm, :]
    preds_u = predict_distribution(model, chrono, scale_win, ctx_eur, Xn_shuf, ctx_covariate, mc)
    placebo["shuffle"] = loss_crps(preds_u, truth) - base_crps_placebo

    rec = {
        "window": int(window),
        "base_crps": float(base_crps),
        "spearman_news": float(rho_news),
        "spearman_covariate": float(rho_covariate),
        "jaccard_top5_news": float(jaccard_mean),
        "curves_news": {k: (v if isinstance(v, list) else float(v)) for k, v in curves_news.items() if k != "order"},
        "curves_covariate": {k: (v if isinstance(v, list) else float(v)) for k, v in curves_covariate.items() if k != "order"},
        "placebo": {k: float(v) for k, v in placebo.items()},
    }
    with open(outdir / FAITH_PER_WINDOW_FILE, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    if window == 1:
        plot_faith_curves_window(outdir, curves_news)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-window aggregation
# ──────────────────────────────────────────────────────────────────────────────
def aggregate(outdir: Path) -> None:
    """
    Read back faithfulness_per_window.jsonl, draw the cross-window figures and
    write faithfulness_summary.json. No-op if the run produced no records.
    """
    faith_path = outdir / FAITH_PER_WINDOW_FILE
    if not faith_path.exists():
        return

    recs = [json.loads(line) for line in open(faith_path)]

    spearman_news = [r["spearman_news"] for r in recs if not np.isnan(r["spearman_news"])]
    spearman_covariate = [r["spearman_covariate"] for r in recs if not np.isnan(r["spearman_covariate"])]
    jacc_news     = [r["jaccard_top5_news"] for r in recs]

    # Deletion/insertion curve means over NEWS (ks assumed identical across windows)
    ks = recs[0]["curves_news"]["ks"]
    del_mat      = np.array([r["curves_news"]["del_curve"] for r in recs])
    del_rand_mat = np.array([r["curves_news"]["del_curve_rand"] for r in recs])
    del_inv_mat  = np.array([r["curves_news"]["del_curve_inv"] for r in recs])
    ins_mat      = np.array([r["curves_news"]["ins_curve"] for r in recs])

    placebo_keys = list(recs[0]["placebo"].keys())
    placebo_mat  = np.array([[r["placebo"][k] for k in placebo_keys] for r in recs])

    plot_faith_aggregates(outdir, ks, del_mat, del_rand_mat, del_inv_mat, ins_mat,
                          spearman_news, spearman_covariate, jacc_news,
                          placebo_keys, placebo_mat)

    faith_summary = {
        "spearman_news_mean": float(np.mean(spearman_news)),
        "spearman_news_median": float(np.median(spearman_news)),
        "spearman_covariate_mean": float(np.mean(spearman_covariate)),
        "jaccard_top5_news_mean": float(np.mean(jacc_news)),
        "placebos_mean": {k: float(np.mean(placebo_mat[:, j])) for j, k in enumerate(placebo_keys)},
    }
    (outdir / FAITH_SUMMARY_FILE).write_text(json.dumps(faith_summary, indent=2))
