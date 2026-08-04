"""
Shared evaluation pipeline: `evaluate` walks one split's windows, scores each
one and draws its forecast panel; `aggregate` reduces the per-window series to
the run-level summary. scripts/validation.py and scripts/test.py are thin
wrappers that point this at different splits.

Scoring functions live in metrics.py, sampling in inference.py, every figure
in plot_utils — this module only decides what gets computed and in which order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model_impl.artifacts_logs.writers import ensure_dir
from model_impl.consts import FORECASTS_DIR
from model_impl.utils.data_loader_utils.transforms import invert_diff
from model_impl.utils.evaluation_utils import faithfulness as faith
from model_impl.utils.evaluation_utils.inference import predict_distribution
from model_impl.utils.evaluation_utils.metrics import (
    build_ece_quantiles, coverage, dm_test, ece_quantiles, interval_score,
    loss_crps, mae, pinball_loss, pit_values, smape,
)
from model_impl.utils.logger_utils.logger import get_logger
from model_impl.utils.plot_utils.forecast_plots import (
    plot_attention_maps, plot_forecast_window,
)
from model_impl.utils.runtime_utils import DEVICE

logger = get_logger(__name__)

if TYPE_CHECKING:  # type-only upward imports — no runtime dependency on higher layers
    import pandas as pd
    from chronos import ChronosPipeline

    from model_impl.utils.schemas import DataConfig, EvaluationConfig
    from model_impl.data_loading.splitting import Split
    from model_impl.data_loading.windowing import Windows
    from model_impl.models.cross_chronos import MultiCrossChronos


# Per-window series accumulated by `evaluate` and reduced by `aggregate`.
_METRIC_KEYS = [
    "mse", "mae", "smape", "crps",
    "wis50", "wis80", "wis90",
    "cov50", "cov80", "cov90",
    "sharp80", "ece_q",
    "mse_naive", "mae_naive",
    "pinball_p10", "pinball_p50", "pinball_p90",
]


def evaluate(model: MultiCrossChronos, chrono: ChronosPipeline,
             windows: Windows, split: Split,
             raw_series: pd.Series | None, outdir: Path, index: str,
             n_windows: int, data_cfg: DataConfig, eval_cfg: EvaluationConfig,
             debug_vis: bool = False,
             faithfulness: bool = False) -> tuple[dict, list[dict], list[float]]:
    """
    Score every window of one split and draw its forecast panel.

    Predictions come out of the model in the space the target was tokenized in.
    When data_cfg.type_of_diff is differenced they are reconstructed to price
    levels via `invert_diff` against the last raw context price, and
    `raw_series` supplies both that anchor and the price-space truth — so
    every metric below is computed on real prices regardless of the target's
    representation.

    Returns
    -------
    metrics : dict[str, list[float]]  per-window series, keys as in _METRIC_KEYS
    fw_rows : list[dict]              one row per (window, horizon) forecast
    pit_all : list[float]             PIT values pooled over windows and horizons
    """
    ctx_len, pred_len, type_of_diff = data_cfg.ctx_len, data_cfg.pred_len, data_cfg.type_of_diff
    alpha_50 = eval_cfg.central_interval.alpha_50
    alpha_80 = eval_cfg.central_interval.alpha_80
    alpha_90 = eval_cfg.central_interval.alpha_90
    ece_grid = build_ece_quantiles(eval_cfg.ece_grid.start, eval_cfg.ece_grid.stop, eval_cfg.ece_grid.steps)

    metrics: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    fw_rows: list[dict] = []
    pit_all: list[float] = []

    # per-window forecast plots go here, named by forecast-start timestamp
    forecasts_dir = outdir / FORECASTS_DIR
    ensure_dir(forecasts_dir)

    for i in tqdm(range(n_windows), desc=f"Evaluating {n_windows} windows", unit="window"):
        # mimic batches in streaming
        ctx_eur  = windows.xe[i].unsqueeze(0).to(DEVICE)
        ctx_news = windows.xn[i].unsqueeze(0).to(DEVICE)
        ctx_covariate = windows.xc[i].unsqueeze(0).to(DEVICE)

        # [0..ctx_len]-->[ctx_len + i .. ctx_len + i + pred_len] , slide each time one day
        start = ctx_len + i
        end   = start + pred_len
        truth = split.prices.iloc[start:end].values

        # One forward to populate CrossBlock.last_weights for the debug/faithfulness views
        model.eval()
        with torch.no_grad():
            _ = model(ctx_eur, ctx_news, ctx_covariate, mc=False)

        # per-window scale — used to decode tokens back to real prices
        scale_win = windows.scales[i:i+1]  # (1,)

        if faithfulness:
            faith.run_window(model, chrono, outdir, window=i + 1, scale_win=scale_win,
                             ctx_eur=ctx_eur, ctx_news=ctx_news, ctx_covariate=ctx_covariate,
                             truth=truth, faith_cfg=eval_cfg.faith, mc_samples=eval_cfg.mc_samples)

        # Debug: attention maps for the very first window
        if debug_vis and i == 0:
            plot_attention_maps(
                outdir,
                weights_news=model.news_eur_q.last_weights.squeeze(0).mean(0).numpy(),
                weights_covariate=model.covariate_eur_q.last_weights.squeeze(0).mean(0).numpy(),
                index=index,
            )

        # MC-Dropout predictive distribution — decode predicted tokens to prices
        preds = predict_distribution(model, chrono, scale_win,
                                     ctx_eur, ctx_news, ctx_covariate, eval_cfg.mc_samples)

        # ── reconstruct to price space when the target is differenced ──
        # Keep diff-space copies for the dual plot; overwrite preds/truth/naive with
        # price levels so every metric below is computed on real prices.
        if type_of_diff != "no_diff":
            fc_dates     = split.prices.index[start:end]
            anchor       = float(raw_series.loc[split.prices.index[start - 1]])
            preds_diff   = preds
            median_diff  = np.quantile(preds_diff, 0.5, axis=0)
            low80_diff   = np.quantile(preds_diff, alpha_80 / 2, axis=0)
            high80_diff  = np.quantile(preds_diff, 1 - alpha_80 / 2, axis=0)
            truth_diff   = truth
            preds        = invert_diff(anchor, preds_diff, type_of_diff)   # (MC, H) price
            truth        = raw_series.loc[fc_dates].values                 # (H,)   price
            naive        = np.repeat(anchor, pred_len)                     # price persistence
        else:
            preds_diff = None
            naive = np.repeat(split.prices.iloc[start - 1], pred_len)

        # Central estimate
        median = np.quantile(preds, 0.5, axis=0)

        # Metrics (point)
        mse_val   = float(np.mean((median - truth)**2))
        mae_val   = mae(truth, median)
        smape_val = smape(truth, median)

        # Pinball losses @ 0.1 / 0.5 / 0.9
        q10 = np.quantile(preds, 0.1, axis=0)
        q50 = np.quantile(preds, 0.5, axis=0)
        q90 = np.quantile(preds, 0.9, axis=0)

        # Intervals & calibration
        low80, high80 = np.quantile(preds, alpha_80/2, axis=0), np.quantile(preds, 1-alpha_80/2, axis=0)
        low50, high50 = np.quantile(preds, alpha_50/2, axis=0), np.quantile(preds, 1-alpha_50/2, axis=0)
        low90, high90 = np.quantile(preds, alpha_90/2, axis=0), np.quantile(preds, 1-alpha_90/2, axis=0)

        # CRPS (proper)
        crps_val = loss_crps(preds, truth)
        if not np.isfinite(crps_val):
            logger.warning(
                "CRPS is not finite. Debug stats:"
                "\n   truth[min/mean/max]=%.6f/%.6f/%.6f"
                "\n   preds[min/mean/max]=%.6f/%.6f/%.6f",
                np.min(truth), np.mean(truth), np.max(truth),
                np.min(preds), np.mean(preds), np.max(preds),
            )

        # PIT values (for uniformity check)
        pit_all.extend(pit_values(truth, preds).tolist())

        # Push metrics
        metrics["mse"].append(mse_val)
        metrics["mae"].append(mae_val)
        metrics["smape"].append(smape_val)
        metrics["crps"].append(float(crps_val))
        metrics["wis50"].append(interval_score(truth, low50, high50, alpha=alpha_50))
        metrics["wis80"].append(interval_score(truth, low80, high80, alpha=alpha_80))
        metrics["wis90"].append(interval_score(truth, low90, high90, alpha=alpha_90))
        metrics["cov50"].append(coverage(truth, low50, high50))
        metrics["cov80"].append(coverage(truth, low80, high80))
        metrics["cov90"].append(coverage(truth, low90, high90))
        metrics["sharp80"].append(float(np.mean(high80 - low80)))
        metrics["ece_q"].append(ece_quantiles(truth, preds, ece_grid))
        metrics["mse_naive"].append(float(np.mean((naive - truth)**2)))
        metrics["mae_naive"].append(mae(truth, naive))
        metrics["pinball_p10"].append(pinball_loss(truth, q10, 0.1))
        metrics["pinball_p50"].append(pinball_loss(truth, q50, 0.5))
        metrics["pinball_p90"].append(pinball_loss(truth, q90, 0.9))

        # Per-window forecast plot: context history + actual vs predicted (+80% band).
        # When the target is differenced, add a top diff-space panel; the price panel
        # uses the raw series for context and the reconstructed price arrays.
        ctx_dates = split.prices.index[start-ctx_len:start]
        if type_of_diff != "no_diff":
            ctx_price_vals = raw_series.loc[ctx_dates].values
            diff_panel = {
                "ctx":    split.prices.iloc[start-ctx_len:start].values,
                "truth":  truth_diff,
                "median": median_diff,
                "low80":  low80_diff,
                "high80": high80_diff,
            }
        else:
            ctx_price_vals = split.prices.iloc[start-ctx_len:start].values
            diff_panel = None

        plot_forecast_window(
            forecasts_dir,
            ctx_dates=ctx_dates,
            ctx_prices=ctx_price_vals,
            fc_dates=split.prices.index[start:end],
            truth=truth, median=median, low80=low80, high80=high80,
            title=f"{index} · start {split.prices.index[start].date()} · MSE={mse_val:.2e}",
            diff_panel=diff_panel,
        )

        # Save per-window forecasts (aggregated into a single CSV later)
        for h in range(pred_len):
            fw_rows.append({
                "window": i+1,
                "horizon": h+1,
                "t": split.prices.index[start+h].strftime("%Y-%m-%d"),
                "truth": float(truth[h]),
                "median": float(median[h]),
                "low80": float(low80[h]),
                "high80": float(high80[h]),
                "q10": float(q10[h]),
                "q50": float(q50[h]),
                "q90": float(q90[h]),
                "naive": float(naive[h]),
            })

    return metrics, fw_rows, pit_all


def aggregate(metrics: dict[str, list[float]], pred_len: int) -> dict:
    """
    Reduce the per-window series to run-level means, skill scores against the
    naïve random walk, and the Diebold-Mariano test on per-window MSE.
    """
    mean = {k: float(np.mean(v)) for k, v in metrics.items()}

    mse_skill = 1.0 - (mean["mse"] / (mean["mse_naive"] + 1e-12))
    mae_skill = 1.0 - (mean["mae"] / (mean["mae_naive"] + 1e-12))
    dm_stat, dm_p = dm_test(np.array(metrics["mse"]), np.array(metrics["mse_naive"]), h=pred_len)

    return {
        "mse": mean["mse"], "mae": mean["mae"], "smape": mean["smape"],
        "crps": mean["crps"],
        "wis50": mean["wis50"], "wis80": mean["wis80"], "wis90": mean["wis90"],
        "cov50": mean["cov50"], "cov80": mean["cov80"], "cov90": mean["cov90"],
        "sharp80": mean["sharp80"], "ece_quantiles": mean["ece_q"],
        "mse_naive": mean["mse_naive"], "mae_naive": mean["mae_naive"],
        "mse_skill_vs_naive": float(mse_skill),
        "mae_skill_vs_naive": float(mae_skill),
        "dm_test": {"stat": dm_stat, "p_value": dm_p, "h": pred_len},
    }
