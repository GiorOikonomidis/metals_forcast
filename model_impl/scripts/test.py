"""
Test stage: final evaluation on the untouched TEST split — scores every window,
aggregates, prints the run banner and persists every table and figure the run
reports on.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd

from model_impl.artifacts_logs import writers
from model_impl.consts import FORECASTS_BY_WINDOW_FILE, METRICS_PER_WINDOW_FILE, SUMMARY_FILE
from model_impl.utils.evaluation_utils import faithfulness as faith
from model_impl.utils.evaluation_utils.eval_pipeline import aggregate, evaluate
from model_impl.utils.evaluation_utils.metrics import (
    build_ece_quantiles, coverage, mae, pinball_loss, smape,
)
from model_impl.utils.logger_utils.logger import get_logger
from model_impl.utils.plot_utils import calibration_plots, forecast_plots
from model_impl.utils.tracking_utils import mlflow_tracker

logger = get_logger(__name__)

if TYPE_CHECKING:
    from chronos import ChronosPipeline

    from model_impl.utils.schemas import DataConfig, EvaluationConfig
    from model_impl.data_loading.splitting import Split
    from model_impl.data_loading.windowing import Windows
    from model_impl.models.cross_chronos import MultiCrossChronos


def print_summary(summary: dict, n_windows: int, n_covariates: int,
                  no_news: bool, elapsed: float) -> None:
    m = summary
    logger.info(
        "\n# ✅ Evaluated %d windows with %d covariates%s"
        "\n#  • MSE    : %.4e (skill vs naïve = %.3f)"
        "\n#  • MAE    : %.4e (skill vs naïve = %.3f)"
        "\n#  • sMAPE  : %.2f%%"
        "\n#  • CRPS   : %.4e"
        "\n#  • WIS50  : %.4e | WIS80: %.4e | WIS90: %.4e"
        "\n#  • Cov@50 : %.3f | Cov@80: %.3f | Cov@90: %.3f"
        "\n#  • Sharp80: %.4e"
        "\n#  • ECE(q) : %.4e"
        "\n#  • DM test (MSE vs naïve): stat=%.3f, p=%.4f"
        "\n# ⏱ Total time: %s\n",
        n_windows, n_covariates, ' and no news' if no_news else '',
        m['mse'], m['mse_skill_vs_naive'],
        m['mae'], m['mae_skill_vs_naive'],
        m['smape'],
        m['crps'],
        m['wis50'], m['wis80'], m['wis90'],
        m['cov50'], m['cov80'], m['cov90'],
        m['sharp80'],
        m['ece_quantiles'],
        m['dm_test']['stat'], m['dm_test']['p_value'],
        time.strftime('%H:%M:%S', time.gmtime(elapsed)),
    )


def run(model: MultiCrossChronos, chrono: ChronosPipeline,
        windows: Windows, split: Split, raw_series: pd.Series | None,
        outdir: Path, index: str, n_covariates: int, no_news: bool, t0: float,
        data_cfg: DataConfig, eval_cfg: EvaluationConfig,
        debug_vis: bool = False, faithfulness_on: bool = False) -> dict:
    """
    Evaluate the test windows, print the metrics banner, and persist all
    artifacts (per-window CSV, forecast CSV, summary.json, every figure).
    Returns the aggregated metrics dict.
    """
    n_windows = len(windows.xe)
    if eval_cfg.windows is not None:
        n_windows = min(n_windows, eval_cfg.windows)

    metrics, fw_rows, pit_all = evaluate(
        model, chrono, windows, split, raw_series, outdir, index,
        n_windows, data_cfg, eval_cfg, debug_vis=debug_vis, faithfulness=faithfulness_on,
    )

    summary_metrics = aggregate(metrics, data_cfg.pred_len)
    print_summary(summary_metrics, n_windows, n_covariates, no_news, time.time() - t0)

    # MLflow: the run-level headline metrics used to rank runs in the UI. Kept
    # to this handful (point error + skill-vs-naive + smape) deliberately — the
    # full metric suite still lands on disk in summary.json / the per-window CSV.
    mlflow_tracker.log_metrics({
        k: summary_metrics[k] for k in
        ("mse", "mae", "mse_skill_vs_naive", "mae_skill_vs_naive", "smape")
    })

    # tables
    per_window_df = pd.DataFrame({"window": np.arange(1, n_windows + 1), **metrics})
    writers.write_csv(outdir, METRICS_PER_WINDOW_FILE, per_window_df)
    forecasts_df = pd.DataFrame(fw_rows)
    writers.write_csv(outdir, FORECASTS_BY_WINDOW_FILE, forecasts_df)

    #[!!] to muchh log all those ....
    """
    # MLflow: metrics grouped by horizon step (not by window — see plan). Only
    # the metrics computable from stored quantiles (low80/high80/q10/q50/q90)
    # are available here; loss_crps/ece_quantiles/pit_values need the full MC
    # sample array, which only exists transiently inside evaluate() and isn't
    # carried in fw_rows, so they can't be broken out per horizon.
    for h, group in forecasts_df.groupby("horizon"):
        truth = group["truth"].to_numpy()
        mlflow_tracker.log_metrics({
            f"mae_h{h}": mae(truth, group["median"].to_numpy()),
            f"smape_h{h}": smape(truth, group["median"].to_numpy()),
            f"coverage80_h{h}": coverage(truth, group["low80"].to_numpy(), group["high80"].to_numpy()),
            f"pinball_p50_h{h}": pinball_loss(truth, group["q50"].to_numpy(), 0.5),
        })
    """
    summary = {
        "index": index,
        "pred_len": data_cfg.pred_len,
        "ctx_len": data_cfg.ctx_len,
        "covariates": n_covariates,
        "windows_evaluated": n_windows,
        "news_mode": "disabled" if no_news else "active",
        "metrics": summary_metrics,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    writers.write_json(outdir, SUMMARY_FILE, summary)

    # figures
    ece_grid = build_ece_quantiles(eval_cfg.ece_grid.start, eval_cfg.ece_grid.stop, eval_cfg.ece_grid.steps)
    forecast_plots.plot_horizon_forecasts(outdir, fw_rows, data_cfg.pred_len, index)
    calibration_plots.plot_metric_per_window(outdir, metrics["mse"], "MSE", "mse_per_window")
    calibration_plots.plot_metric_per_window(outdir, metrics["wis80"], "WIS@80", "wis80_per_window")
    calibration_plots.plot_metric_per_window(outdir, metrics["crps"], "CRPS", "crps_per_window")
    calibration_plots.plot_coverage_summary(outdir, summary_metrics["cov50"],
                                            summary_metrics["cov80"], summary_metrics["cov90"])
    calibration_plots.plot_reliability_curve(outdir, forecasts_df, ece_grid)
    calibration_plots.plot_pit_histogram(outdir, pit_all)
    calibration_plots.plot_skill_vs_naive(outdir, summary_metrics["mse_skill_vs_naive"],
                                          summary_metrics["mae_skill_vs_naive"])
    calibration_plots.plot_pinball_summary(outdir, metrics["pinball_p10"],
                                           metrics["pinball_p50"], metrics["pinball_p90"])

    if faithfulness_on:
        faith.aggregate(outdir)

    return summary_metrics
