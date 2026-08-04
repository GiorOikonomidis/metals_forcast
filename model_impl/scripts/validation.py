"""
Validation stage: the full metric suite on the VAL split, for model selection.
Same pipeline as the test stage, pointed at the validation windows; artifacts
land under <outdir>/validation so they never mix with the final test report.

Not part of main's default flow — a full evaluation runs MC_SAMPLES forward
passes per window, so main gates this stage behind run_validation_suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from model_impl.artifacts_logs import writers
from model_impl.consts import SUMMARY_FILE, VALIDATION_DIR
from model_impl.utils.evaluation_utils.eval_pipeline import aggregate, evaluate

if TYPE_CHECKING:
    import pandas as pd
    from chronos import ChronosPipeline

    from model_impl.utils.schemas import DataConfig, EvaluationConfig
    from model_impl.data_loading.splitting import Split
    from model_impl.data_loading.windowing import Windows
    from model_impl.models.cross_chronos import MultiCrossChronos


def run(model: MultiCrossChronos, chrono: ChronosPipeline,
        windows: Windows, split: Split, raw_series: pd.Series | None,
        outdir: Path, index: str, data_cfg: DataConfig, eval_cfg: EvaluationConfig) -> dict:
    """
    Score every validation window and persist the aggregated summary under
    <outdir>/validation. Returns the summary dict.
    """
    val_dir = outdir / VALIDATION_DIR
    writers.ensure_dir(val_dir)

    n_windows = len(windows.xe)

    metrics, fw_rows, pit_all = evaluate(
        model, chrono, windows, split, raw_series, val_dir, index, n_windows,
        data_cfg, eval_cfg,
    )
    summary = aggregate(metrics, data_cfg.pred_len)
    writers.write_json(val_dir, SUMMARY_FILE, summary)
    return summary


