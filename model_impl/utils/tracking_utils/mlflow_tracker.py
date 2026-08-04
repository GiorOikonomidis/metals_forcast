"""
MLflow experiment tracking: a thin, no-op-when-disabled wrapper so no call
site outside this module needs an `if cfg.mlflow.use` check.

Mirrors the "install once, everything else is a plain function" idiom from
artifacts_logs.run_log (`_installed`) — here it's `_active`, set for the
duration of the `start_run` context manager and checked by every other
function in this module. `start_run` itself is the one place that branches
on `cfg.use`; everywhere else just calls these functions unconditionally.
"""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.pytorch

from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from model_impl.utils.schemas import MLflowConfig, RunConfig

_active = False

_MAX_PARAM_LEN = 500  # MLflow's param value length limit


@contextlib.contextmanager
def start_run(cfg: MLflowConfig, default_run_name: str):
    """
    Open an MLflow run for the duration of the `with` block, or do nothing
    at all if `cfg.use` is False. Every other function in this module reads
    the module-level `_active` flag this sets, so call sites never branch
    on `cfg.use` themselves.
    """
    global _active
    if not cfg.use:
        yield
        return

    if cfg.uri:
        mlflow.set_tracking_uri(cfg.uri)
    if cfg.experiment:
        mlflow.set_experiment(cfg.experiment)

    _active = True
    try:
        with mlflow.start_run(run_name=cfg.run_name or default_run_name):
            yield
    finally:
        _active = False


def _flatten(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """
    Flatten a nested dict (from dataclasses.asdict(RunConfig)) into dotted
    param names, dropping list/dict/None leaves. Fields like `global_covariates`
    (the covariate column list) or `dm_test` stay fully captured in
    config_snapshot.json, which is already logged as an artifact — they don't
    belong as flat MLflow params.
    """
    flat: dict[str, Any] = {}
    for key, val in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(_flatten(val, name))
        elif isinstance(val, (list, tuple)) or val is None:
            continue
        else:
            flat[name] = val
    return flat


def log_params(cfg: RunConfig) -> None:
    """Flatten and log every scalar field of the resolved RunConfig."""
    if not _active:
        return
    flat = _flatten(dataclasses.asdict(cfg))
    flat = {k: (v[:_MAX_PARAM_LEN] if isinstance(v, str) else v) for k, v in flat.items()}
    mlflow.log_params(flat)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a batch of scalar metrics, optionally at a given step."""
    if not _active:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path) -> None:
    """Log a single already-written file as an MLflow artifact."""
    if not _active:
        return
    mlflow.log_artifact(str(path))


def log_figure(fig, artifact_file: str) -> None:
    """
    Log a matplotlib figure straight from memory — used when TRACKING.LOCAL.use
    is off, so no local file ever needs to exist for the artifact to reach MLflow.
    """
    if not _active:
        return
    mlflow.log_figure(fig, artifact_file)


def log_dict(dictionary: dict, artifact_file: str) -> None:
    """Log a dict as a JSON artifact straight from memory (see log_figure)."""
    if not _active:
        return
    mlflow.log_dict(dictionary, artifact_file)


def log_text(text: str, artifact_file: str) -> None:
    """Log a string as a text/CSV artifact straight from memory (see log_figure)."""
    if not _active:
        return
    mlflow.log_text(text, artifact_file)


def log_model(model, artifact_path: str = "model") -> None:
    """
    Log the trained model. This is the first persistence of trained weights
    anywhere in the pipeline — scripts/training.py only ever keeps
    `best_state` in memory (`model.load_state_dict(best_state)`), never on disk.
    """
    if not _active:
        return
    mlflow.pytorch.log_model(model, artifact_path)
