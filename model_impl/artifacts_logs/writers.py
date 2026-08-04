"""
Artifact writers: every file a run leaves behind (figures, tables, json,
config snapshot) is written through here, so the on-disk format and naming
stay in one place. This is also the single hook point for MLflow artifact
logging — every write below reports itself to mlflow_tracker.log_artifact,
which is a no-op when tracking isn't active, so no caller here or upstream
needs to know MLflow exists.

TRACKING.LOCAL.use is likewise handled only here (set once via `configure`,
mirroring artifacts_logs.run_log's "install once" idiom): when it's off, no
directory or file is ever created on disk — figures/json/csv are logged to
MLflow straight from memory instead (mlflow_tracker.log_figure/log_dict/
log_text), so callers never need an `if cfg.tracking.local.use` check either.
`ensure_dir` is the one place directory creation happens, so plot/eval
modules that need a subdirectory (horizons/, forecasts/, validation/) also
route through here instead of calling Path.mkdir themselves.

Known gap: utils/evaluation_utils/faithfulness.py writes its own files
(np.save, open(...).write) directly, bypassing this module entirely — it
predates this toggle and isn't wired into it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_impl.consts import CONFIG_SNAPSHOT_FILE, FIG_DPI
from model_impl.utils.tracking_utils import mlflow_tracker

if TYPE_CHECKING:
    from model_impl.utils.schemas import LocalConfig, RunConfig

_local_enabled = True


def configure(local_cfg: LocalConfig) -> None:
    """
    Set the module-level local-saving toggle for the rest of the run.

    Parameters
    ----------
    local_cfg : LocalConfig
        `cfg.tracking.local` — `local_cfg.use` becomes `_local_enabled`.

    Returns
    -------
    None
        Called once from main(), before the first write of the run.
    """
    global _local_enabled
    _local_enabled = local_cfg.use


def ensure_dir(path: Path) -> None:
    """
    Create a directory on disk, unless local saving is disabled.

    Parameters
    ----------
    path : Path
        Directory to create (with parents), e.g. an `outdir` subfolder
        (horizons/, forecasts/, validation/).

    Returns
    -------
    None
        No-op when `_local_enabled` is False — nothing is created.
    """
    if _local_enabled:
        path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, fig: plt.Figure, name: str, mlflow_ignore: bool = False) -> None:
    """
    Save a matplotlib figure, locally and/or to MLflow depending on config.

    Parameters
    ----------
    path : Path
        Directory the PNG is saved into (only used when local saving is on).
    fig : plt.Figure
        The figure to save; closed before returning either way.
    name : str
        Filename without extension — saved/logged as "<name>.png".
    mlflow_ignore : bool, default False
        When True, this figure is never reported to MLflow (e.g. per-window
        plots too numerous to upload individually) — it still saves locally
        when local saving is on.

    Returns
    -------
    None
        When local saving is on: written to `<path>/<name>.png` on disk, and
        (unless `mlflow_ignore`) reported to MLflow via `log_artifact`.
        When local saving is off: nothing is written to disk; the figure is
        logged to MLflow directly from memory via `log_figure` (unless
        `mlflow_ignore`, in which case it is dropped entirely).
    """
    if _local_enabled:
        fpath = path / f"{name}.png"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, dpi=FIG_DPI, bbox_inches="tight")
        if not mlflow_ignore:
            mlflow_tracker.log_artifact(fpath)
    elif not mlflow_ignore:
        mlflow_tracker.log_figure(fig, f"{name}.png")
    plt.close(fig)


def write_json(outdir: Path, name: str, payload: dict) -> Path | None:
    """
    Write a dict as indented JSON, locally and/or to MLflow depending on config.

    Parameters
    ----------
    outdir : Path
        Directory the JSON file is saved into (only used when local saving is on).
    name : str
        Filename without extension — saved/logged as "<name>.json".
    payload : dict
        The data to serialize.

    Returns
    -------
    Path | None
        The written file's path when local saving is on (and reported to
        MLflow via `log_artifact`); `None` when local saving is off (payload
        is instead logged to MLflow directly from memory via `log_dict`).
    """
    if _local_enabled:
        fpath = outdir / f"{name}.json"
        fpath.write_text(json.dumps(payload, indent=2))
        mlflow_tracker.log_artifact(fpath)
        return fpath
    mlflow_tracker.log_dict(payload, f"{name}.json")
    return None


def write_csv(outdir: Path, name: str, df: "pd.DataFrame") -> Path | None:
    """
    Write a DataFrame as CSV, locally and/or to MLflow depending on config.

    Parameters
    ----------
    outdir : Path
        Directory the CSV file is saved into (only used when local saving is on).
    name : str
        Filename without extension — saved/logged as "<name>.csv".
    df : pd.DataFrame
        The table to write, without its index.

    Returns
    -------
    Path | None
        The written file's path when local saving is on (and reported to
        MLflow via `log_artifact`); `None` when local saving is off (the CSV
        text is instead logged to MLflow directly from memory via `log_text`).
    """
    if _local_enabled:
        fpath = outdir / f"{name}.csv"
        df.to_csv(fpath, index=False)
        mlflow_tracker.log_artifact(fpath)
        return fpath
    mlflow_tracker.log_text(df.to_csv(index=False), f"{name}.csv")
    return None


def save_config_snapshot(outdir: Path, cfg: RunConfig) -> None:
    """
    Dump the run's resolved RunConfig to disk/MLflow as config_snapshot.json.

    Parameters
    ----------
    outdir : Path
        Directory the snapshot is saved into (see `write_json`).
    cfg : RunConfig
        The fully-resolved config (yaml merged over each field's own schema
        default) that produced this run — so any output can always be traced
        back to the exact settings that generated it.

    Returns
    -------
    None
        Delegates to `write_json`; see its Returns for the local/MLflow split.
    """
    write_json(outdir, CONFIG_SNAPSHOT_FILE, dataclasses.asdict(cfg))
