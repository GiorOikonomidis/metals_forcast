"""
Config-file parsing: turns the --config yaml into a typed RunConfig.

Every dataclass field in utils/schemas carries its own default value, so this
module never falls back to consts.py or any other shared "defaults" store —
it only ever passes the keys a yaml section actually specifies. Anything the
yaml omits is left unset, and the dataclass's own default applies. A missing
file, an empty file, or a file missing whole sections all degrade the same
way: everything just falls through to those in-schema defaults.
"""

from __future__ import annotations

import yaml

from model_impl.utils.schemas import (
    CentralIntervalConfig, CrossChronosConfig, DataConfig,
    ECEGridConfig, EarlyStopperConfig, EvaluationConfig, FaithConfig,
    LocalConfig, MLflowConfig, ModelConfig, RunConfig, SchedulerConfig,
    SplitsConfig, TargetConfig, TrackingConfig, TrainingConfig,
)
from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)


def _section(raw: dict, key: str) -> dict:
    """A yaml section, or {} if the section (or the whole file) is absent."""
    return (raw or {}).get(key) or {}


def _pick(section: dict, key_map: dict[str, str]) -> dict:
    """
    Build a kwargs dict from `section`, keyed by the target dataclass field
    name, keeping only yaml keys that are actually present. Anything omitted
    is simply not passed, so the dataclass's own default applies to it.
    """
    return {py_name: section[yaml_key] for yaml_key, py_name in key_map.items() if yaml_key in section}


def _build_data(raw: dict) -> DataConfig:
    d = _section(raw, "DATA")
    target = _section(d, "TARGET")
    splits = _section(d, "SPLITS")

    kwargs = _pick(d, {
        "COVARIATES": "covariates", "GLOBAL_COVARIATES": "global_covariates",
        "NEWS_COL": "news_col",
        "PRED_LEN": "pred_len", "CTX_LEN": "ctx_len",
        "TOKEN_ALL": "token_all", "SHUFFLE_DATA": "shuffle_data",
        "TYPE_OF_DIFF": "type_of_diff",
    })
    target_kwargs = _pick(target, {"ID": "id", "FEATURE": "feature"})
    splits_kwargs = _pick(splits, {"TEST_DAYS": "test_days", "VAL_DAYS": "val_days"})
    if target_kwargs:
        kwargs["target"] = TargetConfig(**target_kwargs)
    if splits_kwargs:
        kwargs["splits"] = SplitsConfig(**splits_kwargs)
    return DataConfig(**kwargs)


def _build_model(raw: dict) -> ModelConfig:
    m = _section(raw, "MODEL")
    cc = _section(m, "CROSS_CHRONOS").get("configs") or {}

    kwargs = _pick(m, {"COMP_ENC": "comp_enc"})
    cc_kwargs = _pick(cc, {
        "EMB_DIM_NEWS": "emb_dim_news", "D_MODEL": "d_model", "N_HEADS": "n_heads",
        "N_LAYERS_TXT": "n_layers_txt", "D_FF": "d_ff", "DROPOUT": "dropout",
        "HEAD": "head",
    })
    if cc_kwargs:
        kwargs["cross_chronos"] = CrossChronosConfig(**cc_kwargs)
    return ModelConfig(**kwargs)


def _build_training(raw: dict) -> TrainingConfig:
    t = _section(raw, "TRAINING")
    kwargs = _pick(t, {
        "EPOCHS": "epochs", "LR": "lr", "WEIGHT_DECAY": "weight_decay", "GRAD_CLIP": "grad_clip",
        "BATCH_SIZE": "batch_size", "LABEL_SMOOTHING": "label_smoothing",
    })
    return TrainingConfig(**kwargs)


def _build_evaluation(raw: dict) -> EvaluationConfig:
    e = _section(raw, "EVALUATION")
    ci = _section(e, "CENTRAL_INTERVAL")
    grid = _section(e, "ECE_QUANTILE_GRID")
    faith = _section(e, "FAITH")

    kwargs = _pick(e, {"WINDOWS": "windows", "MC_SAMPLES": "mc_samples"})
    ci_kwargs = _pick(ci, {"ALPHA_50": "alpha_50", "ALPHA_80": "alpha_80", "ALPHA_90": "alpha_90"})
    grid_kwargs = _pick(grid, {
        "ECE_Q_START": "start", "ECE_Q_STOP": "stop", "ECE_Q_STEPS": "steps",
    })
    faith_kwargs = _pick(faith, {
        "FAITH_MC_SAMPLES": "mc_samples", "FAITH_KS": "ks", "FAITH_TOPK": "topk",
        "FAITH_STABILITY_RUNS": "stability_runs", "FAITH_PLACEBO_SHIFTS": "placebo_shifts",
        "FAITH_MASK_STRATEGY": "mask_strategy", "FAITH_RNG_SEED": "rng_seed",
    })
    if ci_kwargs:
        kwargs["central_interval"] = CentralIntervalConfig(**ci_kwargs)
    if grid_kwargs:
        kwargs["ece_grid"] = ECEGridConfig(**grid_kwargs)
    if faith_kwargs:
        kwargs["faith"] = FaithConfig(**faith_kwargs)
    return EvaluationConfig(**kwargs)


def _build_scheduler(raw: dict) -> SchedulerConfig:
    s = _section(raw, "SCHEDULER")
    c = _section(s, "configs")
    kwargs = _pick(s, {"use": "use"})
    kwargs.update(_pick(c, {
        "LR_COLD_START": "cold_start", "LR_COLD_EPOCHS": "cold_epochs",
        "LR_DECREASE_FACTOR": "decrease_factor",
        "LR_METRIC": "metric", "LR_PLATEAU_PATIENCE": "patience",
    }))
    return SchedulerConfig(**kwargs)


def _build_early_stopper(raw: dict) -> EarlyStopperConfig:
    es = _section(raw, "EARLY_STOPPER")
    c = _section(es, "configs")
    kwargs = _pick(es, {"use": "use"})
    kwargs.update(_pick(c, {"EARLY_STOPPER_PATIENCE": "patience"}))
    return EarlyStopperConfig(**kwargs)


def _build_mlflow(section: dict) -> MLflowConfig:
    c = _section(section, "configs")
    kwargs = _pick(section, {"use": "use"})
    kwargs.update(_pick(c, {
        "URI": "uri", "EXPIREMENT": "experiment", "RUN": "run_name", "LOG_MODEL": "log_model",
    }))
    return MLflowConfig(**kwargs)


def _build_local(section: dict) -> LocalConfig:
    c = _section(section, "configs")
    kwargs = _pick(section, {"use": "use"})
    kwargs.update(_pick(c, {"dir": "dir"}))
    return LocalConfig(**kwargs)


def _build_tracking(raw: dict) -> TrackingConfig:
    t = _section(raw, "TRACKING")
    return TrackingConfig(
        mlflow=_build_mlflow(_section(t, "MLFLOW")),
        local=_build_local(_section(t, "LOCAL")),
    )


def load(path: str) -> RunConfig:
    """
    Parse `path` as yaml into a RunConfig. A missing file, an empty file, or
    a file missing whole sections all degrade gracefully — every gap falls
    through to the matching dataclass field's own default in utils/schemas.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        logger.warning("config file %r not found — using built-in schema defaults for everything", path)
        raw = {}

    kwargs = _pick(raw or {}, {"SEED": "seed"})
    return RunConfig(
        data=_build_data(raw),
        model=_build_model(raw),
        training=_build_training(raw),
        evaluation=_build_evaluation(raw),
        scheduler=_build_scheduler(raw),
        early_stopper=_build_early_stopper(raw),
        tracking=_build_tracking(raw),
        **kwargs,
    )
