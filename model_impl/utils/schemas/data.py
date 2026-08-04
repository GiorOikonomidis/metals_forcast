"""
Data configuration: what to load and how to split/window it.

Every field carries its own default, mirroring the shipped exampl.yaml —
config_file_parser.py only ever passes the keys a yaml actually specifies,
so an omitted key falls straight through to the default declared here. This
is the config system's own fallback layer; it does not reach into consts.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetConfig:
    """The single series to forecast, selected from the long-format file."""
    id: str = "IRON"
    feature: str = "close"


@dataclass(frozen=True)
class SplitsConfig:
    """
    Duration-based, end-anchored splits (row counts on the aligned series):
    test = last `test_days` rows, val = `val_days` rows immediately before,
    train = everything before that.
    """
    test_days: int = 250
    val_days: int = 250


@dataclass(frozen=True)
class DataConfig:
    target: TargetConfig = field(default_factory=TargetConfig)
    # long-file per-id covariates, each entry an (id, [features]) pair
    covariates: list = field(default_factory=list)
    # wide global-file scalar column names
    global_covariates: list[str] = field(default_factory=list)
    news_col: str = "embedding"
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    pred_len: int = 7
    ctx_len: int = 150
    token_all: bool = True
    shuffle_data: bool = True
    type_of_diff: str = "log_diff"  # no_diff | diff | log_diff
