"""The whole resolved run configuration — the root of the schema package."""

from __future__ import annotations

from dataclasses import dataclass, field

from model_impl.utils.schemas.data import DataConfig
from model_impl.utils.schemas.evaluation import EvaluationConfig
from model_impl.utils.schemas.model import ModelConfig
from model_impl.utils.schemas.tracking import TrackingConfig
from model_impl.utils.schemas.training import EarlyStopperConfig, SchedulerConfig, TrainingConfig


@dataclass(frozen=True)
class RunConfig:
    """
    Exists only to be unpacked once in main() into the sections below — no
    function outside main() should take a RunConfig parameter. Everything
    downstream receives just the wrapper it needs (DataConfig, ModelConfig, ...).
    """
    seed: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopper: EarlyStopperConfig = field(default_factory=EarlyStopperConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
