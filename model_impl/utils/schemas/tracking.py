"""
Tracking configuration: the two independent backends a run's artifacts,
params and metrics can be reported to — MLflow and the local output/
directory. Either can be switched off on its own (TRACKING.MLFLOW.use,
TRACKING.LOCAL.use); both stay on by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from model_impl.utils.schemas.local import LocalConfig
from model_impl.utils.schemas.mlflow import MLflowConfig


@dataclass(frozen=True)
class TrackingConfig:
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
