"""MLflow experiment-tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLflowConfig:
    use: bool = True
    uri: str = ""            # empty -> MLflow's own local ./mlruns default
    experiment: str = ""      # empty -> MLflow's own "Default" experiment
    run_name: str = ""        # empty -> caller supplies a default (see main.py)
    log_model: bool = False
