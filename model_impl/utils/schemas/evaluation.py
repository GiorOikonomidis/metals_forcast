"""Evaluation configuration: scoring, calibration grid, and the faithfulness study."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CentralIntervalConfig:
    alpha_50: float = 0.5
    alpha_80: float = 0.2
    alpha_90: float = 0.1


@dataclass(frozen=True)
class ECEGridConfig:
    start: float = 0.05
    stop: float = 0.95
    steps: int = 19


@dataclass(frozen=True)
class FaithConfig:
    mc_samples: int = 100
    ks: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 10, 15, 20, 25, 30])
    topk: int = 5
    stability_runs: int = 5
    placebo_shifts: list[int] = field(default_factory=lambda: [-3, -1, 1, 3])
    mask_strategy: str = "mean"  # mean | zero
    rng_seed: int = 123


@dataclass(frozen=True)
class EvaluationConfig:
    windows: int | None = None
    mc_samples: int = 100
    central_interval: CentralIntervalConfig = field(default_factory=CentralIntervalConfig)
    ece_grid: ECEGridConfig = field(default_factory=ECEGridConfig)
    faith: FaithConfig = field(default_factory=FaithConfig)
