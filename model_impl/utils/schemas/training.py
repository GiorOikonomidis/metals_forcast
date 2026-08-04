"""
Training configuration: the training loop itself, plus the two things that
govern how it stops or slows down — the LR scheduler and early stopping.
Bundled here rather than split out because both are training-process
concerns, not independent top-level sections.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-5           # base learning rate — used directly with no scheduler; the
                               # rate SchedulerConfig's cold-start phase warms up *into*
                               # and its plateau decay works down *from* when one is used.
    weight_decay: float = 1e-4
    grad_clip: float | None = None
    batch_size: int = 256
    label_smoothing: float = 0.15


@dataclass(frozen=True)
class SchedulerConfig:
    """
    Cold-start LR schedule, layered on top of TrainingConfig.lr (not a second,
    independent learning rate — see main.py, where the optimizer's own lr
    already comes from TrainingConfig.lr regardless of whether this is used).
    """
    use: bool = True
    cold_start: float = 1e-5
    cold_epochs: int = 30
    decrease_factor: float = 0.5
    metric: str = "val"  # val | train
    patience: int = 10


@dataclass(frozen=True)
class EarlyStopperConfig:
    use: bool = True
    patience: int = 20
