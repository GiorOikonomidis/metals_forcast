"""Learning-rate scheduling."""

import torch


class ColdStartScheduler:
    """
    Two-phase learning-rate scheduler.

    Cold-start phase (epochs 0 .. cold_epochs-1):
        LR is held fixed at `cold_start_rate`.

    Running phase (epoch >= cold_epochs):
        LR switches to `running_rate`, then reduces by `decrease_factor`
        whenever the observed metric fails to improve for `patience` epochs
        (ReduceLROnPlateau-style, lower-is-better).

    Call `step(metric_value)` once per epoch, after the metric is computed.
    """
    def __init__(self, opt: torch.optim.Optimizer,
                 cold_start_rate: float, cold_epochs: int,
                 running_rate: float, decrease_factor: float,
                 patience: int) -> None:
        self.opt = opt
        self.cold_start_rate = cold_start_rate
        self.cold_epochs = cold_epochs
        self.running_rate = running_rate
        self.decrease_factor = decrease_factor
        self.patience = patience

        self.epoch = 0
        self.best = float('inf')
        self.wait = 0
        self._set_lr(cold_start_rate)  # start in cold-start phase

    def _set_lr(self, lr: float) -> None:
        self.opt.param_groups[0]["lr"] = lr

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def step(self, metric_value: float) -> None:
        self.epoch += 1

        if self.epoch < self.cold_epochs:
            # still in cold start — keep the fixed rate
            self._set_lr(self.cold_start_rate)
            return

        if self.epoch == self.cold_epochs:
            # transition into running phase — reset plateau tracker
            self._set_lr(self.running_rate)
            self.best = metric_value
            self.wait = 0
            return

        # running phase — reduce on plateau
        if metric_value < self.best:
            self.best = metric_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self._set_lr(self.lr * self.decrease_factor)
                self.wait = 0
