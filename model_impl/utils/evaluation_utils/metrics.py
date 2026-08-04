"""
Scoring functions for point, interval and distributional forecasts.

Every function here is pure: arrays in, float out. They carry no notion of the
model, the split or the run — eval_pipeline.py decides what to feed them, and
in particular hands them price-space arrays once a differenced target has been
reconstructed (see data_loader_utils.transforms.invert_diff).
"""

import numpy as np
from properscoring import crps_ensemble
from scipy.stats import t as student_t


def build_ece_quantiles(start: float, stop: float, steps: int) -> np.ndarray:
    """
    The nominal quantile grid shared by ece_quantiles() and the reliability
    (P-P) curve. Built once per run from EvaluationConfig.ece_grid and passed
    down explicitly — this module stays a pure leaf with no config import.
    """
    return np.linspace(start, stop, steps)


# ── point ───────────────────────────────────────────────────────────────────
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))


def mape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((yhat - y) / (np.abs(y) + eps)))) * 100.0


def smape(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(2.0 * np.abs(yhat - y) / (np.abs(y) + np.abs(yhat) + eps))) * 100.0


# ── interval ────────────────────────────────────────────────────────────────
def interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float = 0.2) -> float:
    """
    Interval Score (Gneiting & Raftery). Lower is better.
    """
    width = high - low
    penalty_low = (2/alpha) * np.maximum(low - y, 0)
    penalty_high = (2/alpha) * np.maximum(y - high, 0)
    return float(np.mean(width + penalty_low + penalty_high))


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Empirical coverage of the central interval."""
    return float(np.mean((y >= low) & (y <= high)))


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Pinball loss for quantile q. Lower is better."""
    e = y - q_pred
    return float(np.mean(np.maximum(q*e, (q-1)*e)))


# ── distributional ──────────────────────────────────────────────────────────
def loss_crps(preds: np.ndarray, truth: np.ndarray) -> float:
    """Mean CRPS of an ensemble forecast. preds: (MC, H), truth: (H,)."""
    return float(crps_ensemble(truth, preds.T).mean())


def ece_quantiles(y_true: np.ndarray, samples: np.ndarray,
                  quantiles: np.ndarray) -> float:
    """
    Expected Calibration Error for quantiles.
    We compute empirical frequency of y <= q_pred and average |empirical - nominal|.
    """
    eces = []
    for q in quantiles:
        qhat = np.quantile(samples, q, axis=0)  # shape (H,)
        emp = np.mean(y_true <= qhat)
        eces.append(abs(emp - q))
    return float(np.mean(eces))


def pit_values(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Probability Integral Transform for ensemble forecasts:
    PIT = empirical CDF of the ensemble at the true value.
    Returns a 1D array over horizons (and windows if aggregated outside).
    """
    # Smooth PIT: (rank + 1) / (M + 1)
    M = samples.shape[0]
    pits = []
    for h in range(samples.shape[1]):
        s = np.sort(samples[:, h])
        # rank: number of samples ≤ y_true[h]
        r = np.searchsorted(s, y_true[h], side="right")
        pits.append((r + 1.0) / (M + 1.0))
    return np.array(pits, dtype=np.float32)


# ── model comparison ────────────────────────────────────────────────────────
def dm_test(loss1: np.ndarray, loss2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.
    loss1, loss2: arrays over T items (e.g., per-window aggregated loss).
    h: forecast horizon (we use PRED_LEN here).
    Returns (DM_stat, p_value) with small-sample t approximation.
    """
    d = np.asarray(loss1) - np.asarray(loss2)
    T = d.shape[0]
    d_bar = np.mean(d)

    # Newey-West variance with lag = h-1 to account for autocorrelation
    def autocov(x, lag):
        if lag >= T:
            return 0.0
        x_mean = np.mean(x)
        return np.sum((x[:T-lag] - x_mean) * (x[lag:] - x_mean)) / T

    gamma0 = autocov(d, 0)
    var_hat = gamma0
    for k in range(1, h):
        gamma = autocov(d, k)
        var_hat += 2 * (1 - k/h) * gamma

    dm_denom = np.sqrt(var_hat / T + 1e-12)
    dm_stat = d_bar / dm_denom if dm_denom > 0 else np.inf

    # small-sample correction ~ t_{T-1}
    pval = 2 * (1 - student_t.cdf(np.abs(dm_stat), df=max(T-1, 1)))
    return float(dm_stat), float(pval)
