"""
Synthetic target_variables.parquet with known, controlled signals — sine,
sinc, a square pulse, a modulated sine (trend+AM+FM), a trend+AM-only sine,
and an AR(1)-return process — in the exact schema the model loaders expect
(date, id, open, high, low, close; see model_impl/data_loading/covariates-
structure.md). Useful as a sanity-check dataset: unlike real price series,
these have a known ground-truth periodic structure, so a model that can't
recover the period/shape of a plain sine wave from its own price context is
almost certainly broken at a more basic level than covariate selection.

Dates are copied 1:1 from XCU's own date index (2007-01-25 -> 2026-07-27,
7050 trading days) so the synthetic series plugs into the existing pipeline
(windowing, splits, tokenizer scale) with zero date-alignment work.

Periods are chosen in trading-day units to mirror the three seasonal cycles
the pipeline already encodes elsewhere (generate_date_feat's sin_dow/sin_month/
sin_doy): a trading week (~5 sessions), a trading month (~21 sessions), a
trading year (~252 sessions).

Amplitude/offset are matched to XCU's own close price (mean=3.42, std=0.89,
range [1.25, 6.65]) so the tokenizer's mean-abs scale lands in the same
regime as the real target, rather than an arbitrary unrelated range.

Only `close` carries the actual signal — `open`/`high`/`low` are set equal to
`close` (no intraday range exists for a synthetic series), satisfying the
schema's four required OHLC columns without fabricating meaningless spread.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"C:\Users\GiorgosOikonomidis\Desktop\proakt\imple_ours")
XCU_PATH = REPO_ROOT / "produced_data" / "metals" / "datasets" / "target_variables.parquet"
OUT_PATH = REPO_ROOT / "produced_data" / "synthetic" / "datasets" / "target_variables.parquet"

BASELINE = 3.42   # XCU close mean
AMPLITUDE = 0.89  # XCU close std

PERIOD_WEEK = 5      # trading days per week
PERIOD_MONTH = 21    # trading days per month
PERIOD_QUARTER = 63  # trading days per quarter — used for SINC so its lobe stretches wider
PERIOD_YEAR = 252    # trading days per year

# SINE_MOD: FM/AM/trend parameters are drawn from these candidate sets with a
# fixed seed (reproducible, not re-rolled per run). Shares the other three
# signals' date range (XCU's own 7050 days) rather than an independent one —
# a longer, independent range was tried first, but any run using SINE_MOD as
# the target alongside covariates from a metals-dataset file (7050 days)
# tripped the windower's length-mismatch assert, since nothing in the loader
# chain reindexes across differing date ranges.
SINE_MOD_SEED = 7
CARRIER_PERIOD_CHOICES = [10, 15, 21, 30, 42]
FM_PERIOD_CHOICES = [63, 126, 189, 252]
FM_DEPTH_CHOICES = [0.2, 0.3, 0.4, 0.5]
AM_PERIOD_CHOICES = [126, 189, 252, 378]
AM_DEPTH_CHOICES = [0.3, 0.4, 0.5, 0.6]
TREND_TOTAL_CHOICES = [0.5, 1.0, 1.5, 2.0]

# SINE_TREND_AM: trend + amplitude modulation only (no FM) — fixed carrier
# frequency and trend, randomized amplitude-modulation params only.
SINE_TREND_AM_SEED = 7
TREND_AM_CARRIER_FREQ = 1 / 21   # cycles/day, fixed (no FM)
TREND_AM_TREND_TOTAL = 1.5        # fixed total linear rise
TREND_AM_FREQ_CHOICES = [1 / 126, 1 / 189, 1 / 252, 1 / 378]  # cycles/day
TREND_AM_DEPTH_CHOICES = [0.3, 0.4, 0.5, 0.6]

# AR1_RETURNS: log-returns follow an AR(1) process (r_t = phi*r_{t-1} + eps_t)
# instead of white noise, so — unlike XCU's ~0 return autocorrelation at every
# lag — this signal has a genuine, causal, learnable lag-1 dependency in its
# returns (phi>0 = momentum). sigma matches XCU's own daily log-diff std.
AR1_SEED = 7
AR1_PHI = 0.3
AR1_SIGMA = 0.0135


def make_sine(t: np.ndarray, period: int) -> np.ndarray:
    """Pure sine wave, one full cycle every `period` trading days."""
    return BASELINE + AMPLITUDE * np.sin(2 * np.pi * t / period)


def make_sinc(t: np.ndarray, period: int) -> np.ndarray:
    """
    Periodic sinc: each `period`-day window is one full sinc(x) lobe pattern,
    centered so the central peak falls at the middle of every period.

    Only 2 side-lobes span the whole period (vs. an earlier, denser version
    with 3), and the period itself is stretched to a full trading year so
    those lobes spread across ~252 days instead of ~21-63 — the ringing
    decay gets room to actually taper toward zero before the wraparound,
    instead of being cut off mid-decay by a short period.

    Parameters
    ----------
    t : np.ndarray
        Trading-day index, 0..T-1.
    period : int
        Trading days per repeated sinc lobe.
    """
    phase = (t % period) - period / 2       # wraps to [-period/2, period/2) each cycle
    x = phase / period * 4 * np.pi           # 2 full sinc side-lobes per cycle (was 6*pi/3 lobes)
    return BASELINE + AMPLITUDE * np.sinc(x / np.pi)  # np.sinc is normalized: sin(pi x)/(pi x)


def make_square(t: np.ndarray, period: int) -> np.ndarray:
    """Square pulse: high for the first half of each `period`, low for the second half."""
    phase = (t % period) < (period / 2)
    return np.where(phase, BASELINE + AMPLITUDE, BASELINE - AMPLITUDE)


def draw_sine_mod_params(seed: int) -> dict:
    """One value per FM/AM/trend parameter, drawn from the candidate sets above."""
    rng = np.random.default_rng(seed)
    return {
        "carrier_period": int(rng.choice(CARRIER_PERIOD_CHOICES)),
        "fm_period": int(rng.choice(FM_PERIOD_CHOICES)),
        "fm_depth": float(rng.choice(FM_DEPTH_CHOICES)),
        "am_period": int(rng.choice(AM_PERIOD_CHOICES)),
        "am_depth": float(rng.choice(AM_DEPTH_CHOICES)),
        "trend_total": float(rng.choice(TREND_TOTAL_CHOICES)),
    }


def make_sine_mod(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Sine wave with a linear trend, amplitude modulation, and frequency
    modulation: y(t) = baseline + trend(t) + amplitude_envelope(t) * sin(phase(t)).

    Frequency modulation: instantaneous frequency oscillates around
    2*pi/carrier_period by +/-fm_depth over fm_period trading days; phase is
    the cumulative sum (discrete integral) of that instantaneous frequency,
    not a fixed linear ramp, so the oscillation visibly speeds up and slows
    down rather than holding one period throughout.

    Amplitude modulation: a slow envelope (period am_period, depth am_depth)
    scales the oscillation's height independently of its frequency.

    Parameters
    ----------
    t : np.ndarray
        Trading-day index, 0..T-1.
    p : dict
        carrier_period, fm_period, fm_depth, am_period, am_depth, trend_total
        — see draw_sine_mod_params.
    """
    T = len(t)
    inst_freq = (2 * np.pi / p["carrier_period"]) * (1 + p["fm_depth"] * np.sin(2 * np.pi * t / p["fm_period"]))
    phase = np.cumsum(inst_freq)
    amplitude_env = AMPLITUDE * (1 + p["am_depth"] * np.sin(2 * np.pi * t / p["am_period"]))
    trend = p["trend_total"] * t / T
    return BASELINE + trend + amplitude_env * np.sin(phase)


def draw_sine_trend_am_params(seed: int) -> dict:
    """One value per AM parameter, drawn from the candidate sets above."""
    rng = np.random.default_rng(seed)
    return {
        "am_freq": float(rng.choice(TREND_AM_FREQ_CHOICES)),
        "am_depth": float(rng.choice(TREND_AM_DEPTH_CHOICES)),
    }


def make_sine_trend_am(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Sine wave with a linear trend and amplitude modulation only (no FM):
    y(t) = baseline + trend(t) + amplitude_envelope(t) * sin(2*pi*carrier_freq*t).

    Carrier frequency and trend are fixed; only the amplitude envelope's
    frequency/depth are randomized (see draw_sine_trend_am_params).

    Parameters
    ----------
    t : np.ndarray
        Trading-day index, 0..T-1.
    p : dict
        am_freq, am_depth — see draw_sine_trend_am_params.
    """
    T = len(t)
    amplitude_env = AMPLITUDE * (1 + p["am_depth"] * np.sin(2 * np.pi * p["am_freq"] * t))
    trend = TREND_AM_TREND_TOTAL * t / T
    return BASELINE + trend + amplitude_env * np.sin(2 * np.pi * TREND_AM_CARRIER_FREQ * t)


def make_ar1_returns(t: np.ndarray, phi: float, sigma: float, p0: float, seed: int) -> np.ndarray:
    """
    Price whose log-returns follow an AR(1) process: r_t = phi*r_{t-1} + eps_t,
    eps_t ~ N(0, sigma^2); price_t = price_{t-1} * exp(r_t).

    phi controls momentum (phi>0, today's return pushes tomorrow's the same
    direction) or mean-reversion (phi<0) — a genuine causal, learnable lag-1
    dependency in returns, unlike XCU's own log-diff series whose measured
    autocorrelation is ~0 at every lag.

    Parameters
    ----------
    t : np.ndarray
        Trading-day index, 0..T-1 (only its length is used).
    phi : float
        AR(1) coefficient on the previous day's return.
    sigma : float
        Daily i.i.d. shock std.
    p0 : float
        Starting price.
    seed : int
        RNG seed for the shocks.
    """
    n = len(t)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    log_price = np.log(p0) + np.cumsum(r)
    return np.exp(log_price)


def main() -> None:
    """Build the 4-series synthetic target_variables.parquet."""
    xcu = pd.read_parquet(XCU_PATH)
    dates = xcu[xcu["id"] == "XCU"].sort_values("date")["date"].reset_index(drop=True)
    t = np.arange(len(dates))

    signals = {
        "SINE": (dates, make_sine(t, PERIOD_YEAR)),
        "SINC": (dates, make_sinc(t, PERIOD_YEAR)),
        "SQUARE": (dates, make_square(t, PERIOD_WEEK)),
    }

    # SINE_MOD shares XCU's own 7050-day date range, same as the other three —
    # it originally used its own longer, independent range (1970-2027,
    # 15000 days), but any run using it as the target alongside covariates
    # drawn from a metals-dataset file (7050 days) hits the windower's
    # length-mismatch assert, since nothing in the loader chain reindexes
    # across differing date ranges. Matching the range here is simpler and
    # more robust than reindexing covariates at load time.
    mod_params = draw_sine_mod_params(SINE_MOD_SEED)
    signals["SINE_MOD"] = (dates, make_sine_mod(t, mod_params))
    print(f"SINE_MOD drawn params: {mod_params}")

    trend_am_params = draw_sine_trend_am_params(SINE_TREND_AM_SEED)
    signals["SINE_TREND_AM"] = (dates, make_sine_trend_am(t, trend_am_params))
    print(f"SINE_TREND_AM drawn params: {trend_am_params}")

    signals["AR1_RETURNS"] = (dates, make_ar1_returns(t, AR1_PHI, AR1_SIGMA, BASELINE, AR1_SEED))
    print(f"AR1_RETURNS params: phi={AR1_PHI}, sigma={AR1_SIGMA}")

    rows = []
    for signal_id, (signal_dates, close) in signals.items():
        rows.append(pd.DataFrame({
            "date": signal_dates,
            "id": signal_id,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }))

    out = pd.concat(rows, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows ({out['id'].nunique()} ids) -> {OUT_PATH}")
    print(out.groupby("id")["close"].agg(["min", "max", "mean", "count"]).round(3))


if __name__ == "__main__":
    main()
