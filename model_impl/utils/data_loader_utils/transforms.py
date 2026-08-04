"""
Representation transforms for the target/covariate price streams: differencing
at load time and its exact inverse for reconstruction. Pure functions on
frames/arrays — no file I/O (that belongs to data_loading).
"""

from __future__ import annotations

import numpy as np


def apply_differencing(data: "pd.DataFrame", mode: str) -> "pd.DataFrame":
    """
    Difference every column of an already-interpolated frame, at load time.

    Differencing lives in the model layer (not the dataset build): the parquet
    trees always store raw interpolated levels, and this converts them to the
    representation selected by TYPE_OF_DIFF. The order interpolate → difference
    is what makes reconstruction exact: cumsum(diff(interp(level))) == interp(level),
    so invert_diff (below) telescopes back to the true level over any horizon.

    The leading row (no prior value to difference against) is bfilled, matching
    how the old pre-differenced parquets were loaded (price_series ffill/bfill
    over the stored leading NaN).

    Args:
        data: DataFrame of levels (target Close or covariate panel), no NaN.
        mode: "no_diff"  -> returned unchanged
              "diff"     -> x(t) - x(t-1)
              "log_diff" -> log(x).diff() on strictly-positive columns, first-order
                            fallback on the rest (log undefined for values <= 0)
    """
    if mode == "no_diff":
        return data

    # float64 for the diff arithmetic: the loaders cast levels to float32, and
    # log/diff/cumsum chains in float32 lose enough precision that reconstruction
    # drifts (~1e-2 on NDX-scale prices). In float64 the telescoping is exact to
    # ~1e-11, matching what the old dataset-side differencing produced.
    data = data.astype(np.float64)
    if mode == "diff":
        data[data.columns] = data.diff()
    elif mode == "log_diff":
        for c in data.columns:
            if (data[c] > 0).all():
                data[c] = np.log(data[c]).diff()
            else:
                data[c] = data[c].diff()   # log invalid for non-positive → fall back
    else:
        raise ValueError(f"unknown diff mode: {mode!r} (expected 'no_diff', 'diff' or 'log_diff')")

    return data.ffill().bfill()


def invert_diff(anchor: float, diffs: np.ndarray, mode: str) -> np.ndarray:
    """
    Reconstruct price levels from predicted diffs over the horizon axis (last axis).

    anchor : last raw (undifferenced) context price
    diffs  : predicted differences, shape (..., H)
    mode   : "diff"     -> price(t) = anchor + cumsum(diffs)
             "log_diff" -> price(t) = anchor * exp(cumsum(diffs))
             "no_diff"  -> diffs are already price levels (returned unchanged)
    """
    if mode == "diff":
        return anchor + np.cumsum(diffs, axis=-1)
    if mode == "log_diff":
        return anchor * np.exp(np.cumsum(diffs, axis=-1))
    return diffs
