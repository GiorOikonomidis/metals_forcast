"""
Chronos token-id distribution study for the target series.

Answers one question: what does the tokenizer actually do to our targets under
each of the three representations the model layer can select (no_diff / diff /
log_diff), and how much that answer changes between the two scaling schemes the
windowing code supports.

The two schemes are the same ones `data_loading/windowing.py` implements via
`token_all`:

- ``all``        (token_all=True)  one mean-abs scale computed over the WHOLE
                                   series, every value tokenized under it. One
                                   histogram over every token produced.
- ``per_window`` (token_all=False) every ctx-length window is scaled by its own
                                   mean-abs. Each window gives its own
                                   normalized histogram; those are AVERAGED
                                   across windows, so the result is the
                                   distribution an average window sees rather
                                   than the pooled one.

The distinction matters because the model is a classifier over the token vocab:
under a global scale a trending series drifts across token space (the same price
level maps to different ids early vs late in the sample), while a per-window
scale re-centres every window. The differencing mode interacts with that — a
differenced series is already stationary, so the two schemes should converge,
and how far they do not is what these plots show.

Output layout::

    results/
      <target id>/
        all/         token_distribution.png, token_counts.csv, summary.csv
        per_window/  token_distribution.png, token_counts.csv, summary.csv

Each PNG is one figure holding the three histograms (no_diff, diff, log_diff),
stacked and sharing the token-id axis so the spread is directly comparable.

Run:
    python val_data/tokens_distribution/tokens_distribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig
from chronos.chronos import ChronosConfig, ChronosTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model_impl.utils.data_loader_utils.transforms import apply_differencing

# ── configuration ───────────────────────────────────────────────────────────
TARGET_PARQUET = REPO_ROOT / "target_variables.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

TARGET_IDS = ["XAU", "XAG", "XCU"]
FEATURE = "close"

# Matches DATA.CTX_LEN in exampl.yaml — the per-window scheme is only meaningful
# relative to the window length the model is actually trained with.
CTX_LEN = 30

MODEL_NAME = "amazon/chronos-t5-base"
DIFF_MODES = ["no_diff", "diff", "log_diff"]
CASES = ["all", "per_window"]

MODE_COLORS = {"no_diff": "#1f77b4", "diff": "#d62728", "log_diff": "#2ca02c"}
FIG_DPI = 160

# Fraction of probability mass the shared x-axis must cover. Token ids are drawn
# as one bar each, so the axis is cropped to the bulk of the distribution —
# across the full 4096-wide vocab a single bar is sub-pixel and the histogram
# reads as a continuous curve, which is exactly what it is not. Rare tail ids
# outside the crop are still reported in each panel's title.
X_LIM_MASS = 0.99


# ── tokenizer ───────────────────────────────────────────────────────────────
def build_tokenizer(model_name: str) -> tuple[ChronosTokenizer, ChronosConfig]:
    """
    Build the Chronos tokenizer without loading the model weights.

    `ChronosPipeline.from_pretrained` reads the HF config, constructs a
    `ChronosConfig` from its ``chronos_config`` block and calls
    `create_tokenizer()` — then separately downloads ~1 GB of seq2seq weights we
    have no use for here. This reproduces only the first half, so the tokenizer
    is bit-identical to the one `main.py` hands to the windowing code while
    costing a single small config read.

    `use_eos_token` is switched off to match `main.py`, which does the same right
    after building the pipeline; leaving it on would append an EOS id to every
    context and pollute the histogram with a spike at `eos_token_id`.

    Parameters
    ----------
    model_name : str
        HF model id or local path, e.g. ``"amazon/chronos-t5-base"``.

    Returns
    -------
    tuple[ChronosTokenizer, ChronosConfig]
        tokenizer : the mean-scale uniform-bin tokenizer.
        config    : its config, read for ``n_tokens`` / ``n_special_tokens`` and
                    mutated in place by the callers to set ``context_length``.
    """
    hf_config = AutoConfig.from_pretrained(model_name)
    chronos_config = ChronosConfig(**hf_config.chronos_config)
    object.__setattr__(chronos_config, "use_eos_token", False)
    return chronos_config.create_tokenizer(), chronos_config


# ── data ────────────────────────────────────────────────────────────────────
def load_target(parquet_path: Path, target_id: str, feature: str) -> pd.Series:
    """
    Read one target series out of the long-format target parquet.

    Parameters
    ----------
    parquet_path : Path
        Long-format parquet with columns ``[date, id, open, high, low, close]``.
    target_id : str
        The id to extract, e.g. ``"XAU"``.
    feature : str
        Which price column to profile, e.g. ``"close"``.

    Returns
    -------
    pd.Series
        float64 series indexed by ascending DatetimeIndex, NaNs dropped.
    """
    df = pd.read_parquet(parquet_path, columns=["date", "id", feature])
    sub = df[df["id"] == target_id].copy()
    if sub.empty:
        raise ValueError(f"target id {target_id!r} not present in {parquet_path}")
    sub["date"] = pd.to_datetime(sub["date"])
    ser = sub.set_index("date")[feature].sort_index().astype(np.float64)
    return ser.dropna()


def represent(series: pd.Series, mode: str) -> np.ndarray:
    """
    Apply the model layer's differencing transform to one series.

    Delegates to `model_impl.utils.data_loader_utils.transforms.apply_differencing`
    rather than reimplementing it, so this study profiles exactly the values the
    training pipeline would tokenize — including its leading-row bfill and its
    float64 promotion.

    Parameters
    ----------
    series : pd.Series
        Raw price levels.
    mode : str
        One of ``"no_diff"``, ``"diff"``, ``"log_diff"``.

    Returns
    -------
    np.ndarray
        float32 1-D array of the transformed values, same length as `series`.
    """
    frame = apply_differencing(series.to_frame(name=FEATURE), mode)
    return frame[FEATURE].to_numpy(dtype=np.float32)


# ── tokenization ────────────────────────────────────────────────────────────
def tokens_all(values: np.ndarray, tokenizer: ChronosTokenizer,
               config: ChronosConfig) -> np.ndarray:
    """
    Tokenize the whole series under a single shared scale (token_all=True).

    `context_input_transform` left-truncates anything longer than
    ``config.context_length``, so the config is widened to the series length
    first — the same `object.__setattr__` dance `windowing.sliding_windows_triple`
    performs in its GLOBAL branch.

    Parameters
    ----------
    values : np.ndarray
        1-D transformed series.
    tokenizer : ChronosTokenizer
    config : ChronosConfig
        Mutated in place: ``context_length`` is set to ``len(values)``.

    Returns
    -------
    np.ndarray
        1-D int array of token ids, one per input value.
    """
    object.__setattr__(config, "context_length", len(values))
    ctx = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    token_ids, _, _ = tokenizer.context_input_transform(ctx)
    return token_ids.squeeze(0).numpy()


def tokens_per_window(values: np.ndarray, ctx_len: int, tokenizer: ChronosTokenizer,
                      config: ChronosConfig) -> np.ndarray:
    """
    Tokenize every sliding context window under its own scale (token_all=False).

    Windows are taken at stride 1 over the full series. The pipeline additionally
    reserves the last `pred_len` rows for the forecast target; that reservation is
    dropped here because this study profiles context tokenization only, and
    keeping it would silently discard the most recent windows.

    Parameters
    ----------
    values : np.ndarray
        1-D transformed series.
    ctx_len : int
        Context window length.
    tokenizer : ChronosTokenizer
    config : ChronosConfig
        Mutated in place: ``context_length`` is set to `ctx_len`.

    Returns
    -------
    np.ndarray
        int array of shape ``(n_windows, ctx_len)`` of token ids.
    """
    if len(values) < ctx_len:
        raise ValueError(f"series of length {len(values)} shorter than ctx_len={ctx_len}")

    object.__setattr__(config, "context_length", ctx_len)
    windows = np.stack([values[i - ctx_len:i] for i in range(ctx_len, len(values) + 1)])
    token_ids, _, _ = tokenizer.context_input_transform(torch.tensor(windows, dtype=torch.float32))
    return token_ids.numpy()


# ── distributions ───────────────────────────────────────────────────────────
def distribution_all(token_ids: np.ndarray, n_tokens: int) -> np.ndarray:
    """
    Pooled probability distribution over the vocab.

    Parameters
    ----------
    token_ids : np.ndarray
        Flat or nested int array of token ids.
    n_tokens : int
        Vocabulary size — the length of the returned vector.

    Returns
    -------
    np.ndarray
        float64 vector of length `n_tokens` summing to 1.
    """
    counts = np.bincount(token_ids.ravel(), minlength=n_tokens).astype(np.float64)
    return counts / counts.sum()


def distribution_per_window(token_ids: np.ndarray, n_tokens: int) -> np.ndarray:
    """
    Average of the per-window distributions.

    Each window is normalized on its own before averaging, so every window
    contributes equally regardless of length — this is the "take the avg" step
    that distinguishes the per-window view from simply pooling all its tokens.
    (With a fixed ctx_len the two coincide, but normalizing first is what makes
    the y-axis read as "probability mass in a typical window".)

    Parameters
    ----------
    token_ids : np.ndarray
        int array of shape ``(n_windows, ctx_len)``.
    n_tokens : int
        Vocabulary size.

    Returns
    -------
    np.ndarray
        float64 vector of length `n_tokens` summing to 1.
    """
    n_windows, ctx_len = token_ids.shape
    per_window = np.stack([
        np.bincount(row, minlength=n_tokens) for row in token_ids
    ]).astype(np.float64) / ctx_len
    return per_window.mean(axis=0)


def summarize(dist: np.ndarray, token_ids: np.ndarray, mode: str,
              n_special: int) -> dict:
    """
    Reduce one distribution to the numbers worth comparing across modes.

    Parameters
    ----------
    dist : np.ndarray
        Probability vector over the vocab.
    token_ids : np.ndarray
        The raw ids the distribution was built from (for the observed range).
    mode : str
        Differencing mode the row describes.
    n_special : int
        Number of reserved special-token ids at the bottom of the vocab; ids
        below this are not value tokens and their presence signals padding.

    Returns
    -------
    dict
        One row: token counts, support width, central tendency, entropy and the
        share of mass sitting on special tokens.
    """
    used = np.flatnonzero(dist > 0)
    ids = token_ids.ravel()
    nonzero = dist[dist > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        "diff_mode": mode,
        "n_tokens_emitted": int(ids.size),
        "unique_tokens": int(used.size),
        "vocab_coverage": float(used.size / dist.size),
        "token_id_min": int(ids.min()),
        "token_id_max": int(ids.max()),
        "token_id_mean": float(ids.mean()),
        "token_id_std": float(ids.std()),
        "entropy_bits": entropy,
        "perplexity": float(2.0 ** entropy),
        "special_token_mass": float(dist[:n_special].sum()),
    }


# ── plotting ────────────────────────────────────────────────────────────────
def central_span(dist: np.ndarray, mass: float) -> tuple[int, int]:
    """
    Token-id range between the two symmetric tail quantiles.

    Equal probability is trimmed from each end — `(1 - mass) / 2` per side — so a
    long one-sided tail of near-zero ids cannot drag the range open. A greedy
    outward walk from the mode was tried first and does exactly that: with both
    neighbours at ~0 it wanders to the vocab edge and the crop buys nothing.

    Parameters
    ----------
    dist : np.ndarray
        Probability vector over the vocab.
    mass : float
        Central mass to keep, e.g. ``0.99``.

    Returns
    -------
    tuple[int, int]
        Inclusive ``(lo, hi)`` token ids.
    """
    cdf = np.cumsum(dist)
    tail = (1.0 - mass) / 2.0
    lo = int(np.searchsorted(cdf, tail, side="left"))
    hi = int(np.searchsorted(cdf, 1.0 - tail, side="left"))
    return lo, min(hi, len(dist) - 1)


def plot_distributions(dists: dict[str, np.ndarray], target_id: str, case: str,
                       out_path: Path) -> None:
    """
    Draw the three differencing modes as stacked histograms in one figure.

    One bar per token id — the tokenizer's bins are discrete and the plot says
    so. The x-axis (token id) is shared across the three panels, since the
    comparison being made is how widely each representation spreads over the
    vocabulary, and is cropped to the union of the three modes' central
    `X_LIM_MASS` spans so individual bars stay visible. The y-axes are
    deliberately NOT shared: a differenced series concentrates orders of
    magnitude more mass on a handful of ids, and forcing a common y-scale would
    flatten the level panel into an invisible line.

    Parameters
    ----------
    dists : dict[str, np.ndarray]
        Probability vector per differencing mode, keyed by mode name.
    target_id : str
        Target series id, used in the figure title.
    case : str
        ``"all"`` or ``"per_window"``, used in the figure title.
    out_path : Path
        PNG destination.

    Returns
    -------
    None
        Writes `out_path`.
    """
    n_tokens = len(next(iter(dists.values())))
    spans = [central_span(dists[mode], X_LIM_MASS) for mode in DIFF_MODES]
    lo = max(0, min(s[0] for s in spans) - 2)
    hi = min(n_tokens - 1, max(s[1] for s in spans) + 2)
    ids = np.arange(n_tokens)

    fig, axes = plt.subplots(len(DIFF_MODES), 1, figsize=(11, 8.5), sharex=True)

    for ax, mode in zip(axes, DIFF_MODES):
        dist = dists[mode]
        used = np.flatnonzero(dist > 0)
        shown = dist[lo:hi + 1].sum()
        ax.bar(ids, dist, width=1.0, align="center",
               color=MODE_COLORS[mode], edgecolor="none")
        ax.set_ylabel("probability", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.margins(y=0.05)
        ax.set_title(
            f"{mode}   —   support [{used.min()}, {used.max()}], "
            f"{used.size} of {n_tokens} ids used, "
            f"{shown:.3%} of mass in view",
            fontsize=10, loc="left")

    axes[-1].set_xlim(lo - 0.5, hi + 0.5)
    axes[-1].set_xlabel(f"Chronos token id  (one bar per id; "
                        f"vocab is {n_tokens} wide, view cropped to "
                        f"[{lo}, {hi}])", fontsize=10)

    scheme = ("one scale over the whole series"
              if case == "all" else
              f"per-window scale, averaged over windows (ctx={CTX_LEN})")
    fig.suptitle(f"{target_id} — token-id distribution [{case}]\n{scheme}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(RESULTS_DIR.parent)}")


# ── driver ──────────────────────────────────────────────────────────────────
def run_target(target_id: str, tokenizer: ChronosTokenizer,
               config: ChronosConfig) -> None:
    """
    Produce both cases for one target series.

    Parameters
    ----------
    target_id : str
        Target id to profile.
    tokenizer : ChronosTokenizer
    config : ChronosConfig
        Its ``context_length`` is rewritten per call; nothing else is read from
        it besides ``n_tokens`` / ``n_special_tokens``.

    Returns
    -------
    None
        Writes ``results/<target_id>/{all,per_window}/`` with a PNG, a per-token
        CSV and a summary CSV in each.
    """
    series = load_target(TARGET_PARQUET, target_id, FEATURE)
    n_tokens = config.n_tokens
    print(f"\n[{target_id}] {len(series)} rows, "
          f"{series.index[0].date()} -> {series.index[-1].date()}")

    dists: dict[str, dict[str, np.ndarray]] = {case: {} for case in CASES}
    rows: dict[str, list[dict]] = {case: [] for case in CASES}

    for mode in DIFF_MODES:
        values = represent(series, mode)

        ids_all = tokens_all(values, tokenizer, config)
        dists["all"][mode] = distribution_all(ids_all, n_tokens)
        rows["all"].append(summarize(dists["all"][mode], ids_all, mode,
                                     config.n_special_tokens))

        ids_win = tokens_per_window(values, CTX_LEN, tokenizer, config)
        dists["per_window"][mode] = distribution_per_window(ids_win, n_tokens)
        rows["per_window"].append(summarize(dists["per_window"][mode], ids_win,
                                            mode, config.n_special_tokens))

        print(f"  {mode:<9} all: {rows['all'][-1]['unique_tokens']:>4} ids, "
              f"H={rows['all'][-1]['entropy_bits']:.2f} bits   |   "
              f"per_window ({ids_win.shape[0]} windows): "
              f"{rows['per_window'][-1]['unique_tokens']:>4} ids, "
              f"H={rows['per_window'][-1]['entropy_bits']:.2f} bits")

    for case in CASES:
        out_dir = RESULTS_DIR / target_id / case
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = pd.DataFrame({"token_id": np.arange(n_tokens)})
        for mode in DIFF_MODES:
            counts[mode] = dists[case][mode]
        counts.to_csv(out_dir / "token_counts.csv", index=False)

        pd.DataFrame(rows[case]).to_csv(out_dir / "summary.csv", index=False)
        plot_distributions(dists[case], target_id, case,
                           out_dir / "token_distribution.png")


def main() -> None:
    """
    Build the tokenizer once and profile every configured target.

    Returns
    -------
    None
    """
    tokenizer, config = build_tokenizer(MODEL_NAME)
    print(f"tokenizer: {MODEL_NAME}  vocab={config.n_tokens} "
          f"({config.n_special_tokens} special)")

    for target_id in TARGET_IDS:
        run_target(target_id, tokenizer, config)

    print(f"\nDone. Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
