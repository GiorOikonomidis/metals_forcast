"""
Step 7 — build the model_impl input contract.

Reads a dataset's enriched target / covariates / news and writes one tree of
raw interpolated levels in the shape ``model_impl`` consumes::

    target_variables.parquet   long : date, id, open, high, low, close
    global_covariates.parquet  wide : date + {TICKER}_{feature} + news embedding
    feature_covariates.parquet wide : date + sin/cos calendar encodings (optional)

Which series is the target and which are covariates comes from the dataset
registry, so the same builder serves the index and metals datasets without
branching. Everything is written **raw** — differencing is a model-layer
transform (``model_impl``'s ``apply_differencing``, selected by
``TYPE_OF_DIFF``), so one tree serves every variant.
"""

import ast
import os

import numpy as np
import pandas as pd

from constants import (GLOBAL_COV_SEP, GLOBAL_COVARIATE_PRICE_COLS,
                       GLOBAL_INCLUDE_TECH_FEATURES, INDEX_FEATURES,
                       NEWS_INCLUDE_SENTIMENT, PROB_COLS, PROB_LABELS,
                       TARGET_OHLC_COLS, WRITE_FEATURE_COVARIATES,
                       WRITE_NEWS_TO_GLOBAL,
                       FILE_NAME_FEAT_COV, FILE_NAME_GLOBAL_COV, FILE_NAME_TARGET_VARS)
from scripts.paths import (KIND_COVARIATES, dataset_config, enriched_dir,
                           news_enriched_path, output_dir, target_path)

# Technical-indicator columns = enriched features minus the raw price levels.
# Folded into global_covariates only when GLOBAL_INCLUDE_TECH_FEATURES is set.
TECH_FEATURES = [c for c in INDEX_FEATURES if c not in ("Open", "High", "Low", "Volume")]

# ── loaders ───────────────────────────────────────────────────────────────────

def load_covariates(base_dir: str, dataset: str, min_start: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """
    Load every enriched covariate CSV for a dataset.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.
    min_start : pd.Timestamp or None, optional
        If given, only tickers whose history begins on or before this date are
        kept, and every exclusion is printed by name. Use it to enforce a
        common start date across covariates. None keeps everything.

    Returns
    -------
    dict[str, pd.DataFrame]
        Ticker -> DataFrame indexed by Date, deduplicated (keep last).

    Notes
    -----
    This previously required each ticker's history to begin in exactly 2007,
    which silently discarded any instrument younger than that — for the metals
    dataset that removed REMX (listed 2010) and MP (2020), a third of the
    covariate set, with no warning. The bound is now relative and exclusions
    are always reported.
    """
    enr_dir = enriched_dir(base_dir, dataset, KIND_COVARIATES)
    if not os.path.isdir(enr_dir):
        raise FileNotFoundError(
            f"no enriched covariates for dataset {dataset!r} at {enr_dir} - "
            f"run the covariate download and enrichment steps first"
        )

    covariates, excluded = {}, []
    for f in sorted(os.listdir(enr_dir)):
        if not f.endswith(".csv"):
            continue
        ticker = f[:-len(".csv")]
        df = pd.read_csv(os.path.join(enr_dir, f), index_col="Date", parse_dates=True)
        df = df[~df.index.duplicated(keep="last")]
        if min_start is not None and df.index.min() > min_start:
            excluded.append((ticker, df.index.min().date()))
            continue
        covariates[ticker] = df

    for ticker, first in excluded:
        print(f"  EXCLUDED {ticker}: history starts {first}, after the required {min_start.date()}")
    print(f"Loaded {len(covariates)} covariate series for dataset {dataset!r}")
    if not covariates:
        raise ValueError(
            f"no covariates survived loading for dataset {dataset!r} - "
            f"{len(excluded)} excluded by the min_start bound ({min_start})"
        )
    return covariates


def load_target(base_dir: str, dataset: str) -> pd.DataFrame:
    """
    Load a dataset's enriched target CSV — the prediction target source.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; its ``target_ticker`` names the file.

    Returns
    -------
    pd.DataFrame
        Indexed by Date (trading days only), deduplicated (keep last). Carries
        Open/High/Low/Close/Volume and the technical indicators. ``Close`` is
        the target; the rest are available as covariates.

    Raises
    ------
    FileNotFoundError
        If the target has not been downloaded and enriched for this dataset.
    """
    path = target_path(base_dir, dataset, enriched=True)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no enriched target for dataset {dataset!r} at {path} - "
            f"run the target download and enrichment steps first"
        )
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[~df.index.duplicated(keep="last")]


def load_news(base_dir: str, dataset: str) -> pd.DataFrame | None:
    """
    Load the enriched per-day news for a dataset's topic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; its ``news_topic`` selects the shared cache to read.

    Returns
    -------
    pd.DataFrame or None
        Indexed by Date (days with news), deduplicated, with ``embedding``
        parsed to arrays. None when the news step has not been run — the build
        then proceeds without news, rather than failing.
    """
    path = news_enriched_path(base_dir, dataset)
    if not os.path.isfile(path):
        print(f"  no enriched news at {path} - building without news")
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df[~df.index.duplicated(keep="last")]
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else None
    )
    return df


def has_sentiment(news: pd.DataFrame | None) -> bool:
    """
    Whether a news frame carries sentiment probabilities.

    Parameters
    ----------
    news : pd.DataFrame or None
        As returned by :func:`load_news`.

    Returns
    -------
    bool
        False for embedding-only models such as MiniLM, which produce no
        ``prob_*`` columns. Callers must check this before touching them —
        assuming their presence is what made ``--model minilm`` crash here.
    """
    return news is not None and "prob_positive" in news.columns


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_prefix(name: str) -> str:
    """
    Strip a leading ``^`` from a ticker so it is safe as a column-name prefix.

    Parameters
    ----------
    name : str
        Raw ticker or id (e.g. ``"^NDX"``).

    Returns
    -------
    str
        The name without a leading ``^``. Mirrors the metals convention where
        ``^STOXX50E`` was stored as ``STOXX50E`` (``^`` is disallowed in MLflow
        param/tag keys).
    """
    return name[1:] if name.startswith("^") else name


def build_covariate_global_panel(covariates: dict, include_tech: bool) -> pd.DataFrame:
    """
    Build the wide covariate panel for global_covariates.

    Parameters
    ----------
    covariates : dict[str, pd.DataFrame]
        As returned by :func:`load_covariates`.
    include_tech : bool
        When True, technical-indicator columns are appended beside the price
        columns; when False, only price columns are kept.

    Returns
    -------
    pd.DataFrame
        Indexed by the union of covariate date indices. Columns are
        ``{TICKER}{SEP}{feature}`` — price features lowercased to match the
        metals convention (``AAPL_close``), technical features kept as-is
        (``AAPL_RSI``). Alignment to the output dates is the caller's job.
    """
    cols = list(GLOBAL_COVARIATE_PRICE_COLS) + (TECH_FEATURES if include_tech else [])
    frames = {}
    for ticker, df in covariates.items():
        prefix = _clean_prefix(ticker)
        for col in cols:
            if col not in df.columns:
                continue
            feat = col.lower() if col in GLOBAL_COVARIATE_PRICE_COLS else col
            frames[f"{prefix}{GLOBAL_COV_SEP}{feat}"] = df[col]
    return pd.DataFrame(frames)


def _build_all_dates(target: pd.DataFrame, news: pd.DataFrame | None,
                     cutoff_date: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """
    Build the shared output date range.

    Parameters
    ----------
    target : pd.DataFrame
        Enriched target frame; ``Close`` defines the trading calendar.
    news : pd.DataFrame or None
        Enriched news frame; publication days extend the calendar. When None,
        or when the model produced no sentiment columns, the calendar is the
        trading days alone.
    cutoff_date : pd.Timestamp or None, optional
        If given, the range is truncated to dates strictly before it.

    Returns
    -------
    pd.DatetimeIndex
        Calendar days between the first and last trading day where at least one
        of (market open, news published) is true. Blank weekends are dropped.
    """
    full_range = pd.date_range(
        start=target["Close"].dropna().index.min(),
        end=target["Close"].dropna().index.max(),
        freq="D",
    )
    keep = target["Close"].reindex(full_range).notna()
    if has_sentiment(news):
        keep = keep | news.reindex(full_range)["prob_positive"].notna()
    elif news is not None:
        # Embedding-only model: no prob columns to test, so use presence of a row.
        keep = keep | full_range.isin(news.index)

    all_dates = full_range[keep]
    if cutoff_date is not None:
        all_dates = all_dates[all_dates < cutoff_date]
    return all_dates


# ── news gap filler ───────────────────────────────────────────────────────────

def fill_news_isolated_gaps(news: pd.DataFrame) -> pd.DataFrame:
    """
    Fill isolated single-day news gaps by averaging their neighbours.

    Parameters
    ----------
    news : pd.DataFrame
        As returned by :func:`load_news` — indexed by calendar date, rows only
        where news was published.

    Returns
    -------
    pd.DataFrame
        Same structure as the input, with isolated gap days (missing, but with
        news on both the preceding and following calendar day) added and
        filled: probs and embedding are the element-wise mean of the two
        neighbours, label is the argmax of the filled probs. Runs of two or
        more consecutive missing days are left out. Returned unchanged when the
        frame has no probability columns.
    """
    if not has_sentiment(news):
        return news

    full_range = pd.date_range(news.index.min(), news.index.max(), freq="D")
    df         = news.reindex(full_range).copy()
    idx        = df.index
    filled     = 0

    for i in range(1, len(df) - 1):
        d, d_prev, d_next = idx[i], idx[i - 1], idx[i + 1]
        # current day missing, both neighbours present
        if (pd.isna(df.at[d, "prob_positive"]) and pd.notna(df.at[d_prev, "prob_positive"])
                and pd.notna(df.at[d_next, "prob_positive"])):
            for col in PROB_COLS:
                df.at[d, col] = (df.at[d_prev, col] + df.at[d_next, col]) / 2
            emb_b, emb_a = df.at[d_prev, "embedding"], df.at[d_next, "embedding"]
            if emb_b is not None and emb_a is not None:
                df.at[d, "embedding"] = (np.array(emb_b) + np.array(emb_a)) / 2
            # argmax over PROB_COLS order, so labels must be in that same order
            probs = [df.at[d, c] for c in PROB_COLS]
            df.at[d, "label"] = PROB_LABELS[int(np.argmax(probs))]
            filled += 1

    if filled:
        print(f"  Filled {filled} isolated news gap(s) by neighbour averaging")
    return df[df["prob_positive"].notna()]


# ── parquet writers ───────────────────────────────────────────────────────────

def save_target_variables_long(target_filled: pd.DataFrame, dates: pd.DatetimeIndex,
                               out_dir: str, id: str) -> None:
    """
    Write the long-format target file — one id, OHLC per date.

    Parameters
    ----------
    target_filled : pd.DataFrame
        Interpolated target frame (raw levels); must carry ``TARGET_OHLC_COLS``.
    dates : pd.DatetimeIndex
        Output date sequence.
    out_dir : str
        Directory where ``FILE_NAME_TARGET_VARS`` is written.
    id : str
        Series identifier written into the ``id`` column, from the dataset
        registry's ``target_id``; matched by ``model_impl``'s ``TARGET.ID``.

    Returns
    -------
    None
        Writes ``target_variables.parquet`` with columns
        ``date, id, open, high, low, close`` (OHLC lowercased).
    """
    ohlc = target_filled.reindex(dates)
    out = pd.DataFrame({"date": dates, "id": id})
    for col in TARGET_OHLC_COLS:
        out[col.lower()] = ohlc[col].values
    out = out[["date", "id"] + [c.lower() for c in TARGET_OHLC_COLS]]
    out.to_parquet(os.path.join(out_dir, FILE_NAME_TARGET_VARS), index=False)
    print(f"  Saved {FILE_NAME_TARGET_VARS}  (id={id}, {len(dates)} dates)")


def save_global_covariates_wide(covariate_panel: pd.DataFrame,
                                target_tech: pd.DataFrame | None,
                                news_df: pd.DataFrame | None,
                                dates: pd.DatetimeIndex, out_dir: str) -> None:
    """
    Write the wide global covariate file — covariate panel + optional target
    tech + news, one row per date (no id column, matching the metals form).

    Parameters
    ----------
    covariate_panel : pd.DataFrame
        Wide ``{TICKER}_{feature}`` panel, already reindexed and interpolated.
    target_tech : pd.DataFrame or None
        Optional target technical-indicator columns (``{id}_{FEATURE}``),
        already reindexed. None when ``GLOBAL_INCLUDE_TECH_FEATURES`` is off.
    news_df : pd.DataFrame or None
        News frame. The ``embedding`` column is written **dense** — days
        without news get a zero vector, because model_impl's news loader calls
        ``np.array(v)`` on every row and a NaN cell would break it. Sentiment
        columns are added when ``NEWS_INCLUDE_SENTIMENT`` is on and the model
        produced them. None writes no news columns at all.
    dates : pd.DatetimeIndex
        Output date sequence; every part is reindexed to it.
    out_dir : str
        Directory where ``FILE_NAME_GLOBAL_COV`` is written.

    Returns
    -------
    None
        Writes ``global_covariates.parquet`` (wide) indexed by ``date``.
    """
    parts = [covariate_panel]
    if target_tech is not None:
        parts.append(target_tech)

    if news_df is not None:
        news_cols = ["embedding"]
        if NEWS_INCLUDE_SENTIMENT and has_sentiment(news_df):
            news_cols += PROB_COLS + ["label"]
        news_part = news_df.reindex(index=dates, columns=news_cols)
        # Dense embedding: the model's loader calls np.array(v) on every row, so
        # newsless days must carry a zero vector rather than NaN. Infer the dim
        # from the first real embedding (768 FinBERT/FinancialBERT, 384 MiniLM).
        real = news_part["embedding"].dropna()
        emb_dim = len(real.iloc[0]) if len(real) else 0
        zero = [0.0] * emb_dim
        news_part["embedding"] = news_part["embedding"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else zero
        )
        parts.append(news_part)

    df = pd.concat(parts, axis=1)
    df.index.name = "date"
    df.reset_index().to_parquet(os.path.join(out_dir, FILE_NAME_GLOBAL_COV), index=False)
    print(f"  Saved {FILE_NAME_GLOBAL_COV}  ({df.shape[1]} cols, {len(dates)} dates)")


def save_feature_covariates(dates: pd.DatetimeIndex, out_dir: str) -> None:
    """
    Write the optional date-feature file — six sin/cos calendar encodings.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Output date sequence.
    out_dir : str
        Directory where ``FILE_NAME_FEAT_COV`` is written.

    Returns
    -------
    None
        Writes ``feature_covariates.parquet`` with columns ``date, sin_dow,
        cos_dow, sin_month, cos_month, sin_doy, cos_doy``. These are future
        covariates, known before the market opens.
    """
    df = pd.DataFrame(index=dates)
    df["sin_dow"]   = np.sin(2 * np.pi * dates.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * dates.dayofweek / 7)
    df["sin_month"] = np.sin(2 * np.pi * dates.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * dates.month / 12)
    df["sin_doy"]   = np.sin(2 * np.pi * dates.dayofyear / 365)
    df["cos_doy"]   = np.cos(2 * np.pi * dates.dayofyear / 365)
    df.index.name = "date"
    df.reset_index().to_parquet(os.path.join(out_dir, FILE_NAME_FEAT_COV), index=False)
    print(f"  Saved {FILE_NAME_FEAT_COV}")


# ── builder ───────────────────────────────────────────────────────────────────

def build_dataset(base_dir: str, dataset: str, covariates: dict, target: pd.DataFrame,
                  news: pd.DataFrame | None, cutoff_date: pd.Timestamp | None = None) -> None:
    """
    Build one dataset's output tree (raw interpolated levels).

    Non-trading days are linearly interpolated (``method="time"``, weighted by
    calendar distance) so the model sees a continuous stream; boundary NaN is
    closed with ffill/bfill. News is left on its publication dates and never
    interpolated.

    Parameters
    ----------
    base_dir : str
        Root directory; output goes to ``<base_dir>/<dataset>/datasets``.
    dataset : str
        Dataset key; supplies the output id and column prefix.
    covariates : dict[str, pd.DataFrame]
        From :func:`load_covariates`.
    target : pd.DataFrame
        From :func:`load_target`.
    news : pd.DataFrame or None
        From :func:`fill_news_isolated_gaps`, or None to build without news.
    cutoff_date : pd.Timestamp or None, optional
        Drop output dates on or after this date. None means no cutoff.

    Returns
    -------
    None
        Writes ``target_variables.parquet``, ``global_covariates.parquet`` and
        (when ``WRITE_FEATURE_COVARIATES``) ``feature_covariates.parquet``.
    """
    cfg = dataset_config(dataset)
    target_id = cfg["target_id"]
    print(f"\nBuilding dataset {dataset!r} "
          f"(target {cfg['target_ticker']} -> id {target_id}, {len(covariates)} covariates)...")

    out_dir = output_dir(base_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    all_dates     = _build_all_dates(target, news, cutoff_date=cutoff_date)
    target_filled = target.reindex(all_dates).interpolate(method="time").ffill().bfill()

    # target: OHLC (raw interpolated levels)
    save_target_variables_long(target_filled, all_dates, out_dir, id=target_id)

    # global: wide covariate panel (+ optional tech) + optional target tech + news
    covariate_panel = (build_covariate_global_panel(covariates, GLOBAL_INCLUDE_TECH_FEATURES)
                       .reindex(all_dates).interpolate(method="time").ffill().bfill())
    target_tech = None
    if GLOBAL_INCLUDE_TECH_FEATURES:
        prefix = _clean_prefix(target_id)
        target_tech = target_filled[[c for c in TECH_FEATURES if c in target_filled.columns]]
        target_tech = target_tech.rename(columns={c: f"{prefix}{GLOBAL_COV_SEP}{c}"
                                                  for c in target_tech.columns})
    news_for_global = news if WRITE_NEWS_TO_GLOBAL else None
    save_global_covariates_wide(covariate_panel, target_tech, news_for_global, all_dates, out_dir)

    if WRITE_FEATURE_COVARIATES:
        save_feature_covariates(all_dates, out_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def pipe_line(base_dir: str, dataset: str, cutoff_date: str | None = None,
              min_start: str | None = None) -> None:
    """
    Full merge pipeline for one dataset.

    Loads the enriched sources, fills isolated news gaps, and writes the output
    tree. Always raw levels — differencing is a model-layer transform.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.
    cutoff_date : str or None, optional
        ``"YYYY-MM-DD"``; output dates on or after it are dropped. None (the
        default) means no cutoff. This was previously a hardcoded constant that
        silently truncated every build at a date that quietly went stale.
    min_start : str or None, optional
        ``"YYYY-MM-DD"``; covariates whose history starts after it are excluded
        and reported by name. None keeps every covariate.

    Returns
    -------
    None
    """
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else None
    floor  = pd.Timestamp(min_start) if min_start else None

    print(f"Loading data for dataset {dataset!r}...")
    covariates = load_covariates(base_dir, dataset, min_start=floor)
    target     = load_target(base_dir, dataset)
    news       = load_news(base_dir, dataset)

    if news is not None:
        print("\nFilling isolated news gaps...")
        news = fill_news_isolated_gaps(news)

    build_dataset(base_dir, dataset, covariates, target, news, cutoff_date=cutoff)

    print("\nDone. Output in:", output_dir(base_dir, dataset))
