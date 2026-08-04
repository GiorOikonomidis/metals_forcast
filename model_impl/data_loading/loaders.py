"""
Parquet readers for the three input streams: index prices (target), daily news
embeddings and the covariate OHLC panel. All file I/O of the model pipeline
lives here — everything downstream works on the frames/series these return.
"""

import numpy as np
import pandas as pd

from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)


def covariates(file_path: str,
               keep_covariates: list[tuple[str, list[str]]],
               ) -> tuple[pd.DataFrame, int]:
    """
    Load the covariate panel from the long-format parquet into a wide time series.

    The source parquet is tidy/long — one row per ``(id, date)`` with columns
    ``date, id, open, high, low, close`` (many series stacked as rows, not one
    column per id-feature). For each requested ``(id, [features])`` pair this
    filters that id's rows, keeps its requested feature columns, renames them to
    ``{id}_{feature}``, and concatenates all pairs side by side on the date
    index. The output column order follows the input pairs, features inner:
    ``id1_feat1, id1_feat2, id2_feat1, ...``.

    Ids may have differing date coverage, so the concatenation outer-joins on
    date and the resulting gaps are ffill/bfill-filled (bfill back-fills an id's
    first known value into any leading gap).

    Parameters
    ----------
    file_path : str
        Path to the long-format covariate parquet (columns include
        ``date``, ``id`` and the OHLC feature columns).
    keep_covariates : list[tuple[str, list[str]]]
        One ``(id, [features])`` pair per covariate series, e.g.
        ``[("XAU", ["close"]), ("XAG", ["open", "close"])]``. Feature names must
        match the parquet's column casing (lowercase ``open/high/low/close``).

    Returns
    -------
    tuple[pd.DataFrame, int]
        df : DatetimeIndex (T, total {id}_{feature} columns), dtype float32,
             sorted ascending, ffill/bfill applied. When `keep_covariates` is
             empty, an empty, index-less ``pd.DataFrame()`` — the caller
             (`streams.load_streams`) concatenates this against the
             date-indexed global-covariate frame, which always carries the
             full date range regardless of its own column count, so the
             combined panel still comes out with the correct row count.
        n_covariates : int — the total ``{id}_{feature}`` column count (the panel
             width), not the number of ids. 0 when `keep_covariates` is empty.
    """
    if not keep_covariates:
        logger.info("covariates: no per-id covariates requested (0 columns)")
        return pd.DataFrame(), 0

    feats_needed = sorted({f for _, feats in keep_covariates for f in feats})
    df = pd.read_parquet(file_path, columns=["date", "id"] + feats_needed)
    df["date"] = pd.to_datetime(df["date"])

    frames, cols_order = [], []
    for cid, feats in keep_covariates:
        sub = df[df["id"] == cid].set_index("date")[feats]
        sub.columns = [f"{cid}_{f}" for f in feats]
        cols_order += list(sub.columns)
        frames.append(sub)

    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel[cols_order].astype(np.float32).ffill().bfill()
    n_covariates = len(cols_order)

    logger.info("covariates: ids=%d, cols=%d, T=%d, range=%s -> %s",
                len(keep_covariates), n_covariates, len(panel),
                panel.index[0].date(), panel.index[-1].date())
    return panel, n_covariates


def global_covariates(file_path: str, keep_cols: list[str],
                      embedding: bool = False):
    """
    Load selected columns from the wide global-covariates parquet.

    The global file is wide: one row per date, scalar covariate columns
    (e.g. "AA_close", "eur_usd") plus a news `embedding` column of per-day
    vectors. Both are read the same way; `embedding` toggles the return shape,
    which the caller resolves at the unpacking site.

    Parameters
    ----------
    file_path : str
        Path to the wide global parquet (a `date` column + the requested columns).
    keep_cols : list[str]
        Column names to load, e.g. ["AA_close", "eur_usd"] for scalar covariates,
        or a single embedding column name (e.g. ["embedding"]) for news.
    embedding : bool, default False
        False -> scalar covariates: returns (DataFrame float32, n_cols).
        True  -> news embeddings: returns a pd.Series of float32 vectors
                 (keep_cols must name the single embedding column).

    Returns
    -------
    tuple[pd.DataFrame, int]        when embedding is False
        df : DatetimeIndex (T, len(keep_cols)) float32, sorted, ffill/bfill.
        n  : int — number of columns (the covariate panel width).
    pd.Series                       when embedding is True
        index  : DatetimeIndex (sorted); values : np.ndarray float32, one per day.
    """
    df = pd.read_parquet(file_path, columns=["date"] + list(keep_cols))
    df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"]).sort_index()

    if embedding:
        col = keep_cols[0]
        ser = df[col].apply(lambda v: np.array(v, dtype=np.float32))
        logger.info("global_covariates[news]: T=%d, dim=%d, range=%s -> %s",
                    len(ser), ser.iloc[0].shape[0], ser.index[0].date(), ser.index[-1].date())
        return ser

    df = df[list(keep_cols)].astype(np.float32).ffill().bfill()
    logger.info("global_covariates: cols=%d, T=%d, range=%s -> %s",
                len(keep_cols), len(df), df.index[0].date(), df.index[-1].date())
    return df, len(keep_cols)


