import os
import ast
import numpy as np
import pandas as pd
from config import ENRICHED_DATASETS_DIR, COMPANIES_DIR, INDEX_DIR, NEWS_DIR, DATASETS_DIR, DATE_FEAT_COLS, INDEX_FEATURES

NEWS_CUTOFF_DATE = pd.Timestamp("2026-01-01")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_companies_2007() -> dict[str, pd.DataFrame]:
    """
    What it does:
        Scans the enriched companies directory, reads every CSV, and keeps
        only tickers whose earliest date falls in 2007. This enforces a
        common start date across all company series so every covariate covers
        the full history from 2007-01-xx onward.
        Duplicate dates are dropped (keep last) to guard against CSVs that
        were produced by re-running the enrichment pipeline without clearing
        the output file.

    Input:
        None — reads from ENRICHED_DATASETS_DIR/COMPANIES_DIR (config).

    Output:
        dict[str, pd.DataFrame]
            Keys   : ticker symbol (e.g. "AAPL")
            Values : DataFrame indexed by Date with OHLCV + technical columns,
                     deduplicated. Only tickers with index.min().year == 2007
                     are included.
    """
    enr_dir   = os.path.join(ENRICHED_DATASETS_DIR, COMPANIES_DIR)
    companies = {}
    for f in os.listdir(enr_dir):
        if not f.endswith(".csv"):
            continue
        ticker = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(enr_dir, f), index_col="Date", parse_dates=True)
        df = df[~df.index.duplicated(keep="last")]
        if df.index.min().year == 2007:
            companies[ticker] = df
    print(f"Loaded {len(companies)} companies starting from 2007")
    return companies


def load_index() -> pd.DataFrame:
    """
    What it does:
        Reads the enriched ^NDX index CSV. The Close column is the prediction
        target; all other columns (EMA, RSI, MACD, etc. and sin/cos date
        features) are used as covariates.
        Duplicate dates are dropped (keep last) for the same reason as the
        company and news loaders — re-running the enrichment pipeline without
        clearing the output could produce duplicate rows that would silently
        corrupt any subsequent reindex.

    Input:
        None — reads ENRICHED_DATASETS_DIR/INDEX_DIR/^NDX.csv (config).

    Output:
        pd.DataFrame
            Indexed by Date (trading days only — no weekends or holidays),
            deduplicated. Columns include: Open, High, Low, Close, Volume,
            EMA_12, EMA_26, MACD, RSI, Stoch_K, Stoch_D, Williams_R, ROC,
            Daily_Return, Volatility, Movement, sin_dow, cos_dow, sin_month,
            cos_month, sin_doy, cos_doy.
    """
    path = os.path.join(ENRICHED_DATASETS_DIR, INDEX_DIR, "^NDX.csv")
    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df[~df.index.duplicated(keep="last")]


def load_news() -> pd.DataFrame:
    """
    What it does:
        Reads the enriched news CSV. Drops duplicate dates (keeps last entry
        per date — dedup is required before any reindex to avoid alignment
        errors). Parses the embedding column from its stored string
        representation back into a numpy array.

    Input:
        None — reads ENRICHED_DATASETS_DIR/NEWS_DIR/news_paper2.csv (config).

    Output:
        pd.DataFrame
            Indexed by Date (calendar days on which news was published).
            Columns: prob_positive, prob_negative, prob_neutral  — FinBERT
                     probabilities for that day's aggregated headlines.
                     label    — dominant sentiment string.
                     embedding — np.ndarray of shape (768,), the FinBERT CLS
                                 token averaged over that day's headlines.
            Rows exist only on days with published news; weekends and holidays
            without news are absent from this DataFrame.
    """
    path = os.path.join(ENRICHED_DATASETS_DIR, NEWS_DIR, "news_paper2.csv")
    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    df   = df[~df.index.duplicated(keep="last")]
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else None
    )
    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def get_trading_dates(index: pd.DataFrame) -> pd.DatetimeIndex:
    """
    What it does:
        Derives the true market calendar from the index Close column.
        Any date where Close is NaN (weekend, holiday, or data gap) is
        excluded. This is the single authoritative definition of a trading
        day used across all three cases.

    Input:
        index : pd.DataFrame
            Enriched ^NDX DataFrame as returned by load_index().

    Output:
        pd.DatetimeIndex
            Sorted sequence of dates on which the market was open.
            Approximately 4 881 dates for the 2007–2025 range.
    """
    return index["Close"].dropna().index.sort_values()


def get_all_company_features(companies: dict) -> pd.DataFrame:
    """
    What it does:
        Builds a wide DataFrame of all features for every company using each
        company's original date index. Each (ticker, column) pair becomes one
        column named "{TICKER}__{FEATURE}" (double-underscore separator).
        Alignment to the target date sequence is the caller's responsibility
        via reindex().ffill().bfill().

    Input:
        companies : dict[str, pd.DataFrame]
            As returned by load_companies_2007(). Each DataFrame has OHLCV +
            technical indicator columns indexed by Date.

    Output:
        pd.DataFrame  shape (n_all_dates, n_companies * n_features)
            Index   : union of all company date indices (original trading days).
            Columns : "{TICKER}__{FEATURE}" for every ticker and every column
                      in that ticker's DataFrame (e.g. "AAPL__Close",
                      "AAPL__EMA_12", "MSFT__Close", ...).
    """
    frames = {f"{ticker}__{col}": df[col]
              for ticker, df in companies.items()
              for col in df.columns}
    return pd.DataFrame(frames)


def get_trading_mask(index: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """
    What it does:
        Produces a binary Series aligned to `dates` that marks whether each
        date is a market-open day. Used exclusively in Case 2 as a future
        covariate (the trading calendar is known in advance) so the model
        knows which rows carry a real price observation and which are
        news-only rows.

    Input:
        index : pd.DataFrame
            Enriched ^NDX DataFrame. Only the Close column is used.
        dates : pd.DatetimeIndex
            The full date sequence for Case 2 (trading days + news-only days).

    Output:
        pd.Series  shape (len(dates),)  dtype int64  name="trading_day"
            1 where index["Close"] is non-NaN (market open).
            0 where index["Close"] is NaN (weekend, holiday, or news-only day).
    """
    return index["Close"].reindex(dates).notna().astype(int).rename("trading_day")


# ── news gap filler ───────────────────────────────────────────────────────────

def fill_news_isolated_gaps(news: pd.DataFrame) -> pd.DataFrame:
    """
    What it does:
        Finds isolated calendar days within the news date range where no news
        was published but both the preceding and following calendar day have
        news. Fills each such gap by averaging the neighbour probs and embeddings
        and setting the label to the argmax of the filled probs.

        Only strict single-day gaps with news on both sides are filled; runs
        of two or more consecutive missing days are left as NaN.

    Input:
        news : pd.DataFrame
            As returned by load_news(). Indexed by calendar date, rows only
            where news was published.

    Output:
        pd.DataFrame
            Same structure as input but with isolated gap days added and filled.
            Non-gap NaN days (no news, not isolated) are not included.
    """
    full_range = pd.date_range(news.index.min(), news.index.max(), freq="D")
    df         = news.reindex(full_range).copy()
    prob_cols  = ["prob_positive", "prob_negative", "prob_neutral"]
    idx        = df.index
    filled     = 0

    for i in range(1, len(df) - 1):
        d, d_prev, d_next = idx[i], idx[i - 1], idx[i + 1]
        if (pd.isna(df.at[d, "prob_positive"])
                and pd.notna(df.at[d_prev, "prob_positive"])
                and pd.notna(df.at[d_next, "prob_positive"])):
            for col in prob_cols:
                df.at[d, col] = (df.at[d_prev, col] + df.at[d_next, col]) / 2
            emb_b, emb_a = df.at[d_prev, "embedding"], df.at[d_next, "embedding"]
            if emb_b is not None and emb_a is not None:
                df.at[d, "embedding"] = (np.array(emb_b) + np.array(emb_a)) / 2
            probs = [df.at[d, c] for c in prob_cols]
            df.at[d, "label"] = ["positive", "negative", "neutral"][int(np.argmax(probs))]
            filled += 1

    if filled:
        print(f"  Filled {filled} isolated news gap(s) by neighbour averaging")
    return df[df["prob_positive"].notna()]


# ── parquet writers ───────────────────────────────────────────────────────────

def save_target(index: pd.DataFrame, dates: pd.DatetimeIndex, out_dir: str, id: str = ""):
    """
    What it does:
        Writes the prediction target — the ^NDX Close price series — to
        target.parquet. Uses reindex on the unfilled `index` so that
        non-trading dates (Case 2) get NaN Close, clearly marking rows the
        model should not compute loss on.

    Input:
        index   : pd.DataFrame
            Original (unfilled) enriched ^NDX DataFrame.
        dates   : pd.DatetimeIndex
            Output date sequence. Trading days for Cases 1 & 3;
            all_dates (trading + news-only) for Case 2.
        out_dir : str
            Directory where target.parquet is written.
        id      : str
            Dataset identifier written as a constant column (e.g. "^nsdq").

    Output:
        Writes target.parquet with columns:
            date  — datetime
            Close — float64; NaN on non-trading rows (Case 2 only).
            id    — str constant identifying the dataset.
    """
    df = pd.DataFrame({"id": id, "date": dates, "Close": index["Close"].reindex(dates).values})
    df.to_parquet(os.path.join(out_dir, "target.parquet"), index=False)
    print(f"  Saved target.parquet  ({len(dates)} dates)")


def save_feature_covariates(
    dates:        pd.DatetimeIndex,
    out_dir:      str,
    trading_mask: pd.Series | None = None,
):
    """
    What it does:
        Writes the future covariate file — features known before the market
        opens on any date. Six sin/cos calendar encodings are always written.
        trading_day (0/1) is appended only for case_mask.

    Input:
        dates        : pd.DatetimeIndex  — output date sequence (already cut at cutoff).
        out_dir      : str               — directory to write parquet.
        trading_mask : pd.Series | None  — binary Series from get_trading_mask().

    Output:
        Writes feature_covariates.parquet with columns:
            date
            sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy
            trading_day — int 0/1 (case_mask only)
    """
    df = pd.DataFrame(index=dates)
    df["sin_dow"]   = np.sin(2 * np.pi * dates.dayofweek / 7)
    df["cos_dow"]   = np.cos(2 * np.pi * dates.dayofweek / 7)
    df["sin_month"] = np.sin(2 * np.pi * dates.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * dates.month / 12)
    df["sin_doy"]   = np.sin(2 * np.pi * dates.dayofyear / 365)
    df["cos_doy"]   = np.cos(2 * np.pi * dates.dayofyear / 365)
    df.index.name = "date"
    if trading_mask is not None:
        df["trading_day"] = trading_mask.reindex(dates).astype(int)
    df.reset_index().to_parquet(os.path.join(out_dir, "feature_covariates.parquet"), index=False)
    print(f"  Saved feature_covariates.parquet")


def save_dynamic_covariates(
    index_df:    pd.DataFrame,
    company_df:  pd.DataFrame,
    news_df:     pd.DataFrame | None,
    dates:       pd.DatetimeIndex,
    out_dir:     str,
    id:          str = "",
):
    """
    What it does:
        Assembles and writes the dynamic covariate file — all observed
        (past/present) features that the model conditions on. Three groups
        of columns are joined into a single parquet:

          1. Index technical indicators (11 cols): EMA_12/26, MACD, RSI,
             Stoch_K/D, Williams_R, ROC, Daily_Return, Volatility, Movement.
             Reindexed from index_df (which has already been ffill+bfilled
             by the caller for Case 2).

          2. Company close prices (74 cols, prefixed "Close_<TICKER>"):
             Already ffill+bfilled by the caller; should have zero NaN.

          3. News / FinBERT sentiment (5 cols): prob_positive, prob_negative,
             prob_neutral, label, embedding. NaN where no news exists for
             that date. The embedding column is converted from np.ndarray to
             a Python list before writing — parquet cannot serialise numpy
             arrays natively.

        The embedding column is only written if present in news_df.

    Input:
        index_df   : pd.DataFrame
            Already sliced to INDEX_FEATURES columns by the caller (Cases 1
            & 3 use index[INDEX_FEATURES]; Case 2 uses index_filled[INDEX_FEATURES]
            where index_filled = index.reindex(all_dates).ffill().bfill()).
        company_df : pd.DataFrame
            Wide company feature matrix as returned by get_all_company_features(),
            already ffill+bfilled by the caller.
            Shape (len(dates), n_companies * n_features).
            Columns named "{TICKER}__{FEATURE}" (e.g. "AAPL__Close", "AAPL__EMA_12").
        news_df    : pd.DataFrame | None
            Case 1 : aggregated news DataFrame from aggregate_news_to_trading_days().
            Case 2 : raw news DataFrame (news stays on original publication date).
            Case 3 : news reindexed to trading dates only (weekend news discarded).
            None   : no news columns are written.
        dates      : pd.DatetimeIndex
            Output date sequence; all three input DataFrames are reindexed to this.
        out_dir    : str
            Directory where dynamic_covariates.parquet is written.
        id         : str
            Dataset identifier written as a constant column (e.g. "^nsdq").

    Output:
        Writes dynamic_covariates.parquet indexed by date with columns:
            id                                         — str constant
            EMA_12, EMA_26, MACD, RSI, Stoch_K, Stoch_D, Williams_R,
            ROC, Daily_Return, Volatility, Movement   — float64
            AAPL__Close, AAPL__EMA_12, … TSLA__Movement
                (n_companies * n_features cols)        — float64, no NaN
            prob_positive, prob_negative, prob_neutral — float64, NaN where no news
            label                                      — str or NaN
            embedding                                  — list[float] len=768 or None
    """
    idx_part = index_df.reindex(index=dates, columns=INDEX_FEATURES)

    # news features
    parts = [idx_part, company_df]
    if news_df is not None:
        news_cols = ["prob_positive", "prob_negative", "prob_neutral", "label", "embedding"]
        news_part = news_df.reindex(index=dates, columns=news_cols)
        if "embedding" in news_part.columns:
            news_part["embedding"] = news_part["embedding"].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else None
            )
        parts.append(news_part)

    df = pd.concat(parts, axis=1)
    df.index.name = "date"

    out = df.reset_index()
    out.insert(0, "id", id)
    out.to_parquet(os.path.join(out_dir, "dynamic_covariates.parquet"), index=False)
    print(f"  Saved dynamic_covariates.parquet  ({len(dates)} dates)")


# ── case builders ─────────────────────────────────────────────────────────────

def _build_all_dates(
    index: pd.DataFrame,
    news: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """
    Shared date range for both cases: calendar days between the first and last
    trading day where at least one of (market open, news published) is true.
    Blank weekends (no price, no news) are dropped.
    If cutoff_date is provided the range is truncated to dates < cutoff_date.
    """
    full_range = pd.date_range(
        start=index["Close"].dropna().index.min(),
        end=index["Close"].dropna().index.max(),
        freq="D",
    )
    has_close = index["Close"].reindex(full_range).notna()
    has_news  = news.reindex(full_range)["prob_positive"].notna()
    all_dates = full_range[has_close | has_news]
    if cutoff_date is not None:
        all_dates = all_dates[all_dates < cutoff_date]
    return all_dates


def build_case_mask(companies: dict, index: pd.DataFrame, news: pd.DataFrame, id: str = ""):
    """
    What it does:
        Builds the case_mask dataset — full calendar with trading-day mask and
        forward-fill for non-trading rows.

        Pipeline:
          1. Build all_dates: trading days + days with published news (≈7 027).
          2. Company features: raw frame → reindex to all_dates → ffill+bfill.
          3. Index features:   reindex to all_dates → ffill+bfill.
          4. trading_mask: 1 on market-open rows, 0 on news-only rows.
          5. Write target.parquet    — ^NDX Close, NaN on non-trading rows.
          6. Write feature_covariates.parquet — 6 sin/cos cols + trading_day
             + news_cutoff flag (1 for dates >= 2026-01-01).
          7. Write dynamic_covariates.parquet — index features + company
             features + news on original publication dates.

    Input:
        companies : dict[str, pd.DataFrame]  from load_companies_2007()
        index     : pd.DataFrame             from load_index()
        news      : pd.DataFrame             from fill_news_isolated_gaps()

    Output:
        Writes 3 parquet files to datasets/case_mask/.
    """
    print("\nBuilding case_mask...")
    out_dir = os.path.join(DATASETS_DIR, "case_mask")
    os.makedirs(out_dir, exist_ok=True)

    all_dates        = _build_all_dates(index, news, cutoff_date=NEWS_CUTOFF_DATE)
    trading_dates    = get_trading_dates(index)
    company_features = (get_all_company_features(companies)
                        .reindex(trading_dates).ffill().bfill()
                        .reindex(all_dates))
    trading_mask     = get_trading_mask(index, all_dates)
    index_filled     = index.reindex(all_dates).ffill().bfill()
    index_features   = index_filled[[c for c in INDEX_FEATURES if c in index_filled.columns]]

    save_target(index, all_dates, out_dir, id=id)
    save_feature_covariates(all_dates, out_dir, trading_mask=trading_mask)
    save_dynamic_covariates(index_features, company_features, news, all_dates, out_dir, id=id)


def build_case_interp(companies: dict, index: pd.DataFrame, news: pd.DataFrame, id: str = ""):
    """
    What it does:
        Builds the case_interp dataset — same date range as case_mask but
        with linear time interpolation for non-trading rows instead of a
        forward-fill mask. No trading_day column is written.

        Pipeline:
          1. Build all_dates: identical to case_mask.
          2. Company features: raw frame → reindex → interpolate(time) →
             ffill+bfill (handles boundary NaN before first / after last value).
          3. Index features:   reindex → interpolate(time) → ffill+bfill.
          4. Write target.parquet    — ^NDX Close, NaN on non-trading rows.
          5. Write feature_covariates.parquet — 6 sin/cos cols + news_cutoff
             flag. No trading_day column.
          6. Write dynamic_covariates.parquet — interpolated index features +
             interpolated company features + news on original publication dates.

    Input:
        companies : dict[str, pd.DataFrame]  from load_companies_2007()
        index     : pd.DataFrame             from load_index()
        news      : pd.DataFrame             from fill_news_isolated_gaps()

    Output:
        Writes 3 parquet files to datasets/case_interp/.
    """
    print("\nBuilding case_interp...")
    out_dir = os.path.join(DATASETS_DIR, "case_interp")
    os.makedirs(out_dir, exist_ok=True)

    all_dates        = _build_all_dates(index, news, cutoff_date=NEWS_CUTOFF_DATE)
    company_features = (get_all_company_features(companies)
                        .reindex(all_dates)
                        .interpolate(method="time")
                        .ffill().bfill())
    index_filled     = index.reindex(all_dates).interpolate(method="time").ffill().bfill()
    index_features   = index_filled[[c for c in INDEX_FEATURES if c in index_filled.columns]]

    save_target(index_filled, all_dates, out_dir, id=id)
    save_feature_covariates(all_dates, out_dir)
    save_dynamic_covariates(index_features, company_features, news, all_dates, out_dir, id=id)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    companies = load_companies_2007()
    index     = load_index()
    news      = load_news()

    id = "^nsdq"

    print("\nFilling isolated news gaps...")
    news = fill_news_isolated_gaps(news)

    build_case_mask(companies, index, news, id=id)
    build_case_interp(companies, index, news, id=id)

    print("\nDone. Output in:", DATASETS_DIR)
