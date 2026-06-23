import os
import ast
import numpy as np
import pandas as pd
from config import ENRICHED_DATASETS_DIR, COMPANIES_DIR, INDEX_DIR, NEWS_DIR, DATASETS_DIR, DATE_FEAT_COLS, INDEX_FEATURES


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


def get_all_closes(companies: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    What it does:
        Builds a wide DataFrame of company Close prices aligned to the
        requested date index. Each company is reindexed to `dates`; dates
        where the company has no data will be NaN until the caller applies
        ffill/bfill. The caller is responsible for filling.

    Input:
        companies : dict[str, pd.DataFrame]
            As returned by load_companies_2007().
        dates : pd.DatetimeIndex
            The target date index for the output (trading days for Cases 1 & 3,
            all_dates for Case 2).

    Output:
        pd.DataFrame  shape (len(dates), n_companies)
            Index   : dates
            Columns : "Close_<TICKER>" for each ticker in companies.
            Values  : raw Close prices, NaN where the company had no quote on
                      that date. Caller must ffill().bfill() after this call.
    """
    return pd.DataFrame(
        {f"Close_{t}": df["Close"].reindex(dates) for t, df in companies.items()},
        index=dates,
    )


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


# ── news aggregation (Case 1) ─────────────────────────────────────────────────

def aggregate_news_to_trading_days(
    news: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    What it does:
        Implements the Case 1 news strategy. For each trading day t, collects
        all news rows in the half-open window [t, next_trading_day). This folds
        weekend and holiday news into the preceding Friday (or last trading day).
        Within a window:
          - embedding  : mean of all daily embeddings (preserves direction, not
                         magnitude; shape stays (768,)).
          - probs      : sum across days (so prob_positive can exceed 1.0 when
                         multiple news days fall in one window).
          - label      : argmax of the summed probs.
        If no news exists in a window, all news columns are set to NaN/None.

        The last trading day's window extends to the end of the news series
        (no upper bound), so any trailing news after the last market open is
        captured.

    Input:
        news          : pd.DataFrame
            As returned by load_news(). Indexed by calendar date.
        trading_dates : pd.DatetimeIndex
            Sorted trading days as returned by get_trading_dates().

    Output:
        pd.DataFrame  shape (len(trading_dates), 5)  index name="Date"
            Columns: embedding     — np.ndarray (768,) or None
                     prob_positive — float (sum; may exceed 1.0) or NaN
                     prob_negative — float (sum; may exceed 1.0) or NaN
                     prob_neutral  — float (sum; may exceed 1.0) or NaN
                     label         — "positive" / "negative" / "neutral" or None
    """
    dates_sorted = trading_dates.sort_values()
    rows = []

    for i, t in enumerate(dates_sorted):
        next_t = dates_sorted[i + 1] if i + 1 < len(dates_sorted) else None
        mask   = (
            (news.index >= t) & (news.index < next_t)
            if next_t is not None
            else (news.index >= t)
        )
        window = news[mask].dropna(subset=["embedding"])

        if window.empty:
            rows.append({
                "Date": t, "embedding": None,
                "prob_positive": np.nan, "prob_negative": np.nan,
                "prob_neutral":  np.nan, "label": None,
            })
            continue

        avg_emb  = np.stack(window["embedding"].values).mean(axis=0)
        prob_pos = window["prob_positive"].sum()
        prob_neg = window["prob_negative"].sum()
        prob_neu = window["prob_neutral"].sum()
        label    = ["positive", "negative", "neutral"][
            int(np.argmax([prob_pos, prob_neg, prob_neu]))
        ]
        rows.append({
            "Date": t, "embedding": avg_emb,
            "prob_positive": prob_pos, "prob_negative": prob_neg,
            "prob_neutral":  prob_neu, "label": label,
        })

    return pd.DataFrame(rows).set_index("Date")


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
    index_df:     pd.DataFrame | None = None,
    trading_mask: pd.Series | None = None,
):
    """
    What it does:
        Writes the future covariate file — calendar features that are known
        before the market opens on any date.

        Two code paths:
          - index_df provided (Cases 1 & 3): the six sin/cos columns are
            already precomputed in the enriched index CSV, so they are
            reused directly via reindex. This is only valid because Cases 1
            & 3 operate exclusively on trading dates already present in the
            index.
          - index_df=None (Case 2): all_dates includes non-trading days that
            are absent from the index, so the six columns are recomputed from
            scratch using the dates' dayofweek / month / dayofyear attributes.

        trading_mask is appended as a "trading_day" column only in Case 2.
        It is a future covariate because the market calendar is published in
        advance and is known before any price is observed.

    Input:
        dates        : pd.DatetimeIndex
            Output date sequence.
        out_dir      : str
            Directory where feature_covariates.parquet is written.
        index_df     : pd.DataFrame | None
            Full enriched index DataFrame (Cases 1 & 3) or None (Case 2).
        trading_mask : pd.Series | None
            Binary Series (1=trading, 0=non-trading) as returned by
            get_trading_mask(). Passed only for Case 2.

    Output:
        Writes feature_covariates.parquet indexed by date with columns:
            sin_dow, cos_dow        — day-of-week encoding (period 7)
            sin_month, cos_month    — month encoding (period 12)
            sin_doy, cos_doy        — day-of-year encoding (period 365)
            trading_day             — int 0/1 (Case 2 only)
    """
    if index_df is not None:
        df = index_df[DATE_FEAT_COLS].reindex(dates).copy()
    else:
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
    index_df:   pd.DataFrame,
    all_closes: pd.DataFrame,
    news_df:    pd.DataFrame | None,
    dates:      pd.DatetimeIndex,
    out_dir:    str,
    id:         str = "",
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
        all_closes : pd.DataFrame
            Wide company close matrix as returned by get_all_closes(), already
            ffill+bfilled by the caller. Shape (len(dates), n_companies).
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
            Close_AAPL … Close_XEL (74 cols)          — float64, no NaN
            prob_positive, prob_negative, prob_neutral — float64, NaN where no news
            label                                      — str or NaN
            embedding                                  — list[float] len=768 or None
    """
    df = pd.DataFrame(index=dates)
    df.index.name = "date"

    for col in INDEX_FEATURES:
        df[col] = index_df[col].reindex(dates) if col in index_df.columns else np.nan

    for col in all_closes.columns:
        df[col] = all_closes[col].reindex(dates)

    if news_df is not None:
        for col in ["prob_positive", "prob_negative", "prob_neutral", "label", "embedding"]:
            df[col] = news_df[col].reindex(dates) if col in news_df.columns else None
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else None
            )

    out = df.reset_index()
    out.insert(0, "id", id)
    out.to_parquet(os.path.join(out_dir, "dynamic_covariates.parquet"), index=False)
    print(f"  Saved dynamic_covariates.parquet  ({len(dates)} dates)")


# ── case builders ─────────────────────────────────────────────────────────────

def build_case1(companies: dict, index: pd.DataFrame, news: pd.DataFrame, id: str = ""):
    """
    What it does:
        Builds the Case 1 dataset — "aggregate news forward".

        Pipeline:
          1. Extract trading dates from the index (≈4 881 dates, 2007–2025).
          2. Build company close matrix (4 881 × 74), ffill+bfill to remove NaN
             at the start/end of short-history companies.
          3. Aggregate news into trading-day windows: each trading day t absorbs
             all news from [t, next_trading_day). Weekend and holiday headlines
             are folded into the preceding Friday's window. Probs are summed
             (can exceed 1.0), embeddings are averaged.
          4. Slice index technical features (already on trading dates, no fill
             needed).
          5. Write target.parquet    — ^NDX Close, 4 881 rows, no NaN.
          6. Write feature_covariates.parquet — 6 sin/cos cols reused from the
             enriched index (already precomputed for trading dates).
          7. Write dynamic_covariates.parquet — 11 index features + 74 company
             closes + aggregated news. Trading days with no news in their window
             get NaN for all 5 news columns (117 such days in the 2007–2025 range,
             mostly from the trailing gap after 2025-12-31).

    Input:
        companies : dict[str, pd.DataFrame]  from load_companies_2007()
        index     : pd.DataFrame             from load_index()
        news      : pd.DataFrame             from load_news()

    Output:
        Writes 3 parquet files to datasets/case_1_agg_news/:
            target.parquet               (4 881 × 2)
            feature_covariates.parquet   (4 881 × 6)
            dynamic_covariates.parquet   (4 881 × 87)
    """
    print("\nBuilding case_1_agg_news...")
    out_dir = os.path.join(DATASETS_DIR, "case_1_agg_news")
    os.makedirs(out_dir, exist_ok=True)

    trading_dates  = get_trading_dates(index)
    all_closes     = get_all_closes(companies, trading_dates).ffill().bfill()
    agg_news       = aggregate_news_to_trading_days(news, trading_dates)
    index_features = index[[c for c in INDEX_FEATURES if c in index.columns]]

    save_target(index, trading_dates, out_dir, id=id)
    save_feature_covariates(trading_dates, out_dir, index_df=index)
    save_dynamic_covariates(index_features, all_closes, agg_news, trading_dates, out_dir, id=id)


def build_case2(companies: dict, index: pd.DataFrame, news: pd.DataFrame, id: str = ""):
    """
    What it does:
        Builds the Case 2 dataset — "full calendar with trading mask".

        Pipeline:
          1. Build the full calendar range from the first to the last trading day.
          2. Filter to keep only dates where at least one of:
               - index["Close"] is non-NaN  (trading day), OR
               - news["prob_positive"] is non-NaN  (news was published)
             This drops blank weekends (no price, no news) but keeps weekends
             that carry news. Result: ≈7 027 dates.
          3. Build company close matrix (7 027 × 74), ffill+bfill so non-trading
             rows carry the last known price.
          4. Compute trading_mask (7 027,): 1 on market-open rows, 0 elsewhere.
          5. Fill the index with ffill+bfill across all_dates so EMA, RSI, etc.
             are available on news-only days (last known value carried forward).
          6. Slice index technical features from the filled index.
          7. Write target.parquet    — ^NDX Close, NaN on non-trading rows.
          8. Write feature_covariates.parquet — 6 sin/cos cols recomputed from
             scratch for all_dates (can't reuse index because non-trading days
             are absent from the index) + trading_day column.
          9. Write dynamic_covariates.parquet — 11 filled index features + 74
             company closes + raw news (on original publication dates, no
             aggregation). News NaN where no news published on that date.

    Input:
        companies : dict[str, pd.DataFrame]  from load_companies_2007()
        index     : pd.DataFrame             from load_index()
        news      : pd.DataFrame             from load_news()

    Output:
        Writes 3 parquet files to datasets/case_2_mask/:
            target.parquet               (7 027 × 2)   — NaN Close on 2 146 non-trading rows
            feature_covariates.parquet   (7 027 × 7)   — includes trading_day column
            dynamic_covariates.parquet   (7 027 × 87)
    """
    print("\nBuilding case_2_mask...")
    out_dir = os.path.join(DATASETS_DIR, "case_2_mask")
    os.makedirs(out_dir, exist_ok=True)

    full_range = pd.date_range(
        start=index["Close"].dropna().index.min(),
        end=index["Close"].dropna().index.max(),
        freq="D",
    )
    has_close = index["Close"].reindex(full_range).notna()
    # prob is used to indenty if we have news or not that day
    has_news  = news.reindex(full_range)["prob_positive"].notna()
    all_dates = full_range[has_close | has_news]

    all_closes     = get_all_closes(companies, all_dates).ffill().bfill()
    trading_mask   = get_trading_mask(index, all_dates)
    index_filled   = index.reindex(all_dates).ffill().bfill()
    index_features = index_filled[[c for c in INDEX_FEATURES if c in index_filled.columns]]

    save_target(index, all_dates, out_dir, id=id)
    save_feature_covariates(all_dates, out_dir, trading_mask=trading_mask)
    save_dynamic_covariates(index_features, all_closes, news, all_dates, out_dir, id=id)


def build_case3(companies: dict, index: pd.DataFrame, news: pd.DataFrame, id: str = ""):
    """
    What it does:
        Builds the Case 3 dataset — "discard non-trading-day news".

        Pipeline:
          1. Extract trading dates (≈4 881 dates) — identical to Case 1.
          2. Build company close matrix (4 881 × 74), ffill+bfill.
          3. Reindex news to trading dates only: news published on weekends or
             holidays is silently discarded. Only headlines published on an
             actual trading day survive. This results in sparser news coverage
             than Case 1 (which folds weekend news into the preceding Friday).
          4. Slice index technical features (no fill needed — trading dates
             are already present in the index).
          5. Write target.parquet    — identical to Case 1 (same dates, same values).
          6. Write feature_covariates.parquet — identical structure to Case 1
             (6 sin/cos cols reused from the index, no trading_day column).
          7. Write dynamic_covariates.parquet — 11 index features + 74 company
             closes + sparse news. NaN news on any trading day without a same-day
             headline (includes weekdays where news was only published on adjacent
             non-trading days, plus the same trailing gap as Case 1).

    Input:
        companies : dict[str, pd.DataFrame]  from load_companies_2007()
        index     : pd.DataFrame             from load_index()
        news      : pd.DataFrame             from load_news()

    Output:
        Writes 3 parquet files to datasets/case_3_discard/:
            target.parquet               (4 881 × 2)
            feature_covariates.parquet   (4 881 × 6)
            dynamic_covariates.parquet   (4 881 × 87)
    """
    print("\nBuilding case_3_discard...")
    out_dir = os.path.join(DATASETS_DIR, "case_3_discard")
    os.makedirs(out_dir, exist_ok=True)

    trading_dates  = get_trading_dates(index)
    all_closes     = get_all_closes(companies, trading_dates).ffill().bfill()
    news_trading   = news.reindex(trading_dates)
    index_features = index[[c for c in INDEX_FEATURES if c in index.columns]]

    save_target(index, trading_dates, out_dir, id=id)
    save_feature_covariates(trading_dates, out_dir, index_df=index)
    save_dynamic_covariates(index_features, all_closes, news_trading, trading_dates, out_dir, id=id)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    companies = load_companies_2007()
    index     = load_index()
    news      = load_news()

    id = "^nsdq"

    build_case1(companies, index, news, id=id)
    build_case2(companies, index, news, id=id)
    build_case3(companies, index, news, id=id)

    print("\nDone. Output in:", DATASETS_DIR)
