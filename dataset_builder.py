import os
import ast
import numpy as np
import pandas as pd
from config import ENRICHED_DATASETS_DIR, COMPANIES_DIR, INDEX_DIR, NEWS_DIR , DATASETS_DIR , DATE_FEAT_COLS , INDEX_FEATURES




# ── loaders ───────────────────────────────────────────────────────────────────

def load_companies_2007() -> dict[str, pd.DataFrame]:
    """Load enriched company CSVs, keep only those whose first date is 2007."""
    enr_dir   = os.path.join(ENRICHED_DATASETS_DIR, COMPANIES_DIR)
    companies = {}
    for f in os.listdir(enr_dir):
        if not f.endswith(".csv"):
            continue
        ticker = f.replace(".csv", "")
        df = pd.read_csv(os.path.join(enr_dir, f), index_col="Date", parse_dates=True)
        if df.index.min().year == 2007:
            companies[ticker] = df
    print(f"Loaded {len(companies)} companies starting from 2007")
    return companies


def load_index() -> pd.DataFrame:
    """Load enriched index CSV with all columns (Close used for target, rest for covariates)."""
    path = os.path.join(ENRICHED_DATASETS_DIR, INDEX_DIR, "^NDX.csv")
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_news() -> pd.DataFrame:
    """Load enriched news CSV, drop duplicate dates (keep last), and parse embedding strings back to numpy arrays."""
    path = os.path.join(ENRICHED_DATASETS_DIR, NEWS_DIR, "news_paper2.csv")
    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    df   = df[~df.index.duplicated(keep="last")]
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else None
    )
    return df


# ── helpers ───────────────────────────────────────────────────────────────────

def get_trading_dates(index: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates where the index Close is valid — the true market calendar."""
    return index["Close"].dropna().index.sort_values()


def get_all_closes(companies: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide DataFrame (dates x companies) of Close prices."""
    return pd.DataFrame(
        {f"Close_{t}": df["Close"].reindex(dates) for t, df in companies.items()},
        index=dates,
    )


def get_trading_mask(index: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """1 where the index Close is valid (market open), 0 otherwise."""
    return index["Close"].reindex(dates).notna().astype(int).rename("trading_day")


# ── news aggregation (Case 1) ─────────────────────────────────────────────────

def aggregate_news_to_trading_days(
    news: pd.DataFrame, trading_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    For each trading day t, aggregate news from [t, next_trading_day).
    Embedding  : averaged across days in window (direction, not count).
    Probs      : summed  across days in window (preserves magnitude per day).
    Label      : argmax of summed probs.
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

def save_target(index: pd.DataFrame, dates: pd.DatetimeIndex, out_dir: str):
    """Index Close price — the single time series the model predicts."""
    df = pd.DataFrame({"date": dates, "Close": index["Close"].reindex(dates).values})
    df.to_parquet(os.path.join(out_dir, "target.parquet"), index=False)
    print(f"  Saved target.parquet  ({len(dates)} dates)")


def save_feature_covariates(
    dates:        pd.DatetimeIndex,
    out_dir:      str,
    index_df:     pd.DataFrame | None = None,
    trading_mask: pd.Series | None = None,
):
    """
    Time-varying calendar features known for any future date.
    index_df: if provided (Cases 1 & 3), reuse precomputed date columns from the enriched index.
              if None (Case 2), recompute for the full all_dates range which includes non-trading days.
    trading_mask: added as trading_day column in Case 2 only.
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
        df["trading_day"] = trading_mask.reindex(dates)
    df.to_parquet(os.path.join(out_dir, "feature_covariates.parquet"))
    print(f"  Saved feature_covariates.parquet")


def save_dynamic_covariates(
    index_df:   pd.DataFrame,
    all_closes: pd.DataFrame,
    news_df:    pd.DataFrame | None,
    dates:      pd.DatetimeIndex,
    out_dir:    str,
):
    """Single parquet: index technical features + all company closes + news sentiment/embeddings.
    index_df must already be sliced to INDEX_FEATURES columns (pre-sliced by the caller).
    news_df may be None or contain NaN rows where no news exists for that date."""
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

    df.to_parquet(os.path.join(out_dir, "dynamic_covariates.parquet"))
    print(f"  Saved dynamic_covariates.parquet  ({len(dates)} dates)")


# ── case builders ─────────────────────────────────────────────────────────────

def build_case1(companies: dict, index: pd.DataFrame, news: pd.DataFrame):
    """
    News aggregated forward into each trading day window [t, next_open).
    Embedding averaged across window days; probs summed (so prob > 1.0 is possible).
    Label = argmax of summed probs. Only trading days in output.
    Trading days with no news in their window produce NaN for all news columns.
    """
    print("\nBuilding case_1_agg_news...")
    out_dir = os.path.join(DATASETS_DIR, "case_1_agg_news")
    os.makedirs(out_dir, exist_ok=True)

    trading_dates  = get_trading_dates(index)
    all_closes     = get_all_closes(companies, trading_dates).ffill().bfill()
    agg_news       = aggregate_news_to_trading_days(news, trading_dates)
    index_features = index[[c for c in INDEX_FEATURES if c in index.columns]]

    save_target(index, trading_dates, out_dir)
    save_feature_covariates(trading_dates, out_dir, index_df=index)
    save_dynamic_covariates(index_features, all_closes, agg_news, trading_dates, out_dir)


def build_case2(companies: dict, index: pd.DataFrame, news: pd.DataFrame):
    """
    Keeps trading days AND calendar days that have news, drops blank weekends.
    trading_day=1 marks market-open rows; trading_day=0 marks news-only (non-trading) rows.
    Company closes and index features are ffill+bfill across all kept dates.
    News stays on its original publication date — no aggregation, no fill.
    Date features (sin/cos) recomputed from scratch to cover non-trading days.
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
    has_news  = news.reindex(full_range)["prob_positive"].notna()
    all_dates = full_range[has_close | has_news]

    all_closes     = get_all_closes(companies, all_dates).ffill().bfill()
    trading_mask   = get_trading_mask(index, all_dates)
    index_filled   = index.reindex(all_dates).ffill().bfill()
    index_features = index_filled[[c for c in INDEX_FEATURES if c in index_filled.columns]]

    save_target(index, all_dates, out_dir)
    save_feature_covariates(all_dates, out_dir, trading_mask=trading_mask)
    save_dynamic_covariates(index_features, all_closes, news, all_dates, out_dir)  # index_df omitted — recomputes for non-trading days


def build_case3(companies: dict, index: pd.DataFrame, news: pd.DataFrame):
    """
    Only trading days kept. Weekend and holiday news discarded entirely.
    """
    print("\nBuilding case_3_discard...")
    out_dir = os.path.join(DATASETS_DIR, "case_3_discard")
    os.makedirs(out_dir, exist_ok=True)

    trading_dates  = get_trading_dates(index)
    all_closes     = get_all_closes(companies, trading_dates).ffill().bfill()
    news_trading   = news.reindex(trading_dates)
    index_features = index[[c for c in INDEX_FEATURES if c in index.columns]]

    save_target(index, trading_dates, out_dir)
    save_feature_covariates(trading_dates, out_dir, index_df=index)
    save_dynamic_covariates(index_features, all_closes, news_trading, trading_dates, out_dir)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    companies = load_companies_2007()
    index     = load_index()
    news      = load_news()

    build_case1(companies, index, news)
    build_case2(companies, index, news)
    build_case3(companies, index, news)

    print("\nDone. Output in:", DATASETS_DIR)
