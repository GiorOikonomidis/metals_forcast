# Dataset Builder — Build Logic

## Overview

Reads three enriched sources and produces two parquet dataset variants.

```
data_enriched/
  companies/   <- one CSV per ticker (OHLCV + technical indicators)
  index/       <- ^NDX.csv (OHLCV + technical indicators)
  news/        <- news_paper2.csv (FinBERT embeddings + sentiment probs)

datasets/
  case_mask/
  case_interp/
```

Each case outputs three parquet files:

| File | Contents |
|------|----------|
| `target.parquet` | `(id, date, Close)` — index Close, the prediction target |
| `feature_covariates.parquet` | `(date, sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy [, trading_day])` — known before market opens |
| `dynamic_covariates.parquet` | `(id, date, index_features×11, TICKER__FEATURE×N, news_cols)` — observed covariates |

---

## Constants

```python
NEWS_CUTOFF_DATE = pd.Timestamp("2026-01-01")
```

All datasets are truncated to dates strictly before `NEWS_CUTOFF_DATE`. News data beyond this point is unavailable, so the date range is cut rather than flagged.

---

## Core Definitions

### Trading day
A day is a trading day **if and only if** `index["Close"]` is non-NaN on that date.
Weekends, holidays, and data gaps are excluded by this single condition.

```python
get_trading_dates(index) -> index["Close"].dropna().index
get_trading_mask(index, dates) -> index["Close"].reindex(dates).notna().astype(int)
```

### Shared date range — `_build_all_dates`
Both cases use the same date universe:

```python
full_range = pd.date_range(start=first_trading_day, end=last_trading_day, freq="D")
has_close  = index["Close"].reindex(full_range).notna()
has_news   = news.reindex(full_range)["prob_positive"].notna()
all_dates  = full_range[has_close | has_news]
all_dates  = all_dates[all_dates < NEWS_CUTOFF_DATE]
```

Days where neither a market close nor news exists (most blank weekends) are dropped.
Result: ~6 916 dates covering 2007-01-25 to 2025-12-31.

### Target
Only the ^NDX Close price. Company closes are covariates, not the target.

### `id` column
A string identifier (e.g. `"^nsdq"`) written as the first column in both
`target.parquet` and `dynamic_covariates.parquet`. Allows multiple instruments
to be concatenated into one file downstream.

### Company features — `get_all_company_features`
Builds a wide raw frame with **all OHLCV + technical columns** for every company.
Column naming: `{TICKER}__{FEATURE}` (double underscore separator).

```python
frames = {f"{ticker}__{col}": df[col]
          for ticker, df in companies.items()
          for col in df.columns}
return pd.DataFrame(frames)   # shape: (union of trading dates, n_companies * n_features)
```

~74 companies × ~17 features = ~1 258 company columns.
No alignment is done here — callers call `.reindex(dates)` themselves.

### News gap filling — `fill_news_isolated_gaps`
Before building any case, isolated single-day gaps in the news series are filled
by averaging the neighbouring days:

- Gap day must have news on **both** the day before and the day after.
- `prob_positive/negative/neutral` → arithmetic mean of the two neighbours.
- `embedding` → element-wise mean of the two neighbour arrays (shape 768).
- `label` → argmax of the filled probs.
- Multi-day runs (two or more consecutive missing days) are **not** filled.

---

## Loaders

### `load_companies_2007()`
- Reads all CSVs from `data_enriched/companies/`
- Keeps ticker only if `df.index.min().year == 2007` (common start date)
- Drops duplicate dates (keep last) to guard against re-run artefacts
- Returns `dict[ticker -> DataFrame]`

### `load_index()`
- Reads `data_enriched/index/^NDX.csv`
- Drops duplicate dates (keep last)
- Returns full DataFrame — Close → target, rest → covariates

### `load_news()`
- Reads `data_enriched/news/news_paper2.csv`
- Drops duplicate dates (keep last)
- Parses `embedding` column from stored string `"[0.12, ...]"` back to `np.array`

---

## Feature Covariates

Six sinusoidal encodings of the calendar date, computed for every date in `all_dates`:

```
sin_dow, cos_dow     period = 7   (day of week,  0=Mon ... 6=Sun)
sin_month, cos_month period = 12  (month,        1=Jan ... 12=Dec)
sin_doy, cos_doy     period = 365 (day of year,  1 ... 365)
```

Pairs encode a position on the unit circle — sin alone is ambiguous (day 90 = day 270).
These are **future covariates**: known before observing the market on any date.

`trading_day` (0/1) is appended only in `case_mask` — the market calendar is known
in advance, so it is a legitimate future covariate.

---

## Case mask — trading day flag + NaN on non-trading days

**Concept:** the model is told explicitly which rows carry real observations via the
`trading_day` flag. Non-trading rows are structurally present but explicitly marked
as "no data here."

### Pipeline

```
1. all_dates   <- _build_all_dates(index, news, cutoff=NEWS_CUTOFF_DATE)
2. trading_dates <- get_trading_dates(index)

3. company_features:
     get_all_company_features(companies)
       .reindex(trading_dates).ffill().bfill()   # fill intra-trading gaps
       .reindex(all_dates)                        # NaN on non-trading rows

4. index_features:
     index.reindex(all_dates).ffill().bfill()    # flat carry on non-trading rows

5. trading_mask <- get_trading_mask(index, all_dates)

6. target.parquet       <- index["Close"].reindex(all_dates)
                           NaN on non-trading days

7. feature_covariates   <- sin/cos cols + trading_day

8. dynamic_covariates   <- index_features + company_features + news
```

### Key behaviours

| Data | Trading days | Non-trading days |
|------|-------------|-----------------|
| Company features | Filled (ffill within trading days) | **NaN** |
| Index features | Real values | ffill carried flat |
| Target Close | Real price | **NaN** |
| `trading_day` | 1 | 0 |
| News | On publication date | NaN where no news |

**Two-step reindex for company features** is the critical pattern:
- Step 1 `.reindex(trading_dates).ffill().bfill()` — fills any intra-trading gaps
  (e.g. a company missing one trading day) using neighbouring trading day values.
- Step 2 `.reindex(all_dates)` — expands back to all dates; non-trading dates have
  no match so they receive NaN. No fill is applied here.

This ensures trading days always have values while non-trading days are always NaN.

---

## Case interp — linear time interpolation everywhere

**Concept:** the model sees a smooth continuous signal with no explicit mask.
Non-trading days receive estimated values interpolated linearly between the
surrounding real observations.

### Pipeline

```
1. all_dates <- _build_all_dates(index, news, cutoff=NEWS_CUTOFF_DATE)

2. company_features:
     get_all_company_features(companies)
       .reindex(all_dates)
       .interpolate(method="time")   # proportional to calendar distance
       .ffill().bfill()              # handle boundary NaN (no anchor on one side)

3. index_filled:
     index.reindex(all_dates)
       .interpolate(method="time")
       .ffill().bfill()

4. target.parquet       <- index_filled["Close"]
                           interpolated on non-trading days (no NaN)

5. feature_covariates   <- sin/cos cols only (no trading_day)

6. dynamic_covariates   <- index_features + company_features + news
```

### Key behaviours

| Data | Trading days | Non-trading days |
|------|-------------|-----------------|
| Company features | Real values | Linearly interpolated |
| Index features | Real values | Linearly interpolated |
| Target Close | Real price | Linearly interpolated |
| `trading_day` | Not present | Not present |
| News | On publication date | NaN where no news |

### `interpolate(method="time")` mechanics

Interpolation weights by actual calendar distance, not row count:

```
Friday  Close = 100
Saturday      = 100 + (106-100) * 1/3 = 102.0
Sunday        = 100 + (106-100) * 2/3 = 104.0
Monday  Close = 106
```

`method="time"` requires a non-NaN anchor on **both sides** of a gap. The subsequent
`.ffill().bfill()` handles the boundaries — leading NaN (no left anchor) is filled
backward from the first real value; trailing NaN (no right anchor) is filled forward
from the last real value.

---

## Comparison

| | case_mask | case_interp |
|---|---|---|
| Date range | trading + news days (~6 916) | identical |
| Company features on non-trading | **NaN** | interpolated |
| Index features on non-trading | ffill (flat carry) | interpolated |
| Target Close on non-trading | **NaN** | interpolated |
| `trading_day` column | Yes (feature covariate) | No |
| News | On publication date, NaN elsewhere | identical |
| Data cutoff | < 2026-01-01 | identical |
| Model sees | Explicit mask of valid rows | Continuous smooth stream |

---

## Output Schema

### `target.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `id` | str | constant identifier e.g. `"^nsdq"` |
| `date` | datetime | one row per date in `all_dates` |
| `Close` | float64 | NaN on non-trading rows (case_mask only) |

### `feature_covariates.parquet`

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime | |
| `sin_dow` | float64 | in [-1, 1] |
| `cos_dow` | float64 | in [-1, 1] |
| `sin_month` | float64 | in [-1, 1] |
| `cos_month` | float64 | in [-1, 1] |
| `sin_doy` | float64 | in [-1, 1] |
| `cos_doy` | float64 | in [-1, 1] |
| `trading_day` | int (0/1) | case_mask only |

### `dynamic_covariates.parquet`

| Column group | Count | Notes |
|-------------|-------|-------|
| `id` | 1 | first column, constant |
| `date` | 1 | |
| Index technical features | 11 | EMA_12, EMA_26, MACD, RSI, Stoch_K, Stoch_D, Williams_R, ROC, Daily_Return, Volatility, Movement |
| Company features | ~1 258 | `{TICKER}__{FEATURE}` for ~74 companies × ~17 features |
| `prob_positive` | 1 | NaN where no news |
| `prob_negative` | 1 | NaN where no news |
| `prob_neutral` | 1 | NaN where no news |
| `label` | 1 | "positive"/"negative"/"neutral" or NaN |
| `embedding` | 1 | list[float] len=768 or None |
