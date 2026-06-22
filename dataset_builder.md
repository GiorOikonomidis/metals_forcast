# Dataset Builder — Build Logic

## Overview

Reads three enriched sources and produces three parquet dataset variants.

```
data_enriched/
  companies/   ← one CSV per ticker (OHLCV + technical indicators)
  index/       ← ^NDX.csv (OHLCV + technical indicators)
  news/        ← news_paper2.csv (FinBERT embeddings + sentiment probs)

datasets/
  case_1_agg_news/
  case_2_mask/
  case_3_discard/
```

Each case outputs three parquet files:

| File | Contents |
|------|----------|
| `target.parquet` | `(date, Close)` — index Close, the prediction target |
| `feature_covariates.parquet` | `(date, sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy [, trading_day])` — known before market opens |
| `dynamic_covariates.parquet` | `(date, index_features×11, Close_×74, news_cols)` — observed covariates |

---

## Core Definitions

### Trading day
A day is a trading day **if and only if** `index["Close"]` is non-NaN on that date.  
Weekends, holidays, and data gaps are excluded by this single condition.

```python
get_trading_dates(index) → index["Close"].dropna().index
get_trading_mask(index, dates) → index["Close"].reindex(dates).notna().astype(int)
```

### Target
Only the index (^NDX) Close price. Company closes are covariates.

### Fill policy
| Data type | Fill |
|-----------|------|
| Company closes | `ffill().bfill()` |
| Index features (EMA, RSI, …) | `ffill().bfill()` (Case 2 only — non-trading days) |
| News columns | **no fill** — NaN stays where no news exists |

### Company filter
Only companies whose first date in the enriched CSV is year 2007. Companies with IPOs after 2007 are excluded at load time.

---

## Loaders

### `load_companies_2007()`
- Reads all CSVs from `data_enriched/companies/`
- Keeps ticker only if `df.index.min().year == 2007`
- Returns `dict[ticker → DataFrame]`

### `load_index()`
- Reads `data_enriched/index/^NDX.csv`
- Returns full DataFrame — Close → target, rest → covariates

### `load_news()`
- Reads `data_enriched/news/news_paper2.csv`
- Parses `embedding` column from string `"[0.12, ...]"` back to `np.array` via `ast.literal_eval`

---

## Feature Covariates

Six sinusoidal encodings of the calendar date:

```
sin_dow, cos_dow     period = 7   (day of week,  0=Mon … 6=Sun)
sin_month, cos_month period = 12  (month,        1=Jan … 12=Dec)
sin_doy, cos_doy     period = 365 (day of year,  1 … 365)
```

Pairs encode a position on the unit circle — sin alone is ambiguous (day 90 = day 270).  
These are **future covariates**: known before observing the market on any date.

`trading_day` (0/1) is also a future covariate — the market calendar is published in advance — so it lives here in Case 2, not in dynamic covariates.

### Source of date features per case

`price_feat_gen.py` already computes and saves these 6 columns into the enriched index CSV as part of `generate_features()`. Cases 1 and 3 operate on trading days only — the same dates already present in the index — so they **reuse** those precomputed columns directly via `index_df=index`.

Case 2 covers a wider date range (`full_range[has_close | has_news]`) that includes non-trading days not present in the index, so it **recomputes** the 6 columns from scratch for its date range.

---

## Case 1 — Aggregate news forward

**Date range:** trading days only.

**News strategy:** for each trading day `t`, collect all news from `[t, next_trading_day)`.  
Weekend and holiday news is folded into the preceding trading day's window.

```
Friday window  = [Friday, Monday)  → includes Fri + Sat + Sun headlines
Monday window  = [Monday, Tuesday) → Monday only (or until next gap)
```

Aggregation within the window:
- `embedding` → `mean(axis=0)` across days, shape `(768,)`
- `prob_positive/negative/neutral` → `sum()` across days (preserves magnitude)
- `label` → `argmax([sum_pos, sum_neg, sum_neu])`

If no news in window → all news columns are `NaN/None` for that trading day.

**Output:** trading days only, no blank rows, no fill on news.

---

## Case 2 — Full calendar with trading mask

**Date range:** every calendar day from `index_min` to `index_max`, **filtered** to keep only days where at least one of:
- index Close exists (`trading_day = 1`), or
- news exists for that date

Days with neither (most blank weekends) are dropped.

```python
has_close = index["Close"].reindex(full_range).notna()
has_news  = news.reindex(full_range)["prob_positive"].notna()
all_dates = full_range[has_close | has_news]
```

**News strategy:** news stays on its original publication date — no aggregation, no fill.  
A weekend with news keeps its row; a weekend without news is discarded.

**Index features / company closes:** `ffill().bfill()` across the kept dates so non-trading rows have a valid price context.

**`trading_day`** column in `feature_covariates.parquet` tells the model which rows to compute loss on.

**Output:** trading days + news-bearing non-trading days. Mixed sequence, model attends to all rows but loss only on `trading_day = 1`.

---

## Case 3 — Discard non-trading days

**Date range:** trading days only.

**News strategy:** `news.reindex(trading_dates)` — only news published on an actual trading day survives. Weekend and holiday news is **discarded entirely**.

**Output:** trading days only, simplest output, news is sparse.

---

## Comparison

| | Case 1 | Case 2 | Case 3 |
|---|---|---|---|
| Date range | Trading days | Trading + news days | Trading days |
| Weekend news | Aggregated into Friday | Kept on original date | Discarded |
| News fill | None | None | None |
| Price fill | ffill+bfill on trading days | ffill+bfill across all kept dates | ffill+bfill on trading days |
| trading_day column | No | Yes (feature covariate) | No |
| Sequence length | ~4 800 rows (2007–2024) | ~4 800 + news-only days | ~4 800 rows |
