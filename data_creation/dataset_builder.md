# Dataset Builder — Build Logic

How step 7 (`merge`) turns three enriched sources into the three parquets
`model_impl` consumes. See [`README.md`](README.md) for the pipeline as a whole.

## Overview

```
<base-dir>/<dataset>/data_enriched/
  target/       <- <target_ticker>.csv   OHLCV + technical indicators
  covariates/   <- one CSV per ticker    OHLCV + technical indicators

<base-dir>/news/<topic>/
  news_enriched.csv                      per-day embeddings + sentiment

                    ↓  merge

<base-dir>/<dataset>/datasets/
  target_variables.parquet
  global_covariates.parquet
  feature_covariates.parquet   (optional)
```

| File | Contents |
|------|----------|
| `target_variables.parquet` | **long** `date, id, open, high, low, close` — the prediction target |
| `global_covariates.parquet` | **wide** `date` + `{TICKER}_{feat}` covariate panel + target tech indicators + news |
| `feature_covariates.parquet` | `date` + `sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy` — known before the market opens (optional) |

Which series is the target and which are covariates comes from the **dataset
registry**, not from a branch in the builder. The same code path produces the
index dataset and the metals dataset; nothing in this module knows what an index
or a metal is. Adding a dataset does not touch this file.

Everything is written **raw**. Differencing is a model-layer transform
(`model_impl`'s `apply_differencing`, selected by `TYPE_OF_DIFF`), so one tree
serves every variant.

---

## What comes from the registry

`build_dataset` reads exactly two fields out of `dataset_config(dataset)`:

| Field | Used for |
|-------|----------|
| `target_id` | the `id` column of `target_variables.parquet`, and the prefix of the target's technical-indicator columns in the wide panel |
| `target_ticker` | (via `paths.target_path`) which enriched CSV is the target |

The covariate set is not read from the registry here — it is whatever CSVs the
download and enrichment steps left in `data_enriched/covariates/`. That is
deliberate: it means `--min-start` and a partial re-download behave sensibly,
and the builder never disagrees with what is actually on disk.

---

## Constants that shape the output

Defined in [`constants.py`](constants.py):

| Constant | Default | Effect |
|----------|---------|--------|
| `GLOBAL_INCLUDE_TECH_FEATURES` | `True` | Fold `TECH_FEATURES` for the target and every covariate into `global_covariates`. |
| `WRITE_FEATURE_COVARIATES` | `True` | Emit `feature_covariates.parquet` (sin/cos calendar encodings). |
| `WRITE_NEWS_TO_GLOBAL` | `True` | Fold the news `embedding` into `global_covariates`, written **dense** (see [News](#news)). Off → a news-less global file. |
| `NEWS_INCLUDE_SENTIMENT` | `True` | Also write `prob_positive/negative/neutral` and `label` beside the embedding. |
| `GLOBAL_COV_SEP` | `_` | Wide-column separator, e.g. `AAPL_close`. |
| `GLOBAL_COVARIATE_PRICE_COLS` | `["Open","High","Low","Close"]` | Price columns copied into the wide panel (lowercased on write). |
| `TARGET_OHLC_COLS` | `["Open","High","Low","Close"]` | OHLC columns written to the long target file (lowercased on write). |

```python
TECH_FEATURES = [c for c in INDEX_FEATURES if c not in ("Open","High","Low","Volume")]
```

Note what is **not** here: there is no cutoff-date constant and no dataset-mode
constant.

- The cutoff is `--cutoff-date`, defaulting to none. It was a hardcoded
  `pd.Timestamp("2026-01-01")`, which meant every build silently stopped at a
  date that quietly went stale as time passed the constant.
- `DATASET_MODE` / `MODE_CONFIG` are gone. They were *printed* by the builder as
  though they selected the source mapping, while the body was unconditionally
  index/company logic — decorative toggles documented as working ones, which is
  worse than not having them.

---

## Core definitions

### Trading day
A day is a trading day **iff** `target["Close"]` is non-NaN on that date.
Weekends, holidays, and data gaps are all excluded by this single condition.

### Shared date range — `_build_all_dates`

```python
full_range = pd.date_range(first_trading_day, last_trading_day, freq="D")
keep       = target["Close"].reindex(full_range).notna()
if has_sentiment(news):
    keep = keep | news.reindex(full_range)["prob_positive"].notna()
elif news is not None:
    keep = keep | full_range.isin(news.index)      # embedding-only model
all_dates = full_range[keep]
if cutoff_date is not None:
    all_dates = all_dates[all_dates < cutoff_date]
```

Calendar days between the first and last trading day where at least one of
(market open, news published) is true. Blank weekends are dropped.

The three-way branch exists because **MiniLM writes no `prob_*` columns**. The
old code unconditionally did `news.reindex(...)["prob_positive"]` here, so
`--model minilm` — an advertised choice — died with `KeyError: prob_positive` at
the last step of a multi-hour run. Presence of a row is the fallback test.

### Target
The target's `Close` is what gets predicted. Its own technical indicators are
covariates, and when `GLOBAL_INCLUDE_TECH_FEATURES` is on they are written into
`global_covariates` under a `{target_id}_{FEATURE}` prefix. All other series are
covariates.

### `id` column
The registry's `target_id` (`^nsdq`, `XCU`) is written into the `id` column of
`target_variables.parquet`, and `model_impl` matches it via `TARGET.ID`.
`global_covariates.parquet` carries **no** `id` column — it is wide and shared
across dates.

### Covariate panel — `build_covariate_global_panel`
For every covariate, price columns become `{TICKER}_{feature}` lowercased
(`AAPL_close`) and, when `include_tech` is set, technical indicators are appended
as `{TICKER}_{FEATURE}` kept as-is (`AAPL_RSI`). A leading `^` in a ticker or id
is stripped from the prefix (`_clean_prefix`), matching the metals convention
(`^STOXX50E` → `STOXX50E`) — `^` is disallowed in MLflow param/tag keys.
Alignment to the output dates is the caller's job, not this function's.

### Covariate loading — `load_covariates`
Reads every `.csv` in `data_enriched/covariates/`, deduplicating by date (keep
last). With `--min-start`, tickers whose history begins after that date are
excluded **and every exclusion is printed by name and start date**.

> This used to be `load_companies_2007`, which required `df.index.min().year == 2007`
> — exact equality, silently. For the metals dataset that discarded REMX (listed
> 2010) and MP (2020): a third of the covariate set, one of them
> correlation-validated at `|r| = 0.410` against copper. The bound is now
> relative, optional, and always reported.

### News gap filling — `fill_news_isolated_gaps`
Before the build, isolated single-day gaps in the news series (missing, but with
news on both the preceding and following calendar day) are filled by averaging
the neighbours: `prob_*` and `embedding` → element-wise mean, `label` → argmax of
the filled probs. Runs of two or more consecutive missing days are left unfilled.
Returns the frame unchanged when it has no probability columns.

The argmax maps a position back to a name through `PROB_LABELS`, so
`PROB_COLS` and `PROB_LABELS` must stay in the same order. They are adjacent in
`constants.py` with a comment saying so.

---

## Build — `build_dataset`

Non-trading days are linearly interpolated (`method="time"` — weighted by
calendar distance, not row count); boundary NaN (no anchor on one side) is closed
with `ffill().bfill()`. News is the exception: it stays on its publication dates
and is never interpolated.

```
1. all_dates       <- _build_all_dates(target, news, cutoff_date)
2. target_filled   <- target.reindex(all_dates).interpolate("time").ffill().bfill()

3. target_variables.parquet
     <- save_target_variables_long(target_filled, all_dates, out_dir, id=target_id)
        columns: date, id, open, high, low, close   (OHLC lowercased)

4. covariate_panel <- build_covariate_global_panel(covariates, GLOBAL_INCLUDE_TECH_FEATURES)
                        .reindex(all_dates).interpolate("time").ffill().bfill()
   target_tech     <- target_filled[TECH_FEATURES] renamed {target_id}_{FEATURE}   (if tech on)

5. global_covariates.parquet
     <- save_global_covariates_wide(covariate_panel, target_tech, news, all_dates, out_dir)
        columns: date + {TICKER}_{feat} (+ tech) + news (embedding [+ sentiment])

6. feature_covariates.parquet                          (if WRITE_FEATURE_COVARIATES)
     <- save_feature_covariates(all_dates, out_dir)
        columns: date, sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy
```

### Behaviour summary

| Data | Trading days | Non-trading days |
|------|--------------|------------------|
| Target OHLC | real prices | linearly interpolated |
| Covariate panel / tech | real values | linearly interpolated |
| News (embedding, probs) | on publication date | see [News](#news) |
| Calendar features | deterministic from the date | deterministic from the date |

`interpolate(method="time")` requires a non-NaN anchor on **both** sides of a
gap; the trailing `.ffill().bfill()` closes leading and trailing boundary gaps.

---

## News

News is optional at every level.

- **No news at all.** If `news_enriched.csv` is absent, `load_news` prints
  `no enriched news at <path> - building without news` and returns `None`; the
  build proceeds on the trading calendar alone. This is what makes
  `--skip news,news-feat` a usable way to get a price-only dataset quickly.
- **Embedding-only model.** `has_sentiment()` gates every access to the `prob_*`
  columns. MiniLM output has none, and nothing downstream assumes otherwise.
- **`WRITE_NEWS_TO_GLOBAL` off.** No news columns are written at all.

When news is written:

- **`embedding` is dense.** Days without news get a zero vector, not NaN. This is
  required: `model_impl`'s news loader
  (`loaders.global_covariates(..., embedding=True)`) calls `np.array(v)` on
  **every** row and reads `.shape`, so a NaN cell breaks it. The dimension is
  inferred from the first real vector — 768 for FinBERT and FinancialBERT, 384
  for MiniLM — so nothing here is fixed to a particular model.
- **Sentiment** (`prob_positive/negative/neutral`, `label`) is added when
  `NEWS_INCLUDE_SENTIMENT` is on *and* the model produced it. Left NaN on
  newsless days; the model does not read it.

> **Handshake with `model_impl`.** With `WRITE_NEWS_TO_GLOBAL` off, the global
> parquet has no `embedding` column and `model_impl` must run with
> `no_news=True`. Mismatching the two produces
> `ArrowInvalid: No match for FieldRef.Name(embedding)`.

Which news file feeds this is the dataset's `news_topic`, resolved through
`paths.news_enriched_path`. A metals build reads `news/metals/news_enriched.csv`
and an index build reads `news/stocks/news_enriched.csv` — the filename used to
be a hardcoded `news.csv`, so a metals build enriched and consumed the stocks
feed.

---

## Output schema

### `target_variables.parquet` (long)

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime | one row per date in `all_dates` |
| `id` | str | constant, from the registry's `target_id` (`^nsdq`, `XCU`) |
| `open`, `high`, `low`, `close` | float64 | raw interpolated levels (lowercased) |

### `global_covariates.parquet` (wide, no `id`)

| Column group | Count | Notes |
|--------------|-------|-------|
| `date` | 1 | |
| `{TICKER}_open/high/low/close` | n_covariates × 4 | price panel, lowercased |
| `{TICKER}_{IND}`, `{target_id}_{IND}` | (n_covariates + 1) × 11 | technical indicators — only if `GLOBAL_INCLUDE_TECH_FEATURES` |
| `embedding` | 1 | dense `list[float]` per row — only if `WRITE_NEWS_TO_GLOBAL` |
| `prob_positive/negative/neutral`, `label` | 4 | NaN on newsless days — only if the model produced sentiment and `NEWS_INCLUDE_SENTIMENT` |

Worked example — the index dataset with 101 covariates, tech on, news on:

```
1 (date) + 101×4 (prices) + 102×11 (tech) + 1 (embedding) + 4 (sentiment) = 1532
```

### `feature_covariates.parquet` (optional)

| Column | Type | Notes |
|--------|------|-------|
| `date` | datetime | |
| `sin_dow, cos_dow, sin_month, cos_month, sin_doy, cos_doy` | float64 | cyclic calendar encodings in `[-1, 1]` |

---

## Isolation between datasets

Two datasets built into the same `--base-dir` share nothing except the news cache
when their topics match. Their raw downloads, enriched CSVs and output parquets
live in separate trees, so:

- a covariate ticker cannot leak from one dataset's panel into the other's
- neither can overwrite the other's parquets
- each target is read from its own `<target_ticker>.csv`

That last point is worth stating plainly, because it was not true before: the
builder read a hardcoded `^NDX.csv` as the target regardless of what was being
built, so a metals dataset would have carried the Nasdaq as its prediction
target while labelling it `XCU`.
