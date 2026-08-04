# Covariate Files — Structure Reference

Three parquet files. `target_variables.parquet` and `global_covariates.parquet`
are the two files `model_impl` actually reads, via `--target-covariate-path`
and `--global-covariate-path` in `arg_handler/cli_parser.py`.
`static_covariates.parquet` is documented here for schema reference but has no
matching CLI flag or loader in `model_impl` today — nothing consumes it yet.

Frequency: **daily (`D`)**.

---

## `target_variables.parquet`

One row per `(id, date)`. This is the target series to forecast.

| Column | Type | Description |
| --- | --- | --- |
| `date` | datetime | Trading date |
| `id` | string | Series identifier (ticker/commodity code) |
| `open` | float | Opening price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Closing price |

**Series (`id`) currently in the dataset (13 total):**
`DYS`, `PRA`, `TER`, `LTH`, `DJMc1`, `ND`, `MG`, `TUNGSTEN`, `IRON`, `LCO`, `XAG`, `XAU`, `XCU`

`IRON62` series was removed — its history was too short
(~131 days). It remains though in the target_variables_Original.parquet

---

## `static_covariates.parquet`

One row per `id`. No `date` column (per-series constant).

| Column | Type | Description |
| --- | --- | --- |
| `id` | string | Series identifier — must match `id` values in `target_variables.parquet` |
| `unit` | string (categorical) | Unit of measurement  |

---

## `global_covariates.parquet`

One row per `date`. No `id` column — shared across all series. 42 covariate columns + `date`.

| Column pattern | Description |
| --- | --- |
| `date` | Trading date |
| `<TICKER>_open`, `<TICKER>_high`, `<TICKER>_low`, `<TICKER>_close` |
| `eur_usd` | EUR/USD exchange rate |
| `eur_cny` | EUR/CNY exchange rate |

**Tickers covered (7, × 4 OHLC columns each = 28 columns):**
`AA`, `BHP`, `FCX`, `MP`, `REMX`, `RIO`, `STOXX50E`

**Additional tickers (3, × 4 OHLC columns each = 12 columns):**
`BRENTOIL`, `CL1`, `NG`

Plus `eur_usd`, `eur_cny` (2 columns) → **42 covariate columns total**.

`STOXX50E` was originally named `^STOXX50E`. The `^` was stripped from all column names and from the matching
because MLflow does not allow `^` in tag/param keys.

---

## Notes

  A `filler` preprocessing step (forward-fill) should be applied to **all** target columns, not
  just global covariates, to guard against this generally.

