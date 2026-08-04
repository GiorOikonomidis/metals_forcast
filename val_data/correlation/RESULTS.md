# Global Covariate / Target Correlation — Results

Diagnostic run over `target_variables.parquet` (XAU, XAG, XCU close prices) and
`global_covariates.parquet` (42 feature columns), produced by
[`correlation_attention.py`](correlation_attention.py). All CSV/PNG artifacts
referenced below live in [`results/`](results/).

## Method

1. **Align & difference.** Both parquet files are restricted to their common
   date range (2022-01-01 → 2026-07-12, T=1654 rows) and transformed with
   `log_diff` (`log(price).diff()`), so correlation is measured on returns,
   not on price levels that would trend together spuriously.
2. **Standardize.** Every column is z-scored (zero mean, unit variance).
3. **Correlate.** `Q @ Kᵀ / d_model` between z-scored columns gives the exact
   Pearson `r ∈ [-1, 1]` for every pair — computed both GLOBAL×GLOBAL
   (covariates vs. each other) and GLOBAL×TARGET (covariates vs. XAU/XAG/XCU).
   A temperature-scaled softmax version of the same scores is also saved
   (`*_attention.csv/png`) but correlation is the basis for every selection
   decision below — it's the statistically grounded measure; the attention
   matrix is a derived, hyperparameter-dependent visualization only.

## Stage 1 — Target relevance filter

**Rule:** a feature survives if its correlation with *at least one* target
(not necessarily all three) exceeds `|r| > 0.25`. Reasoning: a feature that's
a strong XCU predictor but weak on XAU is still valuable for forecasting
copper — requiring relevance to every target would throw that away.

**Result: 18 / 42 features survive**, all from the mining-equity block
(FCX/BHP/RIO/AA/REMX). Energy (BRENTOIL/CL1/NG), FX (eur_usd/eur_cny), and
STOXX50E are cut entirely — none clear 0.25 against any target.

| feature | XAU | XAG | XCU | max\|r\| |
|---|---|---|---|---|
| FCX_close | 0.254 | 0.410 | 0.555 | 0.555 |
| FCX_high | 0.217 | 0.371 | 0.498 | 0.498 |
| BHP_close | 0.221 | 0.394 | 0.493 | 0.493 |
| RIO_close | 0.222 | 0.392 | 0.475 | 0.475 |
| FCX_low | 0.205 | 0.344 | 0.453 | 0.453 |
| BHP_high | 0.163 | 0.338 | 0.430 | 0.430 |
| RIO_low | 0.181 | 0.332 | 0.430 | 0.430 |
| BHP_low | 0.177 | 0.335 | 0.419 | 0.419 |
| RIO_high | 0.156 | 0.310 | 0.413 | 0.413 |
| REMX_close | 0.219 | 0.380 | 0.410 | 0.410 |
| AA_close | 0.200 | 0.311 | 0.397 | 0.397 |
| AA_low | 0.174 | 0.272 | 0.351 | 0.351 |
| AA_high | 0.152 | 0.259 | 0.348 | 0.348 |
| REMX_low | 0.172 | 0.319 | 0.334 | 0.334 |
| REMX_high | 0.166 | 0.307 | 0.331 | 0.331 |
| FCX_open | 0.156 | 0.251 | 0.310 | 0.310 |
| RIO_open | 0.125 | 0.248 | 0.304 | 0.304 |
| BHP_open | 0.123 | 0.259 | 0.286 | 0.286 |

Full data: [`stage1_target_survivors.csv`](results/stage1_target_survivors.csv)
Heatmap: [`global_target_correlation.png`](results/global_target_correlation.png)

**Caveat:** nothing here clears `|r|=0.26` against XAU (gold) — gold isn't
well explained linearly by any of these industrial-mining covariates. If
XAU forecasting matters, this panel is weak for it; a macro/FX/rates proxy
would likely do better and isn't present in `global_covariates.parquet`.

## Stage 2 — Redundancy check among survivors

**Rule:** restrict the GLOBAL×GLOBAL matrix to the 18 stage-1 survivors only,
then flag any pair with `|r| > 0.85` (diagonal excluded — self-correlation is
trivially 1.0). This is a report, not an automatic drop — which field to keep
in each cluster (e.g. `_close` vs `_high`) depends on downstream context.

**Result: 14 pairs flagged**, all inside the BHP/RIO block (they move almost
as one instrument across every OHLC field) plus one REMX pair. FCX and AA
never cross 0.85 with anything, including each other.

| feature A | feature B | r |
|---|---|---|
| RIO_low | RIO_high | +0.892 |
| RIO_low | BHP_low | +0.891 |
| BHP_close | RIO_close | +0.890 |
| BHP_high | BHP_low | +0.889 |
| RIO_open | BHP_open | +0.888 |
| BHP_high | RIO_high | +0.882 |
| RIO_high | RIO_open | +0.868 |
| BHP_close | BHP_high | +0.867 |
| RIO_close | RIO_high | +0.864 |
| RIO_low | RIO_open | +0.864 |
| BHP_low | BHP_open | +0.861 |
| RIO_close | RIO_low | +0.858 |
| BHP_close | BHP_low | +0.855 |
| REMX_low | REMX_high | +0.855 |

Full matrix: [`stage2_survivor_redundancy.csv`](results/stage2_survivor_redundancy.csv)
Heatmap (lime border + dot marks flagged cells): [`stage2_survivor_redundancy.png`](results/stage2_survivor_redundancy.png)

## Suggested read

- **BHP and RIO are near-duplicates of each other** across every OHLC field
  (`|r|` 0.86–0.89) — pick one, not both, for any given field.
- **FCX is the single strongest predictor** (`FCX_close`, max\|r\|=0.555,
  driven mostly by XCU) and stays independent enough (`|r| < 0.85` vs.
  everything) to keep alongside whichever of BHP/RIO is chosen.
- **REMX's `_low`/`_high` pair is redundant with itself**; `REMX_close` is
  the strongest of the three and already the one carried in stage 1's top 10.
- **AA is independent of the BHP/RIO/REMX cluster** (`|r| < 0.85` throughout)
  and worth keeping if a fourth mining name is useful.

Config knobs (top of `correlation_attention.py`): `DIFF_MODE`,
`TARGET_HIGHLIGHT_THRESHOLD` (0.25), `REDUNDANCY_THRESHOLD` (0.85),
`ATTENTION_TEMPERATURE` (0.1, softmax visualization only).
