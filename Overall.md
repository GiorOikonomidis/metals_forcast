## Repo overview

Two independent pipelines, linked only by the parquet files the first produces
and the second consumes. This file is the top-level summary; each piece below
has its own, more detailed docs — links inline.

**`data_creation/`** — builds the datasets. Installs as its own package
(`pip install -e .`). Downloads price series and news, enriches prices with
technical indicators (EMA/RSI/MACD/Stochastic/Williams %R/ROC/volatility/etc.)
and news with transformer sentiment + embeddings, then merges everything into
the parquet trees `model_impl` reads. Which dataset gets built is a single
`--dataset` argument, resolved against one registry (`DATASETS` in
`constants.py`) so the target id, its covariates, and its news feed can never
disagree with each other. Docs: [data_creation/README.md](data_creation/README.md)
(usage), [data_creation/dataset_builder.md](data_creation/dataset_builder.md)
(merge/build internals).

**`model_impl/`** — the model itself. A cross-attention architecture on top of
a frozen Chronos-T5 encoder: index prices, daily news embeddings, and a
covariate panel attend to each other, and the fused representation is decoded
into Chronos tokens over a forecast horizon, scored against a naïve random-walk
baseline. Driven by one YAML config file (every key has a default) plus two
CLI-supplied parquet paths. Docs: [model_impl/README.md](model_impl/README.md)
(usage guide, CLI/config reference), [model_impl/code_structure.md](model_impl/code_structure.md)
(architecture, module/API reference), [model_impl/data_loading/covariates-structure.md](model_impl/data_loading/covariates-structure.md)
(exact parquet schemas).

**`produced_data/`** — the parquet trees `data_creation` writes and `model_impl`
reads: one subtree per dataset (`metals`, `index`, `synthetic`, `news`), each
holding raw downloads (`data/`), enriched-with-indicators versions
(`data_enriched/`), and the final merged `datasets/` (`target_variables.parquet`,
`global_covariates.parquet`, `static_covariates.parquet`) — the only files
`model_impl` actually loads.

**`val_data/`** — standalone analysis scripts, not part of either pipeline:
correlation/attention checks between targets and covariates
(`val_data/correlation/`, see [RESULTS.md](val_data/correlation/RESULTS.md)),
the synthetic-signal generators used for sanity-checking the model
(`synthetic_signals.py`), and token-distribution diagnostics
(`val_data/tokens_distribution/`).

**`experiments/`** — sweep configs and the runner script
(`generate_sweep.py`, `run_sweep.sh`) that produced the results this file
draws on; `experiments/xcu_sweep/*.yaml` are the individual per-run configs.

**`output/` / `mlruns/`** — local run artifacts (per-run figures, CSVs,
`summary.json`) and the MLflow tracking store, respectively — both are run
outputs, not source.

---

## Conclusions

Alot of expiremerrnts wher performed and despite hyper-parameter tunning and model architecture changes ,
thre results where same patterned . In the log_diff mode the model always chooses the last day as the new prediction.
To exlude code bugs as the root cause we performed debuggin  and also run the model with synthetic data . The debug show 
no error and also the model performed well on the synthetic data , with actually try of learning the data and no sight of the 
pattern described above , thus leads to data investigation .
In the data examination we found that there apperaing spikes as delta functions where nowhere could be appear , nor there internet or the yahoo finance. APi erorr might corrupted the data . Also person correlation was performed in the targets XCU , XAG , XAU with the globla covs . The results 
show no significant correlation between them (For the embedding pca was performed to reduce their diem).
Our work indicates the current bottle-neck is the data . We suggest for more meaningfull cleanup of the spikes and also the search of covariates with 
better corellation to the target .

**The models prediction pattern is due to the signal nature . In random walk signals the best choice to minimise the pred error is to go with the previius day. We supect that only in the presence of meaningfull covs the model might change the pattern .

---

## Observations from the results sweep

Reorganized results and full per-run breakdown: `results\INDEX.md`
(20 runs total: 11 real XCU, 6 synthetic, 3 incomplete/killed).

| # | Observation |
|---|---|
| 1 | The model **can** learn deterministic structure when it's actually there — SINE/SINC/SQUARE all converge to near-zero error (skill ≈99.9999%). Rules out a basic architecture/training-loop bug as the cause of the XCU pattern. |
| 2 | The model correctly detects genuine causal return structure **only where it exists**: an AR(1)-return synthetic signal (φ=0.3, built specifically to have real lag-1 autocorrelation, unlike XCU) is the only real-return-generating-process signal that beats naive. Every real-XCU config fails to beat naive, consistent with XCU's own measured ≈0 return autocorrelation at every lag. |
| 3 | On real XCU, `log_diff` + transformer/small-context configs get closest to naive parity (best: −0.19% to −0.49% skill); `no_diff` configs are uniformly worse (worst: −317.6%, −127.8%). |
| 4 | Calibration flips between badly **overconfident** (no-news, no-diff runs: 50% interval only covers truth 6–11% of the time) and badly **underconfident** (news-enabled runs: 50% interval covers 80–96% of the time) depending on config — no run found a middle ground. |
| 5 | One `no_diff` + news run shows interval width (`sharp80`) exploding to roughly XCU's entire price range, an order of magnitude above every other run — looks like MC-Dropout variance blowing up specifically in that combination. |
| 6 | The only real-XCU config that beat naive used a 50-day horizon (`pred_len=50`, `ctx_len=365`) instead of 7 — reproducible across two separate runs. Every 7-day-horizon real-XCU run underperforms naive. Worth a deliberate horizon sweep rather than treating as a one-off. |
| 7 | One run's covariate count doesn't add up: its config declares 11 XCU-technical + 49 cross-asset covariates (60 total), but the output folder name and `summary.json` both say 49. Until resolved, it's unclear whether that run (currently the worst real-data result) is a genuine "feature covariates hurt" finding or a silent covariate-drop bug. |
| 8 | The one fully-deterministic modulated-sine synthetic signal underperformed naive (−18.7%), which doesn't fit the pattern set by the other deterministic signals (all ≈+99.9999%) — worth rerunning with more epochs before drawing conclusions from it. |
| 9 | 3 of 20 launched runs never finished (one crashed within seconds with no captured traceback, one was killed with validation loss starting to climb again, one was killed early mid-run) — three different failure modes, each needs its own log inspection. |

## Suggested next steps

1. **Resolve the covariate-count discrepancy (observation 7)** before drawing any conclusion about feature covariates helping or hurting — right now it's not known whether that run actually used them.
2. **Investigate the calibration split (observations 4–5)** before more hyperparameter sweeps: figure out what specifically flips the model between overconfident and underconfident (news on/off looks like the main correlate) rather than tuning further on point-forecast metrics alone.
3. **Follow up on the longer-horizon result (observation 6)** with a deliberate `pred_len` sweep (e.g. 7 / 14 / 30 / 50) instead of leaving it as a one-off — if longer horizons are genuinely more learnable in log-diff space, that changes where effort should go.
4. **Re-run the modulated-sine synthetic (observation 8)** with more epochs / matched settings to the other synthetic signals before treating "the model can't learn AM+FM+trend" as a real finding rather than an undertrained run.
5. **Chase down the 3 incomplete runs** — especially the one that crashed within 5 log lines (observation 9), since a silent crash with no traceback is worth fixing regardless of whether that particular config mattered.
6. Beyond the sweep itself: the data-quality spikes flagged above remain unaddressed, and testing genuine cross-asset **lead-lag** (not just contemporaneous correlation) on the covariate candidates is still an open, unverified assumption behind every covariate-based run in this sweep.
