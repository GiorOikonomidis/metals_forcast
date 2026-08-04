# proakt — model implementation

Forecasts an index's Close price with a cross-attention model on top of a **frozen
Chronos encoder**. Three streams — index prices, daily news embeddings, and a company
panel — attend to each other bidirectionally; the fused representation is decoded into
Chronos tokens over a `pred_len` horizon and scored against a naïve random walk.

Consumes the parquet datasets built by [`data_creation`](../data_creation/README.md).

For architecture, the import hierarchy, and the full module/API reference, see
[code_structure.md](code_structure.md). This file is the usage guide.

---

## Requirements

- **Python ≥ 3.10** (the code uses `X | None` union syntax and dataclass field
  defaults that rely on 3.10+ semantics).
- A virtual environment is strongly recommended.
- Dependencies are declared in [`requirements.txt`](requirements.txt): numpy, pandas,
  scipy, torch, transformers, chronos-forecasting, sentence-transformers, scikit-learn,
  properscoring, matplotlib, seaborn, tqdm, pyarrow, pyyaml, mlflow.
- A CUDA GPU is optional — `utils/runtime_utils.DEVICE` probes for one and falls back to CPU.

## Installation

**Windows (PowerShell)**
```powershell
cd model_impl
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS**
```bash
cd model_impl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> For CUDA, install torch from the PyTorch index first (see the commented line in
> `requirements.txt`), then install the rest.

---

## Quick start

Run from the **repo root** (not from inside `model_impl/`):

```bash
python -m model_impl.main \
  --config exampl.yaml \
  --target-covariate-path /path/to/target_variables.parquet \
  --global-covariate-path /path/to/global_covariates.parquet
```

`python model_impl/main.py ...` also works — `main.py` puts the repo root on
`sys.path` when it detects it wasn't launched via `-m`.

---

## CLI reference

`arg_handler/cli_parser.py` exposes five flags:

| Flag | Required | Behavior |
|---|---|---|
| `--config` | **yes** | Path to a yaml config file. If the file doesn't exist, the run still proceeds — every setting falls back to its built-in default (see [Config file](#config-file)) with a logged warning. |
| `--target-covariate-path` | **yes** | Path to the long-format `target_variables.parquet` — one row per `(id, date)`. Supplies **both** the forecast target (`DATA.TARGET.ID`/`FEATURE`) and any per-id covariates (`DATA.COVARIATES`) from the same file. **Validated at startup** — if the file doesn't exist, the process terminates immediately with `--target-covariate-path: file not found: '...'`. |
| `--global-covariate-path` | **yes** | Path to the wide `global_covariates.parquet` — one row per date, a scalar column per global covariate plus the news `embedding` column. Same fail-fast validation as above. |
| `--dynamic-covariate-path` | no | Accepted but **not read anywhere** — `main.py` never references `opts.dynamic_covariate_path`. Left over from an earlier single-file covariate layout; safe to omit. |
| `--feature-covariate-path` | no (default `""`) | Path to `feature_covariates.parquet`. Accepted but **not currently used** — no loader reads it yet. May be omitted entirely. |

Everything else (which series to forecast, covariate selection, differencing mode,
model hyperparameters, training schedule, evaluation settings) is controlled by
`--config`, not by additional flags. A handful of former debug/ablation flags
(`--no-news`, `--debug`, `--faithfulness`, `-w/--windows`, `-c/--companies`,
`--max-news-per-day`) are commented out in `cli_parser.py`, not deleted —
`main.py` currently hardcodes their old defaults (`no_news=True`, `debug_vis=False`,
`faithfulness=False`, `run_validation_suite=False`, and `WINDOWS` comes from the
config's `EVALUATION.WINDOWS` instead).

---

## Config file

A single yaml file drives everything else about the run. **Every setting has a
built-in default** — an omitted key, an omitted section, or a missing/empty file
all resolve the same way: that value falls back to its default. There is no
required key anywhere in the config; `exampl.yaml` at the repo root is both a
working example and effectively a listing of every default value.

### Structure

```yaml
DATA:
  # The single series to forecast, selected from the long-format file
  # (--target-covariate-path).
  TARGET:
    ID: "IRON"
    FEATURE: "close"

  # Per-id covariates from that same long file: each entry is [id, [features]].
  # [] is valid — see model.no_covariates in code_structure.md.
  COVARIATES:
    - ["XAU", ["close"]]
    - ["XAG", ["close"]]
    - ["XCU", ["close"]]

  # Scalar column names from the wide global file (--global-covariate-path).
  # [] is valid, same as COVARIATES.
  GLOBAL_COVARIATES: ["AA_close", "BHP_close", "eur_usd", ...]

  NEWS_COL: "embedding"            # embedding column name in the global file

  # End-anchored, duration-based splits (row counts on the aligned series):
  # test = last TEST_DAYS rows, val = VAL_DAYS rows before that, train = the rest.
  # (Not calendar dates — there is no TRAIN_DATE/VAL_DATE/TEST_DATE.)
  SPLITS:
    TEST_DAYS: 250
    VAL_DAYS: 250

  PRED_LEN: 7                     # forecast horizon (days)
  CTX_LEN: 150                    # context window (days)
  TOKEN_ALL: true                 # true = one scale for the whole split; false = per-window
  SHUFFLE_DATA: true              # shuffle train/val DataLoader batches
  TYPE_OF_DIFF: "log_diff"        # no_diff | diff | log_diff

SEED: 1

MODEL:
  COMP_ENC: "amazon/chronos-t5-base"
  CROSS_CHRONOS:
    configs:
      EMB_DIM_NEWS: 768
      D_MODEL: 768
      N_HEADS: 8
      N_LAYERS_TXT: 3
      D_FF: 1024
      DROPOUT: 0.2
      HEAD: "linear"              # linear | mlp | lstm | transformer

TRAINING:
  EPOCHS: 1
  LR: 0.00001                     # base learning rate; used as-is if SCHEDULER.use is false,
                                   # otherwise the rate the cold-start schedule warms up into
  WEIGHT_DECAY: 0.0001
  GRAD_CLIP: null                 # null = no clipping
  BATCH_SIZE: 256
  LABEL_SMOOTHING: 0.15

EVALUATION:
  WINDOWS: null                   # null = evaluate every test window
  MC_SAMPLES: 100

  CENTRAL_INTERVAL:
    ALPHA_50: 0.5
    ALPHA_80: 0.2
    ALPHA_90: 0.1

  ECE_QUANTILE_GRID:
    ECE_Q_START: 0.05
    ECE_Q_STOP: 0.95
    ECE_Q_STEPS: 19

  FAITH:
    FAITH_MC_SAMPLES: 100
    FAITH_KS: [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    FAITH_TOPK: 5
    FAITH_STABILITY_RUNS: 5
    FAITH_PLACEBO_SHIFTS: [-3, -1, 1, 3]
    FAITH_MASK_STRATEGY: "mean"   # mean | zero
    FAITH_RNG_SEED: 123

SCHEDULER:
  use: true                       # false = fixed LR (TRAINING.LR), ColdStartScheduler never built
  configs:
    LR_COLD_START: 0.00001
    LR_COLD_EPOCHS: 30
    LR_DECREASE_FACTOR: 0.5
    LR_METRIC: "val"               # val | train
    LR_PLATEAU_PATIENCE: 10
    # No LR_RUNNING key — the plateau decays down from TRAINING.LR, not a
    # second independent rate. An earlier version of this doc listed one;
    # it isn't read by config_file_parser.py.

EARLY_STOPPER:
  use: true                       # false = train for the full EPOCHS, no early exit
  configs:
    EARLY_STOPPER_PATIENCE: 20

# Two independent tracking backends, nested under TRACKING — either can be
# switched off on its own.
TRACKING:
  MLFLOW:
    use: true                     # false = no MLflow calls at all, run behaves exactly as before
    configs:
      URI: ""                    # empty = MLflow's own local ./mlruns
      EXPIREMENT: ""              # empty = MLflow's own "Default" experiment (yes, "EXPIREMENT" — kept as-is, see below)
      RUN: ""                    # empty = same name as the local output dir, e.g. "log_diff_c_74_w_news/20260720"
      LOG_MODEL: false            # true = also log the trained model as an MLflow artifact
  LOCAL:
    use: true                     # false = outdir is still computed (for the MLflow run name)
                                   # but never created and nothing is written to disk locally
    configs:
      dir: ""                    # empty = model_impl.consts.OUTPUT_ROOT is used as the base dir
```

### Options reference

Every key below, its default (from `utils/schemas/`), and what it controls.
An omitted key always falls back to this default — nothing here is required.

**`DATA`**

| Key | Default | Description |
|---|---|---|
| `TARGET.ID` | `"IRON"` | Id of the series to forecast, selected from the long-format file (`--target-covariate-path`). |
| `TARGET.FEATURE` | `"close"` | Column of that id to forecast (must exist in the long file, e.g. `open/high/low/close`). |
| `COVARIATES` | `[]` | Per-id covariates from the same long file. List of `[id, [features]]` pairs, e.g. `[["XAU", ["close"]], ["XCU", ["close"]]]`. `[]` (no per-id covariates) is valid — `MultiCrossChronos` falls back to a learned placeholder embedding. |
| `GLOBAL_COVARIATES` | `[]` | Scalar column names loaded from the wide global file (`--global-covariate-path`), e.g. `["AA_close", "eur_usd"]`. `[]` is valid, same as `COVARIATES`. |
| `NEWS_COL` | `"embedding"` | Name of the news-embedding column in the global file. |
| `SPLITS.TEST_DAYS` | `250` | Row count of the test split — the last `TEST_DAYS` rows of the aligned series. |
| `SPLITS.VAL_DAYS` | `250` | Row count of the val split — the `VAL_DAYS` rows immediately before the test split. Train is everything before that. |
| `PRED_LEN` | `7` | Forecast horizon, in days. |
| `CTX_LEN` | `150` | Context window length, in days. |
| `TOKEN_ALL` | `true` | `true` = one Chronos scale shared across the whole split (train's scale reused for val/test); `false` = each window computes its own scale independently. |
| `SHUFFLE_DATA` | `true` | Shuffle train/val `DataLoader` batches each epoch. |
| `TYPE_OF_DIFF` | `"log_diff"` | Representation applied to target (and, unless disabled, covariates) at load time: `no_diff` \| `diff` (`x(t)-x(t-1)`) \| `log_diff` (`log(x)` then diff; falls back to plain diff on non-positive columns). |

**`SEED`**

| Key | Default | Description |
|---|---|---|
| `SEED` | `1` | Top-level RNG seed (`setup_runtime`), not nested under any section. |

**`MODEL`**

| Key | Default | Description |
|---|---|---|
| `COMP_ENC` | `"amazon/chronos-t5-base"` | Pretrained Chronos checkpoint used as the frozen encoder/tokenizer. |
| `CROSS_CHRONOS.EMB_DIM_NEWS` | `768` | Dimensionality of the news-embedding input (must match what the global file's `embedding` column actually holds). |
| `CROSS_CHRONOS.D_MODEL` | `768` | Cross-attention model width. |
| `CROSS_CHRONOS.N_HEADS` | `8` | Attention head count. |
| `CROSS_CHRONOS.N_LAYERS_TXT` | `3` | Number of cross-attention layers over the news/text stream. |
| `CROSS_CHRONOS.D_FF` | `1024` | Feed-forward hidden width. |
| `CROSS_CHRONOS.DROPOUT` | `0.2` | Dropout rate throughout the cross-attention block (also drives MC-Dropout sampling at inference). |
| `CROSS_CHRONOS.HEAD` | `"linear"` | Output head architecture: `linear` \| `mlp` \| `lstm` \| `transformer`. |

**`TRAINING`**

| Key | Default | Description |
|---|---|---|
| `EPOCHS` | `1` | Max training epochs (subject to `EARLY_STOPPER` if enabled). |
| `LR` | `0.00001` | Base learning rate — used as-is when `SCHEDULER.use` is false; otherwise the rate the cold-start schedule warms up into and its plateau decay works down from (not a second, independent rate). |
| `WEIGHT_DECAY` | `0.0001` | AdamW weight decay. |
| `GRAD_CLIP` | `null` | Gradient-norm clip value; `null` = no clipping. |
| `BATCH_SIZE` | `256` | Training/validation `DataLoader` batch size. |
| `LABEL_SMOOTHING` | `0.15` | Label smoothing applied to the token-classification cross-entropy loss. |

**`EVALUATION`**

| Key | Default | Description |
|---|---|---|
| `WINDOWS` | `null` | Number of test windows to evaluate; `null` = every test window. |
| `MC_SAMPLES` | `100` | Monte-Carlo forward passes per window (MC-Dropout) to build the predictive distribution. |
| `CENTRAL_INTERVAL.ALPHA_50/80/90` | `0.5 / 0.2 / 0.1` | Alpha values defining the 50%/80%/90% central prediction intervals for coverage/sharpness metrics. |
| `ECE_QUANTILE_GRID.ECE_Q_START` | `0.05` | First quantile in the ECE calibration grid. |
| `ECE_QUANTILE_GRID.ECE_Q_STOP` | `0.95` | Last quantile in the ECE calibration grid. |
| `ECE_QUANTILE_GRID.ECE_Q_STEPS` | `19` | Number of quantile points in that grid. |
| `FAITH.FAITH_MC_SAMPLES` | `100` | MC samples used specifically inside the faithfulness study. |
| `FAITH.FAITH_KS` | `[1,2,3,5,8,10,15,20,25,30]` | Top-K values swept in the faithfulness attribution check. |
| `FAITH.FAITH_TOPK` | `5` | Top-K features highlighted in faithfulness output. |
| `FAITH.FAITH_STABILITY_RUNS` | `5` | Repeat count for the faithfulness stability check. |
| `FAITH.FAITH_PLACEBO_SHIFTS` | `[-3,-1,1,3]` | Day-shifts used for the placebo/control check. |
| `FAITH.FAITH_MASK_STRATEGY` | `"mean"` | How masked covariates are filled in the faithfulness study: `mean` \| `zero`. |
| `FAITH.FAITH_RNG_SEED` | `123` | RNG seed local to the faithfulness study (independent of the top-level `SEED`). |

**`SCHEDULER`**

| Key | Default | Description |
|---|---|---|
| `use` | `true` | `false` = fixed LR (`TRAINING.LR`) for the whole run; `ColdStartScheduler` is never constructed. |
| `configs.LR_COLD_START` | `0.00001` | Learning rate held during the cold-start warmup phase. |
| `configs.LR_COLD_EPOCHS` | `30` | Number of epochs the cold-start rate is held before transitioning to `TRAINING.LR`. |
| `configs.LR_DECREASE_FACTOR` | `0.5` | Multiplicative LR decay applied on plateau. |
| `configs.LR_METRIC` | `"val"` | Metric the plateau scheduler watches: `val` \| `train`. |
| `configs.LR_PLATEAU_PATIENCE` | `10` | Epochs of no improvement before the plateau decay fires. |

**`EARLY_STOPPER`**

| Key | Default | Description |
|---|---|---|
| `use` | `true` | `false` = always train the full `EPOCHS`, ignoring `EARLY_STOPPER_PATIENCE`. |
| `configs.EARLY_STOPPER_PATIENCE` | `20` | Epochs of no improvement before stopping early. |

**`TRACKING`**

| Key | Default | Description |
|---|---|---|
| `MLFLOW.use` | `true` | `false` = every MLflow call becomes a no-op; nothing about local behavior changes. |
| `MLFLOW.configs.URI` | `""` | MLflow tracking URI; empty = MLflow's own local `./mlruns`. |
| `MLFLOW.configs.EXPIREMENT` | `""` | MLflow experiment name (yes, spelled `EXPIREMENT` in the yaml — kept as-is); empty = MLflow's `"Default"` experiment. |
| `MLFLOW.configs.RUN` | `""` | MLflow run name; empty = mirrors the local output-dir name, e.g. `log_diff_c_74_w_news/20260720`. |
| `MLFLOW.configs.LOG_MODEL` | `false` | Also log the trained model as an MLflow artifact. |
| `LOCAL.use` | `true` | `false` = the output dir path is still computed (to seed the MLflow run name) but never created; nothing is written to disk locally. |
| `LOCAL.configs.dir` | `""` | Base output directory; empty = `model_impl.consts.OUTPUT_ROOT`. |

### Partial configs are fine

You only need to specify what you're changing. This is a complete, valid config —
save it as its own file and pass `--config that_file.yaml`:

```yaml
DATA:
  TARGET:
    ID: "XCU"
  TYPE_OF_DIFF: "diff"
TRAINING:
  EPOCHS: 10
```

**`exampl.yaml` is not read, merged, or referenced when you do this.** Only the one
file named by `--config` is ever opened. Every key this file omits (model
architecture, scheduler, evaluation settings, the 74-ticker company list, ...)
falls back to a plain Python default declared on the matching dataclass field in
`utils/schemas/` — `exampl.yaml` just happens to spell out those same values as a
working example/template; it plays no special role in the loader itself. Point
`--config` at any yaml file you like, including one that sets nothing at all.

### `use` toggles

`SCHEDULER.use`, `EARLY_STOPPER.use`, `TRACKING.MLFLOW.use`, and
`TRACKING.LOCAL.use` are real switches, not decorative:

- `SCHEDULER.use: false` → `main.py` never constructs a `ColdStartScheduler`;
  the optimizer keeps a fixed learning rate (`TRAINING.LR`) for the whole run.
- `EARLY_STOPPER.use: false` → training always runs the full `EPOCHS`, regardless
  of `EARLY_STOPPER_PATIENCE`.
- `TRACKING.MLFLOW.use: false` → every MLflow call in the pipeline becomes a
  no-op — no run is opened, no `mlruns/` directory is touched, nothing changes
  about how the run behaves or what it writes locally. There's exactly one
  place this is checked (`mlflow_tracker.start_run`); nothing else in the
  codebase branches on it.
- `TRACKING.LOCAL.use: false` → the output directory path is still computed
  (so it can still seed the MLflow run name), but it's never created and
  nothing — `run.log`, `config_snapshot.json`, figures, per-window CSVs — is
  written to disk. Independent of `TRACKING.MLFLOW.use`: you can run with
  MLflow only, local files only, both, or neither.

**Note on `EXPIREMENT`**: yes, that's how the key is spelled in the yaml — kept
as-is rather than silently "corrected" to `EXPERIMENT`, since renaming a yaml
key is a bigger, less obviously-safe change than fixing a spacing typo. The
Python-side field is spelled correctly (`MLflowConfig.experiment`); only the
yaml key itself carries the misspelling.

---

## Inputs

Two parquet files actually feed the pipeline, passed via the CLI flags above.
Full column-level schema for both lives in
[data_loading/covariates-structure.md](data_loading/covariates-structure.md); this
table is just what each one is *for*:

| File | CLI flag | Contents |
|---|---|---|
| `target_variables.parquet` | `--target-covariate-path` | Long format: one row per `(id, date)`, columns `date, id, open, high, low, close`. Supplies **both** the forecast target (`DATA.TARGET.ID`/`FEATURE` selects one id/column) **and** any per-id covariates (`DATA.COVARIATES`, a list of `(id, [features])` pairs pulled from other ids in this same file). |
| `global_covariates.parquet` | `--global-covariate-path` | Wide format: one row per date, a scalar column per global covariate (`DATA.GLOBAL_COVARIATES` selects which ones to load) plus the news `embedding` column (named by `DATA.NEWS_COL`, `"embedding"` by default) holding a 768-dim array per day. |

`static_covariates.parquet` (one row per id, a per-series `unit` column — see
`covariates-structure.md`) and `feature_covariates.parquet`
(`--feature-covariate-path`) are both accepted by other parts of the toolchain
but **not read by anything in `model_impl`** — no loader consumes either.
`--dynamic-covariate-path` is likewise accepted but unused; it's left over from
an earlier single-file covariate layout.

Both `DATA.COVARIATES` and `DATA.GLOBAL_COVARIATES` may be empty lists (or one
entry, or many) — `n_covariates` is derived from however many columns the two
selections add up to, and the model has a dedicated no-covariates path
(`MultiCrossChronos.no_covariates`) for when that total is 0.

---

## Pipeline overview

1. Parse CLI args and the config yaml into a `RunConfig`.
2. Load the three streams, apply differencing per `TYPE_OF_DIFF`, build the output
   directory, snapshot the resolved config.
3. Load the Chronos tokenizer; split into train/val/test; build tokenized sliding
   windows.
4. Build `MultiCrossChronos` and the optimizer/scheduler.
5. Train with early stopping.
6. *(optional)* Run the full metric suite on the validation split.
7. Evaluate the test split: score every window, aggregate, print the summary banner,
   persist every table and figure.

Full detail — every function involved, its signature, and why the module
boundaries sit where they do — is in [code_structure.md](code_structure.md).

---

## Outputs

When `TRACKING.LOCAL.use` is true, everything lands in
`output/<DATA.TARGET.ID>/<TYPE_OF_DIFF>_c_{n_covariates}_w_{news|no_news}/YYYYMMDD/`
(`n_covariates` is the combined width of `DATA.COVARIATES` +
`DATA.GLOBAL_COVARIATES`, not a company count):

| Artifact | Contents |
|---|---|
| `config_snapshot.json` | The fully-resolved `RunConfig` (yaml merged over schema defaults) — traces any output dir back to the exact settings that produced it |
| `run.log` | Everything logged after the output dir exists: window diagnostics, per-epoch lines, the final metrics banner |
| `summary` (`.json`) | Run config + aggregate metrics (means, skill vs naïve, DM test) |
| `metrics_per_window` (`.csv`) | One row per window: mse, mae, smape, crps, wis50/80/90, cov50/80/90, sharp80, ece_q, naïve losses, pinball p10/p50/p90 |
| `forecasts_by_window` (`.csv`) | One row per (window, horizon): truth, median, low80/high80, q10/q50/q90, naive |
| `loss_curve_step.png` / `loss_curve_epoch.png` | Train/val CE per step and per epoch |
| `forecasts/<YYYY-MM-DD>.png` | Per-window context + forecast + 80% band (dual panel when differenced) |
| `horizons/horizon_h{1..pred_len}.png` | h-step-ahead predicted vs actual across all windows |
| `mse_per_window.png`, `wis80_per_window.png`, `crps_per_window.png` | Per-window metric charts |
| `coverage_summary.png`, `reliability_pp_curve.png`, `pit_histogram.png` | Calibration |
| `skill_vs_naive.png`, `pinball_loss_summary.png` | Skill and quantile loss |
| `validation/` | The same tables/figures for the VAL split — only written if `run_validation_suite` is enabled in `main.py` (off by default; it's expensive) |

**With debug mode** (currently hardcoded off in `main.py`): `attn_news_to_index_w1.png`,
`attn_comp_to_eurusd_w1.png`.

**With faithfulness mode** (also hardcoded off): `saliency_{news,comp}_w{n}.npy`,
`faithfulness_per_window.jsonl`, `faithfulness_summary.json`,
`faith_del_curve_news_w1.png`, `faith_ins_curve_news_w1.png`,
`faith_{del,ins}_curve_news_*_mean.png`, `faith_spearman_hist_{news,comp}.png`,
`faith_jaccard_hist_news.png`, `faith_placebos_news.png`.

> `faithfulness_per_window.jsonl` and `run.log` are opened in **append** mode —
> re-running into an existing output dir (same target/config, same day) appends
> rather than overwrites.

**When `TRACKING.MLFLOW.use` is `true`**, every artifact above is also logged
to MLflow the moment it's written — nothing is batched up and pushed at the
end, so a crash mid-run doesn't lose whatever was already tracked. Along with
the artifacts, MLflow gets: the resolved config as flat params (list-valued
fields like the covariate lists are skipped — they're still fully in
`config_snapshot.json`), `train_ce`/`val_ce`/`lr` once per epoch, the
run-level aggregate metrics once, and metrics broken out **by horizon step**
(not by window). The run's name defaults to the same
`<TYPE_OF_DIFF>_c_{n_covariates}_w_{news|no_news}/YYYYMMDD` string used for
the local output dir, so the two are trivially correlated. Set
`TRACKING.MLFLOW.use: false` to skip all of this — the run behaves exactly as
if MLflow weren't installed. This is independent of `TRACKING.LOCAL.use` —
see [`use` toggles](#use-toggles).

---

## Known limitations

The short version — see [code_structure.md § Known issues](code_structure.md#known-issues)
for the full list with file locations:

- Prediction intervals are narrower than they should be (MC-Dropout `argmax`
  discards the model's learned distribution).
- `feature_covariates.parquet` isn't consumed by anything yet.
- No model checkpointing — train/val/test always run in one process.
- A day with a missing news embedding silently corrupts the news tensor's shape
  instead of raising.


