# model_impl — code structure

Architecture reference: directory layout, import hierarchy, and the full
module/API surface. For "how do I run this", see [README.md](README.md).

---

## Design principles

1. **Layered, no cycles.** Every package sits at a layer; imports only ever
   point downward. Two placements exist purely to keep it that way (see
   [Import hierarchy](#import-hierarchy)).
2. **Config flows as typed objects, not module globals.** `arg_handler/config_file_parser.py`
   builds one `RunConfig` in `main()`, immediately unpacks it into six
   section wrappers (`DataConfig`, `ModelConfig`, `TrainingConfig`,
   `EvaluationConfig`, `SchedulerConfig`, `EarlyStopperConfig`), and hands
   each wrapper only to the functions that need it. No function below
   `main()` takes the whole `RunConfig`.
3. **`consts.py` is not a config fallback store.** Every yaml-configurable
   value has its own default colocated on the matching dataclass field in
   `utils/schemas/`. `consts.py` holds only values with no configurable
   counterpart at all — plot DPI, fixed output filenames, a couple of
   not-yet-wired placeholders.
4. **Logging over printing.** Every module gets its logger from
   `utils/logger_utils/logger.get_logger(__name__)`, never `print()` (one
   deliberate, documented exception).

---

## Directory tree

```
model_impl/
  main.py                       — orchestration (entry point); no modelling or scoring logic
  consts.py                     — true constants (see design principle 3)
  code_structure.md             — this file
  README.md                     — usage guide

  arg_handler/
    cli_parser.py                  parse() → argparse Namespace, validates the two required paths
    config_file_parser.py          load(path) → RunConfig, built from yaml + schema defaults

  utils/
    schemas/                       typed config dataclasses, one module per top-level yaml section
      data.py                        DataConfig, DynamicColConfig, SplitsConfig
      model.py                       ModelConfig, CrossChronosConfig
      training.py                    TrainingConfig, SchedulerConfig, EarlyStopperConfig
      evaluation.py                  EvaluationConfig, CentralIntervalConfig, ECEGridConfig, FaithConfig
      run_config.py                  RunConfig — the root
    runtime_utils.py               DEVICE, setup_runtime
    data_loader_utils/
      transforms.py                   apply_differencing, invert_diff
    evaluation_utils/
      metrics.py                      scoring (pure: arrays in, float out) + build_ece_quantiles
      inference.py                    predict_distribution (MC-Dropout sampler)
      eval_pipeline.py                evaluate / aggregate — the shared window loop
      faithfulness.py                 saliency attribution study
    plot_utils/
      training_plots.py               loss curves
      forecast_plots.py               forecasts, horizons, attention maps
      calibration_plots.py            coverage, reliability, PIT, pinball, per-window metrics
      faithfulness_plots.py           saliency figures
    logger_utils/
      logger.py                       get_logger, attach_file_handler — shared logging setup
    tracking_utils/
      mlflow_tracker.py               start_run, log_params, log_metrics, log_artifact, log_model —
                                       no-op unless MLFLOW.use is true

  data_loading/
    loaders.py                     price_series / news_series / company_series (all file I/O)
    splitting.py                   Split, temporal_split
    windowing.py                   Windows, sliding_windows_triple, dataset_windows, print_window_report
    streams.py                     load_streams — composes loaders + transforms

  models/
    blocks.py                      CrossBlock
    cross_chronos.py                MultiCrossChronos

  schedulers/
    cold_start.py                  ColdStartScheduler (two-phase LR)

  scripts/                         the run's stages, called in order by main
    training.py                     run() — fit with early stopping
    validation.py                   run() — full metric suite on VAL (model selection)
    test.py                         run() — final report on TEST + all artifacts

  artifacts_logs/                  everything a run writes to disk
    run_dir.py                      make_output_dir
    writers.py                      savefig, write_json, write_csv, save_config_snapshot
    run_log.py                      install — tees stdout + attaches a log file handler

  requirements.txt
  output/                          run artifacts (created on first run)
```

---

## Import hierarchy

Imports only ever point **downward**:

```
  consts.py                          (leaf: no internal imports)
      │
      ├── utils/schemas/*            typed config dataclasses, each field defaulted
      │
      ├── arg_handler/cli_parser     argparse + path validation
      ├── arg_handler/config_file_parser   yaml → RunConfig, built from schemas
      │
      ├── utils/runtime_utils        DEVICE, setup_runtime
      ├── utils/logger_utils/logger  get_logger, attach_file_handler (imports consts.RUN_LOG_FILE)
      ├── utils/tracking_utils/mlflow_tracker   start_run, log_params/metrics/artifact/model
      ├── utils/data_loader_utils/transforms
      ├── utils/evaluation_utils/metrics     pure leaf
      ├── utils/evaluation_utils/inference   predict_distribution
      │
      ├── artifacts_logs/*           run_dir, writers (imports consts + mlflow_tracker),
      │                              run_log (imports logger + consts)
      │
      ├── data_loading/*             loaders → splitting → windowing → streams
      ├── models/*, schedulers/*     blocks → cross_chronos (takes ModelConfig)
      │
      ├── utils/plot_utils/*         → writers.savefig, consts (dir names)
      │
      ├── utils/evaluation_utils/faithfulness   → inference, metrics, faithfulness_plots
      ├── utils/evaluation_utils/eval_pipeline  → inference, metrics, faithfulness,
      │                                           transforms, forecast_plots, runtime_utils
      │
      ├── scripts/*                  → eval_pipeline, models, schedulers, writers, consts,
      │                                mlflow_tracker (training.py, test.py)
      │
      └── main.py                    → everything; orchestrates only
```

Five placements are deliberate:

- **`predict_distribution` lives in `evaluation_utils/inference.py`**, its own module.
  Both `eval_pipeline` and `faithfulness` call it; homing it in either would make them
  circular. Keeping it out of `metrics.py` also lets that module stay a pure leaf with
  no torch/model dependency.
- **`savefig` lives in `artifacts_logs/writers.py`**, below every plot module that
  calls it. It's an artifact writer, not a figure.
- **`DEVICE` lives in `utils/runtime_utils.py`**, not `consts.py`. It's *probed*
  (`torch.cuda.is_available()`), and `consts.py` holds only literal, undetected values.
- **The train loop lives in `scripts/training.py`, not `models/`.** `models/` is
  architecture only; a trainer there would import `evaluation_utils`, which imports
  the model back — a cycle.
- **MLflow logging is hooked once, into `artifacts_logs/writers.py`**, not scattered
  across every plot module or script. Every artifact in the whole codebase already
  goes through `savefig`/`write_json`/`write_csv`, so that's the single place an
  artifact needs to report itself to `mlflow_tracker` — no plot module or script
  needs to know MLflow exists at all.

---

## Config flow

```
                    --config foo.yaml
                          │
                          ▼
        arg_handler/config_file_parser.load(path)
                          │  reads only the yaml keys present;
                          │  every gap falls through to the dataclass's
                          │  own default in utils/schemas/*.py
                          ▼
                       RunConfig
                (seed, data, model, training, evaluation,
                 scheduler, early_stopper, mlflow)
                          │
                          │  unpacked ONCE, in main()
                          ▼
        ┌─────────┬─────────┬──────────┬────────────┬───────────┬───────────┬───────────┐
        │         │         │          │            │           │           │
   RunConfig  DataConfig ModelConfig TrainingConfig EvaluationConfig SchedulerConfig MLflowConfig
    .seed                                                          EarlyStopperConfig
        │         │         │          │            │           │           │
   setup_runtime  load_streams  MultiCrossChronos  training.run  eval_pipeline.evaluate  ColdStartScheduler  mlflow_tracker
                  temporal_split                                 test.run / validation.run  training.run   .start_run
                  dataset_windows
```

No function outside `main()` ever receives a bare `RunConfig` — every call site
downstream takes the specific wrapper(s) it needs (see the per-module signatures below).

### How a yaml value resolves

`config_file_parser.py` never imports `consts.py`. For each top-level section it:

1. Reads only the keys actually present in that yaml section (`_pick`, keyed by
   `{YAML_KEY: python_field_name}`).
2. Constructs the matching dataclass with just those kwargs.
3. Anything omitted is simply never passed — the dataclass field's own default
   (declared in `utils/schemas/*.py`) applies.

A missing config file, an empty file, and a file missing whole sections all
degrade identically: everything resolves to the schema defaults, which are
the same values `exampl.yaml` ships with.

---

## Module reference

### `arg_handler/cli_parser.py`

`parse() -> argparse.Namespace` — four flags:

| Flag | Required | Validated |
|---|---|---|
| `--config` | yes | not path-checked; `config_file_parser.load` degrades gracefully if missing |
| `--dynamic-covariate-path` | yes | must exist as a file, or `parser.error(...)` terminates the process |
| `--target-covariate-path` | yes | same |
| `--feature-covariate-path` | no (default `""`) | not validated — no loader reads it yet |

The former ablation/debug flags (`--index`, `--windows`, `--no-news`, `--debug`,
`--faithfulness`, `-c`/`--companies`, `--max-news-per-day`) are commented out in
the source, not deleted.

### `arg_handler/config_file_parser.py`

| Function | Purpose |
|---|---|
| `load(path) -> RunConfig` | Parse `path` as yaml, resolve every section |
| `_section(raw, key) -> dict` | A yaml section, or `{}` if absent |
| `_pick(section, key_map) -> dict` | Build kwargs from only the yaml keys present |
| `_build_data/_build_model/_build_training/_build_evaluation/_build_scheduler/_build_early_stopper/_build_mlflow` | One per `RunConfig` section |

### `utils/schemas/` — the typed config

All dataclasses are `@dataclass(frozen=True)`; every field carries its own default.

| Module | Classes |
|---|---|
| `data.py` | `DynamicColConfig(comps_col, news_col)`, `SplitsConfig(train_date, val_date, test_date)`, `DataConfig(target_col, dynamic_col, static_col, splits, pred_len, ctx_len, token_all, shuffle_data, type_of_diff)` |
| `model.py` | `CrossChronosConfig(emb_dim_news, d_model, n_heads, n_layers_txt, d_ff, dropout)`, `ModelConfig(comp_enc, cross_chronos)` |
| `training.py` | `TrainingConfig(epochs, weight_decay, grad_clip, batch_size, label_smoothing)`, `SchedulerConfig(use, cold_start, cold_epochs, running, decrease_factor, metric, patience)`, `EarlyStopperConfig(use, patience)` |
| `evaluation.py` | `CentralIntervalConfig(alpha_50, alpha_80, alpha_90)`, `ECEGridConfig(start, stop, steps)`, `FaithConfig(mc_samples, ks, topk, stability_runs, placebo_shifts, mask_strategy, rng_seed)`, `EvaluationConfig(windows, mc_samples, central_interval, ece_grid, faith)` |
| `mlflow.py` | `MLflowConfig(use, uri, experiment, run_name, log_model)` |
| `run_config.py` | `RunConfig(seed, data, model, training, evaluation, scheduler, early_stopper, mlflow)` |

`utils/schemas/__init__.py` re-exports every class, so callers use
`from model_impl.utils.schemas import X` regardless of which submodule defines it.

### `data_loading/`

| Function | Module | Signature |
|---|---|---|
| `price_series` | `loaders` | `(file_path, keep_features=["Close"]) -> DataFrame` |
| `company_series` | `loaders` | `(file_path, keep_companies, keep_features=["Close"]) -> (DataFrame, int)` |
| `news_series` | `loaders` | `(file_path, col="embedding") -> Series` |
| `temporal_split` | `splitting` | `(prices, news, comp, train_date, val_date, test_date, ctx) -> (Split, Split, Split)` |
| `sliding_windows_triple` | `windowing` | `(prices_ser, news_ser, comp_df, ctx, pred, tokenizer, token_all_, scaler=None) -> (Windows, scaler)` |
| `dataset_windows` | `windowing` | `(train, val, test, ctx, pred, tokenizer, token_all) -> (Windows, Windows, Windows)` |
| `print_window_report` | `windowing` | `(name, wins, split, ctx_len, pred_len) -> None` |
| `load_streams` | `streams` | `(index, no_news, data_cfg, emb_dim_news, dyn_path, tgt_path) -> (prices_target, raw_series, comp_df, news, n_companies)` |

Containers: `Split(prices, news, comp)`, `Windows(xe, xn, xc, y, scales)`.

`load_streams` is the composition point: it reads `dyn_path`/`tgt_path` (the
validated CLI args), applies `transforms.apply_differencing` per
`data_cfg.type_of_diff`, and keeps the pre-differencing series as the
reconstruction anchor.

### `models/`

| Class | Module | Signature |
|---|---|---|
| `CrossBlock` | `blocks` | `__init__(d_model, n_heads, dropout)` — cross-attention + residual + LayerNorm, caches `last_weights (B, heads, Q, K)` |
| `MultiCrossChronos` | `cross_chronos` | `__init__(vocab, n_companies, model_cfg, ctx_len, pred_len)` |

`MultiCrossChronos`: frozen Chronos encoder on the price stream; `TransformerEncoder`
on news/companies; six `CrossBlock`s (all bidirectional pairs); last-step fusion
→ `Linear(d_model*3, pred_len*vocab)` → `(B, pred_len, vocab)` logits.

Two overrides matter: `train(mode)` forces `enc_eur.eval()` on every flip (freezing
stops gradients, not dropout); `mc_dropout(enable)` toggles dropout on the trainable
submodules only, leaving the frozen encoder deterministic during MC sampling.

### `schedulers/cold_start.py`

`ColdStartScheduler(opt, cold_start_rate, cold_epochs, running_rate, decrease_factor, patience)`.
Two phases: hold `cold_start_rate` for `cold_epochs`, then `running_rate` with
reduce-on-plateau. `.step(metric)` once per epoch; `.lr` property reads the current rate.

### `scripts/`

| Function | Signature | Returns |
|---|---|---|
| `training.run` | `(model, opt, tr_loader, va_loader, scheduler, training_cfg, early_stopper_cfg, lr_metric)` | `(tr_losses, va_losses, epoch_starts, ep_tr_means, ep_va_means)` |
| `validation.run` | `(model, chrono, windows, split, raw_series, outdir, index, data_cfg, eval_cfg)` | aggregated VAL summary; writes `<outdir>/validation/summary.json` |
| `test.run` | `(model, chrono, windows, split, raw_series, outdir, index, n_companies, no_news, t0, data_cfg, eval_cfg, debug_vis=False, faithfulness_on=False)` | aggregated TEST metrics; writes every table and figure |

`training.run` accepts `scheduler: ColdStartScheduler | None` — `None` when
`SchedulerConfig.use` is `False`, in which case `.step()` is simply skipped and
the optimizer's LR stays fixed. `early_stopper_cfg.use` replaces the old
`patience > 0` magic-number toggle.

Two losses are built on purpose: `CrossEntropyLoss(label_smoothing=training_cfg.label_smoothing)`
as the training objective, `label_smoothing=0.0` for validation CE — smoothing
raises the loss floor, so the unsmoothed metric is what drives early stopping
and the LR plateau.

`training.run` logs `train_ce`/`val_ce`/`lr` to MLflow once per epoch
(`step=epoch`), at the same point it already logs the equivalent `logger.info`
line. `test.run` logs the run-level aggregate once (`dm_test` flattened into
`dm_test_stat`/`dm_test_pvalue`), plus metrics grouped **by horizon step**
(not by window) using the `fw_rows` DataFrame it already builds — see
`utils/tracking_utils/mlflow_tracker.py` below for what's and isn't
computable at that granularity. Both calls are no-ops when `MLflowConfig.use`
is `False`.

### `utils/evaluation_utils/eval_pipeline.py`

| Function | Signature | Returns |
|---|---|---|
| `evaluate` | `(model, chrono, windows, split, raw_series, outdir, index, n_windows, data_cfg, eval_cfg, debug_vis=False, faithfulness=False)` | `(metrics, fw_rows, pit_all)` |
| `aggregate` | `(metrics, pred_len)` | the `summary.json` metrics block |

Split-agnostic: `scripts/validation.py` and `scripts/test.py` are thin wrappers
pointing this at different windows.

### `utils/evaluation_utils/metrics.py`

Pure functions; no model/split/run awareness.

| Group | Functions |
|---|---|
| Point | `mae`, `mape`, `smape` |
| Interval | `interval_score`, `coverage`, `pinball_loss` |
| Distributional | `loss_crps`, `ece_quantiles(y_true, samples, quantiles)`, `pit_values` |
| Comparison | `dm_test` (Diebold-Mariano) |
| Grid | `build_ece_quantiles(start, stop, steps) -> ndarray` |

`build_ece_quantiles` replaces what used to be a module-level constant — it's
called once per run from `EvaluationConfig.ece_grid` and threaded explicitly
into `ece_quantiles(...)` and `calibration_plots.plot_reliability_curve(...)`.

### `utils/evaluation_utils/inference.py`

`predict_distribution(model, chrono, scale_win, ctx_eur, ctx_news, ctx_comp, mc_samples) -> ndarray (mc_samples, pred_len)`.
Dropout on trainable layers only, `mc_samples` forward passes, `argmax` per
pass, decode via the tokenizer under `scale_win`.

### `utils/evaluation_utils/faithfulness.py`

| Function | Purpose |
|---|---|
| `last_step_temporal_saliency(attn_weights)` | Normalised importance of each key timestep |
| `mask_timesteps(X, idxs, strategy, cached_rep=None)` | Replace timesteps with a `mean`/`zero` baseline |
| `deletion_insertion_curves(model, chrono, scale_win, ctx_eur, ctx_news, ctx_comp, truth, saliency_vec, ks, strategy, mc_samples, rng_seed, stream="news", device="cpu")` | ΔCRPS masking top-k vs random-k vs least-k, plus insertion |
| `loto_deltas(model, chrono, scale_win, ctx_eur, ctx_news, ctx_comp, truth, strategy, mc_samples, stream="news")` | Leave-One-Timestep-Out ΔCRPS |
| `jaccard_topk(sets)` | Pairwise Jaccard over top-k sets across MC runs |
| `run_window(model, chrono, outdir, window, scale_win, ctx_eur, ctx_news, ctx_comp, truth, faith_cfg, mc_samples)` | Per-window driver → appends to `faithfulness_per_window.jsonl` |
| `aggregate(outdir)` | Cross-window figures + `faithfulness_summary.json` |

`faith_cfg` is `EvaluationConfig.faith`; `mc_samples` is `EvaluationConfig.mc_samples`
(the main loop's count), capped inside `run_window` against `faith_cfg.mc_samples`.

### `utils/plot_utils/`

Draws and saves only — callers pass already-scored arrays.

| Module | Functions |
|---|---|
| `training_plots` | `plot_loss_step`, `plot_loss_epoch` |
| `forecast_plots` | `plot_horizon_forecasts`, `plot_forecast_window`, `plot_attention_maps` |
| `calibration_plots` | `plot_metric_per_window`, `plot_coverage_summary`, `plot_reliability_curve(outdir, forecasts_csv, ece_quantiles)`, `plot_pit_histogram`, `plot_skill_vs_naive`, `plot_pinball_summary` |
| `faithfulness_plots` | `plot_faith_curves_window`, `plot_faith_aggregates` |

The headless `Agg` backend is pinned once in `plot_utils/__init__.py`.

### `utils/data_loader_utils/transforms.py`

| Function | Purpose |
|---|---|
| `apply_differencing(data, mode)` | `diff` → `x.diff()`; `log_diff` → `log(x).diff()` (fallback for non-positive cols); upcasts to float64, bfills the leading row |
| `invert_diff(anchor, diffs, mode)` | Inverse: `diff` → `anchor + cumsum`; `log_diff` → `anchor * exp(cumsum)`; `no_diff` → unchanged |

### `utils/runtime_utils.py` · `utils/logger_utils/logger.py`

| Name | Purpose |
|---|---|
| `DEVICE` | `"cuda"` if available else `"cpu"` (probed at import) |
| `setup_runtime(seed)` | Seeds random/numpy/torch/cuda, relaxes SSL for HF downloads |
| `get_logger(name)` | Call with `__name__` from inside a real package module — see the warning below |
| `attach_file_handler(outdir)` | Adds a second handler writing to `<outdir>/run.log` |

**`get_logger(__name__)` pitfall:** when a module is imported normally,
`__name__` is its fully-qualified `model_impl.xxx.yyy` path, which correctly
propagates to the configured `"model_impl"` root logger. But `main.py` is
sometimes *executed directly* (`python model_impl/main.py`), in which case
Python sets `__name__ = "__main__"` for that file — a logger by that name
has no handler and isn't a dotted child of `"model_impl"`, so its `.info()`/
`.warning()` calls are silently dropped by logging's WARNING-only last-resort
handler. `main.py` therefore hardcodes `get_logger("model_impl.main")`
instead of using `__name__`. No other module in the package is ever run
directly, so this is the only place the workaround is needed.

### `artifacts_logs/`

| Name | Module | Purpose |
|---|---|---|
| `make_output_dir(n_companies, no_news, index, type_of_diff)` | `run_dir` | `output/<index>/<type_of_diff>_c_{n}_w_{news}/YYYYMMDD` |
| `savefig(path, fig, name)` | `writers` | PNG at `FIG_DPI`, closes the figure |
| `write_json(outdir, name, payload)` / `write_csv(outdir, name, df)` | `writers` | Return the written path |
| `save_config_snapshot(outdir, cfg)` | `writers` | `dataclasses.asdict(cfg)` → `config_snapshot.json` |
| `install(outdir)` | `run_log` | Attaches the logging file handler and tees stdout, both into `run.log` |

`run_log.install` deliberately leaves stderr out of the tee — tqdm redraws
its bar there and would flood the file with carriage-return frames.

`savefig`, `write_json` and `write_csv` each call `mlflow_tracker.log_artifact`
on the file they just wrote (`save_config_snapshot` gets this for free — it
delegates to `write_json`). This is the single hook point for MLflow artifact
logging in the entire codebase; see `utils/tracking_utils/mlflow_tracker.py`.

### `utils/tracking_utils/mlflow_tracker.py`

MLflow integration, structured so nothing outside this module ever branches
on whether tracking is enabled. A module-level `_active` flag mirrors
`artifacts_logs/run_log.py`'s `_installed` idiom.

| Function | Purpose |
|---|---|
| `start_run(cfg: MLflowConfig, default_run_name)` | Context manager. No-op (`yield` only) if `cfg.use` is `False`. Otherwise sets tracking URI/experiment when non-empty, opens `mlflow.start_run(run_name=cfg.run_name or default_run_name)`, sets `_active = True` for the duration |
| `log_params(cfg: RunConfig)` | Flattens `dataclasses.asdict(cfg)` into dotted param names (`data.pred_len`, `model.cross_chronos.d_model`, ...), dropping list/dict/`None` leaves — the 74-ticker `comps_col`, `FAITH_KS`, etc. stay fully captured in `config_snapshot.json` instead |
| `log_metrics(metrics, step=None)` | Thin wrapper over `mlflow.log_metrics` |
| `log_artifact(path)` | Thin wrapper over `mlflow.log_artifact`, called from `artifacts_logs/writers.py` |
| `log_model(model, artifact_path="model")` | `mlflow.pytorch.log_model` — called from `main.py` only when `MLflowConfig.log_model` is `True`. This is the **first** persistence of trained weights anywhere in the pipeline; `scripts/training.py` only ever keeps `best_state` in memory |

Every function except `start_run` is a plain no-op when `_active` is `False`
— there is exactly one place in the codebase (`start_run`'s own `cfg.use`
check) where MLflow's on/off state is ever branched on.

`log_metrics` is called from two places with two different granularities, by
design (see the `scripts/` section above): per-epoch during training
(`step=epoch`), and per-horizon-step (not per-window) after test evaluation.
`loss_crps`, `ece_quantiles` and `pit_values` are **not** available at the
per-horizon granularity — they need the full `(MC, H)` sample array, which
only exists transiently inside `evaluate()`/`inference.py` and isn't carried
in the `fw_rows` records `test.run` groups by horizon.

### `consts.py`

The only module-level constants left after the config-schema split:

| Name | Value | Why it's a true constant |
|---|---|---|
| `FIG_DPI` | `160` | Cosmetic, never worth varying per run |
| `HEAD_HIDDEN_1/1_5/2` | `64/192/192` | Dead — no code references them; placeholder for a not-yet-built head architecture |
| `OUTPUT_ROOT`, `RUN_LOG_FILE`, `CONFIG_SNAPSHOT_FILE`, `SUMMARY_FILE`, `METRICS_PER_WINDOW_FILE`, `FORECASTS_BY_WINDOW_FILE`, `FORECASTS_DIR`, `HORIZONS_DIR`, `VALIDATION_DIR` | fixed strings | Output artifact layout; `RUN_LOG_FILE` and `SUMMARY_FILE` were previously duplicated as separate literals in two files each |
| `FAITH_PER_WINDOW_FILE`, `FAITH_SUMMARY_FILE` | fixed strings | Faithfulness study output filenames |

Parquet paths are **not** here — they come from `--dynamic-covariate-path`/
`--target-covariate-path` (validated in `cli_parser.parse`) and are threaded
explicitly from `main()` into `data_loading.streams.load_streams`.

---

## Design notes

### Differencing and reconstruction

Differencing is a model-layer transform, not a dataset-build concern — the
parquet trees always store raw interpolated levels. `data_loading.streams.load_streams`
applies `transforms.apply_differencing` (selected by `data_cfg.type_of_diff`)
right after loading, and keeps the raw series as the reconstruction anchor.
`eval_pipeline.evaluate` reconstructs before scoring, so every metric in
`summary.json` is computed on real prices regardless of representation.

| `type_of_diff` | Model predicts | Reconstruction | Naïve baseline |
|---|---|---|---|
| `no_diff` | price levels | none | `price(t−1)` |
| `diff` | `x(t) − x(t−1)` | `anchor + cumsum(preds)` | `anchor` |
| `log_diff` | `log(x(t)/x(t−1))` | `anchor * exp(cumsum(preds))` | `anchor` |

Reconstruction is exact (verified: `0.0` / `~3e-11` max error) because (1)
differencing happens *after* interpolation, so
`cumsum(diff(interp(level))) == interp(level)`, and (2) `apply_differencing`
upcasts to float64 before diffing — the loaders cast levels to float32, and
log/diff/cumsum chains in float32 drift ~1e-2 on NDX-scale prices.

### Tokenization scale (`token_all`)

- `token_all = True` (global): one scale fit on train, threaded into val/test
  — all splits share a token space.
- `token_all = False` (per-window): every window self-normalizes from its
  own context; nothing is shared.

### MC-Dropout uncertainty

`predict_distribution` samples via dropout + `argmax` per pass, not by
sampling the categorical distribution the model actually learned — see
[Known issues](#known-issues).

---

## Known issues

| Issue | Where | Notes |
|---|---|---|
| Over-narrow prediction intervals | `evaluation_utils/inference.py` | `argmax` discards the categorical distribution; only dropout jitter creates spread |
| Faithfulness CRPS is in diff space | `faithfulness.run_window` | Receives `truth` before `evaluate` reconstructs — not comparable to `summary.json`'s price-space CRPS |
| Stability loop has no `torch.no_grad()` | `faithfulness.run_window` | The `stability_runs` forwards build autograd graphs — a likely CUDA OOM contributor |
| Missing embeddings become NaN scalars | `data_loading/loaders.news_series` | `np.array(None, dtype=np.float32)` yields a 0-d NaN, not a 768-vector, so days without news silently corrupt the stream shape instead of failing loudly |
| `--feature-covariate-path` accepted but unused | `main.py` | No loader reads `FEAT_PATH`/`STATIC_COL` yet |
| No local checkpointing | `scripts/training.py` | Best state is restored in-process but never written to a local file — the three stages still can't run as independent processes. `MLFLOW.configs.LOG_MODEL: true` logs the trained model to MLflow, but that's opt-in and only reachable through an active MLflow run, not a general checkpoint mechanism |
| Per-horizon MLflow metrics omit CRPS/ECE/PIT | `scripts/test.py` | `fw_rows` only stores `low80/high80/q10/q50/q90`, not the full `(MC, H)` sample array those three need — only `mae`/`smape`/`coverage80`/`pinball_p50` are broken out by horizon |
| Interpolated weekends inflate metrics | dataset-level | `case_interp` fills non-trading days by interpolation; those targets are synthetic and trivially predictable |
| `HEAD_HIDDEN_*` in `consts.py` are dead | `consts.py` | Placeholder for a head architecture that doesn't exist yet |
