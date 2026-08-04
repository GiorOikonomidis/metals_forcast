"""
Entry point: wires argument parsing, data loading, tokenization, training and
the evaluation stages together. Everything substantial lives elsewhere —
models/ (architecture), schedulers/, data_loading/, scripts/ (train / val /
test stages), utils/ (transforms, metrics, inference, figures), artifacts_logs/
(everything a run writes to disk), arg_handler/ (CLI + config). This file only
orchestrates.

Config flows as typed wrapper classes (arg_handler/schema/), built once here
from the yaml (config_file_parser.load) and handed only to the module that
needs each piece — no function below main() takes the whole RunConfig.

Run from the repo root:
    python -m model_impl.main --config <file> ^
        --dynamic-covariate-path <...>\\dynamic_covariates.parquet ^
        --target-covariate-path <...>\\target.parquet ^
        --feature-covariate-path <...>\\feature_covariates.parquet
"""

# Support `python model_impl/main.py` too: put the repo root on sys.path so the
# package-absolute imports below resolve outside `python -m`.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import torch
from chronos import ChronosPipeline
from torch.utils.data import DataLoader, TensorDataset

from model_impl.arg_handler import config_file_parser
from model_impl.arg_handler.cli_parser import parse
from model_impl.artifacts_logs import run_log, writers
from model_impl.artifacts_logs.run_dir import make_output_dir
from model_impl.data_loading.splitting import temporal_split
from model_impl.data_loading.streams import load_streams
from model_impl.data_loading.windowing import dataset_windows, print_window_report
from model_impl.models.cross_chronos import MultiCrossChronos
from model_impl.schedulers.cold_start import ColdStartScheduler
from model_impl.scripts import test as test_stage
from model_impl.scripts import training, validation
from model_impl.utils.logger_utils.logger import get_logger
from model_impl.utils.plot_utils import training_plots
from model_impl.utils.runtime_utils import DEVICE, setup_runtime
from model_impl.utils.tracking_utils import mlflow_tracker

# Not get_logger(__name__): when this file is run directly (python
# model_impl/main.py, not -m model_impl.main), __name__ is "__main__", not
# "model_impl.main" — a logger by that name has no handler of its own and
# doesn't propagate into the configured "model_impl" root (wrong branch of
# the dotted hierarchy), so every .info()/.warning() here would be silently
# dropped by logging's WARNING-only last-resort handler. Hardcoding the name
# keeps it correct under both invocation styles.
logger = get_logger("model_impl.main")


def main() -> None:
    opts = parse()
    cfg = config_file_parser.load(opts.config)
    setup_runtime(cfg.seed)
    t0 = time.time()

    # feature_covariate_path is accepted but unused — no loader reads FEAT_PATH yet.
    feature_covariate_path = opts.feature_covariate_path

    # index now comes from the config (DATA.TARGET_COL); the rest are former
    # CLI flags, still hardcoded until they move into the config file too.
    index        = cfg.data.target.id
    no_news      = True
    debug_vis    = False
    faithfulness = False
    run_validation_suite = False  # full metric suite on VAL — expensive, off by default

    # 1 · Data
    prices_target, raw_series, covariate_df, news, n_covariates = load_streams(
        index, no_news, cfg.data, cfg.model.cross_chronos.emb_dim_news,
        opts.target_covariate_path, opts.global_covariate_path,
    )
    # outdir is always computed (even with TRACKING.LOCAL.use off) so its name
    # can still seed the MLflow run name below — it just never touches disk
    # when local saving is disabled (make_output_dir's create=..., writers.configure).
    outdir = make_output_dir(n_covariates, no_news, index, cfg.data.type_of_diff,
                             cfg.tracking.local.dir, create=cfg.tracking.local.use)
    writers.configure(cfg.tracking.local)
    if cfg.tracking.local.use:
        run_log.install(outdir)            # from here on, logged output also lands in run.log

    # Run name, when MLFLOW.configs.RUN is empty, mirrors the output dir's own
    # naming so a local run and its MLflow entry are trivially correlated —
    # e.g. "log_diff_c_74_w_news/20260720".
    default_run_name = f"{outdir.parent.name}/{outdir.name}"

    with mlflow_tracker.start_run(cfg.tracking.mlflow, default_run_name):
        mlflow_tracker.log_params(cfg)
        writers.save_config_snapshot(outdir, cfg)   # record the exact settings of this run

        # 2 · Chronos tokenizer — tokenization happens BEFORE training (classification setup)
        chrono = ChronosPipeline.from_pretrained(cfg.model.comp_enc)
        tokenizer = chrono.tokenizer
        object.__setattr__(tokenizer.config, 'use_eos_token', False)

        # 3 · Splits and windows
        train_split, val_split, test_split = temporal_split(
            prices=prices_target["Close"],
            news=news,
            covariate=covariate_df,
            test_days=cfg.data.splits.test_days,
            val_days =cfg.data.splits.val_days,
            ctx=cfg.data.ctx_len,
        )

        train_windows, val_windows, test_windows = dataset_windows(
            train=train_split, val=val_split, test=test_split,
            ctx=cfg.data.ctx_len, pred=cfg.data.pred_len,
            tokenizer=tokenizer,
            token_all=cfg.data.token_all,
        )

        for name, wins, split in [("train", train_windows, train_split),
                                  ("val",   val_windows,   val_split),
                                  ("test",  test_windows,  test_split)]:
            print_window_report(name, wins, split, cfg.data.ctx_len, cfg.data.pred_len)

        # 4 · Model + optimizer
        vocab = tokenizer.config.n_tokens
        model = MultiCrossChronos(vocab, n_covariates, cfg.model, cfg.data.ctx_len, cfg.data.pred_len).to(DEVICE)
        # Base LR always comes from TRAINING.LR; SCHEDULER, when enabled, only
        # layers a cold-start warmup + plateau decay on top of it (see
        # SchedulerConfig / ColdStartScheduler) — it is not a second, independent rate.
        opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg.scheduler.cold_start if cfg.scheduler.use else cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)

        # 5 · Training
        tr_loader = DataLoader(
            TensorDataset(train_windows.xe, train_windows.xn, train_windows.xc, train_windows.y),
            batch_size=cfg.training.batch_size, shuffle=cfg.data.shuffle_data
        )
        va_loader = DataLoader(
            TensorDataset(val_windows.xe, val_windows.xn, val_windows.xc, val_windows.y),
            batch_size=cfg.training.batch_size, shuffle=cfg.data.shuffle_data
        )

        scheduler = None
        if cfg.scheduler.use:
            scheduler = ColdStartScheduler(
                opt,
                cold_start_rate=cfg.scheduler.cold_start, cold_epochs=cfg.scheduler.cold_epochs,
                running_rate=cfg.training.lr, decrease_factor=cfg.scheduler.decrease_factor,
                patience=cfg.scheduler.patience,
            )

        tr_losses, va_losses, epoch_starts, ep_tr_means, ep_va_means = training.run(
            model, opt, tr_loader, va_loader, scheduler,
            cfg.training, cfg.early_stopper, cfg.scheduler.metric,
        )

        if cfg.tracking.mlflow.log_model:
            mlflow_tracker.log_model(model)

        training_plots.plot_loss_step(outdir, tr_losses, va_losses, epoch_starts,
                                      len(tr_loader), len(va_loader))
        training_plots.plot_loss_epoch(outdir, ep_tr_means, ep_va_means)

        # 6 · Validation suite (optional — model selection, not the final report)
        if run_validation_suite:
            validation.run(model, chrono, val_windows, val_split, raw_series, outdir, index,
                           cfg.data, cfg.evaluation)

        # 7 · Test stage — scores, aggregates, prints the banner, persists everything
        test_stage.run(
            model, chrono, test_windows, test_split, raw_series, outdir,
            index, n_covariates, no_news, t0, cfg.data, cfg.evaluation,
            debug_vis=debug_vis, faithfulness_on=faithfulness,
        )
        
    if cfg.tracking.local.use:
        logger.info("# 📦 Artifacts saved in %s\n", outdir)


if __name__ == "__main__":
    main()
