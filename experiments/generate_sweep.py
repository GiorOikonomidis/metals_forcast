"""
Generate the XCU experiment sweep: one yaml per (diff x head x hyperparams)
cell. Edit DEFAULTS / SPECS and re-run to regenerate the whole grid — every
config is emitted from the same template so only the swept fields ever differ.

Run:
    python experiments/generate_sweep.py
"""
from pathlib import Path

OUT_DIR = Path(__file__).parent / "xcu_sweep"
EXPERIMENT = "xcu_sweep_full_Covs"

# clean, copper-correlated covariates (XAU/XAG day-over-day jump 1.1x/1.4x)
COV_ON = '''  COVARIATES:
    - ["XAU", ["close"]]
    - ["XAG", ["close"]]'''
COV_OFF = "  #COVARIATES:   # price-only baseline"

# fields held constant unless a spec overrides them
DEFAULTS = dict(
    comp_enc="amazon/chronos-t5-large", d_model=1024,
    dropout=0.05, label_smoothing=0.0, token_all="false",
    lr=0.00001, ctx_len=30, batch_size=512, cov=True,
)

# name -> overrides. name doubles as the MLflow run name and the file stem.
SPECS = [
    # 1) head x diff, covariates on, default hyperparams
    dict(name="nodiff_linear",       diff="no_diff",  head="linear"),
    dict(name="nodiff_mlp",          diff="no_diff",  head="mlp"),
    dict(name="nodiff_lstm",         diff="no_diff",  head="lstm"),
    dict(name="nodiff_transformer",  diff="no_diff",  head="transformer"),
    dict(name="logdiff_linear",      diff="log_diff", head="linear"),
    dict(name="logdiff_mlp",         diff="log_diff", head="mlp"),
    dict(name="logdiff_lstm",        diff="log_diff", head="lstm"),
    dict(name="logdiff_transformer", diff="log_diff", head="transformer"),

    # 2) price-only baselines (persistence reference)
    dict(name="nodiff_linear_priceonly",  diff="no_diff",  head="linear", cov=False),
    dict(name="logdiff_linear_priceonly", diff="log_diff", head="linear", cov=False),

    # 3) hyperparam variations on log_diff + lstm
    dict(name="logdiff_lstm_do20",   diff="log_diff", head="lstm", dropout=0.2),
    dict(name="logdiff_lstm_ls15",   diff="log_diff", head="lstm", label_smoothing=0.15),
    dict(name="logdiff_lstm_global", diff="log_diff", head="lstm", token_all="true"),
    dict(name="logdiff_lstm_ctx60",  diff="log_diff", head="lstm", ctx_len=60),
    dict(name="logdiff_lstm_lr5e5",  diff="log_diff", head="lstm", lr=0.00005),

    # 4) hyperparam variations on log_diff + transformer
    dict(name="logdiff_transformer_do20", diff="log_diff", head="transformer", dropout=0.2),
    dict(name="logdiff_transformer_ls15", diff="log_diff", head="transformer", label_smoothing=0.15),
]

TEMPLATE = """DATA:
  TARGET:
    ID: "XCU"
    FEATURE: "close"

{cov_block}

  GLOBAL_COVARIATES: [
    "AA_open", "AA_high", "AA_low", "AA_close",
    "BHP_open", "BHP_high", "BHP_low", "BHP_close",
    "FCX_open", "FCX_high", "FCX_low", "FCX_close",
    "MP_open", "MP_high", "MP_low", "MP_close",
    "REMX_open", "REMX_high", "REMX_low", "REMX_close",
    "RIO_open", "RIO_high", "RIO_low", "RIO_close",
    "STOXX50E_open", "STOXX50E_high", "STOXX50E_low", "STOXX50E_close",
    "BRENTOIL_open", "BRENTOIL_high", "BRENTOIL_low", "BRENTOIL_close",
    "CL1_open", "CL1_high", "CL1_low", "CL1_close",
    "NG_open", "NG_high", "NG_low", "NG_close",
    "eur_usd", "eur_cny"
  ]

  NEWS_COL: "embedding"

  SPLITS:
    TEST_DAYS: 230
    VAL_DAYS:  230

  PRED_LEN: 7
  CTX_LEN: {ctx_len}

  TOKEN_ALL: {token_all}
  SHUFFLE_DATA: true
  TYPE_OF_DIFF: "{diff}"  # no_diff | diff | log_diff


SEED: 1


MODEL:
  COMP_ENC: "{comp_enc}"
  CROSS_CHRONOS:
    configs:
      EMB_DIM_NEWS: 768
      D_MODEL: {d_model}
      N_HEADS: 8
      N_LAYERS_TXT: 3
      D_FF: 1024
      DROPOUT: {dropout}
      HEAD: "{head}"  # linear | mlp | lstm | transformer


TRAINING:
  EPOCHS: 300
  LR: {lr}
  WEIGHT_DECAY: 0.000
  GRAD_CLIP: null
  BATCH_SIZE: {batch_size}
  LABEL_SMOOTHING: {label_smoothing}


EVALUATION:
  WINDOWS: 30
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
    FAITH_MASK_STRATEGY: "mean"
    FAITH_RNG_SEED: 123

SCHEDULER:
  use: true
  configs:
    LR_COLD_START: 0.00001
    LR_COLD_EPOCHS: 30
    LR_DECREASE_FACTOR: 0.5
    LR_METRIC: "val"
    LR_PLATEAU_PATIENCE: 10


EARLY_STOPPER:
  use: true
  configs:
    EARLY_STOPPER_PATIENCE: 20


TRACKING:
  MLFLOW:
    use: true
    configs:
      URI: "http://127.0.0.1:5000"
      EXPIREMENT: "{experiment}"
      RUN: "{run}"
      LOG_MODEL: false
  LOCAL:
    use: false
    configs:
      dir: ""
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(SPECS, start=1):
        cfg = {**DEFAULTS, **spec}
        cov_block =COV_OFF
        run = f"{i:02d}_{cfg['name']}"
        # Emit lr in plain decimal, never scientific: PyYAML's implicit-float
        # resolver does NOT match "1e-05" (needs a dot before the exponent) and
        # would load it as a str, which blows up in the optimizer (lr / bc).
        lr_str = f"{cfg['lr']:.8f}".rstrip("0")
        text = TEMPLATE.format(
            cov_block=cov_block, ctx_len=cfg["ctx_len"], token_all=cfg["token_all"],
            diff=cfg["diff"], comp_enc=cfg["comp_enc"], d_model=cfg["d_model"],
            dropout=cfg["dropout"], head=cfg["head"], lr=lr_str,
            batch_size=cfg["batch_size"], label_smoothing=cfg["label_smoothing"],
            experiment=EXPERIMENT, run=run,
        )
        (OUT_DIR / f"{run}.yaml").write_text(text, encoding="utf-8")
    print(f"wrote {len(SPECS)} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
