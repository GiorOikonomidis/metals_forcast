"""
Package-wide constants: directory names, filenames, column lists, and the
dataset registry.

The registry (DATASETS, below) is the single place that ties a dataset's
identity together — its yfinance target ticker, the id stamped into the output
parquet, its covariate tickers, and its news topic. Every step derives what it
needs from one --dataset argument, so those four can no longer drift apart.
Path construction lives in scripts/paths.py, not here.
"""

# ── directory names ───────────────────────────────────────────────────────────
# Layout (see scripts/paths.py, which is the only module that assembles these):
#
#   <base-dir>/
#     news/<topic>/                 shared across datasets, keyed by news topic
#     <dataset>/data/{target,covariates}/
#     <dataset>/data_enriched/{target,covariates}/
#     <dataset>/datasets/*.parquet
#
ORIGINAL_DATASETS_DIR = "data"
ENRICHED_DATASETS_DIR = "data_enriched"
DATASETS_DIR          = "datasets"

# Series role within a dataset tree. Named by role, not by asset type: the
# covariates are Nasdaq companies for one dataset and mining equities for
# another, so "companies" would be wrong for half of them.
TARGET_DIR     = "target"
COVARIATES_DIR = "covariates"
NEWS_DIR       = "news"


# ── filenames ─────────────────────────────────────────────────────────────────
# The news topic directory disambiguates, so the enriched news filenames are
# shared across topics. The raw per-topic news filenames live in
# scripts/load_news/load_news.py's TARGETS table, which also owns the filters.
FILE_NAME_FLAT      = "news_flat.csv"
FILE_NAME_NEWS_ENRH = "news_enriched.csv"

FILE_NAME_TARGET_VARS = "target_variables.parquet"   # long: date,id,open,high,low,close
FILE_NAME_GLOBAL_COV  = "global_covariates.parquet"  # wide: date + {TICKER}_{feat} + embedding
FILE_NAME_FEAT_COV    = "feature_covariates.parquet" # date + sin/cos calendar encodings


# ── covariate ticker pools ────────────────────────────────────────────────────
NASDAQ_100_YAHOO = [
    "ADBE","ADP","AMD","ABNB","ALNY","GOOGL","GOOG","AMZN","AEP","AMGN",
    "ADI","AAPL","AMAT","APP","ARM","ASML","TEAM","ADSK","AXON","BKR",
    "BKNG","AVGO","CDNS","CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG",
    "CPRT","CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DASH","EA",
    "EXC","FAST","FER","FTNT","GEHC","GILD","HON","IDXX","INSM","INTC",
    "INTU","ISRG","KDP","KLAC","KHC","LRCX","LIN","MAR","MRVL","MELI",
    "META","MCHP","MU","MSFT","MSTR","MDLZ","MPWR","MNST","NFLX","NVDA",
    "NXPI","ODFL","ORLY","PCAR","PLTR","PANW","PAYX","PYPL","PDD","PEP",
    "QCOM","REGN","ROP","ROST","STX","SHOP","SBUX","SNPS","TTWO","TSLA",
    "TXN","TRI","TMUS","VRSK","VRTX","WMT","WBD","WDC","WDAY","XEL","ZS"
]

# Mining/metals equity block of the existing global_covariates.parquet. All but
# "MP" are the tickers whose features cleared the |r| > 0.25 relevance filter
# against copper in val_data/correlation/RESULTS.md. This is the *available
# pool* — which columns a run actually consumes stays a per-experiment choice,
# made via GLOBAL_COVARIATES in the model yaml.
#
# The energy (BRENTOIL, CL1, NG), FX (eur_usd, eur_cny) and STOXX50E columns
# also in that parquet are deliberately absent: none cleared the relevance
# filter, and they are the only ones whose column prefix differs from their
# yfinance symbol (CL1 -> CL=F, eur_usd -> EURUSD=X, ...). Keeping them out is
# what lets every ticker below be used verbatim as both download symbol and
# column prefix.
METALS_MINING_YAHOO = ["FCX", "BHP", "RIO", "AA", "REMX", "MP"]


# ── dataset registry ──────────────────────────────────────────────────────────
# One entry per buildable dataset. Adding a third dataset is one entry here and
# nothing else.
#
#   target_ticker : yfinance symbol for the target series (downloaded in step 1
#                   and read back by the merge step as "<target_ticker>.csv")
#   target_id     : identifier written into target_variables.parquet's id
#                   column; matched by model_impl's TARGET.ID
#   covariates    : yfinance symbols downloaded as covariates (step 2). Each is
#                   used verbatim as the wide-panel column prefix, e.g.
#                   "FCX" -> FCX_close
#   news_topic    : key into scripts/load_news/load_news.py's TARGETS table,
#                   selecting the desk/section/tag/keyword filter. News is
#                   cached per topic and shared across datasets.
DATASETS = {
    "index": {
        "target_ticker": "^NDX",
        "target_id":     "^nsdq",
        "covariates":    NASDAQ_100_YAHOO,
        "news_topic":    "stocks",
    },
    "metals": {
        "target_ticker": "HG=F",     # copper front-month future
        "target_id":     "XCU",
        "covariates":    METALS_MINING_YAHOO,
        "news_topic":    "metals",
    },
}


# ── column lists ──────────────────────────────────────────────────────────────
PRICE_COLS     = ["Close", "High", "Low", "Open", "Volume"]

# Order matters: run_day_aggregate and fill_news_isolated_gaps take an argmax
# over these columns and map the winning position back to a label, so the label
# order must match this order. The per-model class order is a separate concern
# handled in news_feat_gen (see MODELS there) — probabilities are mapped into
# these columns by name, never by the model's own index.
PROB_COLS      = ["prob_positive", "prob_negative", "prob_neutral"]
PROB_LABELS    = ["positive", "negative", "neutral"]

DATE_FEAT_COLS = ["sin_dow", "cos_dow", "sin_month", "cos_month", "sin_doy", "cos_doy"]
NEWS_FEAT_COLS = ["embedding", "label", "prob_positive", "prob_negative", "prob_neutral"]
INDEX_FEATURES = [
    "Open","High","Low","Volume",
    "EMA_12", "EMA_26", "MACD", "RSI", "Stoch_K", "Stoch_D",
    "Williams_R", "ROC", "Daily_Return", "Volatility", "Movement",
]


# ── merge-step output shape ───────────────────────────────────────────────────
# Optional feature groups — each gated independently.
GLOBAL_INCLUDE_TECH_FEATURES = True   # fold tech indicators into global_covariates
WRITE_FEATURE_COVARIATES     = True   # emit the sin/cos date-feature parquet
# News into global_covariates. When ON, the merge step folds the enriched news
# `embedding` column in automatically — dense (zero-filled on days without
# news), because model_impl's news loader requires an array on every row. Set to
# False to emit a news-less global file; model_impl must then run with
# no_news=True, or it will fail looking for the embedding column.
WRITE_NEWS_TO_GLOBAL         = True
NEWS_INCLUDE_SENTIMENT       = True   # add prob_*/label beside the embedding column

# Wide-column separator for global_covariates (metals convention: "AA_close").
GLOBAL_COV_SEP = "_"

# Source price columns copied into the wide global panel, per covariate series.
# Written lowercased ({TICKER}_open, ...) to match the metals column casing.
GLOBAL_COVARIATE_PRICE_COLS = ["Open", "High", "Low", "Close"]

# OHLC columns written to the long target_variables file (lowercased on write).
TARGET_OHLC_COLS = ["Open", "High", "Low", "Close"]
