"""
True constants: plain values the code directly defines or uses, with no
configurable counterpart in utils/schemas and no yaml key that could ever
override them. This is NOT a fallback store for the config system — every
value a yaml/config can set has its own default colocated on the matching
dataclass field in utils/schemas, and config_file_parser.py never reads
from this module.

Parquet paths are NOT here: they come from the CLI (--dynamic-covariate-path,
--target-covariate-path, --feature-covariate-path), validated in
arg_handler/cli_parser.py and threaded from main() into data_loading.streams.
"""

# ── plotting ────────────────────────────────────────────────────────────────
FIG_DPI = 160

##TODO [!] not yet , leave for  now  ,need to think the implementation here
### head architecture
HEAD_HIDDEN_1=64
HEAD_HIDDEN_1_5 = HEAD_HIDDEN_1*3
HEAD_HIDDEN_2=HEAD_HIDDEN_1_5

# ── output layout ───────────────────────────────────────────────────────────
# Named once here because several of these are read from more than one file
# (RUN_LOG_FILE by both run_log.py and logger.py; SUMMARY_FILE by both
# scripts/test.py and scripts/validation.py) — a literal in each risked drift.
OUTPUT_ROOT = "output"
RUN_LOG_FILE = "run.log"
CONFIG_SNAPSHOT_FILE = "config_snapshot"
SUMMARY_FILE = "summary"
METRICS_PER_WINDOW_FILE = "metrics_per_window"
FORECASTS_BY_WINDOW_FILE = "forecasts_by_window"
FORECASTS_DIR = "forecasts"
HORIZONS_DIR = "horizons"
VALIDATION_DIR = "validation"

# ── faithfulness study output ────────────────────────────────────────────────
FAITH_PER_WINDOW_FILE = "faithfulness_per_window.jsonl"
FAITH_SUMMARY_FILE = "faithfulness_summary.json"
