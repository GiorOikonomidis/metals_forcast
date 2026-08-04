"""Command-line argument parsing for main.py."""

import argparse
from pathlib import Path

"""
inputs from user :
        config
        dynamic_covariate_path
        target_covariate_path
        featre_covariate_path

"""
def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chronos with full cross-attention (index/news/covariates) + evaluation."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--dynamic-covariate-path", type=str, required=False,
                        help="Path to dynamic_covariates.parquet")
    parser.add_argument("--target-covariate-path", type=str, required=True,
                        help="Path to the long-format file (target + per-id covariates)")
    parser.add_argument("--global-covariate-path", type=str, required=True,
                        help="Path to the wide global_covariates.parquet (global covs + news)")
    parser.add_argument("--feature-covariate-path", type=str, required=False, default="",
                        help="Path to feature_covariates.parquet (optional — no loader reads it yet)")

    # parser.add_argument("--index", type=str, default=DEFAULT_INDEX,
    #                     help=f"Index ticker (default: {DEFAULT_INDEX})")
    # parser.add_argument("-w", "--windows", type=int, default=WINDOWS,
    #                     help=f"Number of windows to evaluate (default {WINDOWS})")
    # parser.add_argument("-c", "--companies", type=int, default=None,
    #                     help="Number of companies to use (default: all)")
    # parser.add_argument("--no-news", action="store_true",
    #                     help="Ignore news embeddings (fill with zeros)")
    # parser.add_argument("--debug", action="store_true",
    #                     help="Enable attention/Grad-CAM visualizations for window 1")
    # parser.add_argument("--faithfulness", action="store_true",
    #                     help="Compute saliency faithfulness metrics (deletion/insertion, LOTO)")
    # parser.add_argument("--max-news-per-day", type=int, default=None,
    #                     help="Cap news articles per day used to build daily embeddings")

    opts = parser.parse_args()

    # Fail fast: these two drive the whole run, so a bad path should stop the
    # process here with a clear message rather than surface as a cryptic
    # pyarrow error deep inside data_loading. feature-covariate-path is
    # exempt — it may be left empty since no loader consumes it yet.
    for flag, value in [("--target-covariate-path", opts.target_covariate_path),
                        ("--global-covariate-path", opts.global_covariate_path)]:
        if not Path(value).is_file():
            parser.error(f"{flag}: file not found: {value!r}")

    return opts
