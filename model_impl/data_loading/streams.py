"""
Stream composition: loads the raw parquet streams and derives the
TYPE_OF_DIFF representation. The one place where loaders and transforms meet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from model_impl.data_loading.loaders import covariates, global_covariates
from model_impl.utils.data_loader_utils.transforms import apply_differencing

if TYPE_CHECKING:
    from model_impl.utils.schemas import DataConfig


def load_streams(index: str, no_news: bool, data_cfg: DataConfig, emb_dim_news: int,
                 tgt_path: str, global_path: str,
                 ) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame, pd.Series, int]:
    """
    Load the input streams and derive the TYPE_OF_DIFF representation.

    `tgt_path` is the long-format file (--target-covariate-path): both the target
    series and the per-id covariates are read from it. `global_path` is the wide
    global file (--global-covariate-path): the scalar global covariates and the
    news `embedding` column. Both are validated (file must exist) in
    arg_handler.cli_parser.parse before they reach here.

    The parquet trees always store raw interpolated levels; differencing is a
    model-layer transform applied here, right after loading (see
    data_loader_utils.transforms.apply_differencing). The raw series is kept
    before differencing — it is the reconstruction anchor, no second parquet
    read needed.

    `emb_dim_news` comes from ModelConfig.cross_chronos rather than DataConfig:
    it is the shape of the model's news_proj input, needed here only to build
    the zero-fill template when `no_news` is set.

    Returns
    -------
    prices_target : target Close, differenced or raw per data_cfg.type_of_diff
    raw_series    : raw (undifferenced) Close, or None when type_of_diff is "no_diff"
                    (prices_target is already raw in that case)
    covariate_df, news, n_covariates
    """
    # Target: one (id, [feature]) from the long file. covariates() names it
    # "{id}_{feature}", but downstream (raw_series below, main.temporal_split)
    # expects a plain "Close" column.
    keep_target = [(data_cfg.target.id, [data_cfg.target.feature])]
    prices_target, _ = covariates(file_path=tgt_path, keep_covariates=keep_target)
    prices_target.columns = ["Close"]

    if prices_target.squeeze().isna().all():
        raise ValueError(f"No price data available for '{index}'.")

    # Keep the raw level before differencing — used as the reconstruction anchor
    # and for scoring/plotting in price space.
    raw_series = None
    if data_cfg.type_of_diff != "no_diff":
        raw_series = prices_target["Close"].copy()

    prices_target = apply_differencing(prices_target, data_cfg.type_of_diff)

    # Covariate panel: long-file per-id covs + wide global covs, concatenated
    # (assume perfectly aligned for now — no fill/reindex). n is their summed width.
    cov_long,   n_long   = covariates(file_path=tgt_path, keep_covariates=data_cfg.covariates)
    cov_global, n_global = global_covariates(file_path=global_path, keep_cols=data_cfg.global_covariates)
    covariate_df = pd.concat([cov_long, cov_global], axis=1)
    n_covariates = n_long + n_global
    # Covariate columns used to get the same representation as the target (they
    # were part of DIFF_FEAT_COLS when differencing lived in the dataset build).
    # Disabled: several covariates (RSI, Stoch_K/D, Williams_R, MACD, ROC,
    # Daily_Return, Volatility, Movement) carry their signal in the current
    # level, not its day-to-day change — differencing them destroyed that.
    # covariate_df = apply_differencing(covariate_df, data_cfg.type_of_diff)

    if no_news:
        # use as template the target
        news = pd.Series([np.zeros(emb_dim_news, dtype=np.float32) for _ in range(len(prices_target))],
                         index=prices_target.index)
    else:
        # news lives in the wide global file as the `embedding` column
        news = global_covariates(file_path=global_path,
                                 keep_cols=[data_cfg.news_col], embedding=True)

    return prices_target, raw_series, covariate_df, news, n_covariates
