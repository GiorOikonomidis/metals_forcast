"""Duration-based (row-count) train/val/test splitting of the aligned streams."""

from collections import namedtuple

import pandas as pd

from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)

Split = namedtuple("Split", ["prices", "news", "covariate"])


def temporal_split(prices: pd.Series, news: pd.Series, covariate: pd.DataFrame,
                   test_days: int, val_days: int, ctx: int) -> tuple[Split, Split, Split]:
    """
    Split the aligned streams into train / val / test by end-anchored row counts.

    "days" are samples/rows on the aligned series (all three streams share one
    index). test is the last `test_days` prediction rows, val the `val_days`
    rows immediately before, train everything before that. val and test each get
    `ctx` rows of lead-in so their first sliding window has full context (mirrors
    the old date-based ctx back-off).

    Prices are kept as a pd.Series (DatetimeIndex) and tokenized per-split in
    dataset_windows, because Chronos tokenization is not 1-to-1 with trading days.

    Parameters
    ----------
    prices : pd.Series      DatetimeIndex Close prices (from load_streams)
    news   : pd.Series      DatetimeIndex embedding per day
    covariate : pd.DataFrame  DatetimeIndex {id}_{feature} columns
    test_days, val_days : int  split durations in rows (end-anchored)
    ctx : int               context length (lead-in rows for val/test)

    Returns
    -------
    tuple[Split, Split, Split]
    """
    n = len(prices)
    test_start = n - test_days
    val_start = n - test_days - val_days
    if val_start - ctx < 0:
        logger.warning(
            "temporal_split: series too short (T=%d) for test_days=%d + val_days=%d + ctx=%d "
            "— train/val lead-in will be clipped", n, test_days, val_days, ctx)

    def _slice(lo: int, hi: int | None) -> Split:
        lo = max(lo, 0)
        sl = slice(lo, hi)
        return Split(prices=prices.iloc[sl], news=news.iloc[sl], covariate=covariate.iloc[sl])

    train = _slice(0, val_start)
    val   = _slice(val_start - ctx, test_start)
    test  = _slice(test_start - ctx, None)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        if len(split.prices) == 0:
            logger.warning("temporal_split [%s]: T=0  (empty)", name)
            continue
        logger.info("temporal_split [%s]: T=%d  %s -> %s",
                    name, len(split.prices), split.prices.index[0].date(), split.prices.index[-1].date())

    return train, val, test
