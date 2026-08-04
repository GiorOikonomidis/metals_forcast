"""
Sliding-window construction over the three aligned streams, with Chronos
tokenization of context and target, plus the per-split window diagnostics.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from chronos import ChronosTokenizer

from model_impl.data_loading.splitting import Split
from model_impl.utils.logger_utils.logger import get_logger

logger = get_logger(__name__)

Windows = namedtuple("Windows", ["xe", "xn", "xc", "y", "scales"])


# first task
# ───── Sliding windows (three streams) ─────
def sliding_windows_triple(prices_ser: pd.Series, news_ser: pd.Series,
                            covariate_df: pd.DataFrame, ctx: int, pred: int,
                            tokenizer: ChronosTokenizer, token_all_: bool ,
                            scaler: torch.Tensor | None = None) -> tuple[Windows, torch.Tensor]:
    """
    Build sliding windows across raw prices, news embeddings and covariate panel,
    tokenizing context and target into Chronos tokens.

    Scaling is controlled by `token_all_`:

    - token_all_ == 1  (GLOBAL): one scale for the whole split. If `scaler` is None it
      is computed once from this split's full price series and returned; otherwise the
      passed-in scaler is reused, so val/test inherit the train scale and land in the
      same token space. Context and target are tokenized via label_input_transform
      under that single shared scale.
    - token_all_ == 0  (PER-WINDOW): every window gets its own scale, computed from its
      own context via context_input_transform. Nothing is shared across windows or
      splits, so `scaler` is ignored and None is returned.

    In both cases the target (N, pred) is tokenized with the same scale as its context,
    and the tokenizer's prediction_length is toggled (ctx, then pred) so each batch
    passes the tokenizer's internal length assert.

    Parameters
    ----------
    prices_ser : pd.Series       raw Close prices with DatetimeIndex
    news_ser   : pd.Series       daily news embeddings (each entry a float32 array)
    covariate_df    : pd.DataFrame    covariate feature panel aligned to same DatetimeIndex
    ctx        : int             context window length (days)
    pred       : int             prediction horizon (days)
    tokenizer  : ChronosTokenizer
    token_all_ : Bool             True = global scale (whole split), False = per-window scale
    scaler     : torch.Tensor | None
                 Only used in GLOBAL mode: if None it is computed from prices_ser and
                 returned; otherwise reused. Ignored in per-window mode.

    Returns
    -------
    tuple[Windows, torch.Tensor | None]
        Windows:
            xe     : LongTensor  (N, ctx)      context token ids (no EOS)
            xn     : FloatTensor (N, ctx, 768) news embeddings
            xc     : FloatTensor (N, ctx, C)   covariate features
            y      : LongTensor  (N, pred)     target token ids, same scale as xe (no EOS)
            scales : FloatTensor (N,)          per-window scale used to tokenize each window
                                               (identical across windows in GLOBAL mode)
        scaler : FloatTensor | None            GLOBAL: the shared scale (reuse downstream);
                                               PER-WINDOW: None
    """
    prices_values = prices_ser.values.astype(np.float32)
    covariate_values   = covariate_df.values


    T = len(prices_values)

    # All three streams are windowed by positional index, so they must be the same
    # length and row-aligned — otherwise a window's news/covariate would not match its prices.
    assert len(news_ser) == T and len(covariate_df) == T, (
        f"stream length mismatch: prices={T}, news={len(news_ser)}, covariate={len(covariate_df)}"
    )

    xs_prices, xs_news, xs_covariate, ys_prices = [], [], [], []

    for i in range(ctx, T - pred + 1):
        xs_prices.append(prices_values[i-ctx:i])
        xs_news.append(np.stack(news_ser.iloc[i-ctx:i].values))
        xs_covariate.append(covariate_values[i-ctx:i])
        ys_prices.append(prices_values[i:i+pred])

    batch_ctx = torch.FloatTensor(np.stack(xs_prices))              # (N, ctx)

    if token_all_ == True:
        # if its train dataset , we have to make the scaler
        if scaler is None:
            object.__setattr__(tokenizer.config, 'context_length', T)
            tok , _ , scaler = tokenizer.context_input_transform(torch.tensor(prices_values, dtype=torch.float32))

        # test or val scaler already calculated , just use it
        object.__setattr__(tokenizer.config, 'prediction_length', ctx)
        num_of_batches = len(batch_ctx)
        # make scaler dim of batches
        scaler_vector_ = scaler.expand(num_of_batches)
        xe, _ = tokenizer.label_input_transform(batch_ctx,scaler_vector_)

    else :
        xe, _ , scaler_vector_ = tokenizer.context_input_transform(batch_ctx)

    object.__setattr__(tokenizer.config, 'prediction_length', pred)
    batch_tgt = torch.FloatTensor(np.stack(ys_prices))
    y, _ = tokenizer.label_input_transform(batch_tgt, scaler_vector_)            # (N, pred)

    return  Windows(
        xe=xe,
        xn=torch.FloatTensor(np.stack(xs_news)),
        xc=torch.FloatTensor(np.stack(xs_covariate)),
        y=y,
        scales=scaler_vector_,
    ) , scaler


#TODO my goal here is
def dataset_windows(train: Split, val: Split, test: Split,
                    ctx: int, pred: int, tokenizer: ChronosTokenizer,
                    token_all: bool) -> tuple[Windows, Windows, Windows]:
    """
    Window all three streams for each split, tokenizing under the scheme set by token_all.

    - token_all == 1 (GLOBAL): the scale is computed once on the train split and threaded
      into val and test, so all three splits share the same token space. The scaler is
      used internally and not returned.
    - token_all == 0 (PER-WINDOW): each split self-normalizes window by window; no scale
      is shared, so the (None) scaler threaded to val/test is simply ignored.

    Parameters
    ----------
    train/val/test : Split    each .prices is a pd.Series of raw Close prices
    ctx, pred      : int      context and prediction lengths
    tokenizer      : ChronosTokenizer
    token_all_ : Bool             True = global scale (whole split), False = per-window scale
    Returns
    -------
    tuple[Windows, Windows, Windows]
    """

    train_wind  , scaler = sliding_windows_triple(prices_ser=train.prices,
                                                  news_ser=train.news,
                                                  covariate_df=train.covariate,
                                                  ctx=ctx,
                                                  pred=pred,
                                                  tokenizer=tokenizer,
                                                  token_all_=token_all)

    val_window , _ = sliding_windows_triple(prices_ser=val.prices,
                                            news_ser=val.news,
                                            covariate_df=val.covariate,
                                            ctx=ctx,
                                            pred=pred,
                                            tokenizer=tokenizer,
                                            token_all_=token_all,
                                            scaler=scaler)

    test_window , _ = sliding_windows_triple(prices_ser=test.prices,
                                             news_ser=test.news,
                                             covariate_df=test.covariate,
                                             ctx=ctx,
                                             pred=pred,
                                             tokenizer=tokenizer,
                                             token_all_=token_all,
                                             scaler=scaler)
    return (
        train_wind,
        val_window,
        test_window
    )


def print_window_report(name: str, wins: Windows, split: Split, ctx_len: int, pred_len: int) -> None:
    """Shape, token-range and date-span diagnostics for one split's windows."""
    N = wins.xe.shape[0]
    first_day = split.prices.index[ctx_len]
    last_day  = split.prices.index[-pred_len]
    # wins.xc has 0 total elements when there are no covariates (shape (N, ctx, 0)) —
    # torch.min/max need an explicit `dim` on an empty tensor, so guard rather
    # than let this crash purely for logging purposes.
    if wins.xc.numel() > 0:
        xc_min, xc_max = float(wins.xc.min()), float(wins.xc.max())
    else:
        xc_min, xc_max = float("nan"), float("nan")
    logger.info(
        "\n[%s]"
        "\n  xe : %s  token id [%d, %d]"
        "\n  xn : %s  embedding norm mean %.3f"
        "\n  xc : %s  price range [%.1f, %.1f]"
        "\n  y  : %s   token id [%d, %d]"
        "\n  scales: %s   scale values [%.5f, %.5f] mean=%.1f"
        "\n  Predictions span: %s -> %s  (N=%d)",
        name,
        tuple(wins.xe.shape), int(wins.xe.min()), int(wins.xe.max()),
        tuple(wins.xn.shape), float(wins.xn.norm(dim=-1).mean()),
        tuple(wins.xc.shape), xc_min, xc_max,
        tuple(wins.y.shape), int(wins.y.min()), int(wins.y.max()),
        tuple(wins.scales.shape), float(wins.scales.min()), float(wins.scales.max()), float(wins.scales.mean()),
        first_day.date(), last_day.date(), N,
    )
