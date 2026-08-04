"""
Step 1-2: download raw OHLCV from Yahoo Finance.

Both the target series and its covariates come from here; which tickers those
are is entirely determined by the dataset registry, so this module contains no
per-dataset knowledge of its own.
"""

import os

import yfinance as yf

from scripts.paths import KIND_COVARIATES, KIND_TARGET, dataset_config, raw_dir, target_ticker


def get_yfinance_ticker(ticker: str, date_start: str, date_end: str, file_path: str) -> None:
    """
    Download OHLCV data for a single ticker and save it as a CSV.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. ``"AAPL"``, ``"HG=F"``).
    date_start : str
        Start date, ``"YYYY-MM-DD"``.
    date_end : str or None
        End date, ``"YYYY-MM-DD"``; None means today.
    file_path : str
        Directory the CSV is written to, as ``<ticker>.csv``.

    Returns
    -------
    None
    """
    df = yf.download(ticker, start=date_start, end=date_end, auto_adjust=True,
                     progress=False, multi_level_index=False)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df = df.reset_index()
    path = os.path.join(file_path, f"{ticker}.csv")
    df.to_csv(path, index=False)
    print(f"Saved {ticker} -> {path}")


def run_target(base_dir: str, dataset: str, date_start: str, date_end: str = None) -> None:
    """
    Download a dataset's target series.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; its ``target_ticker`` is what gets downloaded.
    date_start : str
        Start date, ``"YYYY-MM-DD"``.
    date_end : str or None, optional
        End date, ``"YYYY-MM-DD"``; None means today.

    Returns
    -------
    None
        Writes ``<base_dir>/<dataset>/data/target/<target_ticker>.csv``.
    """
    out_dir = raw_dir(base_dir, dataset, KIND_TARGET)
    os.makedirs(out_dir, exist_ok=True)
    get_yfinance_ticker(target_ticker(dataset), date_start, date_end, out_dir)


def run_covariates(base_dir: str, dataset: str, date_start: str, date_end: str = None) -> None:
    """
    Download a dataset's covariate series, one CSV per ticker.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; its ``covariates`` list is what gets downloaded.
    date_start : str
        Start date, ``"YYYY-MM-DD"``.
    date_end : str or None, optional
        End date, ``"YYYY-MM-DD"``; None means today.

    Returns
    -------
    None
        Writes ``<base_dir>/<dataset>/data/covariates/<ticker>.csv`` per ticker.
    """
    out_dir = raw_dir(base_dir, dataset, KIND_COVARIATES)
    os.makedirs(out_dir, exist_ok=True)
    for ticker in dataset_config(dataset)["covariates"]:
        get_yfinance_ticker(ticker, date_start, date_end, out_dir)
