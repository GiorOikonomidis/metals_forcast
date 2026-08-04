"""Entry point for steps 1-2: download target / covariate OHLCV."""

import argparse

from scripts.cli import add_base_dir, add_dataset, add_dates
from scripts.load_symbols.load_symb import run_covariates, run_target


def run(mode: int, base_dir: str, dataset: str, date_start: str = None, date_end: str = None) -> None:
    """
    Download one role of a dataset's raw price data.

    Parameters
    ----------
    mode : int
        ``0`` downloads the target series, ``1`` downloads the covariates.
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; determines both the target ticker and the covariate list.
    date_start : str, optional
        Start date, ``"YYYY-MM-DD"``.
    date_end : str, optional
        End date, ``"YYYY-MM-DD"``; None means today.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``mode`` is neither 0 nor 1.
    """
    if mode == 0:
        run_target(base_dir, dataset, date_start, date_end)
    elif mode == 1:
        run_covariates(base_dir, dataset, date_start, date_end)
    else:
        raise ValueError(f"unknown mode {mode!r} - expected 0 (target) or 1 (covariates)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download target/covariate OHLCV from Yahoo Finance")
    add_base_dir(parser)
    add_dataset(parser)
    add_dates(parser)
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1],
                        help="0=target series, 1=covariates")
    args = parser.parse_args()
    run(mode=args.mode, base_dir=args.base_dir, dataset=args.dataset,
        date_start=args.date_start, date_end=args.date_end)
