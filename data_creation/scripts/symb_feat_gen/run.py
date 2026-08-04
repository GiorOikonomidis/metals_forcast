"""Entry point for steps 4-5: enrich raw OHLCV with technical indicators."""

import argparse

from scripts.cli import add_base_dir, add_dataset
from scripts.symb_feat_gen.symb_feat_gen import pipe_line


def run(mode: int, base_dir: str, dataset: str) -> None:
    """
    Enrich one role of a dataset's downloaded price data.

    Always produces raw (undifferenced) levels — differencing is a model-layer
    transform applied at load time in ``model_impl``, so one enriched tree
    serves every variant.

    Parameters
    ----------
    mode : int
        ``0`` enriches the target series, ``1`` enriches the covariates.
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.

    Returns
    -------
    None
    """
    pipe_line(base_dir=base_dir, dataset=dataset, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich OHLCV with technical indicators (raw levels; differencing happens in model_impl)")
    add_base_dir(parser)
    add_dataset(parser)
    parser.add_argument("--mode", type=int, default=0, choices=[0, 1],
                        help="0=target series, 1=covariates")
    args = parser.parse_args()
    run(mode=args.mode, base_dir=args.base_dir, dataset=args.dataset)
