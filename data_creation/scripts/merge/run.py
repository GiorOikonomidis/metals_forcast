"""Entry point for step 7: build the output parquets."""

import argparse

from scripts.cli import add_base_dir, add_cutoff, add_dataset
from scripts.merge.dataset_builder import pipe_line


def run(base_dir: str, dataset: str, cutoff_date: str | None = None,
        min_start: str | None = None) -> None:
    """
    Build a dataset's three output parquets.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; supplies the target ticker, output id and covariate set.
    cutoff_date : str or None, optional
        ``"YYYY-MM-DD"``; drop output dates on or after it. None means none.
    min_start : str or None, optional
        ``"YYYY-MM-DD"``; exclude covariates whose history starts after it.

    Returns
    -------
    None
    """
    pipe_line(base_dir=base_dir, dataset=dataset, cutoff_date=cutoff_date, min_start=min_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build target_variables / global_covariates / feature_covariates parquets")
    add_base_dir(parser)
    add_dataset(parser)
    add_cutoff(parser)
    parser.add_argument("--min-start", default=None,
                        help="Exclude covariates whose history starts after this date (YYYY-MM-DD)")
    args = parser.parse_args()
    run(base_dir=args.base_dir, dataset=args.dataset,
        cutoff_date=args.cutoff_date, min_start=args.min_start)
