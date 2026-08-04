"""Entry point for step 6: run the NLP model over fetched news."""

import argparse

from scripts.cli import add_base_dir, add_dataset, add_model
from scripts.news_feat_gen.news_feat_gen import pipe_line


def run(base_dir: str, dataset: str, run_flat: int = 1, model: str = "finbert") -> None:
    """
    Produce per-article and per-day NLP features for a dataset's news topic.

    Parameters
    ----------
    base_dir : str
        Root directory holding the shared ``news/`` cache.
    dataset : str
        Dataset key; its ``news_topic`` selects which cache is processed.
    run_flat : int, optional
        When truthy, re-run the per-article pass; when falsy, reuse the
        existing flat file and only redo the per-day aggregation.
    model : str, optional
        ``finbert``, ``financialbert`` or ``minilm``.

    Returns
    -------
    None
    """
    pipe_line(base_dir=base_dir, dataset=dataset, run_flat=run_flat, model=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLP on news -> flat CSV -> per-day aggregate CSV")
    add_base_dir(parser)
    add_dataset(parser)
    add_model(parser)
    args = parser.parse_args()
    run(base_dir=args.base_dir, dataset=args.dataset,
        run_flat=int(args.run_flat), model=args.model)
