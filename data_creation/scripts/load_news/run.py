"""Entry point for step 3: fetch news from the NYT Archive API."""

import argparse

from scripts.cli import add_base_dir, add_dataset, add_news_range
from scripts.load_news.load_news import fetch_news
from scripts.paths import dataset_config


def run(base_dir: str, dataset: str, start_year: int = 2007, end_year: int | None = None) -> None:
    """
    Fetch and filter the news for a dataset's topic.

    The topic is taken from the dataset registry rather than a separate flag,
    so a metals build can no longer fetch (or silently reuse) the stocks feed.

    Parameters
    ----------
    base_dir : str
        Root directory holding the shared ``news/`` cache.
    dataset : str
        Dataset key; its ``news_topic`` selects the filter and output directory.
    start_year : int, optional
        First year of the NYT archive to fetch.
    end_year : int or None, optional
        Last year to fetch; None means through the last fully completed month.

    Returns
    -------
    None
        Writes into ``<base_dir>/news/<topic>/``, shared across every dataset
        using that topic. Resumable via per-month checkpoints.
    """
    topic = dataset_config(dataset)["news_topic"]
    fetch_news(target=topic, start_year=start_year, end_year=end_year, base_dir=base_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch news data from the NYT Archive API")
    add_base_dir(parser)
    add_dataset(parser)
    add_news_range(parser)
    args = parser.parse_args()
    run(base_dir=args.base_dir, dataset=args.dataset,
        start_year=args.start_year, end_year=args.end_year)
