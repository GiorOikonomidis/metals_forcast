"""
Every filesystem path the pipeline reads or writes.

Each step used to assemble its own ``os.path.join(base, CONST, CONST)``, which
is why the pipeline could only ever build one dataset: there was no single seam
at which to insert a "which dataset" level. Centralising path construction here
is what lets ``--dataset`` reach all seven steps.

Layout
------
::

    <base-dir>/
      news/<topic>/                     shared across datasets, keyed by topic
        news.csv | news_metals.csv      raw, per topic (name from load_news)
        <topic checkpoints>/            resumable per-month NYT cache
        news_flat.csv                   per-article NLP output
        news_enriched.csv               per-day aggregated NLP output
      <dataset>/
        data/{target,covariates}/       raw OHLCV, one CSV per ticker
        data_enriched/{target,covariates}/
        datasets/                       the three output parquets

News sits outside the per-dataset trees on purpose: it depends only on the news
topic, and both fetching it (NYT is rate-limited to roughly 45 minutes for a
2007-2026 span) and embedding it (a transformer pass over every headline) are
far too expensive to repeat for each dataset that shares a topic.
"""

from __future__ import annotations

import os

from constants import (
    DATASETS, DATASETS_DIR, ENRICHED_DATASETS_DIR, ORIGINAL_DATASETS_DIR,
    COVARIATES_DIR, NEWS_DIR, TARGET_DIR,
    FILE_NAME_FLAT, FILE_NAME_NEWS_ENRH,
)

# Series roles a dataset tree holds, used as the `kind` argument below.
KIND_TARGET     = TARGET_DIR
KIND_COVARIATES = COVARIATES_DIR
_KINDS = (KIND_TARGET, KIND_COVARIATES)


def dataset_config(dataset: str) -> dict:
    """
    Look up a dataset's registry entry.

    Parameters
    ----------
    dataset : str
        Dataset key, e.g. ``"index"`` or ``"metals"``. Must be a key of
        ``constants.DATASETS``.

    Returns
    -------
    dict
        The registry entry, carrying ``target_ticker``, ``target_id``,
        ``covariates`` and ``news_topic``.

    Raises
    ------
    KeyError
        If ``dataset`` is not registered, listing the valid keys. An unknown
        dataset is a configuration mistake, so it fails loudly rather than
        silently producing an empty tree.
    """
    try:
        return DATASETS[dataset]
    except KeyError:
        raise KeyError(
            f"unknown dataset {dataset!r} - valid datasets are {sorted(DATASETS)}. "
            f"Add an entry to DATASETS in constants.py to support a new one."
        ) from None


def _kind(kind: str) -> str:
    """
    Validate a series-role argument.

    Parameters
    ----------
    kind : str
        Either ``KIND_TARGET`` or ``KIND_COVARIATES``.

    Returns
    -------
    str
        The validated role, unchanged.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the two roles.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown series kind {kind!r} - expected one of {list(_KINDS)}")
    return kind


def dataset_root(base_dir: str, dataset: str) -> str:
    """
    Root of a dataset's own tree.

    Parameters
    ----------
    base_dir : str
        Root directory holding every dataset tree and the shared news cache.
    dataset : str
        Dataset key; validated against the registry.

    Returns
    -------
    str
        ``<base_dir>/<dataset>``.
    """
    dataset_config(dataset)
    return os.path.join(base_dir, dataset)


def raw_dir(base_dir: str, dataset: str, kind: str) -> str:
    """
    Directory of downloaded (unenriched) OHLCV CSVs for one series role.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.
    kind : str
        ``KIND_TARGET`` or ``KIND_COVARIATES``.

    Returns
    -------
    str
        ``<base_dir>/<dataset>/data/<kind>``.
    """
    return os.path.join(dataset_root(base_dir, dataset), ORIGINAL_DATASETS_DIR, _kind(kind))


def enriched_dir(base_dir: str, dataset: str, kind: str) -> str:
    """
    Directory of technical-indicator-enriched CSVs for one series role.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.
    kind : str
        ``KIND_TARGET`` or ``KIND_COVARIATES``.

    Returns
    -------
    str
        ``<base_dir>/<dataset>/data_enriched/<kind>``.
    """
    return os.path.join(dataset_root(base_dir, dataset), ENRICHED_DATASETS_DIR, _kind(kind))


def output_dir(base_dir: str, dataset: str) -> str:
    """
    Directory the three output parquets are written to.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.

    Returns
    -------
    str
        ``<base_dir>/<dataset>/datasets``. There is no ``case_interp`` level —
        it once separated two output variants and only the interpolated one is
        produced now, so the dataset name is the sole discriminator.
    """
    return os.path.join(dataset_root(base_dir, dataset), DATASETS_DIR)


def news_dir(base_dir: str, topic: str) -> str:
    """
    Directory of the shared news cache for one topic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    topic : str
        News topic, e.g. ``"stocks"`` or ``"metals"`` — a key of
        ``load_news.TARGETS``.

    Returns
    -------
    str
        ``<base_dir>/news/<topic>``. Shared deliberately: two datasets on the
        same topic reuse one fetch and one set of embeddings.
    """
    return os.path.join(base_dir, NEWS_DIR, topic)


def news_dir_for(base_dir: str, dataset: str) -> str:
    """
    Shared news directory for the topic a dataset uses.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key; its ``news_topic`` selects the directory.

    Returns
    -------
    str
        ``<base_dir>/news/<news_topic>``.
    """
    return news_dir(base_dir, dataset_config(dataset)["news_topic"])


def news_flat_path(base_dir: str, dataset: str) -> str:
    """
    Path of the per-article NLP output for a dataset's news topic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.

    Returns
    -------
    str
        ``<base_dir>/news/<topic>/news_flat.csv``.
    """
    return os.path.join(news_dir_for(base_dir, dataset), FILE_NAME_FLAT)


def news_enriched_path(base_dir: str, dataset: str) -> str:
    """
    Path of the per-day aggregated NLP output for a dataset's news topic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.

    Returns
    -------
    str
        ``<base_dir>/news/<topic>/news_enriched.csv``.
    """
    return os.path.join(news_dir_for(base_dir, dataset), FILE_NAME_NEWS_ENRH)


def target_ticker(dataset: str) -> str:
    """
    The yfinance symbol of a dataset's target series.

    Parameters
    ----------
    dataset : str
        Dataset key.

    Returns
    -------
    str
        e.g. ``"^NDX"`` for the index dataset, ``"HG=F"`` for metals.
    """
    return dataset_config(dataset)["target_ticker"]


def target_filename(dataset: str) -> str:
    """
    Filename of a dataset's downloaded target CSV.

    Parameters
    ----------
    dataset : str
        Dataset key.

    Returns
    -------
    str
        ``"<target_ticker>.csv"`` — matching what the download step writes.
        Replaces the former hardcoded ``FILE_NAME_INDEX = "^NDX.csv"``, which
        made every dataset read the Nasdaq's file as its target.
    """
    return f"{target_ticker(dataset)}.csv"


def target_path(base_dir: str, dataset: str, enriched: bool = False) -> str:
    """
    Full path of a dataset's target CSV.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.
    enriched : bool, optional
        Select the enriched tree instead of the raw download tree.

    Returns
    -------
    str
        Path to ``<target_ticker>.csv`` under the chosen tree.
    """
    directory = (enriched_dir if enriched else raw_dir)(base_dir, dataset, KIND_TARGET)
    return os.path.join(directory, target_filename(dataset))
