"""
Full-pipeline entry point.

The pipeline is an ordered table of named steps (``STEPS``) rather than a fixed
sequence of calls, so any subset can be run with ``--only`` / ``--skip`` without
editing code. Selected steps always execute in table order regardless of the
order they are listed in, because each depends on the outputs of those before
it — ``--only merge,news`` runs news first, not merge.

Every step takes the same ``--dataset`` key and derives its target ticker,
output id, covariate set and news topic from the registry in ``constants.py``,
so the parts of a build cannot disagree with one another.
"""

import argparse
import os

from scripts.cli import (add_base_dir, add_cutoff, add_dataset, add_dates,
                         add_model, add_news_range, add_step_selection)
from scripts.load_news.load_news import TARGETS
from scripts.paths import (KIND_COVARIATES, KIND_TARGET, dataset_config,
                           enriched_dir, news_dir, news_flat_path, raw_dir,
                           target_path)


# ── preconditions ─────────────────────────────────────────────────────────────
# Each returns None when the step can run, or a string explaining what is
# missing. Without these, running a step whose inputs are absent fails deep
# inside pandas with a bare FileNotFoundError, which is what made --only/--skip
# unsafe to use.

def _needs_dir(path: str, what: str, produced_by: str) -> str | None:
    """
    Check that a directory exists and holds at least one CSV.

    Parameters
    ----------
    path : str
        Directory expected to hold the inputs.
    what : str
        Human-readable description of the missing inputs.
    produced_by : str
        Step key that produces them, named in the message.

    Returns
    -------
    str or None
        An explanatory message, or None when the inputs are present.
    """
    if not os.path.isdir(path) or not any(f.endswith(".csv") for f in os.listdir(path)):
        return f"no {what} at {path} - run step {produced_by!r} first"
    return None


def _needs_file(path: str, what: str, produced_by: str) -> str | None:
    """
    Check that a single expected file exists.

    Parameters
    ----------
    path : str
        File expected to be present.
    what : str
        Human-readable description of the missing input.
    produced_by : str
        Step key that produces it, named in the message.

    Returns
    -------
    str or None
        An explanatory message, or None when the file is present.
    """
    if not os.path.isfile(path):
        return f"no {what} at {path} - run step {produced_by!r} first"
    return None


def _raw_news_path(base_dir: str, dataset: str) -> str:
    """
    Path of the raw fetched news CSV for a dataset's topic.

    Parameters
    ----------
    base_dir : str
        Root directory.
    dataset : str
        Dataset key.

    Returns
    -------
    str
        ``<base_dir>/news/<topic>/<topic out_name>``.
    """
    topic = dataset_config(dataset)["news_topic"]
    return os.path.join(news_dir(base_dir, topic), TARGETS[topic]["out_name"])


# ── step runners ──────────────────────────────────────────────────────────────
# Each imports its module lazily. The NLP step pulls in torch/transformers,
# which costs minutes; importing that at module scope would make --help, step
# selection and every non-NLP step pay for it too.

def _run_target(a) -> None:
    """Download the dataset's target series. Returns None."""
    from scripts.load_symbols.run import run
    run(mode=0, base_dir=a.base_dir, dataset=a.dataset,
        date_start=a.date_start, date_end=a.date_end)


def _run_covariates(a) -> None:
    """Download the dataset's covariate series. Returns None."""
    from scripts.load_symbols.run import run
    run(mode=1, base_dir=a.base_dir, dataset=a.dataset,
        date_start=a.date_start, date_end=a.date_end)


def _run_news(a) -> None:
    """Fetch the dataset topic's news into the shared cache. Returns None."""
    from scripts.load_news.run import run
    run(base_dir=a.base_dir, dataset=a.dataset,
        start_year=a.start_year, end_year=a.end_year)


def _run_target_feat(a) -> None:
    """Enrich the target series with technical indicators. Returns None."""
    from scripts.symb_feat_gen.run import run
    run(mode=0, base_dir=a.base_dir, dataset=a.dataset)


def _run_covariates_feat(a) -> None:
    """Enrich the covariate series with technical indicators. Returns None."""
    from scripts.symb_feat_gen.run import run
    run(mode=1, base_dir=a.base_dir, dataset=a.dataset)


def _run_news_feat(a) -> None:
    """Run the NLP model over the fetched news. Returns None."""
    from scripts.news_feat_gen.run import run
    run(base_dir=a.base_dir, dataset=a.dataset, run_flat=int(a.run_flat), model=a.model)


def _run_merge(a) -> None:
    """Build the dataset's output parquets. Returns None."""
    from scripts.merge.run import run
    run(base_dir=a.base_dir, dataset=a.dataset,
        cutoff_date=a.cutoff_date, min_start=a.min_start)


# ── step table ────────────────────────────────────────────────────────────────
# (key, label, run, precondition) — precondition is None for steps that only
# download and so depend on nothing local.

STEPS = [
    ("target", "Downloading target OHLCV", _run_target, None),

    ("covariates", "Downloading covariate OHLCVs", _run_covariates, None),

    ("news", "Fetching news", _run_news, None),

    ("target-feat", "Enriching target with technical indicators", _run_target_feat,
     lambda a: _needs_file(target_path(a.base_dir, a.dataset), "downloaded target", "target")),

    ("covariates-feat", "Enriching covariates with technical indicators", _run_covariates_feat,
     lambda a: _needs_dir(raw_dir(a.base_dir, a.dataset, KIND_COVARIATES),
                          "downloaded covariates", "covariates")),

    ("news-feat", "Running NLP on news", _run_news_feat,
     lambda a: (_needs_file(_raw_news_path(a.base_dir, a.dataset), "fetched news", "news")
                if a.run_flat else
                _needs_file(news_flat_path(a.base_dir, a.dataset),
                            "existing flat news (needed with --no-run-flat)", "news-feat"))),

    ("merge", "Building dataset parquets", _run_merge,
     lambda a: (_needs_file(target_path(a.base_dir, a.dataset, enriched=True),
                            "enriched target", "target-feat")
                or _needs_dir(enriched_dir(a.base_dir, a.dataset, KIND_COVARIATES),
                              "enriched covariates", "covariates-feat"))),
]

STEP_KEYS = [key for key, _, _, _ in STEPS]


def select_steps(args) -> list:
    """
    Filter the step table according to ``--only`` / ``--skip``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; ``only`` and ``skip`` are sets of step keys or None.

    Returns
    -------
    list
        The selected ``STEPS`` entries, always in table order — the order they
        are listed in on the command line is ignored, since later steps consume
        earlier ones' outputs.

    Raises
    ------
    SystemExit
        If any named step key is unknown, listing the valid keys.
    """
    requested = args.only or args.skip or set()
    unknown = requested - set(STEP_KEYS)
    if unknown:
        raise SystemExit(
            f"unknown step(s): {sorted(unknown)}\nvalid steps are: {', '.join(STEP_KEYS)}"
        )

    if args.only:
        return [s for s in STEPS if s[0] in args.only]
    if args.skip:
        return [s for s in STEPS if s[0] not in args.skip]
    return list(STEPS)


def parse():
    """
    Build and parse the full-pipeline argument set.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for every step in the table.
    """
    parser = argparse.ArgumentParser(description="Full data-creation pipeline")
    add_base_dir(parser)
    add_dataset(parser)
    add_dates(parser)
    add_news_range(parser)
    add_model(parser)
    add_cutoff(parser)
    add_step_selection(parser, STEP_KEYS)
    parser.add_argument("--min-start", default=None,
                        help="Exclude covariates whose history starts after this date (YYYY-MM-DD)")
    return parser.parse_args()


def run(args) -> None:
    """
    Run the selected pipeline steps for one dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, as produced by :func:`parse`.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If a selected step's inputs are missing, naming the step that produces
        them.
    """
    selected = select_steps(args)
    cfg = dataset_config(args.dataset)
    total = len(selected)

    print(f"dataset : {args.dataset}  (target {cfg['target_ticker']} -> id {cfg['target_id']}, "
          f"{len(cfg['covariates'])} covariates, news topic {cfg['news_topic']!r})")
    print(f"base dir: {args.base_dir}")
    print(f"steps   : {', '.join(key for key, _, _, _ in selected)}")

    for i, (key, label, fn, precondition) in enumerate(selected, 1):
        problem = precondition(args) if precondition else None
        if problem:
            raise SystemExit(f"\ncannot run step {key!r} for dataset {args.dataset!r}: {problem}")
        print(f"\n=== [{i}/{total}] {label}  ({key}) ===")
        fn(args)

    print("\n=== Done ===")


if __name__ == "__main__":
    run(parse())
