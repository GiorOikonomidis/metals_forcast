"""
Shared argparse helpers.

Every entry point builds its parser from these, so a flag means the same thing
and carries the same default everywhere. Validation of values against the
dataset registry happens in scripts/paths.py; this module only declares flags.
"""

import argparse
from pathlib import Path

from constants import DATASETS


def resolve_path(p: str) -> str:
    """
    Normalise a path argument to an absolute path.

    Parameters
    ----------
    p : str
        Path as typed on the command line, possibly relative.

    Returns
    -------
    str
        The absolute form, so a step's output location does not depend on the
        working directory it was launched from.
    """
    return str(Path(p).resolve())


def add_base_dir(parser: argparse.ArgumentParser) -> None:
    """
    Add ``--base-dir``: the one root every step reads from and writes to.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.

    Returns
    -------
    None
    """
    parser.add_argument("--base-dir", required=True, type=resolve_path,
                        help="Root directory holding every dataset tree and the shared news cache")


def add_dataset(parser: argparse.ArgumentParser) -> None:
    """
    Add ``--dataset``: which dataset to build.

    Replaces the former ``--index`` / ``--target`` / ``--id`` trio. Those were
    three flags naming one concept, and because each reached only some of the
    seven steps they could silently disagree — a run could download copper and
    label the output as the Nasdaq. One key into ``constants.DATASETS`` now
    determines the target ticker, the output id, the covariate set and the news
    topic together.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.

    Returns
    -------
    None
    """
    parser.add_argument("--dataset", required=True, choices=list(DATASETS),
                        help="Which dataset to build (see DATASETS in constants.py)")


def add_dates(parser: argparse.ArgumentParser) -> None:
    """
    Add the download date-range arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    None
    """
    parser.add_argument("--date-start", default="2007-01-03", help="Download start date (YYYY-MM-DD)")
    parser.add_argument("--date-end",   default=None,         help="Download end date (YYYY-MM-DD), default: today")


def add_news_range(parser: argparse.ArgumentParser) -> None:
    """
    Add the NYT archive year-range arguments.

    The news *topic* is no longer a flag — it comes from the dataset registry,
    so it cannot disagree with the dataset being built.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    None
    """
    parser.add_argument("--start-year", type=int, default=2007, help="First year of NYT archive to fetch")
    parser.add_argument("--end-year",   type=int, default=None,
                        help="Last year to fetch (default: last fully completed calendar month)")


def add_model(parser: argparse.ArgumentParser) -> None:
    """
    Add the NLP model selection arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.

    Returns
    -------
    None
    """
    parser.add_argument("--model", default="finbert", choices=["finbert", "financialbert", "minilm"],
                        help="NLP model to use (see MODELS in news_feat_gen.py)")
    parser.add_argument("--run-flat", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-run flat per-article NLP analysis (use --no-run-flat to skip)")


def add_cutoff(parser: argparse.ArgumentParser) -> None:
    """
    Add ``--cutoff-date``: an optional upper bound on the built date range.

    Defaults to no cutoff. This was previously a hardcoded module constant,
    which silently truncated every dataset at a date that quietly went stale.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the argument to.

    Returns
    -------
    None
    """
    parser.add_argument("--cutoff-date", default=None,
                        help="Drop output dates on/after this date (YYYY-MM-DD); default: no cutoff")


def add_step_selection(parser: argparse.ArgumentParser, step_keys: list[str]) -> None:
    """
    Add the mutually exclusive ``--only`` / ``--skip`` step filters.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add the arguments to.
    step_keys : list[str]
        Valid step keys, listed in the help text. Membership is checked by the
        caller, which owns the step table.

    Returns
    -------
    None
        Both flags parse to a ``set[str]``; at most one can be given.
    """
    keys = ", ".join(step_keys)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only", type=_comma_set, default=None,
                       help=f"Run only these comma-separated steps ({keys})")
    group.add_argument("--skip", type=_comma_set, default=None,
                       help=f"Run every step except these ({keys})")


def _comma_set(value: str) -> set[str]:
    """
    Parse a comma-separated flag value into a set of step keys.

    Parameters
    ----------
    value : str
        Raw flag value, e.g. ``"news, merge"``.

    Returns
    -------
    set[str]
        Non-empty, whitespace-stripped entries.
    """
    return {part.strip() for part in value.split(",") if part.strip()}
