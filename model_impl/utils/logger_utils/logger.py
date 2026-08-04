"""
Shared logging setup.

Every module reports through a logger obtained here (`get_logger(__name__)`)
rather than print(), so format and level stay uniform everywhere. The handler
writes to stderr, which artifacts_logs.run_log deliberately does not tee (tqdm
redraws its bar there) — run.log captures logger output separately by adding
a file handler once the output directory exists, via attach_file_handler.
"""

import logging
from pathlib import Path

from model_impl.consts import RUN_LOG_FILE

_FORMAT = "[%(asctime)s]  %(levelname)s %(name)s: %(message)s"

_ROOT = logging.getLogger("model_impl")
_ROOT.setLevel(logging.INFO)
# Don't bubble up to the root logger: dependencies (transformers/mlflow) call
# logging.basicConfig(), which installs a root handler that would re-emit every
# record in the default "INFO:name:message" format — the duplicate log lines.
_ROOT.propagate = False
if not _ROOT.handlers:
    _stream = logging.StreamHandler()
    _stream.setFormatter(logging.Formatter(_FORMAT))
    _ROOT.addHandler(_stream)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for `name` (call with `__name__`).

    Callers already pass a fully-qualified "model_impl.…" module name, so this
    is a plain logging.getLogger — not _ROOT.getChild(name), which would
    prepend "model_impl." a second time. The returned logger has no handler of
    its own; it propagates up the dotted hierarchy to the "model_impl" logger
    configured above, so every module gets consistent formatting for free.
    """
    return logging.getLogger(name)


def attach_file_handler(outdir: Path) -> None:
    """
    Add a file handler writing to <outdir>/run.log, in addition to the
    existing stream handler. Called once by artifacts_logs.run_log.install
    after the output directory exists.
    """
    handler = logging.FileHandler(outdir / RUN_LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    _ROOT.addHandler(handler)
