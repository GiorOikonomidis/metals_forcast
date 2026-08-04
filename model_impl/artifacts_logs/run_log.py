"""
Run log: persist the run's console report into <outdir>/run.log so it
survives after the terminal closes.

Two sources feed it:
- every model_impl logger (utils.logger_utils.get_logger) gets a second file
  handler pointed at run.log — this is where window diagnostics, epoch lines
  and the final metrics banner actually land;
- stdout is additionally teed, as a catch-all for anything third-party prints
  directly rather than through logging.

stderr is deliberately left out of the tee — tqdm redraws its progress bar
there and would flood the file with carriage-return frames. Installed once
per process by main, right after the output directory exists; anything
logged or printed before that (e.g. data loading diagnostics) only reaches
the console.
"""

import sys
from pathlib import Path

from model_impl.consts import RUN_LOG_FILE
from model_impl.utils.logger_utils.logger import attach_file_handler


class _Tee:
    """Duplicate every write to the original stream and the log file."""

    def __init__(self, stream, fh) -> None:
        self._stream = stream
        self._fh = fh

    def write(self, s: str):
        n = self._stream.write(s)
        self._fh.write(s)
        return n

    def flush(self) -> None:
        self._stream.flush()
        self._fh.flush()

    def __getattr__(self, name):
        # anything else (encoding, isatty, ...) is answered by the real stream
        return getattr(self._stream, name)


_installed = False


def install(outdir: Path) -> None:
    """
    Attach a file handler to every model_impl logger and tee stdout, both
    into <outdir>/run.log. Idempotent.
    """
    global _installed
    if _installed:
        return
    attach_file_handler(outdir)
    fh = open(outdir / RUN_LOG_FILE, "a", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, fh)
    _installed = True
