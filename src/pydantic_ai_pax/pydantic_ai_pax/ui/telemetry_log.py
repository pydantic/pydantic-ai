"""Keep telemetry SDK log records, and on a terminal everything written to standard error, out of the session's display.

Logfire and OpenTelemetry report export problems, such as a request to Logfire timing out, through `logging`.
With no handler configured, Python's last-resort handler writes them to stderr, over the editor. While the
interactive session runs they go to a small rotating file instead, and on a POSIX terminal so does standard error.
"""

import logging
import os
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console

from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.stderr_relay import stderr_to_file

_LOGGERS = ('logfire', 'opentelemetry')
_MAX_BYTES = 1_000_000


class _LogFile(RotatingFileHandler):
    """A rotating log file that remembers whether it wrote anything and never reports its own failures."""

    wrote = False
    _failed = False

    def emit(self, record: logging.LogRecord) -> None:
        self._failed = False
        super().emit(record)
        self.wrote = self.wrote or not self._failed

    def _open(self) -> TextIOWrapper:
        # Exporter errors can quote endpoint details, so the file (and each rotation's new one) is owner-only.
        os.close(os.open(self.baseFilename, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600))
        os.chmod(self.baseFilename, 0o600)
        return super()._open()

    def handleError(self, record: logging.LogRecord) -> None:
        # The default prints a traceback to stderr, over the editor; losing the record is the lesser harm.
        self._failed = True


def _writes(logger: logging.Logger) -> bool:
    """Whether the logger has a handler that writes records out.

    A `NullHandler`, such as the one the `pax` entry point adds for the whole process, only discards them.
    """
    return any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def _shown(logger: logging.Logger | None) -> bool:
    """Whether the logger, or an ancestor its records propagate to, writes records out."""
    while logger is not None:
        if _writes(logger):
            return True
        logger = logger.parent if logger.propagate else None
    return False


def _unhandled_below(record: logging.LogRecord) -> bool:
    """Whether no logger between the record's own and the top-level one it propagated to writes it out."""
    parts = record.name.split('.')
    loggers = logging.root.manager.loggerDict
    return not any(
        isinstance(logger := loggers.get('.'.join(parts[:depth])), logging.Logger) and _writes(logger)
        for depth in range(2, len(parts) + 1)
    )


@contextmanager
def telemetry_log(path: Path, *, console: Console) -> Generator[None]:
    """Send warnings and errors from the Logfire and OpenTelemetry loggers to `path`, and name it on exit if written.

    A logger that already reaches a handler, one the embedding application configured on it, an ancestor, or a
    descendant, is left alone: only records no handler would write out are redirected, whether they would fall
    through to `logging.lastResort` or be discarded by a `NullHandler`. When `console` is the terminal, everything
    written to standard error goes to `path` too (`stderr_to_file`), including what such a handler writes there.
    """
    # `delay` opens the file on the first record, so a session without problems creates nothing.
    handler = _LogFile(path, maxBytes=_MAX_BYTES, backupCount=1, encoding='utf-8', delay=True)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    handler.addFilter(_unhandled_below)
    # `lastResort` only fires when no handler at all is on a record's path, so a handler on these loggers is enough;
    # records still propagate, so a handler the application adds to the root later sees them too.
    loggers = [logger for name in _LOGGERS if not _shown(logger := logging.getLogger(name))]
    for logger in loggers:
        logger.addHandler(handler)
    stderr_written: Callable[[], bool] = _nothing
    try:
        with stderr_to_file(path) if console.is_terminal else nullcontext(_nothing) as stderr_written:
            yield
    finally:
        for logger in loggers:
            logger.removeHandler(handler)
        handler.close()
        if stderr_written():
            notice = f'Warnings and errors from this session are in {path}.'
        elif handler.wrote:
            notice = f'Logfire or OpenTelemetry reported problems, such as failed exports. See {path}.'
        else:
            notice = None
        if notice is not None:
            console.print(notice, style=theme.color(theme.MUTED), markup=False)


def _nothing() -> bool:
    return False
