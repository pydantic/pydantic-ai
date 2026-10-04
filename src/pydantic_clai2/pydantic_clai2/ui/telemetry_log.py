"""Keep telemetry SDK log records out of the interactive terminal.

Logfire and OpenTelemetry report export problems, such as a request to Logfire timing out, through `logging`.
With no handler configured, Python's last-resort handler writes them to stderr, over the editor. While the
interactive session runs they go to a small rotating file instead.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console

from pydantic_clai2.ui.rendering import theme

_LOGGERS = ('logfire', 'opentelemetry')
_MAX_BYTES = 1_000_000


@contextmanager
def telemetry_log(path: Path, *, console: Console) -> Iterator[None]:
    """Send warnings and errors from the Logfire and OpenTelemetry loggers to `path`, and name it on exit if used.

    A logger that already reaches a handler, one the embedding application configured, is left alone: only
    records that would otherwise fall through to `logging.lastResort` are redirected.
    """
    # `delay` opens the file on the first record, so a session without problems creates nothing.
    handler = RotatingFileHandler(path, maxBytes=_MAX_BYTES, backupCount=1, encoding='utf-8', delay=True)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    wrote = False

    def note(record: logging.LogRecord) -> bool:
        nonlocal wrote
        wrote = True
        return True

    handler.addFilter(note)
    loggers = [logger for name in _LOGGERS if not (logger := logging.getLogger(name)).hasHandlers()]
    propagate = [logger.propagate for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        # A record that still propagated would find no handler and reach `logging.lastResort` after all.
        logger.propagate = False
    try:
        yield
    finally:
        for logger, previous in zip(loggers, propagate):
            logger.removeHandler(handler)
            logger.propagate = previous
        handler.close()
        if wrote:
            console.print(
                f'Logfire or OpenTelemetry reported problems, such as failed exports, this session. See {path}.',
                style=theme.color(theme.MUTED),
                markup=False,
            )
