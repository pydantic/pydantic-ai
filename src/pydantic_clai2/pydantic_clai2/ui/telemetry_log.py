"""Keep telemetry SDK log records out of the interactive terminal.

Logfire and OpenTelemetry report export problems, such as a timed-out request that Logfire retries from disk,
as `logging` warnings. CLAI configures no logging handlers, so Python's last-resort handler would write them to
stderr, over the editor. While the interactive session runs they go to a small rotating file instead.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console

from pydantic_clai2.ui.rendering import theme

LOGGERS = ('logfire', 'opentelemetry')
"""The logger hierarchies whose records go to the file instead of the terminal."""
MAX_BYTES = 1_000_000
"""The file's size before it rotates; one previous file is kept."""


@contextmanager
def telemetry_log(path: Path, *, console: Console) -> Iterator[None]:
    """Send warnings from `LOGGERS` to `path`, and say on exit where to find any this session wrote."""
    # `delay` opens the file on the first record, so a session without problems creates nothing.
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=1, encoding='utf-8', delay=True)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    wrote = False

    def note(record: logging.LogRecord) -> bool:
        nonlocal wrote
        wrote = True
        return True

    handler.addFilter(note)
    loggers = [logging.getLogger(name) for name in LOGGERS]
    propagate = [logger.propagate for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        # Without this a record would still reach the root logger, and `logging.lastResort` when it has no handler.
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
