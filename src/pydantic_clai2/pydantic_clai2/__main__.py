"""Import-light entry point: animate before loading the agent/UI dependencies."""

import json
import logging
import os
import sqlite3
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

TELEMETRY_LOGGERS = ('logfire', 'opentelemetry')
"""Loggers of the telemetry SDKs, whose export retries, timeouts, and batch failures run on background threads."""


@contextmanager
def quiet_telemetry_logs() -> Generator[None]:
    """Keep telemetry SDK log records off stderr, where they would tear through the live display.

    Without a handler anywhere on a record's path, `logging` falls back to printing it on stderr. A `NullHandler`
    on these loggers stops only that fallback: handlers a plugin or host configures still receive every record.
    """
    handler = logging.NullHandler()
    loggers = [logging.getLogger(name) for name in TELEMETRY_LOGGERS]
    for logger in loggers:
        logger.addHandler(handler)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeHandler(handler)


def main() -> None:
    """Load the project environment, then cover heavyweight imports with the splash."""
    dotenv_path = '.env'
    try:
        dotenv_path = find_dotenv(usecwd=True)
        # Named pipes must not block automatic startup.
        if Path(dotenv_path).is_file():
            load_dotenv(dotenv_path)
    except (OSError, UnicodeDecodeError) as exc:
        print(f'Ignoring `.env` at {dotenv_path!r}: {exc}', file=sys.stderr)

    # Splash imports configuration, which must see the loaded environment.
    from pydantic_clai2.ui.rendering.splash import Splash

    os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
    enabled = len(sys.argv) == 1 and not os.getenv('CLAI_NO_SPLASH')
    database = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2/config.db'
    if enabled and database.exists():
        try:
            connection = sqlite3.connect(f'{database.as_uri()}?mode=ro', uri=True)
            try:
                row = connection.execute("SELECT value_json FROM settings WHERE key = 'display.splash'").fetchone()
                enabled = row is None or json.loads(row[0]) is True
            finally:
                connection.close()
        except (sqlite3.Error, ValueError):
            enabled = False
    splash = Splash(enabled=bool(enabled))
    splash.start()
    try:
        from pydantic_clai2.cli._cli import run

        with warnings.catch_warnings(), quiet_telemetry_logs():
            if not sys.warnoptions:
                # Library warnings are advice for the developer who wired the agent, not the person at the
                # prompt, and stderr output tears through the live display. Not every one is a `UserWarning`
                # (Logfire's `InspectArgumentsFailedWarning` subclasses `Warning`). `-W` or `PYTHONWARNINGS`
                # restores them.
                warnings.simplefilter('ignore', Warning)
            run(splash=splash)
    finally:
        splash.stop()


if __name__ == '__main__':
    main()
