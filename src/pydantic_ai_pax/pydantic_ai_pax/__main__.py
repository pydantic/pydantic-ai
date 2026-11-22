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

LAZY_IMPORTERS = frozenset({'pydantic_ai', 'pydantic_ai_harness', 'pydantic_ai_pax', 'pydantic_graph'})
"""Packages whose module-level imports Python 3.15 makes lazy for a launch: this repository's, not a dependency's."""
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
    """Load the project environment, then paint the last launch's first frame or cover heavyweight imports with the splash."""
    dotenv_path = '.env'
    try:
        dotenv_path = find_dotenv(usecwd=True)
        # Named pipes must not block automatic startup.
        if Path(dotenv_path).is_file():
            load_dotenv(dotenv_path)
    except (OSError, UnicodeDecodeError) as exc:
        print(f'Ignoring `.env` at {dotenv_path!r}: {exc}', file=sys.stderr)

    from pydantic_ai_pax.ui import first_frame

    database = Path(os.getenv('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'pydantic-clai2/config.db'
    bare = len(sys.argv) == 1
    if bare:
        first_frame.current.paint(database.with_name('first-frame'))
    try:
        with _lazy_imports():
            _main(database, animate=bare and not first_frame.current.holding)
    finally:
        first_frame.current.close()


@contextmanager
def _lazy_imports() -> Generator[None]:
    """On Python 3.15, make module-level imports in `LAZY_IMPORTERS` bind on first use while the CLI runs.

    Startup then skips the modules the prompt never needs. Every other package imports as usual: some import
    modules only for their side effects. Only in a fresh process: one that imported Pydantic AI already, such as a
    test run, keeps its imports as they are.
    """
    if sys.version_info < (3, 15) or 'pydantic_ai' in sys.modules:
        yield
        return
    # Typeshed has no stubs for PEP 810's functions yet.
    sys.set_lazy_imports_filter(_lazy_import)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    sys.set_lazy_imports('all')  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    try:
        yield
    finally:
        sys.set_lazy_imports('normal')  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
        sys.set_lazy_imports_filter(None)  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]


def _lazy_import(importer: str, name: str, fromlist: tuple[str, ...] | None) -> bool:
    """Whether an import may bind on first use: only in our own packages, never in a dependency.

    `from .name import name` stays eager: a lazy `name` would be replaced by the submodule `name` once that loads.
    """
    return importer.partition('.')[0] in LAZY_IMPORTERS and name.rpartition('.')[2] not in (fromlist or ())


def _main(database: Path, *, animate: bool) -> None:
    """Cover heavyweight imports with the splash when `animate`, then run the CLI."""
    # Splash imports configuration, which must see the loaded environment.
    from pydantic_ai_pax.ui.rendering.splash import Splash

    os.environ['PYDANTIC_AI_NO_BANNER'] = '1'
    enabled = animate and not os.getenv('CLAI_NO_SPLASH')
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
        from pydantic_ai_pax.cli.cli import run

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
