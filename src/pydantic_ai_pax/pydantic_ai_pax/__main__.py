"""Import-light entry point: animate before loading the agent/UI dependencies."""

import json
import logging
import os
import socket
import sqlite3
import sys
import warnings
from collections.abc import Callable, Generator
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


_WSAECONNRESET = 10054
"""Windows' `winerror` for a connection the peer reset."""


class _ResetTolerantSocket:
    """A socket whose `shutdown` ignores the peer having reset the connection already."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def fileno(self) -> int:
        return self.sock.fileno()

    def close(self) -> None:
        self.sock.close()

    def shutdown(self, how: int) -> None:
        try:
            self.sock.shutdown(how)
        except ConnectionResetError as exc:
            if getattr(exc, 'winerror', None) != _WSAECONNRESET:
                raise


@contextmanager
def quiet_proactor_resets() -> Generator[None]:
    """Let Windows Proactor transports finish teardown when the peer already reset the connection.

    CPython shuts the socket down before closing it (python/cpython#87419). A reset raises there, which skips the
    close and prints a callback traceback over the live display.
    """
    if sys.platform != 'win32':
        yield
        return
    from asyncio import proactor_events

    # Typeshed declares neither the private method nor `_sock`.
    transport_class: type = vars(proactor_events)['_ProactorBasePipeTransport']
    original: Callable[[object, BaseException | None], None] = vars(transport_class)['_call_connection_lost']

    def call_connection_lost(self: object, exc: BaseException | None) -> None:
        state = vars(self)
        sock: object = state['_sock']
        if isinstance(sock, socket.socket):
            state['_sock'] = _ResetTolerantSocket(sock)
        try:
            original(self, exc)
        finally:
            if state['_sock'] is not None:
                state['_sock'] = sock

    setattr(transport_class, '_call_connection_lost', call_connection_lost)
    try:
        yield
    finally:
        setattr(transport_class, '_call_connection_lost', original)


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
    from pydantic_ai_pax.ui.rendering.splash import Splash

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
        from pydantic_ai_pax.cli.cli import run

        with warnings.catch_warnings(), quiet_telemetry_logs(), quiet_proactor_resets():
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
