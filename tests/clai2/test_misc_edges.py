"""Storage and public API error boundaries."""

import io
import logging
import sqlite3
import sys
import warnings
from contextlib import closing
from pathlib import Path

import pytest
from opentelemetry.exporter.otlp.proto.http import trace_exporter
from opentelemetry.sdk.trace import export

import pydantic_clai2
import pydantic_clai2.__main__
import pydantic_clai2.cli._cli
from pydantic_clai2.commands import config_completions
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.rendering.splash import Splash


def test_public_errors(tmp_path: Path) -> None:
    with pytest.raises(AttributeError):
        assert pydantic_clai2.missing
    assert callable(pydantic_clai2.__main__.main)
    assert list(config_completions(['set', '']))
    path = tmp_path / 'future.db'
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('PRAGMA user_version = 99')
    with pytest.raises(ValueError, match='Unsupported'):
        SettingsStore(path)


def test_splash_broken_stream_and_replaced_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            raise OSError('closed terminal')

    monkeypatch.setenv('COLUMNS', '80')
    monkeypatch.setenv('LINES', '30')
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('COLORTERM', '16color')
    stream = Terminal()
    monkeypatch.setattr(sys, 'stdout', stream)
    splash = Splash()
    assert '\x1b[35m' in splash.frame(10)
    monkeypatch.setenv('COLORTERM', 'truecolor')
    assert '\x1b[38;2;' in splash.frame(10)
    splash.start()
    monkeypatch.setattr(sys, 'stdout', io.StringIO())
    monkeypatch.setattr(sys, 'stderr', io.StringIO())
    with pytest.raises(OSError):
        splash.stop()


class LibraryWarning(Warning):
    """Like Logfire's `InspectArgumentsFailedWarning`: a warning that is not a `UserWarning`."""


@pytest.mark.parametrize(('warnoptions', 'shown'), [([], []), (['default'], ['for developers', 'from a library'])])
def test_entry_point_quiets_warnings_unless_requested(
    monkeypatch: pytest.MonkeyPatch, warnoptions: list[str], shown: list[str]
) -> None:
    def run(*, splash: Splash | None = None) -> None:
        warnings.warn('for developers', UserWarning)
        warnings.warn('from a library', LibraryWarning)

    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')
    monkeypatch.setattr(sys, 'argv', ['clai2', 'config'])
    monkeypatch.setattr(sys, 'warnoptions', warnoptions)
    monkeypatch.setattr(pydantic_clai2.cli._cli, 'run', run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        filters = list(warnings.filters)
        pydantic_clai2.__main__.main()
        assert warnings.filters == filters
    assert [str(warning.message) for warning in caught] == shown


def test_entry_point_keeps_telemetry_export_logs_off_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logfire export retries and timeouts, logged from background threads, painted over the live panel."""
    stderr = io.StringIO()
    received: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            received.append(record.getMessage())

    def log_exports(configured: logging.Handler | None) -> None:
        if configured:
            logging.root.addHandler(configured)
        try:
            logging.getLogger('logfire').warning('Currently retrying %s failed export(s) (%s bytes)', 1, 1773)
            logging.getLogger(trace_exporter.__name__).error(
                'Failed to export span batch code: %s, reason: %s', None, 'Read timed out. (read timeout=10)'
            )
            logging.getLogger(export.__name__).error('Exception while exporting Span.')
        finally:
            if configured:
                logging.root.removeHandler(configured)

    def run(*, splash: Splash | None = None) -> None:
        log_exports(None)
        log_exports(Collect())
        logging.getLogger('pydantic_clai2.example').warning('not telemetry')

    monkeypatch.setenv('PYDANTIC_AI_NO_BANNER', '1')
    monkeypatch.setattr(sys, 'argv', ['clai2', 'config'])
    monkeypatch.setattr(sys, 'stderr', stderr)
    # pytest's log capture sits on the root logger; without it, `logging` falls back to stderr as in the CLI.
    monkeypatch.setattr(logging.root, 'handlers', [])
    monkeypatch.setattr(pydantic_clai2.cli._cli, 'run', run)
    loggers = [logging.getLogger(name) for name in pydantic_clai2.__main__.TELEMETRY_LOGGERS]
    before = [list(logger.handlers) for logger in loggers]
    pydantic_clai2.__main__.main()
    assert stderr.getvalue() == 'not telemetry\n'
    # Records still reach handlers that are configured, so logs and observability keep them.
    assert received == [
        'Currently retrying 1 failed export(s) (1773 bytes)',
        'Failed to export span batch code: None, reason: Read timed out. (read timeout=10)',
        'Exception while exporting Span.',
    ]
    assert [logger.handlers for logger in loggers] == before
