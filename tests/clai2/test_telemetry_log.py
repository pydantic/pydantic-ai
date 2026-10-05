"""Telemetry SDK warnings, such as failed Logfire exports, go to a file instead of over the interactive editor."""

import io
import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from tests.clai2.test_app_edges import inputs

_METRICS = 'opentelemetry.exporter.otlp.proto.http.metric_exporter'
_NOTICE = 'Logfire or OpenTelemetry reported problems, such as failed exports. See {}.\n'


class _Exports(AbstractCapability[None]):
    """`/fail` logs what Logfire's and OpenTelemetry's exporters log when a request to Logfire times out."""

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        def fail(args: list[str]) -> str:
            logging.getLogger('logfire').warning('Currently retrying %s failed export(s) (%s bytes)', 1, 955)
            logging.getLogger(_METRICS).error('Failed to export metrics batch code: %s, reason: %s', None, 'timed out')
            logging.getLogger('logfire').info('Not a problem')
            return 'Failed.'

        return [Command(name='fail', description='Log export failures', handler=fail)]


@contextmanager
def unconfigured_logging() -> Generator[None]:
    """Logging as the CLI leaves it, with no handler, inside the test body.

    pytest's own root handlers would hide `logging.lastResort`, and it adds them again for the call phase, so a
    fixture cannot remove them; restoring the same list before the call phase ends lets pytest remove its own.
    """
    root = logging.getLogger()
    handlers = root.handlers
    root.handlers = []
    try:
        yield
    finally:
        root.handlers = handlers


async def _chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands: list[str]) -> str:
    inputs(monkeypatch, [*commands, '/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        plugins=[_Exports()],
        settings=Settings(model='test'),
        console=Console(file=output, width=1000),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    return output.getvalue()


async def test_export_problems_go_to_a_file_not_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loggers = [logging.getLogger(name) for name in ('logfire', 'opentelemetry')]
    before = [(logger.propagate, list(logger.handlers)) for logger in loggers]

    # Even with INFO enabled on the logger, only warnings and errors reach the file, as only they reach `lastResort`.
    logging.getLogger('logfire').setLevel(logging.INFO)
    try:
        with unconfigured_logging():
            text = await _chat(tmp_path, monkeypatch, ['/fail'])
    finally:
        logging.getLogger('logfire').setLevel(logging.NOTSET)
    with unconfigured_logging():
        # Nothing fell through to `logging.lastResort`, which writes to stderr, over the editor.
        assert capsys.readouterr().err == ''
        # The session leaves logging as it found it, so later records reach stderr again.
        assert [(logger.propagate, list(logger.handlers)) for logger in loggers] == before
        logging.getLogger('logfire').warning('After the session')
        assert capsys.readouterr().err == 'After the session\n'

    log = tmp_path / 'telemetry.log'
    assert [line.split(' ', 2)[2] for line in log.read_text().splitlines()] == [
        'WARNING logfire: Currently retrying 1 failed export(s) (955 bytes)',
        f'ERROR {_METRICS}: Failed to export metrics batch code: None, reason: timed out',
    ]
    assert 'Currently retrying' not in text
    assert text.endswith(_NOTICE.format(log))


async def test_handlers_the_application_configured_still_get_their_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An application embedding `chat()` that handles `logfire` itself keeps those records; the rest are redirected."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logging.getLogger('logfire'), 'handlers', [handler])

    with unconfigured_logging():
        text = await _chat(tmp_path, monkeypatch, ['/fail'])

    assert stream.getvalue() == 'Currently retrying 1 failed export(s) (955 bytes)\n'
    log = tmp_path / 'telemetry.log'
    assert [line.split(' ', 2)[2] for line in log.read_text().splitlines()] == [
        f'ERROR {_METRICS}: Failed to export metrics batch code: None, reason: timed out',
    ]
    assert text.endswith(_NOTICE.format(log))


async def test_a_session_without_problems_creates_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with unconfigured_logging():
        text = await _chat(tmp_path, monkeypatch, [])
    assert not (tmp_path / 'telemetry.log').exists()
    assert 'Logfire or OpenTelemetry' not in text


async def test_an_unwritable_file_stays_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record the file cannot take is dropped: a logging traceback on stderr would land over the editor."""
    (tmp_path / 'telemetry.log').mkdir()
    with unconfigured_logging():
        text = await _chat(tmp_path, monkeypatch, ['/fail'])
        assert capsys.readouterr().err == ''
    assert 'Logfire or OpenTelemetry' not in text
