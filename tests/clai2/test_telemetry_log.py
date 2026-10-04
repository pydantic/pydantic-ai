"""Telemetry SDK warnings, such as failed Logfire exports, go to a file instead of over the interactive editor."""

import io
import logging
from collections.abc import Iterator, Sequence
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
_NOTICE = 'Logfire or OpenTelemetry reported problems, such as failed exports, this session. See {}.\n'


class _Exports(AbstractCapability[None]):
    """`/fail` logs what Logfire's and OpenTelemetry's exporters log when a request to Logfire times out."""

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        def fail(args: list[str]) -> str:
            logging.getLogger('logfire').warning('Currently retrying %s failed export(s) (%s bytes)', 1, 955)
            logging.getLogger(_METRICS).error('Failed to export metrics batch code: %s, reason: %s', None, 'timed out')
            logging.getLogger('logfire').info('Not a problem')
            return 'Failed.'

        return [Command(name='fail', description='Log export failures', handler=fail)]


@pytest.fixture
def unconfigured_logging() -> Iterator[None]:
    """Logging as the CLI leaves it, with no handler: pytest's own root handlers would hide `logging.lastResort`."""
    root = logging.getLogger()
    handlers = root.handlers[:]
    for handler in handlers:
        root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in handlers:
            root.addHandler(handler)


async def _chat_failing_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    inputs(monkeypatch, ['/fail', '/exit'])
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


@pytest.mark.usefixtures('unconfigured_logging')
async def test_export_problems_go_to_a_file_not_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loggers = [logging.getLogger(name) for name in ('logfire', 'opentelemetry')]
    before = [(logger.propagate, list(logger.handlers)) for logger in loggers]

    text = await _chat_failing_exports(tmp_path, monkeypatch)

    log = tmp_path / 'telemetry.log'
    assert [line.split(' ', 2)[2] for line in log.read_text().splitlines()] == [
        'WARNING logfire: Currently retrying 1 failed export(s) (955 bytes)',
        f'ERROR {_METRICS}: Failed to export metrics batch code: None, reason: timed out',
    ]
    # Nothing fell through to `logging.lastResort`, which writes to stderr, over the editor.
    assert capsys.readouterr().err == ''
    assert 'Currently retrying' not in text
    assert text.endswith(_NOTICE.format(log))
    # The session leaves logging as it found it, so later records reach stderr again.
    assert [(logger.propagate, list(logger.handlers)) for logger in loggers] == before
    logging.getLogger('logfire').warning('After the session')
    assert capsys.readouterr().err == 'After the session\n'


@pytest.mark.usefixtures('unconfigured_logging')
async def test_handlers_the_application_configured_still_get_their_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An application embedding `chat()` that handles `logfire` itself keeps those records; the rest are redirected."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(logging.getLogger('logfire'), 'handlers', [handler])

    text = await _chat_failing_exports(tmp_path, monkeypatch)

    assert stream.getvalue() == 'Currently retrying 1 failed export(s) (955 bytes)\n'
    log = tmp_path / 'telemetry.log'
    assert [line.split(' ', 2)[2] for line in log.read_text().splitlines()] == [
        f'ERROR {_METRICS}: Failed to export metrics batch code: None, reason: timed out',
    ]
    assert text.endswith(_NOTICE.format(log))


@pytest.mark.usefixtures('unconfigured_logging')
async def test_a_session_without_problems_creates_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/exit'])
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test'),
        console=Console(file=output, width=1000),
        store=SettingsStore(tmp_path / 'config.db'),
    )
    assert not (tmp_path / 'telemetry.log').exists()
    assert 'Logfire or OpenTelemetry' not in output.getvalue()
