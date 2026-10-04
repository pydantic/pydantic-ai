"""Telemetry SDK warnings, such as failed Logfire exports, go to a file instead of over the interactive editor."""

import io
import logging
from collections.abc import Sequence
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

_EXPORTERS = ('logfire', 'opentelemetry.exporter.otlp.proto.http.metric_exporter')


class _Exports(AbstractCapability[None]):
    """`/fail` logs what Logfire's and OpenTelemetry's exporters log when a request to Logfire times out."""

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        def fail(args: list[str]) -> str:
            logging.getLogger(_EXPORTERS[0]).warning('Currently retrying %s failed export(s) (%s bytes)', 1, 955)
            logging.getLogger(_EXPORTERS[1]).warning('Failed to export metrics batch code: None, reason: timed out')
            logging.getLogger(_EXPORTERS[0]).info('Not a problem')
            return 'Failed.'

        return [Command(name='fail', description='Log export failures', handler=fail)]


async def test_export_warnings_go_to_a_file_not_the_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    inputs(monkeypatch, ['/fail', '/exit'])
    output = io.StringIO()
    caplog.set_level(logging.INFO)
    loggers = [logging.getLogger(name) for name in ('logfire', 'opentelemetry')]
    before = [(logger.propagate, list(logger.handlers)) for logger in loggers]
    await chat(
        Agent(TestModel()),
        deps=None,
        plugins=[_Exports()],
        settings=Settings(model='test'),
        console=Console(file=output, width=1000),
        store=SettingsStore(tmp_path / 'config.db'),
    )

    log = tmp_path / 'telemetry.log'
    lines = log.read_text().splitlines()
    assert [line.split(' ', 2)[2] for line in lines] == [
        'WARNING logfire: Currently retrying 1 failed export(s) (955 bytes)',
        'WARNING opentelemetry.exporter.otlp.proto.http.metric_exporter: Failed to export metrics batch code: None, '
        'reason: timed out',
    ]
    # Nothing propagated to the root logger, so `logging.lastResort` had nothing to write to stderr.
    assert not [record for record in caplog.records if record.name.startswith(('logfire', 'opentelemetry'))]
    assert capsys.readouterr().err == ''
    text = output.getvalue()
    assert 'Currently retrying' not in text
    assert text.endswith(
        f'Logfire or OpenTelemetry reported problems, such as failed exports, this session. See {log}.\n'
    )
    # The session leaves logging as it found it.
    assert [(logger.propagate, list(logger.handlers)) for logger in loggers] == before


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
