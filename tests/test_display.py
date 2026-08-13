from __future__ import annotations

import importlib.util
import sys
from io import StringIO
from typing import Any

import pytest

import pydantic_ai._display as _display
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class TTYStream(StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def stderr() -> TTYStream:
    return TTYStream()


@pytest.fixture(autouse=True)
def reset_banner(monkeypatch: pytest.MonkeyPatch):
    # Process-wide state has to be reset so each test starts at its first run.
    _display._banner_displayed = False  # pyright: ignore[reportPrivateUsage]
    _display.BANNER_ENABLED = True
    monkeypatch.delenv('PYDANTIC_AI_NO_BANNER', raising=False)
    monkeypatch.delenv('CI', raising=False)
    yield
    _display._banner_displayed = False  # pyright: ignore[reportPrivateUsage]
    _display.BANNER_ENABLED = True


def display_banner(**overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        'name': 'support_agent',
        'model': 'openai:gpt-5.6-sol',
        'output_type': str,
        'tools': 2,
        'instrumented': False,
    }
    kwargs.update(overrides)
    _display.display_agent_banner(**kwargs)


def test_display_banner_with_logfire(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def find_logfire(name: str) -> object:
        return object()

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setattr(importlib.util, 'find_spec', find_logfire)

    display_banner()

    assert stderr.getvalue() == (
        f'pydantic-ai v{_display.__version__} • Python {_display.platform.python_version()}\n'
        'agent: support_agent • model: openai:gpt-5.6-sol • output: str • tools: 2\n'
        'observability: not configured — set `instrument=True` and run `logfire.configure()` to see\n'
        '  this agent live: every model call, tool call, and cost. Free with a Logfire account —\n'
        '  sign up: `uvx logfire auth` or https://logfire.pydantic.dev\n'
        '  (or use any OpenTelemetry backend)\n'
        '  docs: https://pydantic.dev/docs/ai/logfire/ • hide this banner: PYDANTIC_AI_NO_BANNER=1\n'
    )


def test_display_banner_without_logfire(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def find_logfire(name: str) -> None:
        return None

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.delitem(sys.modules, 'logfire', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', find_logfire)

    display_banner(name=None, output_type=list[str])

    assert stderr.getvalue() == (
        f'pydantic-ai v{_display.__version__} • Python {_display.platform.python_version()}\n'
        'agent: (unnamed) • model: openai:gpt-5.6-sol • output: list[str] • tools: 2\n'
        'observability: not configured — install `logfire` and set `instrument=True`.\n'
        '  Then run `logfire.configure()` to see this agent live:\n'
        '  Every model call, tool call, and cost. Free with a Logfire account —\n'
        '  sign up: `uvx logfire auth` or https://logfire.pydantic.dev\n'
        '  (or use any OpenTelemetry backend)\n'
        '  docs: https://pydantic.dev/docs/ai/logfire/ • hide this banner: PYDANTIC_AI_NO_BANNER=1\n'
    )


def test_display_banner_with_loaded_logfire(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def fail_find_logfire(name: str) -> None:
        pytest.fail('find_spec should not be called')  # pragma: no cover

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setitem(sys.modules, 'logfire', None)
    monkeypatch.setattr(importlib.util, 'find_spec', fail_find_logfire)

    display_banner()

    assert 'set `instrument=True` and run `logfire.configure()`' in stderr.getvalue()


@pytest.mark.parametrize(
    ('condition', 'value'),
    [
        ('PYDANTIC_AI_NO_BANNER', ''),
        ('CI', ''),
        ('instrumented', True),
        ('tty', False),
        ('enabled', False),
    ],
)
def test_display_banner_suppressed(
    condition: str, value: str | bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    kwargs: dict[str, Any] = {}
    if condition in {'PYDANTIC_AI_NO_BANNER', 'CI'}:
        monkeypatch.setenv(condition, str(value))
    elif condition == 'instrumented':
        kwargs['instrumented'] = value
    elif condition == 'tty':
        monkeypatch.setattr(sys.stderr, 'isatty', lambda: value)
    else:
        monkeypatch.setattr(_display, 'BANNER_ENABLED', value)

    display_banner(**kwargs)

    assert capsys.readouterr().err == ''


def test_display_banner_once_per_process(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    display_banner()
    assert stderr.getvalue()

    stderr.seek(0)
    stderr.truncate()
    display_banner(name='second_agent')

    assert stderr.getvalue() == ''


def test_banner_is_shown_by_agent_run(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel(), name='support_agent')

    agent.run_sync('hello')

    output = stderr.getvalue()
    assert 'agent: support_agent • model: test:test • output: str • tools: 0' in output


def test_instrumented_agent_run_is_silent(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel())
    agent.instrument = True

    agent.run_sync('hello')

    assert stderr.getvalue() == ''
