from __future__ import annotations

import importlib.util
import sys
from importlib import metadata
from io import StringIO
from typing import Any

import pytest
from inline_snapshot import snapshot

import pydantic_ai._display as _display
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel

_find_spec = importlib.util.find_spec


class TTYStream(StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def stderr() -> TTYStream:
    return TTYStream()


@pytest.fixture(autouse=True)
def reset_banner(monkeypatch: pytest.MonkeyPatch):
    def find_spec_without_harness(name: str) -> object | None:
        return None if name == 'pydantic_ai_harness' else _find_spec(name)

    # Process-wide state has to be reset so each test starts at its first run.
    _display._banner_displayed = False  # pyright: ignore[reportPrivateUsage]
    _display.BANNER_ENABLED = True
    monkeypatch.delenv('PYDANTIC_AI_NO_BANNER', raising=False)
    monkeypatch.delenv('CI', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec_without_harness)
    yield
    _display._banner_displayed = False  # pyright: ignore[reportPrivateUsage]
    _display.BANNER_ENABLED = True


def render(**overrides: Any) -> str:
    """Render with a fixed heading, so the layout can be asserted without the versions in play."""
    kwargs: dict[str, Any] = {
        'heading': 'HEADING',
        'name': 'support_agent',
        'model': 'openai:gpt-5.6-sol',
        'output_type': str,
        'tools': 2,
        'capabilities': 0,
    }
    kwargs.update(overrides)
    return _display.render_banner(**kwargs)


def display_banner(**overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        'name': 'support_agent',
        'model': 'openai:gpt-5.6-sol',
        'output_type': str,
        'tools': 2,
        'capabilities': 0,
        'instrumented': False,
    }
    kwargs.update(overrides)
    _display.display_agent_banner(**kwargs)


def test_render_banner_with_logfire():
    assert render() == snapshot("""\
                      HEADING
         / \\          agent: support_agent • model: openai:gpt-5.6-sol • tools: 2
       /     \\        observability: not configured — set `instrument=True` and run
     /____.____\\        `logfire.configure()` to see this agent live: every model call, tool call,
   /      |      \\      and cost. Free with a Logfire account — sign up: `uvx logfire auth` or
 /        |        \\    https://logfire.pydantic.dev (or use any OpenTelemetry backend)
  ·.______|______.·     docs: https://pydantic.dev/docs/ai/logfire/
                        hide this banner: PYDANTIC_AI_NO_BANNER=1\
""")


def find_nothing(name: str) -> None:
    return None


def find_anything(name: str) -> object:
    return object()


def test_render_banner_without_logfire(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, 'logfire', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', find_nothing)

    # An agent with no name of its own drops the `agent:` segment rather than inventing a name.
    assert render(name=None, output_type=list[str], capabilities=3) == snapshot("""\
                      HEADING
         / \\          model: openai:gpt-5.6-sol • output: list[str] • tools: 2 • capabilities: 3
       /     \\        observability: not configured — install `logfire`, set `instrument=True`, and
     /____.____\\        run `logfire.configure()` to see this agent live: every model call, tool
   /      |      \\      call, and cost. Free with a Logfire account — sign up: `uvx logfire auth` or
 /        |        \\    https://logfire.pydantic.dev (or use any OpenTelemetry backend)
  ·.______|______.·     docs: https://pydantic.dev/docs/ai/logfire/
                        hide this banner: PYDANTIC_AI_NO_BANNER=1\
""")


def test_render_banner_without_observability():
    """What `clai` shows: the same banner, minus advice it has already acted on."""
    assert render(observability=False) == snapshot("""\
         / \\
       /     \\
     /____.____\\      HEADING
   /      |      \\    agent: support_agent • model: openai:gpt-5.6-sol • tools: 2
 /        |        \\
  ·.______|______.·\
""")


def test_render_banner_with_plain_output():
    assert 'output: int • tools: 2 • capabilities: 1' in render(output_type=int, capabilities=1)


def test_display_banner_writes_versions(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)

    display_banner()

    assert stderr.getvalue().split('\n')[0].strip() == snapshot(
        f'pydantic-ai v{_display.__version__} • Python {_display.platform.python_version()}'
    )


def test_display_banner_with_harness(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def distribution_version(distribution: str) -> str:
        return '1.2.3'

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setattr(importlib.util, 'find_spec', find_anything)
    monkeypatch.setattr(metadata, 'version', distribution_version)

    display_banner()

    assert (
        f'pydantic-ai v{_display.__version__} • pydantic-ai-harness v1.2.3 • Python '
        f'{_display.platform.python_version()}'
    ) in stderr.getvalue()


def test_display_banner_with_harness_module_but_no_distribution(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def missing_distribution(distribution: str) -> str:
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setattr(importlib.util, 'find_spec', find_anything)
    monkeypatch.setattr(metadata, 'version', missing_distribution)

    display_banner()

    assert 'pydantic-ai-harness' not in stderr.getvalue()


def test_display_banner_with_loaded_logfire(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def fail_find_logfire(name: str) -> None:
        if name == 'logfire':
            pytest.fail('find_spec should not be called')  # pragma: no cover
        return None

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setitem(sys.modules, 'logfire', None)
    monkeypatch.setattr(importlib.util, 'find_spec', fail_find_logfire)

    display_banner()

    # An already-imported `logfire` takes the branch that doesn't tell the user to install it.
    assert 'install `logfire`' not in stderr.getvalue()
    assert 'set `instrument=True` and run' in stderr.getvalue()


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


def test_claimed_banner_is_not_displayed(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """How `clai` stops a run from printing a second banner over the answer to the first prompt."""
    monkeypatch.setattr(sys, 'stderr', stderr)
    assert _display.claim_banner() is True
    assert _display.claim_banner() is False

    display_banner()

    assert stderr.getvalue() == ''


def test_banner_is_shown_by_agent_run(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel(), name='support_agent')

    agent.run_sync('hello')

    assert 'agent: support_agent • model: test:test • tools: 0' in stderr.getvalue()


def test_registered_capabilities_are_counted(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """Only what the user registered — every agent gets infrastructure capabilities injected."""

    class Coder(AbstractCapability[object]):
        pass

    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel(), capabilities=[Coder()])

    agent.run_sync('hello')

    assert 'tools: 0 • capabilities: 1' in stderr.getvalue()


def test_run_capabilities_are_counted(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    class Coder(AbstractCapability[object]):
        pass

    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel(), capabilities=[Coder()])

    agent.run_sync('hello', capabilities=[Coder()])

    assert 'tools: 0 • capabilities: 2' in stderr.getvalue()


def test_agent_without_capabilities_omits_them(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)

    Agent(TestModel()).run_sync('hello')

    assert 'capabilities' not in stderr.getvalue().split('observability')[0]


def test_instrumented_agent_run_is_silent(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel())
    agent.instrument = True

    agent.run_sync('hello')

    assert stderr.getvalue() == ''
