from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from importlib import metadata
from io import StringIO
from typing import Any

import pytest
from pydantic import BaseModel

import pydantic_ai._display as _display
from pydantic_ai import Agent, ModelMessage, ModelRequest, UserPromptPart, __version__
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset

from ._inline_snapshot import snapshot
from .continuation_utils import ScriptedContinuationModel, scripted_response

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
    # Colour is asserted on its own; everywhere else it would only obscure what's being asserted.
    monkeypatch.setenv('NO_COLOR', '1')
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec_without_harness)
    yield
    _display._banner_displayed = False  # pyright: ignore[reportPrivateUsage]
    _display.BANNER_ENABLED = True


@pytest.fixture
def render(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Render with the version line pinned, so the layout can be asserted without the versions in play."""
    monkeypatch.setattr(_display, '_version_line', lambda: 'HEADING')

    def render_with(**overrides: Any) -> str:
        kwargs: dict[str, Any] = {
            'name': 'support_agent',
            'model': 'openai:gpt-5.6-sol',
            'output_type': str,
            'tools': 2,
            'capabilities': 0,
            'color': False,
        }
        kwargs.update(overrides)
        return _display.render_banner(**kwargs)

    return render_with


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


def find_nothing(name: str) -> None:
    return None


def find_anything(name: str) -> object:
    return object()


def test_render_banner_with_logfire(monkeypatch: pytest.MonkeyPatch, render: Callable[..., str]):
    # Pin the branch via `sys.modules` rather than the install, so the slim jobs — where `logfire`
    # genuinely isn't importable — assert the same banner as the full ones.
    monkeypatch.setitem(sys.modules, 'logfire', None)

    assert render() == snapshot("""\
                      HEADING
         / \\          agent: support_agent • model: openai:gpt-5.6-sol • tools: 2 • capabilities: 0
       /     \\        observability: off — to see every model and tool call live with its cost,
     /____.____\\        pass `instrument=True` to `Agent()` and run `logfire.configure()`.
   /      |      \\      sign up for Pydantic Logfire for free via `uvx logfire auth` or
 /        |        \\    https://logfire.pydantic.dev, or use any OpenTelemetry backend.
  ·.______|______.·     docs: https://pydantic.dev/docs/ai/logfire/
                      hide this banner: PYDANTIC_AI_NO_BANNER=1\
""")


def test_render_banner_without_logfire(monkeypatch: pytest.MonkeyPatch, render: Callable[..., str]):
    monkeypatch.delitem(sys.modules, 'logfire', raising=False)
    monkeypatch.setattr(importlib.util, 'find_spec', find_nothing)

    # An agent with no name of its own drops the `agent:` segment rather than inventing a name.
    assert render(name=None, output_type=list[str], capabilities=3) == snapshot("""\
                      HEADING
         / \\          model: openai:gpt-5.6-sol • output: list[str] • tools: 2 • capabilities: 3
       /     \\        observability: off — to see every model and tool call live with its cost,
     /____.____\\        install `logfire`, pass `instrument=True` to `Agent()`, and run
   /      |      \\      `logfire.configure()`.
 /        |        \\    sign up for Pydantic Logfire for free via `uvx logfire auth` or
  ·.______|______.·     https://logfire.pydantic.dev, or use any OpenTelemetry backend.
                        docs: https://pydantic.dev/docs/ai/logfire/
                      hide this banner: PYDANTIC_AI_NO_BANNER=1\
""")


def test_render_banner_without_observability(render: Callable[..., str]):
    """What `clai` shows: the same banner, minus advice it has already acted on."""
    assert render(observability=False) == snapshot("""\
         / \\
       /     \\
     /____.____\\      HEADING
   /      |      \\    agent: support_agent • model: openai:gpt-5.6-sol • tools: 2 • capabilities: 0
 /        |        \\
  ·.______|______.·\
""")


def test_render_banner_wraps_long_details(render: Callable[..., str]):
    """Details too wide for the column continue on the next line rather than overflowing it."""
    assert render(
        name='the-agent-that-has-a-rather-long-name',
        model='bedrock:us.anthropic.claude-fable-5-20260101-v1:0',
        observability=False,
    ) == snapshot("""\
         / \\
       /     \\        HEADING
     /____.____\\      agent: the-agent-that-has-a-rather-long-name
   /      |      \\      model: bedrock:us.anthropic.claude-fable-5-20260101-v1:0 • tools: 2
 /        |        \\    capabilities: 0
  ·.______|______.·\
""")


def test_render_banner_colors_the_logo_and_identity(monkeypatch: pytest.MonkeyPatch, render: Callable[..., str]):
    """The logo takes `clai`'s magenta, and what identifies the agent takes the green it used."""
    banner = render(color=True, observability=False)

    assert '\x1b[35m         / \\\x1b[0m' in banner
    assert 'agent: \x1b[32msupport_agent\x1b[0m • model: \x1b[32mopenai:gpt-5.6-sol\x1b[0m' in banner
    # What the agent was given is counted plainly; only its identity is highlighted.
    assert 'tools: 2 • capabilities: 0' in banner


@pytest.mark.parametrize(
    ('output_type', 'expected'),
    [
        pytest.param(int, 'output: int', id='class'),
        pytest.param(list[str], 'output: list[str]', id='parameterized'),
        pytest.param([int, str], 'output: int | str', id='list-of-types'),
        pytest.param(find_nothing, 'output: find_nothing', id='output-function'),
        pytest.param(
            list[dict[str, list[tuple[int, str, bytes, float, complex, bool]]]],
            'output: list[dict[str, list[tuple[int, str, byt…',
            id='cut-off-when-too-long',
        ),
    ],
)
def test_render_banner_names_the_output_type(output_type: Any, expected: str, render: Callable[..., str]):
    assert expected in render(output_type=output_type)


def test_display_banner_writes_versions(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)

    display_banner()

    assert stderr.getvalue().split('\n')[0].strip() == snapshot(
        f'pydantic-ai v{__version__} • Python {_display.platform.python_version()}'
    )


def test_display_banner_with_harness(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    def distribution_version(distribution: str) -> str:
        return '1.2.3'

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setattr(importlib.util, 'find_spec', find_anything)
    monkeypatch.setattr(metadata, 'version', distribution_version)

    display_banner()

    assert (
        f'pydantic-ai v{__version__} • pydantic-ai-harness v1.2.3 • Python {_display.platform.python_version()}'
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

    # An already-imported `logfire` takes the branch that doesn't tell the user to install it, but
    # still tells them how to switch it on. Matched on the identifiers rather than the prose around
    # them, which is the banner's to reword.
    assert 'install `logfire`' not in stderr.getvalue()
    assert '`instrument=True`' in stderr.getvalue()


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


@pytest.mark.parametrize(
    ('condition', 'value'),
    [
        ('PYDANTIC_AI_NO_BANNER', ''),
        ('CI', ''),
        ('tty', False),
        ('enabled', False),
    ],
)
def test_a_banner_that_can_never_be_shown_stops_being_offered(
    condition: str, value: str | bool, monkeypatch: pytest.MonkeyPatch
):
    """Otherwise every run in a production process gathers a banner's details all over again."""
    if condition in {'PYDANTIC_AI_NO_BANNER', 'CI'}:
        monkeypatch.setenv(condition, str(value))
    elif condition == 'tty':
        monkeypatch.setattr(sys.stderr, 'isatty', lambda: value)
    else:
        monkeypatch.setattr(_display, 'BANNER_ENABLED', value)

    display_banner()

    assert _display.banner_pending() is False


def test_an_instrumented_run_leaves_the_banner_for_another_agent(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """Instrumentation is the agent's, not the process's, so it doesn't speak for the ones after it."""
    monkeypatch.setattr(sys, 'stderr', stderr)

    display_banner(instrumented=True)
    assert _display.banner_pending() is True

    display_banner(name='uninstrumented_agent')
    assert 'agent: uninstrumented_agent' in stderr.getvalue()


class BrokenStream(TTYStream):
    """A terminal that can't encode the banner, as `LC_ALL=C` gives you."""

    def write(self, s: str) -> int:
        raise UnicodeEncodeError('ascii', s, 0, 1, 'ordinal not in range(128)')


def test_a_banner_that_cannot_be_written_is_dropped(monkeypatch: pytest.MonkeyPatch):
    """A courtesy that fails is not worth an agent run: this used to raise straight through `iter()`."""
    monkeypatch.setattr(sys, 'stderr', BrokenStream())
    agent = Agent(TestModel())

    result = agent.run_sync('hello')

    assert result.output == snapshot('success (no tool calls)')


def test_a_banner_is_not_written_to_a_stderr_that_is_not_there(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`sys.stderr` is `None` under `pythonw`, where `print(file=None)` would divert to `stdout`."""
    monkeypatch.setattr(sys, 'stderr', None)

    display_banner()

    assert capsys.readouterr().out == ''


def test_a_stderr_that_cannot_be_asked_is_not_a_terminal(monkeypatch: pytest.MonkeyPatch):
    """`isatty()` raises on a closed stream, which is where a long-lived process can leave `stderr`."""

    class ClosedStream(TTYStream):
        def isatty(self) -> bool:
            raise ValueError('I/O operation on closed file')

    monkeypatch.setattr(sys, 'stderr', ClosedStream())

    display_banner()

    assert _display.banner_pending() is False


def test_an_unnameable_harness_is_left_out_of_the_versions(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """`find_spec` raises for a module whose `__spec__` is None, and for what a custom importer hates."""

    def find_spec_that_raises(name: str) -> None:
        raise ValueError(f'{name}.__spec__ is None')

    monkeypatch.setattr(sys, 'stderr', stderr)
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec_that_raises)
    monkeypatch.setitem(sys.modules, 'logfire', None)

    display_banner()

    assert 'pydantic-ai-harness' not in stderr.getvalue()
    assert f'pydantic-ai v{__version__}' in stderr.getvalue()


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

    assert 'agent: support_agent • model: test:test • tools: 0 • capabilities: 0' in stderr.getvalue()


def test_banner_counts_every_tool_the_model_is_offered(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """Tools reach an agent by more routes than `@agent.tool`, and the run resolves all of them."""

    def double(value: int) -> int:
        return value * 2

    def halve(value: int) -> int:
        return value // 2

    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel(), toolsets=[FunctionToolset([double, halve])])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    agent.run_sync('hello')

    assert 'tools: 3' in stderr.getvalue()


def test_banner_is_shown_by_a_run_that_resumes_a_suspended_turn(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """Resuming a paused turn reaches the first request by its own path, which used to skip this."""
    monkeypatch.setattr(sys, 'stderr', stderr)
    model = ScriptedContinuationModel(
        responses=[scripted_response(texts=['done'], provider_response_id='c2', input_tokens=1, output_tokens=1)]
    )
    agent = Agent(model, name='resumed_agent')
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('go')]),
        scripted_response(
            texts=['part one '], state='suspended', provider_response_id='c1', input_tokens=1, output_tokens=1
        ),
    ]

    agent.run_sync(message_history=history)

    assert 'agent: resumed_agent' in stderr.getvalue()


def test_banner_does_not_count_output_tools(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """The output type is reported in its own right, so counting its tool would double-count it."""

    class Answer(BaseModel):
        answer: str

    monkeypatch.setattr(sys, 'stderr', stderr)

    Agent(TestModel(), output_type=Answer).run_sync('hello')

    assert 'output: Answer • tools: 0' in stderr.getvalue()


def test_banner_reports_the_model_the_run_selected(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """A run-level `model=` wins over the agent's own, as the run's first step is what settles it."""
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel())

    agent.run_sync('hello', model=TestModel(custom_output_text='hi'))

    assert 'model: test:test' in stderr.getvalue()


def test_banner_reports_the_run_output_type(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """A run-level `output_type=` overrides what the agent was built with."""
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel())

    agent.run_sync('hello', output_type=int)

    assert 'output: int' in stderr.getvalue()


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


def test_agent_without_capabilities_counts_zero(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    """Zero is worth saying: the infrastructure capabilities every agent gets aren't the user's."""
    monkeypatch.setattr(sys, 'stderr', stderr)

    Agent(TestModel()).run_sync('hello')

    assert 'capabilities: 0' in stderr.getvalue()


def test_instrumented_agent_run_is_silent(monkeypatch: pytest.MonkeyPatch, stderr: TTYStream):
    monkeypatch.setattr(sys, 'stderr', stderr)
    agent = Agent(TestModel())
    agent.instrument = True

    agent.run_sync('hello')

    assert stderr.getvalue() == ''
