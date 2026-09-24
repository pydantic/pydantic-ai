"""Offline tests for the composite action's runner (`action/runner.py`).

Run:  uv run pytest action/test_runner.py
"""

import sys
from pathlib import Path

import logfire
import pytest
from pytest import CaptureFixture, MonkeyPatch

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# `action/` isn't on `sys.path` by default. The runtime equivalent is the action running
# `python3 runner.py`, which puts the script's own directory first.
sys.path.insert(0, str(Path(__file__).parent))

import runner

# The agents the `module:variable` tests resolve. This module is importable under its own name, so
# they double as the module a target names, and what they resolve to stays typed.
agent = Agent(instructions='Be helpful.')
agent_with_model = Agent('test', instructions='Be helpful.')
not_an_agent = 'nope'
# Stands in for a model that answers with workflow commands instead of an answer.
agent_that_forges_commands = Agent(TestModel(custom_output_text='::error::forged\n::add-mask::secret'))

SPEC = """
name: reviewer
instructions: Be helpful.
"""


@pytest.fixture
def inputs(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    """Clear every input the action sets, so each test names only the ones it cares about."""
    names = ('PAI_AGENT', 'PAI_MODEL', 'PAI_PROMPT', 'PAI_PROMPT_FILE', 'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY')
    # A developer's own Logfire credentials would otherwise trace the test suite's runs.
    for name in (*names, 'LOGFIRE_TOKEN', 'OTEL_EXPORTER_OTLP_ENDPOINT', 'GITHUB_ACTIONS'):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / 'reviewer.yml').write_text(SPEC)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_prompt_input(inputs: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    assert runner._read_prompt() == 'Review the diff.'  # pyright: ignore[reportPrivateUsage]


def test_prompt_file_input(inputs: Path, monkeypatch: MonkeyPatch):
    (inputs / 'prompt.md').write_text('Review the diff.\n')
    monkeypatch.setenv('PAI_PROMPT_FILE', 'prompt.md')
    assert runner._read_prompt() == 'Review the diff.\n'  # pyright: ignore[reportPrivateUsage]


def test_prompt_and_prompt_file_are_exclusive(inputs: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    monkeypatch.setenv('PAI_PROMPT_FILE', 'prompt.md')
    with pytest.raises(ValueError, match='exactly one'):
        runner._read_prompt()  # pyright: ignore[reportPrivateUsage]


def test_prompt_is_required(inputs: Path):
    with pytest.raises(ValueError, match='exactly one'):
        runner._read_prompt()  # pyright: ignore[reportPrivateUsage]


def test_missing_prompt_file(inputs: Path, monkeypatch: MonkeyPatch):
    monkeypatch.setenv('PAI_PROMPT_FILE', 'absent.md')
    with pytest.raises(ValueError, match='could not read prompt file'):
        runner._read_prompt()  # pyright: ignore[reportPrivateUsage]


def test_empty_prompt_file(inputs: Path, monkeypatch: MonkeyPatch):
    (inputs / 'empty.md').write_text('')
    monkeypatch.setenv('PAI_PROMPT_FILE', 'empty.md')
    with pytest.raises(ValueError, match='is empty'):
        runner._read_prompt()  # pyright: ignore[reportPrivateUsage]


def test_resolve_module_target(inputs: Path):
    assert runner._resolve_agent('test_runner:agent') is agent  # pyright: ignore[reportPrivateUsage]


def test_resolve_spec_file(inputs: Path):
    assert runner._resolve_agent('reviewer.yml').name == 'reviewer'  # pyright: ignore[reportPrivateUsage]


def test_resolve_target_without_variable(inputs: Path):
    with pytest.raises(ValueError, match='expected a `module:variable` target'):
        runner._resolve_agent('test_runner')  # pyright: ignore[reportPrivateUsage]


def test_resolve_target_that_is_not_an_agent(inputs: Path):
    with pytest.raises(TypeError, match=r'resolved to a builtins\.str'):
        runner._resolve_agent('test_runner:not_an_agent')  # pyright: ignore[reportPrivateUsage]


def test_run(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    monkeypatch.setenv('GITHUB_OUTPUT', str(inputs / 'output'))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(inputs / 'summary'))

    assert runner.main() == 0

    output = capsys.readouterr().out.strip()
    assert output == 'success (no tool calls)'

    published = (inputs / 'output').read_text()
    assert published.startswith('result<<EOF_')
    assert published.splitlines()[1] == output
    delimiter = published.splitlines()[0].removeprefix('result<<')
    assert published.splitlines()[2] == delimiter

    assert (inputs / 'summary').read_text() == f'{output}\n'


def test_run_without_github_environment(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """Outside a workflow there is nowhere to publish to, and the run still prints what it said."""
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0
    assert capsys.readouterr().out.strip() == 'success (no tool calls)'


def test_run_uses_the_agents_own_model(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """An agent that carries a model runs without the `model` input."""
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent_with_model')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0
    assert capsys.readouterr().out.strip() == 'success (no tool calls)'


def test_run_without_any_model(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """Neither the agent nor the input names a model, so the run fails with Pydantic AI's own error."""
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 1
    assert 'error: agent run failed:' in capsys.readouterr().err


def test_run_without_agent(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 2
    assert 'error: `agent` is required' in capsys.readouterr().err


def test_run_without_prompt(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')

    assert runner.main() == 2
    assert 'error: set exactly one of `prompt` and `prompt-file`' in capsys.readouterr().err


def test_run_with_unimportable_agent(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'absent_module:agent')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 2
    assert "error: could not resolve agent 'absent_module:agent'" in capsys.readouterr().err


def test_run_with_unwritable_github_output(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    monkeypatch.setenv('GITHUB_OUTPUT', str(inputs / 'absent' / 'output'))

    assert runner.main() == 1
    assert 'error: could not write `GITHUB_OUTPUT`' in capsys.readouterr().err


def test_run_with_unwritable_step_summary(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(inputs / 'absent' / 'summary'))

    assert runner.main() == 1
    assert 'error: could not write `GITHUB_STEP_SUMMARY`' in capsys.readouterr().err


def test_multiline_output_is_published_whole(inputs: Path, monkeypatch: MonkeyPatch):
    """A delimited value keeps a model's line breaks, which a `name=value` output would not."""
    runner._append_github_output(str(inputs / 'output'), 'first\nsecond')  # pyright: ignore[reportPrivateUsage]

    lines = (inputs / 'output').read_text().splitlines()
    assert lines[1:3] == ['first', 'second']
    assert lines[3] == lines[0].removeprefix('result<<')


def _record_logfire(monkeypatch: MonkeyPatch) -> list[object]:
    """Stand in for the two Logfire calls the runner makes, recording them in order."""
    calls: list[object] = []

    def configure(**kwargs: object) -> None:
        calls.append(kwargs)

    def instrument_pydantic_ai() -> None:
        calls.append('instrumented')

    monkeypatch.setattr(logfire, 'configure', configure)
    monkeypatch.setattr(logfire, 'instrument_pydantic_ai', instrument_pydantic_ai)
    return calls


def test_tracing_is_configured_when_a_token_is_set(inputs: Path, monkeypatch: MonkeyPatch):
    calls = _record_logfire(monkeypatch)
    monkeypatch.setenv('LOGFIRE_TOKEN', 'pylf_v1_us_fake')

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert calls == [{'send_to_logfire': 'if-token-present', 'console': False}, 'instrumented']


def test_tracing_is_configured_for_an_otlp_endpoint(inputs: Path, monkeypatch: MonkeyPatch):
    calls = _record_logfire(monkeypatch)
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert calls == [{'send_to_logfire': 'if-token-present', 'console': False}, 'instrumented']


def test_tracing_is_left_alone_without_a_destination(inputs: Path, monkeypatch: MonkeyPatch):
    """Nothing in the environment says where traces would go, so Logfire is not touched at all."""
    calls = _record_logfire(monkeypatch)

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert calls == []


def test_tracing_without_logfire_installed(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """The run goes ahead untraced rather than failing because a package is missing."""
    monkeypatch.setenv('LOGFIRE_TOKEN', 'pylf_v1_us_fake')
    monkeypatch.setitem(sys.modules, 'logfire', None)

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert 'warning: the environment asks for tracing' in capsys.readouterr().err


def test_output_cannot_forge_workflow_commands(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """A model that answers with workflow commands gets printed as text, not obeyed."""
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent_that_forges_commands')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0

    lines = capsys.readouterr().out.strip().splitlines()
    token = lines[0].removeprefix('::stop-commands::')
    assert lines[0] == f'::stop-commands::{token}'
    assert lines[-1] == f'::{token}::'
    # The forged commands are still there, inert, between the markers.
    assert lines[1:-1] == ['::error::forged', '::add-mask::secret']


def test_output_is_printed_plainly_outside_actions(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """Nothing reads workflow commands off a terminal, so the markers would only be noise."""
    monkeypatch.setenv('PAI_AGENT', 'test_runner:agent_that_forges_commands')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0
    assert capsys.readouterr().out == '::error::forged\n::add-mask::secret\n'
