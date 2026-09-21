"""Offline tests for the composite action's runner (`action/runner.py`).

Run:  uv run pytest action/test_runner.py
"""

import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

# `action/` isn't on `sys.path` by default. The runtime equivalent is the action running
# `python3 runner.py`, which puts the script's own directory first.
sys.path.insert(0, str(Path(__file__).parent))

import runner

AGENT_MODULE = """
from pydantic_ai import Agent

agent = Agent(instructions='Be helpful.')
with_model = Agent('test', instructions='Be helpful.')
not_an_agent = 'nope'
"""

SPEC = """
name: reviewer
instructions: Be helpful.
"""


@pytest.fixture
def inputs(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    """Clear every input the action sets, so each test names only the ones it cares about."""
    names = ('PAI_AGENT', 'PAI_MODEL', 'PAI_PROMPT', 'PAI_PROMPT_FILE', 'GITHUB_OUTPUT', 'GITHUB_STEP_SUMMARY')
    # A developer's own Logfire credentials would otherwise trace the test suite's runs.
    for name in (*names, 'LOGFIRE_TOKEN', 'OTEL_EXPORTER_OTLP_ENDPOINT'):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / 'my_agents.py').write_text(AGENT_MODULE)
    (tmp_path / 'reviewer.yml').write_text(SPEC)
    monkeypatch.syspath_prepend(str(tmp_path))
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
    import my_agents  # pyright: ignore[reportMissingImports]

    assert runner._resolve_agent('my_agents:agent') is my_agents.agent  # pyright: ignore[reportPrivateUsage]


def test_resolve_spec_file(inputs: Path):
    assert runner._resolve_agent('reviewer.yml').name == 'reviewer'  # pyright: ignore[reportPrivateUsage]


def test_resolve_target_without_variable(inputs: Path):
    with pytest.raises(ValueError, match='expected a `module:variable` target'):
        runner._resolve_agent('my_agents')  # pyright: ignore[reportPrivateUsage]


def test_resolve_target_that_is_not_an_agent(inputs: Path):
    with pytest.raises(TypeError, match=r'resolved to a builtins\.str'):
        runner._resolve_agent('my_agents:not_an_agent')  # pyright: ignore[reportPrivateUsage]


def test_run(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')
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
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0
    assert capsys.readouterr().out.strip() == 'success (no tool calls)'


def test_run_uses_the_agents_own_model(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """An agent that carries a model runs without the `model` input."""
    monkeypatch.setenv('PAI_AGENT', 'my_agents:with_model')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 0
    assert capsys.readouterr().out.strip() == 'success (no tool calls)'


def test_run_without_any_model(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """Neither the agent nor the input names a model, so the run fails with Pydantic AI's own error."""
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 1
    assert 'error: agent run failed:' in capsys.readouterr().err


def test_run_without_agent(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 2
    assert 'error: `agent` is required' in capsys.readouterr().err


def test_run_without_prompt(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')

    assert runner.main() == 2
    assert 'error: set exactly one of `prompt` and `prompt-file`' in capsys.readouterr().err


def test_run_with_unimportable_agent(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'absent_module:agent')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')

    assert runner.main() == 2
    assert "error: could not resolve agent 'absent_module:agent'" in capsys.readouterr().err


def test_run_with_unwritable_github_output(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')
    monkeypatch.setenv('PAI_MODEL', 'test')
    monkeypatch.setenv('PAI_PROMPT', 'Review the diff.')
    monkeypatch.setenv('GITHUB_OUTPUT', str(inputs / 'absent' / 'output'))

    assert runner.main() == 1
    assert 'error: could not write `GITHUB_OUTPUT`' in capsys.readouterr().err


def test_run_with_unwritable_step_summary(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    monkeypatch.setenv('PAI_AGENT', 'my_agents:agent')
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


def test_tracing_is_configured_when_a_token_is_set(inputs: Path, monkeypatch: MonkeyPatch):
    import logfire

    configured: list[object] = []
    monkeypatch.setattr(logfire, 'configure', lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(logfire, 'instrument_pydantic_ai', lambda: configured.append('instrumented'))
    monkeypatch.setenv('LOGFIRE_TOKEN', 'pylf_v1_us_fake')

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert configured == [{'send_to_logfire': 'if-token-present', 'console': False}, 'instrumented']


def test_tracing_is_configured_for_an_otlp_endpoint(inputs: Path, monkeypatch: MonkeyPatch):
    import logfire

    configured: list[object] = []
    monkeypatch.setattr(logfire, 'configure', lambda **kwargs: configured.append(kwargs))
    monkeypatch.setattr(logfire, 'instrument_pydantic_ai', lambda: configured.append('instrumented'))
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert configured == [{'send_to_logfire': 'if-token-present', 'console': False}, 'instrumented']


def test_tracing_is_left_alone_without_a_destination(inputs: Path, monkeypatch: MonkeyPatch):
    """Nothing in the environment says where traces would go, so Logfire is not touched at all."""
    import logfire

    monkeypatch.delenv('LOGFIRE_TOKEN', raising=False)
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    monkeypatch.setattr(logfire, 'configure', lambda **kwargs: pytest.fail('configured Logfire'))

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]


def test_tracing_without_logfire_installed(inputs: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]):
    """The run goes ahead untraced rather than failing because a package is missing."""
    monkeypatch.setenv('LOGFIRE_TOKEN', 'pylf_v1_us_fake')
    monkeypatch.setitem(sys.modules, 'logfire', None)

    runner._configure_observability()  # pyright: ignore[reportPrivateUsage]

    assert 'warning: the environment asks for tracing' in capsys.readouterr().err
