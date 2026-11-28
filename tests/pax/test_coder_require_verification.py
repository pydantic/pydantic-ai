"""The `coder` plugin's opt-in `require_verification` setting, over harness `RequireVerification`."""

import io
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.require_verification import (
    DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS,
    RequireVerification,
    VerificationRequiredPart,
)
from pydantic_ai_pax import Session
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_ai_pax.plugins import PluginHost, collect

Step = ToolCallPart | str
"""What the model does in one response: call a tool, or answer."""

WRITE = ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1\n'})


def check(passes: bool = True) -> ToolCallPart:
    """A verification command: a stand-in `pytest` in the workspace that passes or fails."""
    return ToolCallPart('shell', {'command': f'{"ok" if passes else "bad"}/pytest -q'})


def coder_capabilities(**settings: JsonValue) -> list[AgentCapability[None]]:
    """What the `coder` plugin binds with `settings`, without skills or delegation."""
    host = PluginHost[None](
        name='coder',
        console=Console(file=io.StringIO()),
        settings={'repo_context': False, 'sub_agents': False, 'skill_folders': [], **settings},
    )
    return list(collect(CoderPlugin[None].from_host(host)).capabilities)


def scripted(steps: Sequence[Step]) -> FunctionModel:
    """A model that takes `steps` in order, one per response of the session."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        step = steps[sum(isinstance(message, ModelResponse) for message in messages)]
        if isinstance(step, str):
            yield step
        else:
            yield {0: DeltaToolCall(name=step.tool_name, json_args=step.args_as_json_str())}

    return FunctionModel(stream_function=stream)


def redirects(messages: Sequence[ModelMessage]) -> list[list[str]]:
    """The verification instructions each model request carried, one list per request."""
    requests: list[list[str]] = [[]]
    for message in messages:
        if isinstance(message, ModelResponse):
            requests.append([])
        else:
            requests[-1].extend(
                str(part.content) for part in message.parts if isinstance(part, VerificationRequiredPart)
            )
    return requests[:-1]


def answer(messages: Sequence[ModelMessage]) -> str:
    """The text of the conversation's last response."""
    last = messages[-1]
    assert isinstance(last, ModelResponse)
    return ''.join(part.content for part in last.parts if isinstance(part, TextPart))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A project with a stand-in `pytest` that passes (`ok/`) and one that fails (`bad/`)."""
    for folder, status in (('ok', 0), ('bad', 1)):
        script = tmp_path / folder / 'pytest'
        script.parent.mkdir()
        script.write_text(f'#!/bin/sh\nexit {status}\n')
        script.chmod(0o755)
    return tmp_path


def session(workspace: Path, steps: Sequence[Step], plugins: Sequence[AgentCapability[None]]) -> Session[None, str]:
    return Session(create_stock_agent(scripted(steps)), deps=None, workspace=workspace, plugins=list(plugins))


def test_off_by_default_binds_only_coder() -> None:
    assert CoderSettings().require_verification is False
    [coder] = coder_capabilities()
    assert isinstance(coder, Coder)
    [coder, verification] = coder_capabilities(require_verification=True)
    assert isinstance(coder, Coder) and isinstance(verification, RequireVerification)


@pytest.mark.parametrize('enabled', [False, True])
async def test_an_unverified_edit_is_sent_back_to_verify(workspace: Path, enabled: bool) -> None:
    """With the setting on, an answer after an edit gets the harness's instruction; a passing check then ends it."""
    conversation = session(
        workspace, [WRITE, 'Done.', check(), 'Verified.'], coder_capabilities(require_verification=enabled)
    )
    await conversation.prompt('set x')
    assert (workspace / 'app.py').read_text() == 'x = 1\n'
    if enabled:
        assert redirects(conversation.messages) == [[], [], [DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS], []]
        assert answer(conversation.messages) == 'Verified.'
    else:
        assert redirects(conversation.messages) == [[], []]
        assert answer(conversation.messages) == 'Done.'


async def test_a_check_after_the_edit_finishes_normally(workspace: Path) -> None:
    conversation = session(workspace, [WRITE, check(), 'Done.'], coder_capabilities(require_verification=True))
    await conversation.prompt('set x and test it')
    assert redirects(conversation.messages) == [[], [], []]
    assert answer(conversation.messages) == 'Done.'


async def test_a_turn_without_edits_is_not_sent_back(workspace: Path) -> None:
    read = ToolCallPart('read_file', {'path': 'ok/pytest'})
    conversation = session(workspace, [read, 'It exits 0.'], coder_capabilities(require_verification=True))
    await conversation.prompt('what does it do?')
    assert redirects(conversation.messages) == [[], []]
    assert answer(conversation.messages) == 'It exits 0.'


async def test_a_turn_that_cannot_verify_still_ends_with_its_answer(workspace: Path) -> None:
    """Redirects are new requests, not output retries: two failed checks end the turn with the model's answer."""
    steps = [WRITE, 'Done.', check(passes=False), 'Tests fail.', check(passes=False), 'Tests still fail.', 'Hi.']
    conversation = session(workspace, steps, coder_capabilities(require_verification=True))
    await conversation.prompt('set x')
    instruction = [DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS]
    assert redirects(conversation.messages) == [[], [], instruction, [], instruction, []]
    assert answer(conversation.messages) == 'Tests still fail.'
    # The next turn starts with no edits, so its answer ends it.
    await conversation.prompt('thanks')
    assert answer(conversation.messages) == 'Hi.'
    assert sum(isinstance(message, ModelRequest) for message in conversation.messages) == 7


async def test_shell_commands_inside_run_code_count(workspace: Path) -> None:
    """With code mode, a `shell` edit and a `shell` check called from `run_code` count like direct calls."""
    pytest.importorskip('pydantic_monty')
    from pydantic_ai_pax.runtime.speculation import SpeculationCounters
    from pydantic_ai_pax.runtime.speculative_mode import speculative_capabilities

    capabilities = coder_capabilities(require_verification=True)
    plugins = [*capabilities, *speculative_capabilities(SpeculationCounters(), capabilities, speculate=False)]
    edit = ToolCallPart('run_code', {'code': 'await shell(command="echo x = 1 > app.py")'})
    test = ToolCallPart('run_code', {'code': 'await shell(command="ok/pytest -q")'})
    conversation = session(workspace, [edit, 'Done.', test, 'Verified.'], plugins)
    await conversation.prompt('set x')
    assert (workspace / 'app.py').read_text() == 'x = 1\n'
    assert redirects(conversation.messages) == [[], [], [DEFAULT_REQUIRE_VERIFICATION_INSTRUCTIONS], []]
    assert answer(conversation.messages) == 'Verified.'


def test_the_menu_saves_the_setting() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append
    )
    source = CoderSource(host)
    row = next(row for row in source.rows() if row.key == 'require_verification')
    assert (row.label, source.current(row)) == ('Require verification', 'false')
    assert source.apply(row, 'true') == 'Saved Require verification.'
    assert saved == [{'require_verification': True}]
    assert source.current(row) == 'true'
