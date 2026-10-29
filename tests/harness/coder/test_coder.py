import copy
import importlib
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import NoReturn

import pytest

import pydantic_ai.capabilities.local_workspace
import pydantic_ai_harness.coder
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability, LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceRef
from pydantic_ai_harness.coder import FILE_TOOL_NAMES, Coder, coder_agent
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.subagents import SubAgents, TaskControl
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits


def test_coder_agent_is_model_less_and_composed() -> None:
    assert isinstance(coder_agent, Agent)
    assert coder_agent.model is None
    assert coder_agent.name == 'coder'


async def test_bundled_coder_agent_supplies_import_time_working_directory() -> None:
    result = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'))

    assert result.output == 'done'
    assert await result.workspace.working_dir() == Path.cwd().resolve().as_posix()


async def test_bundled_coder_agent_continues_its_local_ref_from_history() -> None:
    first = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'))
    assert first.workspace.ref == WorkspaceRef(provider='local', id=str(Path.cwd()))

    second = await coder_agent.run(
        'again', model=TestModel(call_tools=[], custom_output_text='done'), message_history=first.all_messages()
    )

    assert second.workspace.ref == first.workspace.ref


async def test_bundled_coder_agent_declines_a_local_ref_for_another_directory(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await coder_agent.run(
            'go',
            model=TestModel(call_tools=[], custom_output_text='done'),
            workspace=WorkspaceRef(provider='local', id=str(tmp_path)),
        )


async def test_bundled_coder_agent_preserves_explicit_workspace_identity(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(working_dir=tmp_path)
    workspace = ReadOnlyWorkspace(Workspace(backend))
    result = await coder_agent.run('go', model=TestModel(call_tools=[], custom_output_text='done'), workspace=workspace)

    assert result.workspace is workspace


def _posix_only(*_: object, **__: object) -> NoReturn:
    raise NotImplementedError('LocalWorkspaceBackend requires a POSIX host.')


async def test_coder_agent_imports_on_a_non_posix_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # `LocalWorkspace` refuses a non-POSIX host when constructed; the bundled agent must still import there.
    import pydantic_ai_harness.coder._agent as module

    monkeypatch.setattr(pydantic_ai.capabilities.local_workspace, 'LocalWorkspaceBackend', _posix_only)
    try:
        agent = importlib.reload(module).coder_agent
    finally:
        monkeypatch.undo()
        importlib.reload(module)
    assert not any(isinstance(item, LocalWorkspace) for item in agent.root_capability.capabilities)
    with pytest.raises(UserError, match='`Coder` needs a workspace'):
        await agent.run('go', model=TestModel(call_tools=[]))


@pytest.mark.subprocess(reason='asserts what a fresh interpreter imports')
def test_coder_agent_export_is_lazy() -> None:
    # A fresh interpreter, so neither `pydantic_ai_harness` nor `pydantic_ai_harness.coder` can import the
    # bundled agent without the test seeing it. Re-importing the package in this process instead would
    # define its typed tool parts a second time, leaving the tool kind registry pointing at classes the
    # restored modules no longer hold, which breaks pickling those parts in every later test.
    code = 'import sys, pydantic_ai_harness.coder; print("pydantic_ai_harness.coder._agent" in sys.modules)'
    env: dict[str, str] = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == 'False'


def test_coder_unknown_export() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.coder.__getattr__('missing')


def test_coder_uses_a_given_repo_context_in_place_of_the_bundled_one() -> None:
    mine = RepoContext[object](expose_inventory_tool=False)
    coder = Coder[object](repo_context=mine)
    names = [type(capability).__name__ for capability in coder.capabilities]
    assert names.index('RepoContext') == 4
    assert [capability for capability in coder.capabilities if isinstance(capability, RepoContext)] == [mine]


def test_coder_members_and_parameters() -> None:
    coder = Coder(instructions='Custom instructions')
    assert [type(capability).__name__ for capability in coder.capabilities] == [
        'RequireWorkspace',
        'Capability',
        'FileSystem',
        'Shell',
        'RepoContext',
        'SubAgents',
        'ClearToolResults',
        '_BoundToolOutputs',
        'RepairToolArguments',
        '_DefaultCaching',
        '_SetToolStrict',
    ]
    files = next(item for item in coder.capabilities if isinstance(item, FileSystem))
    assert (files.root_dir, files.content_hashes, files.tools) == (None, False, FILE_TOOL_NAMES)
    shell = next(item for item in coder.capabilities if isinstance(item, Shell))
    assert (shell.tools, shell.denied_commands, shell.allow_interactive) == (['shell'], [], True)
    assert (shell.env, shell.denied_env_patterns) == (None, [])
    guidance = next(item for item in coder.capabilities if isinstance(item, Capability))
    instructions = str(guidance.get_instructions())
    for text in ('Custom instructions', 'DRY', 'YAGNI', 'SOLID', 'Zen of Python'):
        assert text in instructions
    repo = next(item for item in coder.capabilities if isinstance(item, RepoContext))
    assert (
        repo.root_markers,
        repo.nested_traversal,
        repo.nested_inject,
        repo.nested_max_chars,
        repo.expose_inventory_tool,
    ) == (('.git',), True, 'contents', 32_000, False)
    delegation = next(item for item in coder.capabilities if isinstance(item, SubAgents))
    assert (delegation.include_self, delegation.agents, delegation.agent_folders) == (True, (), None)
    limits = next(item for item in coder.capabilities if isinstance(item, ToolOutputLimits))
    assert limits.id == 'coder_tool_output_limits'
    assert isinstance(coder.for_agent(Agent(TestModel())), Coder)


def test_bound_tool_outputs_sharing_an_id_raise_instead_of_merging() -> None:
    limits = next(item for item in Coder().capabilities if isinstance(item, ToolOutputLimits))
    with pytest.raises(UserError, match="'coder_tool_output_limits' is used by multiple capabilities"):
        Agent(TestModel(), capabilities=[limits, copy.copy(limits)])


async def test_no_workspace_fails_the_run_naming_coder() -> None:
    with pytest.raises(UserError, match='`Coder` needs a workspace') as raised:
        await Agent(TestModel(), capabilities=[Coder()]).run('go')
    assert 'from pydantic_ai.capabilities import LocalWorkspace' in str(raised.value)
    assert '- LocalWorkspace: .' in str(raised.value)


CONTROL_TOOLS = {'list_tasks', 'message_task', 'answer_task', 'cancel_task'}
CODER_TOOLS = {*FILE_TOOL_NAMES, 'shell', 'delegate_task', *CONTROL_TOOLS}


async def _sent_tools(tmp_path: Path, coder: Coder[object]) -> dict[str, ToolDefinition]:
    model = TestModel(call_tools=[])
    agent = Agent[object](model, capabilities=[LocalWorkspace(tmp_path), coder])

    @agent.tool_plain(strict=True)
    def strict_tool() -> str:
        return 'ok'  # pragma: no cover

    @agent.tool_plain
    def inferred_tool() -> str:
        return 'ok'  # pragma: no cover

    await agent.run('go')
    assert model.last_model_request_parameters is not None
    return {tool.name: tool for tool in model.last_model_request_parameters.function_tools}


async def test_coder_tools_are_not_strict_by_default(tmp_path: Path) -> None:
    sent = await _sent_tools(tmp_path, Coder[object]())

    assert set(sent) == CODER_TOOLS - CONTROL_TOOLS | {'strict_tool', 'inferred_tool'}
    assert {name: tool.strict for name, tool in sent.items()} == {
        **dict.fromkeys(CODER_TOOLS - CONTROL_TOOLS, False),
        'strict_tool': True,
        'inferred_tool': None,
    }


@pytest.mark.parametrize('strict', [False, True, None])
async def test_coder_strict_applies_to_its_own_tools_only(tmp_path: Path, strict: bool | None) -> None:
    sent = await _sent_tools(tmp_path, Coder[object](strict=strict, task_control=TaskControl()))

    assert {name: tool.strict for name, tool in sent.items()} == {
        **dict.fromkeys(CODER_TOOLS, strict),
        'strict_tool': True,
        'inferred_tool': None,
    }


@pytest.mark.parametrize('strict', [False, True, None])
async def test_coder_strict_reaches_the_delegates_ask_parent(tmp_path: Path, strict: bool | None) -> None:
    delegate_tools: dict[str, bool | None] = {}

    def delegate(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Run it'})])

    async def run_delegated(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        delegate_tools.update((tool.name, tool.strict) for tool in info.function_tools)
        yield 'done'

    coder = Coder[object](strict=strict, task_control=TaskControl(), repo_context=False)
    model = FunctionModel(delegate, stream_function=run_delegated)
    await Agent[object](model, capabilities=[LocalWorkspace(tmp_path), coder]).run('go')

    assert 'ask_parent' in delegate_tools
    assert set(delegate_tools.values()) == {strict}
