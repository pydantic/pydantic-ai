"""`CapabilityStorage` moves the files harness capabilities keep for themselves out of the run's workspace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import CombinedCapability, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import CapabilityStorage, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME, Band, Spill, WorkspaceStore

PAYLOAD = '\n'.join(f'line {i}' for i in range(500))


def _returns(messages: Sequence[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def _spill_then_read(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call `big_tool`, read two lines of its spill back, then answer with them."""
    if read := _returns(messages, READ_TOOL_NAME):
        return ModelResponse(parts=[TextPart(str(read[0].content))])
    if spilled := _returns(messages, 'big_tool'):
        assert spilled[0].metadata is not None
        handle = spilled[0].metadata['overflow_handle']
        return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
    return ModelResponse(parts=[ToolCallPart('big_tool', {})])


def _agent(*capabilities: CapabilityStorage | ToolOutputLimits | LocalWorkspace | CombinedCapability) -> Agent:
    agent = Agent(FunctionModel(_spill_then_read), capabilities=list(capabilities))

    @agent.tool_plain
    def big_tool() -> str:
        return PAYLOAD

    return agent


def _spill_handle(messages: Sequence[ModelMessage]) -> str:
    [spilled] = _returns(messages, 'big_tool')
    assert spilled.metadata is not None
    return spilled.metadata['overflow_handle']


def _limits() -> ToolOutputLimits:
    return ToolOutputLimits(bands=[Band(over=100, action=Spill())])


async def test_spills_go_to_storage_without_a_run_workspace(tmp_path: Path):
    agent = _agent(CapabilityStorage(LocalWorkspaceBackend(tmp_path)), _limits())

    result = await agent.run('go')

    handle = _spill_handle(result.all_messages())
    assert handle.startswith(f'{tmp_path.resolve()}/.pydantic-ai-harness/tool-output/')
    assert Path(handle).read_text(encoding='utf-8') == PAYLOAD
    assert result.output == f'[handle {handle!r}: 500 matching line(s); showing 2]\nline 0\nline 1'


async def test_storage_keeps_spills_out_of_the_run_workspace(tmp_path: Path):
    work = tmp_path / 'work'
    storage = tmp_path / 'storage'
    work.mkdir()
    storage.mkdir()
    agent = _agent(LocalWorkspace(work), CapabilityStorage(LocalWorkspaceBackend(storage)), _limits())

    result = await agent.run('go')

    assert _spill_handle(result.all_messages()).startswith(f'{storage.resolve()}/.pydantic-ai-harness/tool-output/')
    assert not (work / '.pydantic-ai-harness').exists()
    assert result.output.endswith('line 0\nline 1')


async def test_spills_work_when_the_run_workspace_is_read_only(tmp_path: Path):
    work = tmp_path / 'work'
    storage = tmp_path / 'storage'
    work.mkdir()
    storage.mkdir()
    agent = _agent(LocalWorkspace(work, read_only=True), CapabilityStorage(LocalWorkspaceBackend(storage)), _limits())

    result = await agent.run('go')

    assert Path(_spill_handle(result.all_messages())).read_text(encoding='utf-8') == PAYLOAD
    assert list(work.iterdir()) == []


async def test_storage_passed_per_run(tmp_path: Path):
    agent = _agent(_limits())

    result = await agent.run('go', capabilities=[CapabilityStorage(LocalWorkspaceBackend(tmp_path))])

    assert _spill_handle(result.all_messages()).startswith(f'{tmp_path.resolve()}/')


async def test_storage_inside_a_combined_capability(tmp_path: Path):
    agent = _agent(CombinedCapability([CapabilityStorage(LocalWorkspaceBackend(tmp_path)), _limits()]))

    result = await agent.run('go')

    assert _spill_handle(result.all_messages()).startswith(f'{tmp_path.resolve()}/')


async def test_an_explicit_store_workspace_takes_precedence(tmp_path: Path):
    storage = tmp_path / 'storage'
    spills = tmp_path / 'spills'
    storage.mkdir()
    spills.mkdir()
    limits = ToolOutputLimits(
        bands=[Band(over=100, action=Spill())], store=WorkspaceStore(workspace=LocalWorkspaceBackend(spills))
    )
    agent = _agent(CapabilityStorage(LocalWorkspaceBackend(storage)), limits)

    result = await agent.run('go')

    assert _spill_handle(result.all_messages()).startswith(f'{spills.resolve()}/')
    assert list(storage.iterdir()) == []


def test_workspace_must_be_a_backend(tmp_path: Path):
    with pytest.raises(TypeError, match=r'`CapabilityStorage\(workspace=...\)` takes a workspace backend'):
        CapabilityStorage(LocalWorkspace(tmp_path))  # pyright: ignore[reportArgumentType]
