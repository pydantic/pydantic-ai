"""Workspace spills use general file tools when they can read the spill path."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME, Band, Spill, ToolOutputLimits, WorkspaceStore


def _returns(messages: list[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


async def test_workspace_spill_uses_read_file(tmp_path: Path) -> None:
    payload = '\n'.join(f'line {i}' for i in range(500))
    offered: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append({tool.name for tool in info.function_tools})
        if read := _returns(messages, 'read_file'):
            return ModelResponse(parts=[TextPart(str(read[0].content))])
        if spilled := _returns(messages, 'big_tool'):
            assert spilled[0].metadata is not None
            return ModelResponse(
                parts=[ToolCallPart('read_file', {'path': spilled[0].metadata['overflow_handle'], 'limit': 2})]
            )
        return ModelResponse(parts=[ToolCallPart('big_tool', {})])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[ToolOutputLimits(bands=[Band(over=100, action=Spill())]), LocalWorkspace(tmp_path), FileSystem()],
    )

    @agent.tool_plain
    def big_tool() -> str:
        return payload

    result = await agent.run('go')
    assert 'line 1' in result.output
    assert all(READ_TOOL_NAME not in tools for tools in offered)
    assert all('read_file' in tools for tools in offered)
    [spilled] = _returns(result.all_messages(), 'big_tool')
    assert spilled.metadata is not None
    assert f'Read it with read_file(path={spilled.metadata["overflow_handle"]!r}' in str(spilled.content)


@pytest.mark.parametrize('case', ['absent', 'inactive', 'custom-store', 'unreadable'])
async def test_read_tool_result_stays_when_file_tools_cannot_replace_it(tmp_path: Path, case: str) -> None:
    store: WorkspaceStore | None = None
    file_system: FileSystem[None] | None = None
    if case == 'inactive':
        file_system = FileSystem(id='files', defer_loading=True)
    elif case == 'custom-store':
        custom = tmp_path / 'custom'
        custom.mkdir()
        store = WorkspaceStore(workspace=LocalWorkspaceBackend(custom))
        file_system = FileSystem()
    elif case == 'unreadable':
        file_system = FileSystem(denied_patterns=['**/.pydantic-ai-harness/**'])

    offered: set[str] = set()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[
            ToolOutputLimits(store=store),
            LocalWorkspace(tmp_path),
            *([file_system] if file_system is not None else []),
        ],
    )
    await agent.run('go')

    assert READ_TOOL_NAME in offered
