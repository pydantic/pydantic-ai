"""`read_tool_result` steps aside for `FileSystem`'s `read_file` when spills are in the run's workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, PrefixTools
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import READ_FILE_KIND, FileSystem
from pydantic_ai_harness.tool_output_limits import (
    READ_TOOL_NAME,
    READ_TOOL_RESULT_KIND,
    Band,
    LocalFileStore,
    OverflowStore,
    Spill,
    ToolOutputLimits,
    WorkspaceStore,
)

_PAYLOAD = '\n'.join(f'line {i}' for i in range(500))


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


class _Recorder:
    """A `FunctionModel` that calls `big_tool`, then reads the spill back with whichever reader is offered."""

    def __init__(self, *, read_back: bool = True) -> None:
        self.read_back = read_back
        self.offered: list[dict[str, str | None]] = []
        self.marker: str | None = None

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.offered.append({tool.name: tool.tool_kind for tool in info.function_tools})
        returns = _returns(messages)
        if reads := [part for part in returns if part.tool_name != 'big_tool']:
            return ModelResponse(parts=[TextPart(str(reads[0].content))])
        if spilled := [part for part in returns if part.tool_name == 'big_tool']:
            self.marker = str(spilled[0].content)
            assert spilled[0].metadata is not None
            handle = spilled[0].metadata['overflow_handle']
            if not self.read_back:
                return ModelResponse(parts=[TextPart('done')])
            tools = self.offered[-1]
            if reader := next((name for name, kind in tools.items() if kind == READ_TOOL_RESULT_KIND.id), None):
                return ModelResponse(parts=[ToolCallPart(reader, {'handle': handle})])
            file_reader = next(name for name, kind in tools.items() if kind == READ_FILE_KIND.id)
            return ModelResponse(parts=[ToolCallPart(file_reader, {'path': handle})])
        return ModelResponse(parts=[ToolCallPart('big_tool', {})])


async def _run(
    workspace: Path, *capabilities: AbstractCapability[None], read_back: bool = True
) -> tuple[_Recorder, str]:
    recorder = _Recorder(read_back=read_back)
    agent = Agent(
        FunctionModel(recorder.respond), deps_type=type(None), capabilities=[LocalWorkspace(workspace), *capabilities]
    )

    @agent.tool_plain
    def big_tool() -> str:
        return _PAYLOAD

    result = await agent.run('go')
    return recorder, result.output


def _limits(store: OverflowStore | WorkspaceStore | None = None) -> ToolOutputLimits[None]:
    return ToolOutputLimits(bands=[Band(over=100, action=Spill())], store=store)


class TestReaderStepsAside:
    async def test_read_file_replaces_read_tool_result(self, tmp_path: Path) -> None:
        recorder, output = await _run(tmp_path, _limits(), FileSystem())
        assert all(READ_TOOL_NAME not in tools for tools in recorder.offered)
        assert recorder.marker is not None
        assert 'Read it with read_file(path=' in recorder.marker
        assert READ_TOOL_NAME not in recorder.marker
        # The read of the spill is not itself spilled, though it is over the band's threshold.
        assert 'line 100' in output
        assert 'Tool output too large' not in output

    async def test_prefixed_read_file_is_named(self, tmp_path: Path) -> None:
        # A renamed `FileSystem` tool can be offered but not called yet (#10927), so this checks what is offered.
        recorder, _ = await _run(tmp_path, _limits(), PrefixTools(FileSystem(), prefix='repo'), read_back=False)
        assert all(READ_TOOL_NAME not in tools and 'repo_read_file' in tools for tools in recorder.offered)
        assert recorder.marker is not None
        assert 'Read it with repo_read_file(path=' in recorder.marker

    async def test_read_file_of_other_files_is_still_reduced(self, tmp_path: Path) -> None:
        (tmp_path / 'big.txt').write_text(_PAYLOAD)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if returns := _returns(messages):
                return ModelResponse(parts=[TextPart(str(returns[0].content))])
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'big.txt'})])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[LocalWorkspace(tmp_path), _limits(), FileSystem()],
        )
        result = await agent.run('go')
        assert result.output.startswith('[Tool output too large')
        assert 'Read it with read_file(path=' in result.output


class TestReaderKept:
    async def test_without_file_system(self, tmp_path: Path) -> None:
        recorder, output = await _run(tmp_path, _limits())
        assert all(READ_TOOL_NAME in tools for tools in recorder.offered)
        assert recorder.marker is not None
        assert f'Read it with {READ_TOOL_NAME}(handle=' in recorder.marker
        assert 'line 100' in output

    @pytest.mark.parametrize('own_store', ['workspace', 'local'])
    async def test_store_outside_the_run_workspace(self, tmp_path: Path, own_store: str) -> None:
        spills = tmp_path / 'spills'
        spills.mkdir()
        store = (
            WorkspaceStore(workspace=LocalWorkspaceBackend(spills))
            if own_store == 'workspace'
            else LocalFileStore(base_dir=spills)
        )
        work = tmp_path / 'work'
        work.mkdir()
        recorder, output = await _run(work, _limits(store=store), FileSystem())
        assert all(READ_TOOL_NAME in tools and 'read_file' in tools for tools in recorder.offered)
        assert recorder.marker is not None
        assert f'Read it with {READ_TOOL_NAME}(handle=' in recorder.marker
        assert 'line 100' in output

    async def test_prefixed_read_tool_result_is_named(self, tmp_path: Path) -> None:
        recorder, output = await _run(tmp_path, PrefixTools(_limits(), prefix='limits'))
        assert recorder.marker is not None
        assert f'Read it with limits_{READ_TOOL_NAME}(handle=' in recorder.marker
        # The prefixed reader is exempt from reduction by its kind.
        assert 'line 100' in output
        assert 'Tool output too large' not in output
