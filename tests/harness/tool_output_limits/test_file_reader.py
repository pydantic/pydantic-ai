"""A spill in the run's workspace is read with the agent's file tool when one can read it."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME, Band, Spill, ToolOutputLimits

pytestmark = pytest.mark.anyio

PAYLOAD = '\n'.join(f'line {i}' for i in range(500))


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _returns(messages: Sequence[ModelMessage], tool_name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


def _agent(work: Path, model: FunctionModel, *capabilities: AbstractCapability[None]) -> Agent[None, str]:
    agent = Agent(
        model,
        deps_type=type(None),
        capabilities=[
            ToolOutputLimits(bands=[Band(over=100, action=Spill())]),
            LocalWorkspace(work),
            *capabilities,
        ],
    )

    @agent.tool_plain
    def big_tool() -> str:
        return PAYLOAD

    return agent


def _call_big_tool_once(offered: list[set[str]]) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    """Call `big_tool` once, then finish, recording the tools offered at each step."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append({tool.name for tool in info.function_tools})
        return ModelResponse(
            parts=[TextPart('done') if _returns(messages, 'big_tool') else ToolCallPart('big_tool', {})]
        )

    return respond


class TestFileReader:
    async def test_read_file_replaces_read_tool_result(self, tmp_path: Path):
        offered: list[set[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append({tool.name for tool in info.function_tools})
            if read := _returns(messages, 'read_file'):
                return ModelResponse(parts=[TextPart(str(read[0].content))])
            if spilled := _returns(messages, 'big_tool'):
                assert spilled[0].metadata is not None
                return ModelResponse(
                    parts=[ToolCallPart('read_file', {'path': spilled[0].metadata['overflow_handle']})]
                )
            return ModelResponse(parts=[ToolCallPart('big_tool', {})])

        result = await _agent(tmp_path, FunctionModel(respond), FileSystem(max_read_chars=10_000)).run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert spilled.metadata is not None
        handle = spilled.metadata['overflow_handle']
        assert str(spilled.content).splitlines()[0] == (
            f'[Tool output too large ({len(PAYLOAD):,} chars); saved to the file {handle!r}. Read it with `read_file`.]'
        )
        assert all(READ_TOOL_NAME not in tools for tools in offered)
        # The whole file came back: reading a spill is exempt from being spilled again.
        assert '\tline 250\n' in result.output

    async def test_coders_read_file_reads_spills(self, tmp_path: Path):
        offered: list[set[str]] = []

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            # Coder's capabilities wrap the event stream, so the run streams its model requests.
            offered.append({tool.name for tool in info.function_tools})
            yield 'done' if _returns(messages, 'big_tool') else {0: DeltaToolCall(name='big_tool', json_args='{}')}

        result = await _agent(tmp_path, FunctionModel(stream_function=stream), Coder()).run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert str(spilled.content).splitlines()[0].endswith('Read it with `read_file`.]')
        assert all(READ_TOOL_NAME not in tools for tools in offered)

    @pytest.mark.parametrize(
        'file_system',
        [
            pytest.param(None, id='no-file-tool'),
            pytest.param(FileSystem[None](max_read_chars=None), id='uncapped-reads'),
            pytest.param(
                FileSystem[None](max_read_chars=10_000, denied_patterns=['.pydantic-ai-harness/**']), id='denied'
            ),
        ],
    )
    async def test_read_tool_result_without_a_file_tool_that_reads_spills(
        self, tmp_path: Path, file_system: FileSystem[None] | None
    ):
        offered: list[set[str]] = []
        extra = [] if file_system is None else [file_system]

        result = await _agent(tmp_path, FunctionModel(_call_big_tool_once(offered)), *extra).run('go')

        [spilled] = _returns(result.all_messages(), 'big_tool')
        assert f'Read it with {READ_TOOL_NAME}(' in str(spilled.content).splitlines()[0]
        assert all(READ_TOOL_NAME in tools for tools in offered)

    async def test_read_tool_result_stays_once_a_marker_named_it(self, tmp_path: Path):
        earlier = await _agent(tmp_path, FunctionModel(_call_big_tool_once([]))).run('go')
        offered: list[set[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append({tool.name for tool in info.function_tools})
            return ModelResponse(parts=[TextPart('done')])

        agent = _agent(tmp_path, FunctionModel(respond), FileSystem(max_read_chars=10_000))
        await agent.run('continue', message_history=earlier.all_messages())

        assert READ_TOOL_NAME in offered[0]
