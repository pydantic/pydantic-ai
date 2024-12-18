from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from datetime import timezone

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ArgsDict,
    ArgsJson,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Usage

from .conftest import IsNow

pytestmark = pytest.mark.anyio


async def test_streamed_text_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with test_agent.run_stream('Hello') as result:
        assert test_agent.name == 'test_agent'
        assert not result.is_structured
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args=ArgsDict(args_dict={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=5,
                total_tokens=108,
            )
        )
        response = await result.get_data()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=11,
                total_tokens=114,
            )
        )
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args=ArgsDict(args_dict={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
                ),
                ModelResponse.from_text(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
            ]
        )


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, result_type=tuple[str, str], name='fig_jam')

    async with agent.run_stream('') as result:
        assert agent.name == 'fig_jam'
        assert result.is_structured
        assert not result.is_complete
        response = await result.get_data()
        assert response == snapshot(('a', 'a'))
        assert result.is_complete


async def test_structured_response_iter():
    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.result_tools is not None
        assert len(agent_info.result_tools) == 1
        name = agent_info.result_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), result_type=list[int])

    chunks: list[list[int]] = []
    async with agent.run_stream('') as result:
        async for structured_response, last in result.stream_structured(debounce_by=None):
            response_data = await result.validate_structured_result(structured_response, allow_partial=not last)
            chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4]])

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_text\(\) can only be used with text responses'):
            async for _ in result.stream_text():
                pass


async def test_streamed_text_stream():
    m = TestModel(custom_result_text='The cat sat on the mat.')

    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        assert not result.is_structured
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream()]
        # one chunk due to group_by_temporal
        assert chunks == snapshot(['The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            [
                'The ',
                'The cat ',
                'The cat sat ',
                'The cat sat on ',
                'The cat sat on the ',
                'The cat sat on the mat.',
            ]
        )

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
            ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
        )

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_structured\(\) can only be used with structured responses'):
            async for _ in result.stream_structured():
                pass


async def test_plain_response():
    call_index = 0

    async def text_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_index

        call_index += 1
        yield 'hello '
        yield 'world'

    agent = Agent(FunctionModel(stream_function=text_stream), result_type=tuple[str, str])

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
        async with agent.run_stream(''):
            pass

    assert call_index == 2


async def test_call_tool():
    async def stream_structured_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            assert agent_info.function_tools is not None
            assert len(agent_info.function_tools) == 1
            name = agent_info.function_tools[0].name
            first = messages[0]
            assert isinstance(first, ModelRequest)
            assert isinstance(first.parts[0], UserPromptPart)
            json_string = json.dumps({'x': first.parts[0].content})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_string[:3])}
            yield {0: DeltaToolCall(json_args=json_string[3:])}
        else:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            assert isinstance(last.parts[0], ToolReturnPart)
            assert agent_info.result_tools is not None
            assert len(agent_info.result_tools) == 1
            name = agent_info.result_tools[0].name
            json_data = json.dumps({'response': [last.parts[0].content, 2]})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_data[:5])}
            yield {0: DeltaToolCall(json_args=json_data[5:])}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        assert x == 'hello'
        return f'{x} world'

    async with agent.run_stream('hello') as result:
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args=ArgsJson(args_json='{"x": "hello"}'))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name='ret_a', content='hello world', timestamp=IsNow(tz=timezone.utc))]
                ),
            ]
        )
        assert await result.get_data() == snapshot(('hello world', 2))
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args=ArgsJson(args_json='{"x": "hello"}'))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[ToolReturnPart(tool_name='ret_a', content='hello world', timestamp=IsNow(tz=timezone.utc))]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args=ArgsJson(args_json='{"response": ["hello world", 2]}'),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ]
                ),
            ]
        )


async def test_call_tool_empty():
    async def stream_structured_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    with pytest.raises(UnexpectedModelBehavior, match='Received empty tool call message'):
        async with agent.run_stream('hello'):
            pass


async def test_call_tool_wrong_name():
    async def stream_structured_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='foobar', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    @agent.tool_plain
    async def ret_a(x: str) -> str:  # pragma: no cover
        return x

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
        async with agent.run_stream('hello'):
            pass
    assert agent.last_run_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args=ArgsJson(args_json='{}'))],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Unknown tool name: 'foobar'. Available tools: ret_a, final_result",
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


class ResultType(BaseModel):
    """Result type used by all tests."""

    value: str


async def test_early_strategy_stops_after_first_final_result():
    """Test that 'early' strategy stops processing regular tools after first final result."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.result_tools is not None
        yield {1: DeltaToolCall('final_result', '{"value": "final"}')}
        yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}

    agent = Agent(FunctionModel(stream_function=sf), result_type=ResultType, end_strategy='early')

    @agent.tool_plain
    def regular_tool(x: int) -> int:  # pragma: no cover
        """A regular tool that should not be called."""
        tool_called.append('regular_tool')
        return x

    @agent.tool_plain
    def another_tool(y: int) -> int:  # pragma: no cover
        """Another tool that should not be called."""
        tool_called.append('another_tool')
        return y

    async with agent.run_stream('test early strategy') as result:
        response = await result.get_data()
        assert response.value == snapshot('final')
        messages = result.all_messages()

    # Verify no tools were called after final result
    assert tool_called == []

    # Verify we got tool returns for all calls
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='test early strategy', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"value": "final"}')),
                    ToolCallPart(tool_name='regular_tool', args=ArgsJson(args_json='{"x": 1}')),
                    ToolCallPart(tool_name='another_tool', args=ArgsJson(args_json='{"y": 2}')),
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    ToolReturnPart(
                        tool_name='regular_tool',
                        content='Tool not executed - a final result was already processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    ToolReturnPart(
                        tool_name='another_tool',
                        content='Tool not executed - a final result was already processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
        ]
    )


async def test_early_strategy_uses_first_final_result():
    """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.result_tools is not None
        yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
        yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

    agent = Agent(FunctionModel(stream_function=sf), result_type=ResultType, end_strategy='early')

    async with agent.run_stream('test multiple final results') as result:
        response = await result.get_data()
        assert response.value == snapshot('first')
        messages = result.all_messages()

    # Verify we got appropriate tool returns
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"value": "first"}')),
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"value": "second"}')),
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    ),
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Result tool not used - a final result was already processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
        ]
    )


async def test_exhaustive_strategy_executes_all_tools():
    """Test that 'exhaustive' strategy executes all tools while using first final result."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.result_tools is not None
        yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
        yield {2: DeltaToolCall('regular_tool', '{"x": 42}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
        yield {4: DeltaToolCall('final_result', '{"value": "second"}')}
        yield {5: DeltaToolCall('unknown_tool', '{"value": "???"}')}

    agent = Agent(FunctionModel(stream_function=sf), result_type=ResultType, end_strategy='exhaustive')

    @agent.tool_plain
    def regular_tool(x: int) -> int:
        """A regular tool that should be called."""
        tool_called.append('regular_tool')
        return x

    @agent.tool_plain
    def another_tool(y: int) -> int:
        """Another tool that should be called."""
        tool_called.append('another_tool')
        return y

    async with agent.run_stream('test exhaustive strategy') as result:
        response = await result.get_data()
        assert response.value == snapshot('first')
        messages = result.all_messages()

    # Verify we got tool returns in the correct order
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"value": "first"}')),
                    ToolCallPart(tool_name='regular_tool', args=ArgsJson(args_json='{"x": 42}')),
                    ToolCallPart(tool_name='another_tool', args=ArgsJson(args_json='{"y": 2}')),
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"value": "second"}')),
                    ToolCallPart(tool_name='unknown_tool', args=ArgsJson(args_json='{"value": "???"}')),
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Result tool not used - a final result was already processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    RetryPromptPart(
                        content="Unknown tool name: 'unknown_tool'. Available tools: regular_tool, another_tool, final_result",
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    ToolReturnPart(tool_name='regular_tool', content=42, timestamp=IsNow(tz=timezone.utc)),
                    ToolReturnPart(tool_name='another_tool', content=2, timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
        ]
    )


@pytest.mark.xfail(reason='final result tool not first is not yet supported')
async def test_early_strategy_with_final_result_in_middle():
    """Test that 'early' strategy stops at first final result, regardless of position."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.result_tools is not None
        yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}
        yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
        yield {4: DeltaToolCall('unknown_tool', '{"value": "???"}')}

    agent = Agent(FunctionModel(stream_function=sf), result_type=ResultType, end_strategy='early')

    @agent.tool_plain
    def regular_tool(x: int) -> int:  # pragma: no cover
        """A regular tool that should not be called."""
        tool_called.append('regular_tool')
        return x

    @agent.tool_plain
    def another_tool(y: int) -> int:  # pragma: no cover
        """A tool that should not be called."""
        tool_called.append('another_tool')
        return y

    async with agent.run_stream('test early strategy with final result in middle') as result:
        response = await result.get_data()
        assert response.value == snapshot('first')
        messages = result.all_messages()

    # Verify no tools were called
    assert tool_called == []

    # Verify we got appropriate tool returns
    assert messages == snapshot()


async def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool():
    """Test that 'early' strategy does not apply to tool calls without final tool."""
    tool_called: list[str] = []
    agent = Agent(TestModel(), result_type=ResultType, end_strategy='early')

    @agent.tool_plain
    def regular_tool(x: int) -> int:
        """A regular tool that should be called."""
        tool_called.append('regular_tool')
        return x

    async with agent.run_stream('test early strategy with regular tool calls') as result:
        response = await result.get_data()
        assert response.value == snapshot('a')
        messages = result.all_messages()

    assert tool_called == ['regular_tool']

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='test early strategy with regular tool calls', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='regular_tool', args=ArgsDict(args_dict={'x': 0}))],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='regular_tool', content=0, timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args=ArgsDict(args_dict={'value': 'a'}))],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )
