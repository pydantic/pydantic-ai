"""Tests for AG-UI implementation."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from dirty_equals import IsStr
from pydantic import BaseModel

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.function import (
    AgentInfo,
    BuiltinToolCallsReturns,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT, ToolDefinition

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsSameStr, try_import

with try_import() as imports_successful:
    from ag_ui.core import (
        ActivityMessage,
        ActivitySnapshotEvent,
        AssistantMessage,
        BaseEvent,
        BinaryInputContent,
        CustomEvent,
        DeveloperMessage,
        EventType,
        FunctionCall,
        Message,
        RunAgentInput,
        StateSnapshotEvent,
        SystemMessage,
        TextInputContent,
        ThinkingEndEvent,
        Tool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from pydantic_ai.ag_ui import (
        SSE_CONTENT_TYPE,
        AGUIAdapter,
        OnCompleteFunc,
        StateDeps,
        handle_ag_ui_request,
        run_ag_ui,
    )
    from pydantic_ai.ui.ag_ui import AGUIEventStream


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='ag-ui-protocol not installed'),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


def simple_result() -> Any:
    return snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': 'success '},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': '(no tool calls)',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def run_and_collect_events(
    agent: Agent[AgentDepsT, OutputDataT],
    *run_inputs: RunAgentInput,
    deps: AgentDepsT = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
) -> list[dict[str, Any]]:
    events = list[dict[str, Any]]()
    for run_input in run_inputs:
        async for event in run_ag_ui(agent, run_input, deps=deps, on_complete=on_complete):
            events.append(json.loads(event.removeprefix('data: ')))
    return events


class StateInt(BaseModel):
    """Example state class for testing purposes."""

    value: int = 0


def get_weather(name: str = 'get_weather') -> Tool:
    return Tool(
        name=name,
        description='Get the weather for a given location',
        parameters={
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The location to get the weather for',
                },
            },
            'required': ['location'],
        },
    )


def current_time() -> str:
    """Get the current time in ISO format.

    Returns:
        The current UTC time in ISO format string.
    """
    return '2023-06-21T12:08:45.485981+00:00'


async def send_snapshot() -> StateSnapshotEvent:
    """Display the recipe to the user.

    Returns:
        StateSnapshotEvent.
    """
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot={'key': 'value'},
    )


async def send_custom() -> ToolReturn:
    return ToolReturn(
        return_value='Done',
        metadata=[
            CustomEvent(
                type=EventType.CUSTOM,
                name='custom_event1',
                value={'key1': 'value1'},
            ),
            CustomEvent(
                type=EventType.CUSTOM,
                name='custom_event2',
                value={'key2': 'value2'},
            ),
        ],
    )


def uuid_str() -> str:
    """Generate a random UUID string."""
    return uuid.uuid4().hex


def create_input(
    *messages: Message, tools: list[Tool] | None = None, thread_id: str | None = None, state: Any = None
) -> RunAgentInput:
    """Create a RunAgentInput for testing."""
    thread_id = thread_id or uuid_str()
    return RunAgentInput(
        thread_id=thread_id,
        run_id=uuid_str(),
        messages=list(messages),
        state=dict(state) if state else {},
        context=[],
        tools=tools or [],
        forwarded_props=None,
    )


async def simple_stream(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
    """A simple function that returns a text response without tool calls."""
    yield 'success '
    yield '(no tool calls)'


async def test_agui_adapter_state_none() -> None:
    """Ensure adapter exposes `None` state when no frontend state provided."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )

    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        messages=[],
        state=None,
        context=[],
        tools=[],
        forwarded_props=None,
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=None)

    assert adapter.state is None


async def test_basic_user_message() -> None:
    """Test basic user message with text response."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        )
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == simple_result()


async def test_empty_messages() -> None:
    """Test handling of empty messages."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError
        yield 'no messages'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
    )

    run_input = create_input()
    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': IsStr(),
                'runId': IsStr(),
            },
            {
                'type': 'RUN_ERROR',
                'timestamp': IsInt(),
                'message': 'No message history, user prompt, or instructions provided',
            },
        ]
    )


async def test_multiple_messages() -> None:
    """Test with multiple different message types."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='First message',
        ),
        AssistantMessage(
            id='msg_2',
            content='Assistant response',
        ),
        SystemMessage(
            id='msg_3',
            content='System message',
        ),
        DeveloperMessage(
            id='msg_4',
            content='Developer note',
        ),
        UserMessage(
            id='msg_5',
            content='Second message',
        ),
        ActivityMessage(
            id='msg_6',
            activity_type='testing',
            content={
                'test_field': None,
            },
        ),
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == simple_result()


async def test_messages_with_history() -> None:
    """Test with multiple user messages (conversation history)."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='First message',
        ),
        UserMessage(
            id='msg_2',
            content='Second message',
        ),
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == simple_result()


async def test_tool_ag_ui() -> None:
    """Test AG-UI tool call."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First call - make a tool call
            yield {0: DeltaToolCall(name='get_weather', json_args='{"location": ')}
            yield {0: DeltaToolCall(json_args='"Paris"}')}
        else:
            # Second call - return text result
            yield '{"get_weather": "Tool result"}'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        tools=[send_snapshot, send_custom, current_time],
    )

    thread_id = uuid_str()
    run_inputs = [
        create_input(
            UserMessage(
                id='msg_1',
                content='Please call get_weather for Paris',
            ),
            tools=[get_weather()],
            thread_id=thread_id,
        ),
        create_input(
            UserMessage(
                id='msg_1',
                content='Please call get_weather for Paris',
            ),
            AssistantMessage(
                id='msg_2',
                tool_calls=[
                    ToolCall(
                        id='pyd_ai_00000000000000000000000000000003',
                        type='function',
                        function=FunctionCall(
                            name='get_weather',
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_3',
                content='Tool result',
                tool_call_id='pyd_ai_00000000000000000000000000000003',
            ),
            thread_id=thread_id,
        ),
    ]

    events = await run_and_collect_events(agent, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': IsStr(),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': tool_call_id,
                'delta': '{"location": ',
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': '"Paris"}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_tool_ag_ui_multiple() -> None:
    """Test multiple AG-UI tool calls in sequence."""
    run_count = 0

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        nonlocal run_count
        run_count += 1

        if run_count == 1:
            # First run - make multiple tool calls
            yield {0: DeltaToolCall(name='get_weather')}
            yield {0: DeltaToolCall(json_args='{"location": "Paris"}')}
            yield {1: DeltaToolCall(name='get_weather_parts')}
            yield {1: DeltaToolCall(json_args='{"location": "')}
            yield {1: DeltaToolCall(json_args='Paris"}')}
        else:
            # Second run - process tool results
            yield '{"get_weather": "Tool result", "get_weather_parts": "Tool result"}'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
    )

    tool_call_id1 = uuid_str()
    tool_call_id2 = uuid_str()
    run_inputs = [
        (
            first_input := create_input(
                UserMessage(
                    id='msg_1',
                    content='Please call get_weather and get_weather_parts for Paris',
                ),
                tools=[get_weather(), get_weather('get_weather_parts')],
            )
        ),
        create_input(
            UserMessage(
                id='msg_1',
                content='Please call get_weather for Paris',
            ),
            AssistantMessage(
                id='msg_2',
                tool_calls=[
                    ToolCall(
                        id=tool_call_id1,
                        type='function',
                        function=FunctionCall(
                            name='get_weather',
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_3',
                content='Tool result',
                tool_call_id=tool_call_id1,
            ),
            AssistantMessage(
                id='msg_4',
                tool_calls=[
                    ToolCall(
                        id=tool_call_id2,
                        type='function',
                        function=FunctionCall(
                            name='get_weather_parts',
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_5',
                content='Tool result',
                tool_call_id=tool_call_id2,
            ),
            tools=[get_weather(), get_weather('get_weather_parts')],
            thread_id=first_input.thread_id,
        ),
    ]

    events = await run_and_collect_events(agent, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': (parent_message_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': tool_call_id,
                'delta': '{"location": "Paris"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather_parts',
                'parentMessageId': parent_message_id,
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': tool_call_id,
                'delta': '{"location": "',
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': 'Paris"}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result", "get_weather_parts": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_tool_ag_ui_parts() -> None:
    """Test AG-UI tool call with streaming/parts (same as tool_call_with_args_streaming)."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First call - make a tool call with streaming args
            yield {0: DeltaToolCall(name='get_weather')}
            yield {0: DeltaToolCall(json_args='{"location":"')}
            yield {0: DeltaToolCall(json_args='Paris"}')}
        else:
            # Second call - return text result
            yield '{"get_weather": "Tool result"}'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_inputs = [
        (
            first_input := create_input(
                UserMessage(
                    id='msg_1',
                    content='Please call get_weather_parts for Paris',
                ),
                tools=[get_weather('get_weather_parts')],
            )
        ),
        create_input(
            UserMessage(
                id='msg_1',
                content='Please call get_weather_parts for Paris',
            ),
            AssistantMessage(
                id='msg_2',
                tool_calls=[
                    ToolCall(
                        id='pyd_ai_00000000000000000000000000000003',
                        type='function',
                        function=FunctionCall(
                            name='get_weather_parts',
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_3',
                content='Tool result',
                tool_call_id='pyd_ai_00000000000000000000000000000003',
            ),
            tools=[get_weather('get_weather_parts')],
            thread_id=first_input.thread_id,
        ),
    ]
    events = await run_and_collect_events(agent, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': IsStr(),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': tool_call_id,
                'delta': '{"location":"',
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': 'Paris"}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': """\
Unknown tool name: 'get_weather'. Available tools: 'get_weather_parts'

Fix the errors and try again.\
""",
                'role': 'tool',
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_tool_local_single_event() -> None:
    """Test local tool call that returns a single event."""

    encoder = EventEncoder()

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First call - make a tool call
            yield {0: DeltaToolCall(name='send_snapshot')}
            yield {0: DeltaToolCall(json_args='{}')}
        else:
            # Second call - return text result
            yield encoder.encode(await send_snapshot())

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        tools=[send_snapshot],
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call send_snapshot',
        ),
    )
    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'send_snapshot',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': '{"type":"STATE_SNAPSHOT","timestamp":null,"raw_event":null,"snapshot":{"key":"value"}}',
                'role': 'tool',
            },
            {'type': 'STATE_SNAPSHOT', 'timestamp': IsInt(), 'snapshot': {'key': 'value'}},
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': """\
data: {"type":"STATE_SNAPSHOT","snapshot":{"key":"value"}}

""",
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_tool_local_multiple_events() -> None:
    """Test local tool call that returns multiple events."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First call - make a tool call
            yield {0: DeltaToolCall(name='send_custom')}
            yield {0: DeltaToolCall(json_args='{}')}
        else:
            # Second call - return text result
            yield 'success send_custom called'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        tools=[send_custom],
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call send_custom',
        ),
    )
    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'send_custom',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': 'Done',
                'role': 'tool',
            },
            {'type': 'CUSTOM', 'timestamp': IsInt(), 'name': 'custom_event1', 'value': {'key1': 'value1'}},
            {'type': 'CUSTOM', 'timestamp': IsInt(), 'name': 'custom_event2', 'value': {'key2': 'value2'}},
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'success send_custom called',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_tool_local_parts() -> None:
    """Test local tool call with streaming/parts."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First call - make a tool call with streaming args
            yield {0: DeltaToolCall(name='current_time')}
            yield {0: DeltaToolCall(json_args='{}')}
        else:
            # Second call - return text result
            yield 'success current_time called'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        tools=[send_snapshot, send_custom, current_time],
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call current_time',
        ),
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'current_time',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': '2023-06-21T12:08:45.485981+00:00',
                'role': 'tool',
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'success current_time called',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_thinking() -> None:
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='')}
        yield "Let's do some thinking"
        yield ''
        yield ' and some more'
        yield {1: DeltaThinkingPart(content='Thinking ')}
        yield {1: DeltaThinkingPart(content='about the weather')}
        yield {2: DeltaThinkingPart(content='')}
        yield {3: DeltaThinkingPart(content='')}
        yield {3: DeltaThinkingPart(content='Thinking about the meaning of life')}
        yield {4: DeltaThinkingPart(content='Thinking about the universe')}

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Think about the weather',
        ),
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            {
                'type': 'ACTIVITY_SNAPSHOT',
                'timestamp': IsInt(),
                'activityType': 'pydantic_ai_thinking',
                'messageId': IsStr(),
                'content': {'content': ''},
                'replace': True,
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': "Let's do some thinking",
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': ' and some more',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'Thinking '},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'about the weather'},
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {
                'type': 'THINKING_TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'delta': 'Thinking about the meaning of life',
            },
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {
                'type': 'THINKING_TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'delta': 'Thinking about the universe',
            },
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            {
                'type': 'ACTIVITY_SNAPSHOT',
                'timestamp': IsInt(),
                'activityType': 'pydantic_ai_thinking',
                'messageId': IsStr(),
                'content': {'content': 'Thinking about the universe'},
                'replace': True,
            },
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_thinking_with_signature() -> None:
    """Test that ActivitySnapshotEvent is emitted after ThinkingEndEvent with metadata."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Thinking deeply', signature='sig_abc123')}
        yield 'Here is my response'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_input = create_input(
        UserMessage(id='msg_1', content='Think about something'),
    )

    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'Thinking deeply'},
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            {
                'type': 'ACTIVITY_SNAPSHOT',
                'timestamp': IsInt(),
                'activityType': 'pydantic_ai_thinking',
                'messageId': IsStr(),
                'content': {
                    'content': 'Thinking deeply',
                    'signature': 'sig_abc123',
                    'provider_name': 'function',
                },
                'replace': True,
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'Here is my response',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {'type': 'RUN_FINISHED', 'timestamp': IsInt(), 'threadId': thread_id, 'runId': run_id},
        ]
    )


def test_activity_message_thinking_roundtrip() -> None:
    """Test that ActivityMessage with pydantic_ai_thinking converts to ThinkingPart."""
    messages = AGUIAdapter.load_messages(
        [
            ActivityMessage(
                id='activity-1',
                activity_type='pydantic_ai_thinking',
                content={
                    'content': 'Let me think about this...',
                    'id': 'thinking-1',
                    'signature': 'sig_abc123',
                    'provider_name': 'anthropic',
                    'provider_details': {'some': 'details'},
                },
            ),
            AssistantMessage(id='msg-1', content='Here is my response'),
        ]
    )

    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Let me think about this...',
                        id='thinking-1',
                        signature='sig_abc123',
                        provider_name='anthropic',
                        provider_details={'some': 'details'},
                    ),
                    TextPart(content='Here is my response'),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_thinking_end_event_with_all_metadata() -> None:
    """Test that ActivitySnapshotEvent includes all metadata fields (id, signature, provider_name, provider_details)."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE)

    part = ThinkingPart(
        content='Thinking content',
        id='thinking-123',
        signature='sig_xyz',
        provider_name='anthropic',
        provider_details={'model': 'claude-sonnet-4-5'},
    )

    events = [e async for e in event_stream.handle_thinking_end(part, followed_by_thinking=False)]

    assert events == snapshot(
        [
            ThinkingEndEvent(),
            ActivitySnapshotEvent(
                message_id='thinking-123',
                activity_type='pydantic_ai_thinking',
                content={
                    'content': 'Thinking content',
                    'id': 'thinking-123',
                    'signature': 'sig_xyz',
                    'provider_name': 'anthropic',
                    'provider_details': {'model': 'claude-sonnet-4-5'},
                },
            ),
        ]
    )


def test_activity_message_other_types_ignored() -> None:
    """Test that ActivityMessage with other activity types are ignored."""
    messages = AGUIAdapter.load_messages(
        [
            ActivityMessage(
                id='activity-1',
                activity_type='some_other_activity',
                content={'foo': 'bar'},
            ),
            AssistantMessage(id='msg-1', content='Response'),
        ]
    )

    assert messages == snapshot([ModelResponse(parts=[TextPart(content='Response')], timestamp=IsDatetime())])


def _sync_part_timestamps(original_part: Any, new_part: Any) -> None:
    """Sync timestamp attribute if both parts have it."""
    if hasattr(new_part, 'timestamp') and hasattr(original_part, 'timestamp'):
        object.__setattr__(new_part, 'timestamp', original_part.timestamp)


def _sync_timestamps(original: list[ModelMessage], reloaded: list[ModelMessage]) -> None:
    """Sync timestamps between original and reloaded messages for comparison."""
    for o, n in zip(original, reloaded):
        if isinstance(n, ModelResponse) and isinstance(o, ModelResponse):
            n.timestamp = o.timestamp
            for op, np in zip(o.parts, n.parts):
                _sync_part_timestamps(op, np)
        elif isinstance(n, ModelRequest) and isinstance(o, ModelRequest):  # pragma: no branch
            for op, np in zip(o.parts, n.parts):
                _sync_part_timestamps(op, np)


def test_dump_load_roundtrip_basic() -> None:
    """Test that load_messages(dump_messages(msgs)) preserves basic messages."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='You are helpful'), UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart(content='Hi!')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_thinking() -> None:
    """Test full round-trip for thinking parts with all metadata."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Think about this')]),
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='Deep thoughts...',
                    id='think-001',
                    signature='sig_xyz',
                    provider_name='anthropic',
                    provider_details={'model': 'claude-sonnet-4-5'},
                ),
                TextPart(content='Conclusion'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_tools() -> None:
    """Test full round-trip for tool calls and returns."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Call tool')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', tool_call_id='call_abc', args='{"x": 1}')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', tool_call_id='call_abc', content='result')]),
        ModelResponse(parts=[TextPart(content='Done')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_multiple_thinking_parts() -> None:
    """Test round-trip preserves multiple ThinkingParts with their metadata."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Think hard')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='First thought', id='think-1', signature='sig_1'),
                ThinkingPart(content='Second thought', id='think-2', signature='sig_2'),
                TextPart(content='Final answer'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_binary_content() -> None:
    """Test round-trip for binary content in user prompts (images, documents, etc.)."""
    original: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Describe this image',
                        ImageUrl(url='https://example.com/image.png', media_type='image/png'),
                        BinaryContent(data=b'raw image data', media_type='image/jpeg'),
                    ]
                ),
            ]
        ),
        ModelResponse(parts=[TextPart(content='I see an image.')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_file_part() -> None:
    """Test round-trip for FilePart in model responses.

    Note: BinaryImage is used because from_data_uri() returns BinaryImage for image/* media types.
    """
    file_data = b'generated file content'
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Generate an image')]),
        ModelResponse(
            parts=[
                FilePart(
                    content=BinaryImage(data=file_data, media_type='image/png'),
                    id='file-001',
                    provider_name='openai',
                    provider_details={'model': 'gpt-image'},
                ),
                TextPart(content='Here is your generated image.'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_builtin_tool_return() -> None:
    """Test round-trip for builtin tool calls with their return values.

    Note: The round-trip reorders parts within ModelResponse because AG-UI's AssistantMessage
    has separate content and tool_calls fields. TextPart comes first (from content), then
    BuiltinToolCallPart (from tool_calls), then BuiltinToolReturnPart (from subsequent ToolMessage).
    """
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search for info')]),
        ModelResponse(
            parts=[
                TextPart(content='Based on the search...'),
                BuiltinToolCallPart(
                    tool_name='web_search',
                    tool_call_id='call_123',
                    args='{"query": "test"}',
                    provider_name='anthropic',
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    tool_call_id='call_123',
                    content='Search results here',
                    provider_name='anthropic',
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_cache_point() -> None:
    """Test that CachePoint is filtered out during round-trip (it's metadata only)."""
    original: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content=['Hello', CachePoint(), 'world']),
            ]
        ),
        ModelResponse(parts=[TextPart(content='Hi!')]),
    ]
    expected: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['Hello', 'world'])]),
        ModelResponse(parts=[TextPart(content='Hi!')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(expected, reloaded)

    assert reloaded == expected


def test_dump_load_roundtrip_retry_prompt_with_tool() -> None:
    """Test round-trip for RetryPromptPart with tool_name (converted to ToolMessage with error)."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Call tool')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', tool_call_id='call_1', args='{}')]),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    tool_name='my_tool',
                    tool_call_id='call_1',
                    content='Invalid args',
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content='OK')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    # RetryPromptPart becomes ToolReturnPart on reload (same tool_call_id mapping)
    assert len(reloaded) == 4
    assert isinstance(reloaded[2], ModelRequest)
    retry_part = reloaded[2].parts[0]
    assert isinstance(retry_part, ToolReturnPart)
    assert retry_part.tool_name == 'my_tool'
    assert retry_part.tool_call_id == 'call_1'


def test_dump_load_roundtrip_retry_prompt_without_tool() -> None:
    """Test round-trip for RetryPromptPart without tool_name (converted to UserMessage)."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Do something')]),
        ModelResponse(parts=[TextPart(content='Done')]),
        ModelRequest(parts=[RetryPromptPart(content='Please try again')]),
        ModelResponse(parts=[TextPart(content='OK')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    # RetryPromptPart without tool becomes UserPromptPart on reload
    # Content is formatted by RetryPromptPart.model_response()
    assert len(reloaded) == 4
    assert isinstance(reloaded[2], ModelRequest)
    retry_part = reloaded[2].parts[0]
    assert isinstance(retry_part, UserPromptPart)
    assert 'Please try again' in str(retry_part.content)


def test_dump_load_roundtrip_file_part_minimal() -> None:
    """Test round-trip for FilePart without optional attributes (id, provider_name, provider_details)."""
    file_data = b'minimal file'
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Generate')]),
        ModelResponse(
            parts=[
                FilePart(content=BinaryImage(data=file_data, media_type='image/png')),
                TextPart(content='Done'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_file_part_only() -> None:
    """Test round-trip for response with only FilePart (no text, no tool calls)."""
    file_data = b'only file'
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Generate image only')]),
        ModelResponse(parts=[FilePart(content=BinaryImage(data=file_data, media_type='image/png'))]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


async def test_tool_local_then_ag_ui() -> None:
    """Test mixed local and AG-UI tool calls."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            # First - call local tool (current_time)
            yield {0: DeltaToolCall(name='current_time')}
            yield {0: DeltaToolCall(json_args='{}')}
            # Then - call AG-UI tool (get_weather)
            yield {1: DeltaToolCall(name='get_weather')}
            yield {1: DeltaToolCall(json_args='{"location": "Paris"}')}
        else:
            # Final response with results
            yield 'current time is 2023-06-21T12:08:45.485981+00:00 and the weather in Paris is bright and sunny'

    tool_call_id1 = uuid_str()
    tool_call_id2 = uuid_str()
    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        tools=[current_time],
    )

    run_inputs = [
        (
            first_input := create_input(
                UserMessage(
                    id='msg_1',
                    content='Please tell me the time and then call get_weather for Paris',
                ),
                tools=[get_weather()],
            )
        ),
        create_input(
            UserMessage(
                id='msg_1',
                content='Please call get_weather for Paris',
            ),
            AssistantMessage(
                id='msg_2',
                tool_calls=[
                    ToolCall(
                        id=tool_call_id1,
                        type='function',
                        function=FunctionCall(
                            name='current_time',
                            arguments='{}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_3',
                content='Tool result',
                tool_call_id=tool_call_id1,
            ),
            AssistantMessage(
                id='msg_4',
                tool_calls=[
                    ToolCall(
                        id=tool_call_id2,
                        type='function',
                        function=FunctionCall(
                            name='get_weather',
                            arguments='{"location": "Paris"}',
                        ),
                    ),
                ],
            ),
            ToolMessage(
                id='msg_5',
                content='Bright and sunny',
                tool_call_id=tool_call_id2,
            ),
            tools=[get_weather()],
            thread_id=first_input.thread_id,
        ),
    ]
    events = await run_and_collect_events(agent, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (first_tool_call_id := IsSameStr()),
                'toolCallName': 'current_time',
                'parentMessageId': (parent_message_id := IsSameStr()),
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': first_tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': first_tool_call_id},
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (second_tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': parent_message_id,
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': second_tool_call_id,
                'delta': '{"location": "Paris"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': second_tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': first_tool_call_id,
                'content': '2023-06-21T12:08:45.485981+00:00',
                'role': 'tool',
            },
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'current time is 2023-06-21T12:08:45.485981+00:00 and the weather in Paris is bright and sunny',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_request_with_state() -> None:
    """Test request with state modification."""

    seen_states: list[int] = []

    async def store_state(
        ctx: RunContext[StateDeps[StateInt]], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        seen_states.append(ctx.deps.state.value)
        ctx.deps.state.value += 1
        return tool_defs

    agent: Agent[StateDeps[StateInt], str] = Agent(
        model=FunctionModel(stream_function=simple_stream),
        deps_type=StateDeps[StateInt],
        prepare_tools=store_state,
    )

    run_inputs = [
        create_input(
            UserMessage(
                id='msg_1',
                content='Hello, how are you?',
            ),
            state=StateInt(value=41),
        ),
        create_input(
            UserMessage(
                id='msg_2',
                content='Hello, how are you?',
            ),
        ),
        create_input(
            UserMessage(
                id='msg_3',
                content='Hello, how are you?',
            ),
        ),
        create_input(
            UserMessage(
                id='msg_4',
                content='Hello, how are you?',
            ),
            state=StateInt(value=42),
        ),
    ]

    seen_deps_states: list[int] = []

    for run_input in run_inputs:
        events = list[dict[str, Any]]()
        deps = StateDeps(StateInt(value=0))

        async def on_complete(result: AgentRunResult[Any]):
            seen_deps_states.append(deps.state.value)

        async for event in run_ag_ui(agent, run_input, deps=deps, on_complete=on_complete):
            events.append(json.loads(event.removeprefix('data: ')))

        assert events == simple_result()
    assert seen_states == snapshot([41, 0, 0, 42])
    assert seen_deps_states == snapshot([42, 1, 1, 43])


async def test_request_with_state_without_handler() -> None:
    agent = Agent(model=FunctionModel(stream_function=simple_stream))

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        ),
        state=StateInt(value=41),
    )

    with pytest.warns(
        UserWarning,
        match='State was provided but `deps` of type `NoneType` does not implement the `StateHandler` protocol, so the state was ignored. Use `StateDeps\\[\\.\\.\\.\\]` or implement `StateHandler` to receive AG-UI state.',
    ):
        events = list[dict[str, Any]]()
        async for event in run_ag_ui(agent, run_input):
            events.append(json.loads(event.removeprefix('data: ')))

    assert events == simple_result()


async def test_request_with_empty_state_without_handler() -> None:
    agent = Agent(model=FunctionModel(stream_function=simple_stream))

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        ),
        state={},
    )

    events = list[dict[str, Any]]()
    async for event in run_ag_ui(agent, run_input):
        events.append(json.loads(event.removeprefix('data: ')))

    assert events == simple_result()


async def test_request_with_state_with_custom_handler() -> None:
    @dataclass
    class CustomStateDeps:
        state: dict[str, Any]

    seen_states: list[dict[str, Any]] = []

    async def store_state(ctx: RunContext[CustomStateDeps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen_states.append(ctx.deps.state)
        return tool_defs

    agent: Agent[CustomStateDeps, str] = Agent(
        model=FunctionModel(stream_function=simple_stream),
        deps_type=CustomStateDeps,
        prepare_tools=store_state,
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        ),
        state={'value': 42},
    )

    async for _ in run_ag_ui(agent, run_input, deps=CustomStateDeps(state={'value': 0})):
        pass

    assert seen_states[-1] == {'value': 42}


async def test_concurrent_runs() -> None:
    """Test concurrent execution of multiple runs."""
    import asyncio

    agent: Agent[StateDeps[StateInt], str] = Agent(
        model=TestModel(),
        deps_type=StateDeps[StateInt],
    )

    @agent.tool
    async def get_state(ctx: RunContext[StateDeps[StateInt]]) -> int:
        return ctx.deps.state.value

    concurrent_tasks: list[asyncio.Task[list[dict[str, Any]]]] = []

    for i in range(5):  # Test with 5 concurrent runs
        run_input = create_input(
            UserMessage(
                id=f'msg_{i}',
                content=f'Message {i}',
            ),
            state=StateInt(value=i),
            thread_id=f'test_thread_{i}',
        )

        task = asyncio.create_task(run_and_collect_events(agent, run_input, deps=StateDeps(StateInt())))
        concurrent_tasks.append(task)

    results = await asyncio.gather(*concurrent_tasks)

    # Verify all runs completed successfully
    for i, events in enumerate(results):
        assert events == [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': f'test_thread_{i}',
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_state',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': str(i),
                'role': 'tool',
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': '{"get_s'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'tate":' + str(i) + '}',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {'type': 'RUN_FINISHED', 'timestamp': IsInt(), 'threadId': f'test_thread_{i}', 'runId': run_id},
        ]


@pytest.mark.anyio
async def test_to_ag_ui() -> None:
    """Test the agent.to_ag_ui method."""

    agent = Agent(model=FunctionModel(stream_function=simple_stream), deps_type=StateDeps[StateInt])

    deps = StateDeps(StateInt(value=0))
    app = agent.to_ag_ui(deps=deps)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://localhost:8000'
            run_input = create_input(
                UserMessage(
                    id='msg_1',
                    content='Hello, world!',
                ),
                state=StateInt(value=42),
            )
            async with client.stream(
                'POST',
                '/',
                content=run_input.model_dump_json(),
                headers={'Content-Type': 'application/json', 'Accept': SSE_CONTENT_TYPE},
            ) as response:
                assert response.status_code == HTTPStatus.OK, f'Unexpected status code: {response.status_code}'
                events: list[dict[str, Any]] = []
                async for line in response.aiter_lines():
                    if line:
                        events.append(json.loads(line.removeprefix('data: ')))

            assert events == simple_result()

    # Verify the state was not mutated by the run
    assert deps.state.value == 0


async def test_callback_sync() -> None:
    """Test that sync callbacks work correctly."""

    captured_results: list[AgentRunResult[Any]] = []

    def sync_callback(run_result: AgentRunResult[Any]) -> None:
        captured_results.append(run_result)

    agent = Agent(TestModel())
    run_input = create_input(
        UserMessage(
            id='msg1',
            content='Hello!',
        )
    )

    events = await run_and_collect_events(agent, run_input, on_complete=sync_callback)

    # Verify callback was called
    assert len(captured_results) == 1
    run_result = captured_results[0]

    # Verify we can access messages
    messages = run_result.all_messages()
    assert len(messages) >= 1

    # Verify events were still streamed normally
    assert len(events) > 0
    assert events[0]['type'] == 'RUN_STARTED'
    assert events[-1]['type'] == 'RUN_FINISHED'


async def test_callback_async() -> None:
    """Test that async callbacks work correctly."""

    captured_results: list[AgentRunResult[Any]] = []

    async def async_callback(run_result: AgentRunResult[Any]) -> None:
        captured_results.append(run_result)

    agent = Agent(TestModel())
    run_input = create_input(
        UserMessage(
            id='msg1',
            content='Hello!',
        )
    )

    events = await run_and_collect_events(agent, run_input, on_complete=async_callback)

    # Verify callback was called
    assert len(captured_results) == 1
    run_result = captured_results[0]

    # Verify we can access messages
    messages = run_result.all_messages()
    assert len(messages) >= 1

    # Verify events were still streamed normally
    assert len(events) > 0
    assert events[0]['type'] == 'RUN_STARTED'
    assert events[-1]['type'] == 'RUN_FINISHED'


async def test_messages(image_content: BinaryContent, document_content: BinaryContent) -> None:
    messages = [
        SystemMessage(
            id='msg_1',
            content='System message',
        ),
        DeveloperMessage(
            id='msg_2',
            content='Developer message',
        ),
        UserMessage(
            id='msg_3',
            content='User message',
        ),
        UserMessage(
            id='msg_4',
            content='User message',
        ),
        UserMessage(
            id='msg_1',
            content=[
                TextInputContent(text='this is an image:'),
                BinaryInputContent(url=image_content.data_uri, mime_type=image_content.media_type),
            ],
        ),
        UserMessage(
            id='msg2',
            content=[BinaryInputContent(url='http://example.com/image.png', mime_type='image/png')],
        ),
        UserMessage(
            id='msg3',
            content=[BinaryInputContent(url='http://example.com/video.mp4', mime_type='video/mp4')],
        ),
        UserMessage(
            id='msg4',
            content=[BinaryInputContent(url='http://example.com/audio.mp3', mime_type='audio/mpeg')],
        ),
        UserMessage(
            id='msg5',
            content=[BinaryInputContent(url='http://example.com/doc.pdf', mime_type='application/pdf')],
        ),
        UserMessage(
            id='msg6', content=[BinaryInputContent(data=document_content.base64, mime_type=document_content.media_type)]
        ),
        AssistantMessage(
            id='msg_5',
            tool_calls=[
                ToolCall(
                    id='pyd_ai_builtin|function|search_1',
                    function=FunctionCall(
                        name='web_search',
                        arguments='{"query": "Hello, world!"}',
                    ),
                ),
            ],
        ),
        ToolMessage(
            id='msg_6',
            content='{"results": [{"title": "Hello, world!", "url": "https://en.wikipedia.org/wiki/Hello,_world!"}]}',
            tool_call_id='pyd_ai_builtin|function|search_1',
        ),
        AssistantMessage(
            id='msg_7',
            content='Assistant message',
        ),
        AssistantMessage(
            id='msg_8',
            tool_calls=[
                ToolCall(
                    id='tool_call_1',
                    function=FunctionCall(
                        name='tool_call_1',
                        arguments='{}',
                    ),
                ),
            ],
        ),
        AssistantMessage(
            id='msg_9',
            tool_calls=[
                ToolCall(
                    id='tool_call_2',
                    function=FunctionCall(
                        name='tool_call_2',
                        arguments='{}',
                    ),
                ),
            ],
        ),
        ToolMessage(
            id='msg_10',
            content='Tool message',
            tool_call_id='tool_call_1',
        ),
        ToolMessage(
            id='msg_11',
            content='Tool message',
            tool_call_id='tool_call_2',
        ),
        UserMessage(
            id='msg_12',
            content='User message',
        ),
        AssistantMessage(
            id='msg_13',
            content='Assistant message',
        ),
    ]

    assert AGUIAdapter.load_messages(messages) == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='System message',
                        timestamp=IsDatetime(),
                    ),
                    SystemPromptPart(
                        content='Developer message',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='User message',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='User message',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=['this is an image:', image_content],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            ImageUrl(
                                url='http://example.com/image.png', _media_type='image/png', media_type='image/png'
                            )
                        ],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            VideoUrl(
                                url='http://example.com/video.mp4', _media_type='video/mp4', media_type='video/mp4'
                            )
                        ],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            AudioUrl(
                                url='http://example.com/audio.mp3', _media_type='audio/mpeg', media_type='audio/mpeg'
                            )
                        ],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            DocumentUrl(
                                url='http://example.com/doc.pdf',
                                _media_type='application/pdf',
                                media_type='application/pdf',
                            )
                        ],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[document_content],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args='{"query": "Hello, world!"}',
                        tool_call_id='search_1',
                        provider_name='function',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='{"results": [{"title": "Hello, world!", "url": "https://en.wikipedia.org/wiki/Hello,_world!"}]}',
                        tool_call_id='search_1',
                        timestamp=IsDatetime(),
                        provider_name='function',
                    ),
                    TextPart(content='Assistant message'),
                    ToolCallPart(tool_name='tool_call_1', args='{}', tool_call_id='tool_call_1'),
                    ToolCallPart(tool_name='tool_call_2', args='{}', tool_call_id='tool_call_2'),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='tool_call_1',
                        content='Tool message',
                        tool_call_id='tool_call_1',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='tool_call_2',
                        content='Tool message',
                        tool_call_id='tool_call_2',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='User message',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Assistant message')],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_builtin_tool_call() -> None:
    """Test back-to-back builtin tool calls share the same parent_message_id.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4098:
    When a model performs multiple builtin tool calls (e.g. web searches) in
    the same response, the BuiltinToolReturn handling would mutate the shared
    message_id, causing subsequent tool calls to reference a non-existent
    parent message.
    """

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        yield {
            0: BuiltinToolCallPart(
                tool_name=WebSearchTool.kind,
                args='{"query":',
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield {
            0: DeltaToolCall(
                json_args='"Hello world"}',
                tool_call_id='search_1',
            )
        }
        yield {
            1: BuiltinToolReturnPart(
                tool_name=WebSearchTool.kind,
                content={
                    'results': [
                        {
                            'title': '"Hello, World!" program',
                            'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                        }
                    ]
                },
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield {
            2: BuiltinToolCallPart(
                tool_name=WebSearchTool.kind,
                args='{"query": "Hello world history"}',
                tool_call_id='search_2',
                provider_name='function',
            )
        }
        yield {
            3: BuiltinToolReturnPart(
                tool_name=WebSearchTool.kind,
                content={
                    'results': [
                        {
                            'title': 'History of Hello World',
                            'url': 'https://en.wikipedia.org/wiki/Hello_World_history',
                        }
                    ]
                },
                tool_call_id='search_2',
                provider_name='function',
            )
        }
        yield 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". '

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Tell me about Hello World',
        ),
    )
    events = await run_and_collect_events(agent, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'toolCallName': 'web_search',
                'parentMessageId': (parent_message_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'delta': '{"query":',
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'delta': '"Hello world"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'pyd_ai_builtin|function|search_1'},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'content': '{"results":[{"title":"\\"Hello, World!\\" program","url":"https://en.wikipedia.org/wiki/%22Hello,_World!%22_program"}]}',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'pyd_ai_builtin|function|search_2',
                'toolCallName': 'web_search',
                'parentMessageId': parent_message_id,
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'pyd_ai_builtin|function|search_2',
                'delta': '{"query": "Hello world history"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'pyd_ai_builtin|function|search_2'},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'pyd_ai_builtin|function|search_2',
                'content': '{"results":[{"title":"History of Hello World","url":"https://en.wikipedia.org/wiki/Hello_World_history"}]}',
                'role': 'tool',
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': message_id,
                'delta': 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_event_stream_back_to_back_text():
    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'), next_part_kind='text')
        yield PartStartEvent(index=1, part=TextPart(content='Goodbye'), previous_part_kind='text')
        yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=1, part=TextPart(content='Goodbye world'))

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Tell me about Hello World',
        ),
    )
    event_stream = AGUIEventStream(run_input=run_input)
    events = [
        json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': 'Hello'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': 'Goodbye'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_event_stream_multiple_responses_with_tool_calls():
    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'), next_part_kind='tool-call')

        yield PartStartEvent(
            index=1,
            part=ToolCallPart(tool_name='tool_call_1', args='{}', tool_call_id='tool_call_1'),
            previous_part_kind='text',
        )
        yield PartDeltaEvent(
            index=1, delta=ToolCallPartDelta(args_delta='{"query": "Hello world"}', tool_call_id='tool_call_1')
        )
        yield PartEndEvent(
            index=1,
            part=ToolCallPart(tool_name='tool_call_1', args='{"query": "Hello world"}', tool_call_id='tool_call_1'),
            next_part_kind='tool-call',
        )

        yield PartStartEvent(
            index=2,
            part=ToolCallPart(tool_name='tool_call_2', args='{}', tool_call_id='tool_call_2'),
            previous_part_kind='tool-call',
        )
        yield PartDeltaEvent(
            index=2, delta=ToolCallPartDelta(args_delta='{"query": "Goodbye world"}', tool_call_id='tool_call_2')
        )
        yield PartEndEvent(
            index=2,
            part=ToolCallPart(tool_name='tool_call_2', args='{"query": "Hello world"}', tool_call_id='tool_call_2'),
            next_part_kind=None,
        )

        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_1', args='{"query": "Hello world"}', tool_call_id='tool_call_1')
        )
        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_2', args='{"query": "Goodbye world"}', tool_call_id='tool_call_2')
        )

        yield FunctionToolResultEvent(
            result=ToolReturnPart(tool_name='tool_call_1', content='Hi!', tool_call_id='tool_call_1')
        )
        yield FunctionToolResultEvent(
            result=ToolReturnPart(tool_name='tool_call_2', content='Bye!', tool_call_id='tool_call_2')
        )

        yield PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name='tool_call_3', args='{}', tool_call_id='tool_call_3'),
            previous_part_kind=None,
        )
        yield PartDeltaEvent(
            index=0, delta=ToolCallPartDelta(args_delta='{"query": "Hello world"}', tool_call_id='tool_call_3')
        )
        yield PartEndEvent(
            index=0,
            part=ToolCallPart(tool_name='tool_call_3', args='{"query": "Hello world"}', tool_call_id='tool_call_3'),
            next_part_kind='tool-call',
        )

        yield PartStartEvent(
            index=1,
            part=ToolCallPart(tool_name='tool_call_4', args='{}', tool_call_id='tool_call_4'),
            previous_part_kind='tool-call',
        )
        yield PartDeltaEvent(
            index=1, delta=ToolCallPartDelta(args_delta='{"query": "Goodbye world"}', tool_call_id='tool_call_4')
        )
        yield PartEndEvent(
            index=1,
            part=ToolCallPart(tool_name='tool_call_4', args='{"query": "Goodbye world"}', tool_call_id='tool_call_4'),
            next_part_kind=None,
        )

        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_3', args='{"query": "Hello world"}', tool_call_id='tool_call_3')
        )
        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_4', args='{"query": "Goodbye world"}', tool_call_id='tool_call_4')
        )

        yield FunctionToolResultEvent(
            result=ToolReturnPart(tool_name='tool_call_3', content='Hi!', tool_call_id='tool_call_3')
        )
        yield FunctionToolResultEvent(
            result=ToolReturnPart(tool_name='tool_call_4', content='Bye!', tool_call_id='tool_call_4')
        )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Tell me about Hello World',
        ),
    )
    event_stream = AGUIEventStream(run_input=run_input)
    events = [
        json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': (message_id := IsSameStr()),
                'role': 'assistant',
            },
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': 'Hello'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_1',
                'toolCallName': 'tool_call_1',
                'parentMessageId': message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': 'tool_call_1', 'delta': '{}'},
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_1',
                'delta': '{"query": "Hello world"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'tool_call_1'},
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_2',
                'toolCallName': 'tool_call_2',
                'parentMessageId': message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': 'tool_call_2', 'delta': '{}'},
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_2',
                'delta': '{"query": "Goodbye world"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'tool_call_2'},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'tool_call_1',
                'content': 'Hi!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': (result_message_id := IsSameStr()),
                'toolCallId': 'tool_call_2',
                'content': 'Bye!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_3',
                'toolCallName': 'tool_call_3',
                'parentMessageId': (new_message_id := IsSameStr()),
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': 'tool_call_3', 'delta': '{}'},
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_3',
                'delta': '{"query": "Hello world"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'tool_call_3'},
            {
                'type': 'TOOL_CALL_START',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_4',
                'toolCallName': 'tool_call_4',
                'parentMessageId': new_message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': 'tool_call_4', 'delta': '{}'},
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': 'tool_call_4',
                'delta': '{"query": "Goodbye world"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'tool_call_4'},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'tool_call_3',
                'content': 'Hi!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'tool_call_4',
                'content': 'Bye!',
                'role': 'tool',
            },
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )

    assert result_message_id != new_message_id


async def test_timestamps_are_set():
    """Test that all AG-UI events have timestamps set."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        )
    )

    events = await run_and_collect_events(agent, run_input)

    # All events should have timestamps
    for event in events:
        assert 'timestamp' in event, f'Event {event["type"]} missing timestamp'
        assert isinstance(event['timestamp'], int), (
            f'Event {event["type"]} timestamp should be int, got {type(event["timestamp"])}'
        )
        assert event['timestamp'] > 0, f'Event {event["type"]} timestamp should be positive'


async def test_tool_returns_event_with_timestamp_preserved():
    """Test that tools can return BaseEvents with pre-set timestamps that are preserved."""
    custom_timestamp = 1234567890000

    async def event_generator():
        yield FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name='get_status',
                content='Status retrieved',
                tool_call_id='call_1',
                metadata=CustomEvent(name='status_update', value={'status': 'ok'}, timestamp=custom_timestamp),
            )
        )

    run_input = create_input(UserMessage(id='msg_1', content='Check status'))
    event_stream = AGUIEventStream(run_input=run_input)
    events = [
        json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    custom_event = next((e for e in events if e.get('type') == 'CUSTOM'), None)
    assert custom_event is not None
    assert custom_event['timestamp'] == custom_timestamp


async def test_handle_ag_ui_request():
    agent = Agent(model=TestModel())
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Tell me about Hello World',
        ),
    )

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': run_input.model_dump_json().encode('utf-8')}

    starlette_request = Request(
        scope={
            'type': 'http',
            'method': 'POST',
            'headers': [
                (b'content-type', b'application/json'),
            ],
        },
        receive=receive,
    )

    response = await handle_ag_ui_request(agent, starlette_request)

    assert isinstance(response, StreamingResponse)

    chunks: list[MutableMapping[str, Any]] = []

    async def send(data: MutableMapping[str, Any]) -> None:
        if body := data.get('body'):
            data['body'] = json.loads(body.decode('utf-8').removeprefix('data: '))
        chunks.append(data)

    await response.stream_response(send)

    assert chunks == snapshot(
        [
            {
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'text/event-stream; charset=utf-8')],
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'RUN_STARTED',
                    'timestamp': IsInt(),
                    'threadId': (thread_id := IsSameStr()),
                    'runId': (run_id := IsSameStr()),
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_START',
                    'timestamp': IsInt(),
                    'messageId': (message_id := IsSameStr()),
                    'role': 'assistant',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'timestamp': IsInt(),
                    'messageId': message_id,
                    'delta': 'success ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'timestamp': IsInt(),
                    'messageId': message_id,
                    'delta': '(no ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'timestamp': IsInt(),
                    'messageId': message_id,
                    'delta': 'tool ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'timestamp': IsInt(),
                    'messageId': message_id,
                    'delta': 'calls)',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'RUN_FINISHED',
                    'timestamp': IsInt(),
                    'threadId': thread_id,
                    'runId': run_id,
                },
                'more_body': True,
            },
            {'type': 'http.response.body', 'body': b'', 'more_body': False},
        ]
    )
