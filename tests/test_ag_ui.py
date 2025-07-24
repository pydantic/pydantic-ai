"""Tests for AG-UI implementation."""

# pyright: reportPossiblyUnboundVariable=none
from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from dirty_equals import IsStr
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT

from .conftest import IsSameStr

has_ag_ui: bool = False
with contextlib.suppress(ImportError):
    from ag_ui.core import (
        AssistantMessage,
        CustomEvent,
        DeveloperMessage,
        EventType,
        FunctionCall,
        Message,
        RunAgentInput,
        StateSnapshotEvent,
        SystemMessage,
        Tool,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from ag_ui.encoder import EventEncoder

    from pydantic_ai.ag_ui import (
        SSE_CONTENT_TYPE,
        StateDeps,
        _Adapter,  # type: ignore[reportPrivateUsage]
    )

    has_ag_ui = True


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not has_ag_ui, reason='ag-ui-protocol not installed'),
]


def simple_result() -> Any:
    return snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'success '},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '(no tool calls)',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def collect_events_from_adapter(
    adapter: _Adapter[AgentDepsT, OutputDataT], *run_inputs: RunAgentInput, deps: AgentDepsT = None
) -> list[dict[str, Any]]:
    """Helper function to collect events from an AG-UI adapter run."""
    events = list[dict[str, Any]]()
    for run_input in run_inputs:
        async for event in adapter.run(run_input, deps=deps):
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


async def send_custom() -> list[CustomEvent]:
    """Display the recipe to the user.

    Returns:
        StateSnapshotEvent.
    """
    return [
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
    ]


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
        state=state,
        context=[],
        tools=tools or [],
        forwarded_props=None,
    )


async def simple_stream(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
    """A simple function that returns a text response without tool calls."""
    yield 'success '
    yield '(no tool calls)'


async def test_basic_user_message() -> None:
    """Test basic user message with text response."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )
    adapter = _Adapter(agent=agent)
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        )
    )

    events = await collect_events_from_adapter(adapter, run_input)

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
    adapter = _Adapter(agent=agent)
    run_input = create_input()
    events = await collect_events_from_adapter(adapter, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': IsStr(),
                'runId': IsStr(),
            },
            {'type': 'RUN_ERROR', 'message': 'no messages found in the input', 'code': 'no_messages'},
        ]
    )


async def test_multiple_messages() -> None:
    """Test with multiple different message types."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )
    adapter = _Adapter(agent=agent)
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
    )

    events = await collect_events_from_adapter(adapter, run_input)

    assert events == simple_result()


async def test_messages_with_history() -> None:
    """Test with multiple user messages (conversation history)."""
    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )
    adapter = _Adapter(agent=agent)
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

    events = await collect_events_from_adapter(adapter, run_input)

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
    adapter = _Adapter(agent=agent)
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

    events = await collect_events_from_adapter(adapter, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': IsStr(),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'toolCallId': tool_call_id,
                'delta': '{"location": ',
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': '"Paris"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
    adapter = _Adapter(agent=agent)
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

    events = await collect_events_from_adapter(adapter, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': (parent_message_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'toolCallId': tool_call_id,
                'delta': '{"location": "Paris"}',
            },
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather_parts',
                'parentMessageId': parent_message_id,
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'toolCallId': tool_call_id,
                'delta': '{"location": "',
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': 'Paris"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result", "get_weather_parts": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
    adapter = _Adapter(agent=agent)
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
    events = await collect_events_from_adapter(adapter, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': IsStr(),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'toolCallId': tool_call_id,
                'delta': '{"location":"',
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': 'Paris"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': '{"get_weather": "Tool result"}',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
    adapter = _Adapter(agent=agent)
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call send_snapshot',
        ),
    )
    events = await collect_events_from_adapter(adapter, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'send_snapshot',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': '{"type":"STATE_SNAPSHOT","timestamp":null,"raw_event":null,"snapshot":{"key":"value"}}',
                'role': 'tool',
            },
            {'type': 'STATE_SNAPSHOT', 'snapshot': {'key': 'value'}},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': """\
data: {"type":"STATE_SNAPSHOT","snapshot":{"key":"value"}}

""",
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
    adapter = _Adapter(agent=agent)
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call send_custom',
        ),
    )
    events = await collect_events_from_adapter(adapter, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'send_custom',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': '[{"type":"CUSTOM","timestamp":null,"raw_event":null,"name":"custom_event1","value":{"key1":"value1"}},{"type":"CUSTOM","timestamp":null,"raw_event":null,"name":"custom_event2","value":{"key2":"value2"}}]',
                'role': 'tool',
            },
            {'type': 'CUSTOM', 'name': 'custom_event1', 'value': {'key1': 'value1'}},
            {'type': 'CUSTOM', 'name': 'custom_event2', 'value': {'key2': 'value2'}},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': 'success send_custom called',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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

    adapter = _Adapter(agent=agent)
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Please call current_time',
        ),
    )

    events = await collect_events_from_adapter(adapter, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'current_time',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': '2023-06-21T12:08:45.485981+00:00',
                'role': 'tool',
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': 'success current_time called',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_thinking() -> None:
    """Test thinking events - now supported by FunctionModel."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Thinking ')}
        yield {0: DeltaThinkingPart(content='about the weather')}
        yield 'Thought about the weather'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
    )
    adapter = _Adapter(agent=agent)
    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Think about the weather',
        ),
    )

    events = await collect_events_from_adapter(adapter, run_input)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'Thinking '},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'about the weather'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': 'Thought about the weather',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


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
    adapter = _Adapter(agent=agent)
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
    events = await collect_events_from_adapter(adapter, *run_inputs)

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (first_tool_call_id := IsSameStr()),
                'toolCallName': 'current_time',
                'parentMessageId': (parent_message_id := IsSameStr()),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': first_tool_call_id, 'delta': '{}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': first_tool_call_id},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (second_tool_call_id := IsSameStr()),
                'toolCallName': 'get_weather',
                'parentMessageId': parent_message_id,
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'toolCallId': second_tool_call_id,
                'delta': '{"location": "Paris"}',
            },
            {'type': 'TOOL_CALL_END', 'toolCallId': second_tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': first_tool_call_id,
                'content': '2023-06-21T12:08:45.485981+00:00',
                'role': 'tool',
            },
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'RUN_STARTED',
                'threadId': thread_id,
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': 'current time is 2023-06-21T12:08:45.485981+00:00 and the weather in Paris is bright and sunny',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_request_with_state() -> None:
    """Test request with state modification."""

    agent: Agent[StateDeps[StateInt], str] = Agent(
        model=FunctionModel(stream_function=simple_stream),
        deps_type=StateDeps[StateInt],  # type: ignore[reportUnknownArgumentType]
    )
    adapter = _Adapter(agent=agent)
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
            state=StateInt(value=42),
        ),
    ]

    deps = StateDeps(StateInt())

    last_value = deps.state.value
    for run_input in run_inputs:
        events = list[dict[str, Any]]()
        async for event in adapter.run(run_input, deps=deps):
            events.append(json.loads(event.removeprefix('data: ')))

        assert events == simple_result()
        assert deps.state.value == run_input.state.value if run_input.state is not None else last_value
        last_value = deps.state.value

    assert deps.state.value == 42


async def test_concurrent_runs() -> None:
    """Test concurrent execution of multiple runs."""
    import asyncio

    agent = Agent(
        model=FunctionModel(stream_function=simple_stream),
    )
    adapter = _Adapter(agent=agent)
    concurrent_tasks: list[asyncio.Task[list[dict[str, Any]]]] = []

    for i in range(5):  # Test with 5 concurrent runs
        run_input = create_input(
            UserMessage(
                id=f'msg_{i}',
                content=f'Message {i}',
            ),
            thread_id=f'test_thread_{i}',
        )

        task = asyncio.create_task(collect_events_from_adapter(adapter, run_input))
        concurrent_tasks.append(task)

    results = await asyncio.gather(*concurrent_tasks)

    # Verify all runs completed successfully
    for i, events in enumerate(results):
        assert events == [
            {'type': 'RUN_STARTED', 'threadId': f'test_thread_{i}', 'runId': (run_id := IsSameStr())},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'success '},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': '(no tool calls)'},
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {'type': 'RUN_FINISHED', 'threadId': f'test_thread_{i}', 'runId': run_id},
        ]


@pytest.mark.anyio
async def test_to_ag_ui() -> None:
    """Test the agent.to_ag_ui method."""

    agent = Agent(model=FunctionModel(stream_function=simple_stream))
    app = agent.to_ag_ui()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://localhost:8000'
            run_input = create_input(
                UserMessage(
                    id='msg_1',
                    content='Hello, world!',
                ),
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
