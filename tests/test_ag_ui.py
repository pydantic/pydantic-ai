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
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.exceptions import UserError
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

from .conftest import IsDatetime, IsSameStr, try_import

with try_import() as imports_successful:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
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
                'threadId': IsStr(),
                'runId': IsStr(),
            },
            {'type': 'RUN_ERROR', 'message': 'No message history, user prompt, or instructions provided'},
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
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': """\
Unknown tool name: 'get_weather'. Available tools: 'get_weather_parts'

Fix the errors and try again.\
""",
                'role': 'tool',
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
                'content': 'Done',
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
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'THINKING_START'},
            {'type': 'THINKING_END'},
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': "Let's do some thinking",
            },
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': ' and some more',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {'type': 'THINKING_START'},
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'Thinking '},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'about the weather'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'Thinking about the meaning of life'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'Thinking about the universe'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'THINKING_END'},
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

    with pytest.raises(
        UserError,
        match='State is provided but `deps` of type `NoneType` does not implement the `StateHandler` protocol: it needs to be a dataclass with a non-optional `state` field.',
    ):
        async for _ in run_ag_ui(agent, run_input):
            pass


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
            {'type': 'RUN_STARTED', 'threadId': f'test_thread_{i}', 'runId': (run_id := IsSameStr())},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': (tool_call_id := IsSameStr()),
                'toolCallName': 'get_state',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_END', 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': str(i),
                'role': 'tool',
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': '{"get_s'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'tate":' + str(i) + '}'},
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {'type': 'RUN_FINISHED', 'threadId': f'test_thread_{i}', 'runId': run_id},
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


async def test_messages() -> None:
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
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'toolCallName': 'web_search',
                'parentMessageId': IsStr(),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'pyd_ai_builtin|function|search_1', 'delta': '{"query":'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'pyd_ai_builtin|function|search_1', 'delta': '"Hello world"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'pyd_ai_builtin|function|search_1'},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': 'pyd_ai_builtin|function|search_1',
                'content': '{"results":[{"title":"\\"Hello, World!\\" program","url":"https://en.wikipedia.org/wiki/%22Hello,_World!%22_program"}]}',
                'role': 'tool',
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {
                'type': 'TEXT_MESSAGE_CONTENT',
                'messageId': message_id,
                'delta': 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
            },
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'Hello'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'Goodbye'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'RUN_FINISHED',
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
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'TEXT_MESSAGE_START', 'messageId': (message_id := IsSameStr()), 'role': 'assistant'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': 'Hello'},
            {'type': 'TEXT_MESSAGE_CONTENT', 'messageId': message_id, 'delta': ' world'},
            {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': 'tool_call_1',
                'toolCallName': 'tool_call_1',
                'parentMessageId': message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_1', 'delta': '{}'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_1', 'delta': '{"query": "Hello world"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'tool_call_1'},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': 'tool_call_2',
                'toolCallName': 'tool_call_2',
                'parentMessageId': message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_2', 'delta': '{}'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_2', 'delta': '{"query": "Goodbye world"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'tool_call_2'},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': 'tool_call_1',
                'content': 'Hi!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': (result_message_id := IsSameStr()),
                'toolCallId': 'tool_call_2',
                'content': 'Bye!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': 'tool_call_3',
                'toolCallName': 'tool_call_3',
                'parentMessageId': (new_message_id := IsSameStr()),
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_3', 'delta': '{}'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_3', 'delta': '{"query": "Hello world"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'tool_call_3'},
            {
                'type': 'TOOL_CALL_START',
                'toolCallId': 'tool_call_4',
                'toolCallName': 'tool_call_4',
                'parentMessageId': new_message_id,
            },
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_4', 'delta': '{}'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'tool_call_4', 'delta': '{"query": "Goodbye world"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'tool_call_4'},
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': 'tool_call_3',
                'content': 'Hi!',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_RESULT',
                'messageId': IsStr(),
                'toolCallId': 'tool_call_4',
                'content': 'Bye!',
                'role': 'tool',
            },
            {
                'type': 'RUN_FINISHED',
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )

    assert result_message_id != new_message_id


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
                    'threadId': (thread_id := IsSameStr()),
                    'runId': (run_id := IsSameStr()),
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_START',
                    'messageId': (message_id := IsSameStr()),
                    'role': 'assistant',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'messageId': message_id,
                    'delta': 'success ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'messageId': message_id,
                    'delta': '(no ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'messageId': message_id,
                    'delta': 'tool ',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'TEXT_MESSAGE_CONTENT',
                    'messageId': message_id,
                    'delta': 'calls)',
                },
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {'type': 'TEXT_MESSAGE_END', 'messageId': message_id},
                'more_body': True,
            },
            {
                'type': 'http.response.body',
                'body': {
                    'type': 'RUN_FINISHED',
                    'threadId': thread_id,
                    'runId': run_id,
                },
                'more_body': True,
            },
            {'type': 'http.response.body', 'body': b'', 'more_body': False},
        ]
    )
