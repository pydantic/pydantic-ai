"""Tests for AG-UI implementation."""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import uuid
import warnings
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    CachePoint,
    DocumentUrl,
    FilePart,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
    capture_run_messages,
)
from pydantic_ai._deferred_capabilities import parse_loaded_capabilities
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent, AgentRunResult
from pydantic_ai.capabilities import Capability, PrepareTools
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
)
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
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
)
from pydantic_ai.toolsets._tool_search import parse_discovered_tools

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsSameStr, IsStr, message, message_part, try_import

with try_import() as imports_successful:
    from ag_ui.core import (
        ActivityMessage,
        AssistantMessage,
        AudioInputContent,
        BaseEvent,
        BinaryInputContent,
        CustomEvent,
        DeveloperMessage,
        DocumentInputContent,
        EventType,
        FunctionCall,
        ImageInputContent,
        InputContentDataSource,
        InputContentUrlSource,
        Message,
        ReasoningMessage,
        RunAgentInput,
        StateSnapshotEvent,
        SystemMessage,
        TextInputContent,
        Tool,
        ToolCall,
        ToolMessage,
        UserMessage,
        VideoInputContent,
    )
    from ag_ui.encoder import EventEncoder
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from pydantic_ai.ui import SSE_CONTENT_TYPE, OnCompleteFunc, StateDeps
    from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream
    from pydantic_ai.ui.ag_ui._utils import (
        BUILTIN_TOOL_CALL_ID_PREFIX,
        detect_ag_ui_version,
        parse_ag_ui_version,
    )

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as interrupts_imports_successful:
    # `ResumeEntry` and the interrupt-aware run lifecycle were added in ag-ui-protocol 0.1.19
    # (PR #1569). On older installs, the dedicated interrupt tests below are skipped.
    from ag_ui.core import ResumeEntry


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='ag-ui-protocol not installed'),
]


def simple_result(*, outcome: dict[str, Any] | None = None) -> Any:
    """Expected event sequence for `simple_stream`.

    Pass `outcome={'type': 'success'}` for callers that run against `ag-ui-protocol >= 0.1.19`
    (where the adapter emits `RunFinishedEvent.outcome`). Older negotiated versions
    (e.g. `ag_ui_version='0.1.10'`) suppress the field, so the default `outcome=None`
    matches a bare `RUN_FINISHED`.
    """
    thread_id = IsSameStr()
    run_id = IsSameStr()
    message_id = IsSameStr()
    run_finished: dict[str, Any] = {
        'type': 'RUN_FINISHED',
        'timestamp': IsInt(),
        'threadId': thread_id,
        'runId': run_id,
    }
    if outcome is not None:
        run_finished['outcome'] = outcome
    return snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
            {
                'type': 'TEXT_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': message_id,
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
            run_finished,
        ]
    )


def test_manage_system_prompt_visible_in_ag_ui_from_request_signature() -> None:
    from_request_parameters = inspect.signature(AGUIAdapter.from_request).parameters

    assert 'manage_system_prompt' in from_request_parameters
    assert from_request_parameters['manage_system_prompt'].default == 'server'


async def run_and_collect_events(
    agent: Agent[AgentDepsT, OutputDataT],
    *run_inputs: RunAgentInput,
    deps: AgentDepsT = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
    ag_ui_version: Literal['0.1.10', '0.1.13'] = '0.1.10',
) -> list[dict[str, Any]]:
    events = list[dict[str, Any]]()
    for run_input in run_inputs:
        adapter = AGUIAdapter(agent=agent, run_input=run_input, ag_ui_version=ag_ui_version)
        async for event in adapter.encode_stream(adapter.run_stream(deps=deps, on_complete=on_complete)):
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

    # The frontend-sent `SystemMessage` is stripped by the default server mode; verify
    # that doesn't change the event stream (which is driven by the assistant's output).
    with pytest.warns(UserWarning, match='manage_system_prompt'):
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


async def test_output_tool() -> None:
    """Output tool calls emit `TOOL_CALL_RESULT` via `handle_output_tool_result`."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"query":"hello"}', tool_call_id='out_1')}

    def web_search(query: str) -> dict[str, str]:
        return {'result': f'Searched for {query}'}

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=web_search)

    run_input = create_input(UserMessage(id='msg_1', content='Tell me about hello'))

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
                'toolCallName': 'final_result',
                'parentMessageId': IsStr(),
            },
            {
                'type': 'TOOL_CALL_ARGS',
                'timestamp': IsInt(),
                'toolCallId': tool_call_id,
                'delta': '{"query":"hello"}',
            },
            {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': tool_call_id},
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': tool_call_id,
                'content': 'Final result processed.',
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
            # Part 0: empty thinking — skipped (no content, no metadata)
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
            # Part 1: "Thinking about the weather"
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'Thinking '},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'about the weather'},
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            # Part 2: empty thinking — skipped (no content, no metadata)
            # Part 3: "Thinking about the meaning of life"
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {
                'type': 'THINKING_TEXT_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'delta': 'Thinking about the meaning of life',
            },
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            # Part 4: "Thinking about the universe"
            {'type': 'THINKING_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_START', 'timestamp': IsInt()},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'timestamp': IsInt(), 'delta': 'Thinking about the universe'},
            {'type': 'THINKING_TEXT_MESSAGE_END', 'timestamp': IsInt()},
            {'type': 'THINKING_END', 'timestamp': IsInt()},
            {
                'type': 'RUN_FINISHED',
                'timestamp': IsInt(),
                'threadId': thread_id,
                'runId': run_id,
            },
        ]
    )


async def test_thinking_with_signature() -> None:
    """Test that ReasoningEncryptedValueEvent is emitted with thinking metadata."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Thinking deeply', signature='sig_abc123')}
        yield 'Here is my response'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_input = create_input(
        UserMessage(id='msg_1', content='Think about something'),
    )

    events = await run_and_collect_events(agent, run_input, ag_ui_version='0.1.13')

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            {'type': 'REASONING_START', 'timestamp': IsInt(), 'messageId': (reasoning_id := IsSameStr())},
            {
                'type': 'REASONING_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': reasoning_id,
                'role': 'reasoning',
            },
            {
                'type': 'REASONING_MESSAGE_CONTENT',
                'timestamp': IsInt(),
                'messageId': reasoning_id,
                'delta': 'Thinking deeply',
            },
            {'type': 'REASONING_MESSAGE_END', 'timestamp': IsInt(), 'messageId': reasoning_id},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'timestamp': IsInt(),
                'subtype': 'message',
                'entityId': reasoning_id,
                'encryptedValue': IsStr(),
            },
            {'type': 'REASONING_END', 'timestamp': IsInt(), 'messageId': reasoning_id},
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


async def test_thinking_consecutive_signatures() -> None:
    """Test that consecutive ThinkingParts each preserve their own metadata via separate REASONING blocks."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='First thought', signature='sig_aaa')}
        yield {1: DeltaThinkingPart(content='Second thought', signature='sig_bbb')}
        yield {2: DeltaThinkingPart(content='Third thought', signature='sig_ccc')}
        yield 'Final answer'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_input = create_input(
        UserMessage(id='msg_1', content='Think deeply'),
    )

    events = await run_and_collect_events(agent, run_input, ag_ui_version='0.1.13')

    assert events == snapshot(
        [
            {
                'type': 'RUN_STARTED',
                'timestamp': IsInt(),
                'threadId': (thread_id := IsSameStr()),
                'runId': (run_id := IsSameStr()),
            },
            # Part 0: signature=sig_aaa
            {'type': 'REASONING_START', 'timestamp': IsInt(), 'messageId': (r0 := IsSameStr())},
            {
                'type': 'REASONING_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': r0,
                'role': 'reasoning',
            },
            {'type': 'REASONING_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': r0, 'delta': 'First thought'},
            {'type': 'REASONING_MESSAGE_END', 'timestamp': IsInt(), 'messageId': r0},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'timestamp': IsInt(),
                'subtype': 'message',
                'entityId': r0,
                'encryptedValue': IsStr(),
            },
            {'type': 'REASONING_END', 'timestamp': IsInt(), 'messageId': r0},
            # Part 1: signature=sig_bbb
            {'type': 'REASONING_START', 'timestamp': IsInt(), 'messageId': (r1 := IsSameStr())},
            {
                'type': 'REASONING_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': r1,
                'role': 'reasoning',
            },
            {'type': 'REASONING_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': r1, 'delta': 'Second thought'},
            {'type': 'REASONING_MESSAGE_END', 'timestamp': IsInt(), 'messageId': r1},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'timestamp': IsInt(),
                'subtype': 'message',
                'entityId': r1,
                'encryptedValue': IsStr(),
            },
            {'type': 'REASONING_END', 'timestamp': IsInt(), 'messageId': r1},
            # Part 2: signature=sig_ccc
            {'type': 'REASONING_START', 'timestamp': IsInt(), 'messageId': (r2 := IsSameStr())},
            {
                'type': 'REASONING_MESSAGE_START',
                'timestamp': IsInt(),
                'messageId': r2,
                'role': 'reasoning',
            },
            {'type': 'REASONING_MESSAGE_CONTENT', 'timestamp': IsInt(), 'messageId': r2, 'delta': 'Third thought'},
            {'type': 'REASONING_MESSAGE_END', 'timestamp': IsInt(), 'messageId': r2},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'timestamp': IsInt(),
                'subtype': 'message',
                'entityId': r2,
                'encryptedValue': IsStr(),
            },
            {'type': 'REASONING_END', 'timestamp': IsInt(), 'messageId': r2},
            # Text response
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
                'delta': 'Final answer',
            },
            {'type': 'TEXT_MESSAGE_END', 'timestamp': IsInt(), 'messageId': message_id},
            {'type': 'RUN_FINISHED', 'timestamp': IsInt(), 'threadId': thread_id, 'runId': run_id},
        ]
    )


def test_reasoning_message_thinking_roundtrip() -> None:
    """Test that ReasoningMessage converts to ThinkingPart with metadata from encrypted_value."""
    messages = AGUIAdapter.load_messages(
        [
            ReasoningMessage(
                id='reasoning-1',
                content='Let me think about this...',
                encrypted_value=json.dumps(
                    {
                        'id': 'thinking-1',
                        'signature': 'sig_abc123',
                        'provider_name': 'anthropic',
                        'provider_details': {'some': 'details'},
                    }
                ),
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


async def test_reasoning_events_with_all_metadata() -> None:
    """Test that REASONING_* events emit encryptedValue with all metadata fields."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    part = ThinkingPart(
        content='Thinking content',
        id='thinking-123',
        signature='sig_xyz',
        provider_name='anthropic',
        provider_details={'model': 'claude-sonnet-4-5'},
    )

    events: list[BaseEvent] = []
    async for e in event_stream.handle_thinking_start(part):
        events.append(e)
    async for e in event_stream.handle_thinking_end(part):
        events.append(e)

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {'type': 'REASONING_MESSAGE_START', 'message_id': IsStr(), 'role': 'reasoning'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'Thinking content'},
            {'type': 'REASONING_MESSAGE_END', 'message_id': IsStr()},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'subtype': 'message',
                'entity_id': IsStr(),
                'encrypted_value': '{"id": "thinking-123", "signature": "sig_xyz", "provider_name": "anthropic", "provider_details": {"model": "claude-sonnet-4-5"}}',
            },
            {'type': 'REASONING_END', 'message_id': IsStr()},
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


@pytest.mark.parametrize(
    'encrypted_value',
    [
        pytest.param('not valid json{{{', id='invalid-json'),
        pytest.param('"just a string"', id='non-dict-string'),
        pytest.param('[1, 2, 3]', id='non-dict-list'),
        pytest.param('42', id='non-dict-number'),
    ],
)
def test_reasoning_message_malformed_encrypted_value(encrypted_value: str) -> None:
    """Test that malformed or non-dict encrypted_value is handled gracefully."""
    messages = AGUIAdapter.load_messages(
        [
            ReasoningMessage(id='r-1', content='Thinking...', encrypted_value=encrypted_value),
            AssistantMessage(id='msg-1', content='Done'),
        ]
    )

    assert messages == snapshot(
        [
            ModelResponse(
                parts=[ThinkingPart(content='Thinking...'), TextPart(content='Done')],
                timestamp=IsDatetime(),
            )
        ]
    )


def test_activity_message_file_part_missing_url() -> None:
    """Test that ActivityMessage(pydantic_ai_file) with empty url raises ValueError."""
    with pytest.raises(ValueError, match='must have a non-empty url'):
        AGUIAdapter.load_messages(
            [
                ActivityMessage(
                    id='activity-1',
                    activity_type='pydantic_ai_file',
                    content={'url': '', 'media_type': 'image/png'},
                ),
            ],
            preserve_file_data=True,
        )


_TIMESTAMPED_PARTS = (UserPromptPart, RetryPromptPart, ToolReturnPart, NativeToolReturnPart, SystemPromptPart)


def _sync_part_timestamps(
    original_part: ModelRequestPart | ModelResponsePart,
    new_part: ModelRequestPart | ModelResponsePart,
) -> None:
    """Sync timestamp attribute if both parts are request parts (which carry timestamps)."""
    if isinstance(new_part, _TIMESTAMPED_PARTS) and isinstance(original_part, _TIMESTAMPED_PARTS):
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

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
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


def test_dump_load_roundtrip_load_capability() -> None:
    """Typed `load_capability` parts keep their identity through dump/load on >= 0.1.11.

    The `encrypted_value` carrier landed in 0.1.11, so `tool_kind` survives from there; without it a
    resuming agent would forget its loaded capabilities. Dumped at exactly 0.1.11 to pin the floor.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(tool_call_id='load-foobar', args='{"id": "foobar"}')]),
        ModelRequest(
            parts=[LoadCapabilityReturnPart(tool_call_id='load-foobar', content={'instructions': '# Foo Bar'})]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.11')
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == original
    assert parse_loaded_capabilities(reloaded) == {'foobar'}


def test_dump_load_roundtrip_load_capability_invalid_args() -> None:
    """Invalid `load_capability` args never count as loaded after a roundtrip.

    AG-UI args are JSON strings, which always satisfy the typed part's `str` arm — unlike
    Vercel's structured args the call still promotes, and invalidity surfaces as
    `capability_id is None`.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(tool_call_id='load-foobar', args='{"name": "foobar"}')]),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    tool_name='load_capability',
                    tool_call_id='load-foobar',
                    content='Field required: id',
                )
            ]
        ),
        ModelResponse(parts=[LoadCapabilityCallPart(tool_call_id='load-foobar', args='{"id": "foobar"}')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)

    reloaded_call = message_part(reloaded, LoadCapabilityCallPart)
    assert reloaded_call.capability_id is None
    assert parse_loaded_capabilities(reloaded) == set()


def test_dump_load_roundtrip_load_capability_old_version() -> None:
    """On < 0.1.11, `tool_kind` is skipped (no `encrypted_value` field) and typed parts reload as base classes.

    Dumping a typed part below the floor warns, since the round-trip silently forgets loaded state.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(tool_call_id='load-foobar', args='{"id": "foobar"}')]),
        ModelRequest(
            parts=[LoadCapabilityReturnPart(tool_call_id='load-foobar', content={'instructions': '# Foo Bar'})]
        ),
    ]

    with pytest.warns(UserWarning, match=r'ag-ui-protocol 0\.1\.10 predates the `encrypted_value` field'):
        ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.10')

    assistant_msg = ag_ui_msgs[0]
    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.tool_calls is not None
    # Omitted entirely (not set to `None`), so a pre-0.1.11 client never sees an unexpected field.
    assert 'encrypted_value' not in assistant_msg.tool_calls[0].model_fields_set
    tool_msg = ag_ui_msgs[1]
    assert isinstance(tool_msg, ToolMessage)
    assert 'encrypted_value' not in tool_msg.model_fields_set

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    assert type(reloaded[0].parts[0]) is ToolCallPart
    assert type(reloaded[1].parts[0]) is ToolReturnPart
    assert parse_loaded_capabilities(reloaded) == set()


def test_dump_omits_encrypted_value_without_tool_kind() -> None:
    """On a supported version, a plain call/return with no `tool_kind` still omits `encrypted_value`
    rather than emitting a bare `null` — the field is only set when there's a claim to carry."""
    original: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='regular', tool_call_id='c1', args='{}')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='regular', tool_call_id='c1', content='ok')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')

    assistant_msg = ag_ui_msgs[0]
    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.tool_calls is not None
    assert 'encrypted_value' not in assistant_msg.tool_calls[0].model_fields_set
    tool_msg = ag_ui_msgs[1]
    assert isinstance(tool_msg, ToolMessage)
    assert 'encrypted_value' not in tool_msg.model_fields_set


def test_dump_load_roundtrip_native_tool_search() -> None:
    """Native tool-search parts keep their typed identity through dump/load on >= 0.1.11."""
    original: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(tool_call_id='search-1', args='{"queries": ["refund"]}'),
                NativeToolSearchReturnPart(
                    tool_call_id='search-1',
                    content={'discovered_tools': [{'name': 'refund_tool'}]},
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)

    assert parse_discovered_tools(reloaded) == {'refund_tool'}
    # A non-empty result proves the return part kept its typed identity; the call part's
    # identity matters to Anthropic history replay, so pin it too.
    assert isinstance(reloaded[0].parts[0], NativeToolSearchCallPart)


def test_dump_load_roundtrip_interrupted_tool_return() -> None:
    """A synthesized interrupted return keeps `outcome='interrupted'` through dump/load on >= 0.1.11.

    `ToolMessage` has no outcome slot (`'failed'`/`'denied'` map to `error`, `'success'` is the
    default), so the claim rides the `encrypted_value` carrier like `tool_kind` does — and below
    0.1.11 there is no carrier, so the outcome silently degrades to `'success'`.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', tool_call_id='c1', args='{}')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='my_tool',
                    tool_call_id='c1',
                    content='The tool call was interrupted before a result was produced.',
                    outcome='interrupted',
                )
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
    tool_msg = next(msg for msg in ag_ui_msgs if isinstance(msg, ToolMessage))
    assert tool_msg.encrypted_value == '{"pydantic_ai": {"outcome": "interrupted"}}'

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    reloaded_return = next(
        part
        for message in reloaded
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    assert reloaded_return.outcome == 'interrupted'

    old_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.10')
    old_tool_msg = next(msg for msg in old_msgs if isinstance(msg, ToolMessage))
    assert 'encrypted_value' not in old_tool_msg.model_fields_set


def test_load_forged_encrypted_outcome_stays_success() -> None:
    """Only `'interrupted'` ever rides the carrier; any other client-supplied outcome claim is ignored."""
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[ToolCall(id='c1', function=FunctionCall(name='my_tool', arguments='{}'))],
            ),
            ToolMessage(
                id='msg-2',
                tool_call_id='c1',
                content='done',
                encrypted_value='{"pydantic_ai": {"outcome": "failed"}}',
            ),
        ]
    )

    return_part = loaded[1].parts[0]
    assert isinstance(return_part, ToolReturnPart)
    assert return_part.outcome == 'success'


def test_load_tool_kind_falls_back_to_call_claim() -> None:
    """A ToolMessage without its own `encrypted_value` narrows via the paired call's claim.

    Streamed results have no metadata slot, so client-assembled histories only carry
    the claim on the `ToolCall`.
    """
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[
                    ToolCall(
                        id='load-foobar',
                        function=FunctionCall(name='load_capability', arguments='{"id": "foobar"}'),
                        encrypted_value='{"pydantic_ai": {"tool_kind": "capability-load"}}',
                    )
                ],
            ),
            ToolMessage(id='msg-2', tool_call_id='load-foobar', content='{"instructions": "# Foo Bar"}'),
        ]
    )

    assert isinstance(loaded[0].parts[0], LoadCapabilityCallPart)
    assert isinstance(loaded[1].parts[0], LoadCapabilityReturnPart)
    assert parse_loaded_capabilities(loaded) == {'foobar'}


def test_load_tool_kind_error_result_stays_plain() -> None:
    """A ToolMessage with `error` set never narrows: typed return parts imply success to their readers."""
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[
                    ToolCall(
                        id='load-foobar',
                        function=FunctionCall(name='load_capability', arguments='{"name": "foobar"}'),
                        encrypted_value='{"pydantic_ai": {"tool_kind": "capability-load"}}',
                    )
                ],
            ),
            ToolMessage(
                id='msg-2',
                tool_call_id='load-foobar',
                content='Field required: id',
                error='Field required: id',
            ),
        ]
    )

    assert type(loaded[1].parts[0]) is ToolReturnPart
    assert parse_loaded_capabilities(loaded) == set()


@pytest.mark.parametrize(
    'encrypted_value',
    [
        'not json',
        '"a string"',
        '[1]',
        # A genuine provider blob or an un-namespaced claim (no `pydantic_ai` key) is never honored.
        '{"tool_kind": "capability-load"}',
        # A namespaced claim with an unknown kind is rejected by the `ToolPartKind` filter.
        '{"pydantic_ai": {"tool_kind": "unknown-kind"}}',
    ],
)
def test_load_tool_kind_garbage_encrypted_value(encrypted_value: str) -> None:
    """`encrypted_value` is client-supplied: anything malformed or un-namespaced loads as a plain part."""
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[
                    ToolCall(
                        id='call-1',
                        function=FunctionCall(name='load_capability', arguments='{"id": "foobar"}'),
                        encrypted_value=encrypted_value,
                    )
                ],
            ),
        ]
    )

    assert type(loaded[0].parts[0]) is ToolCallPart


def test_load_tool_kind_unparseable_result_content_stays_plain() -> None:
    """A claimed return whose content isn't valid JSON degrades to the plain string part."""
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[
                    ToolCall(
                        id='load-foobar',
                        function=FunctionCall(name='load_capability', arguments='{"id": "foobar"}'),
                        encrypted_value='{"pydantic_ai": {"tool_kind": "capability-load"}}',
                    )
                ],
            ),
            ToolMessage(id='msg-2', tool_call_id='load-foobar', content='not json'),
        ]
    )

    return_part = loaded[1].parts[0]
    assert type(return_part) is ToolReturnPart
    assert return_part.content == 'not json'


def test_load_malformed_builtin_tool_call_id_degrades_to_plain() -> None:
    """A client-supplied id starting with the builtin prefix but missing its `|` segments
    degrades to plain tool call/return parts instead of raising on the tuple unpack."""
    loaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg-1',
                tool_calls=[
                    ToolCall(
                        id=BUILTIN_TOOL_CALL_ID_PREFIX,
                        function=FunctionCall(name='web_search', arguments='{}'),
                    )
                ],
            ),
            ToolMessage(id='msg-2', tool_call_id=BUILTIN_TOOL_CALL_ID_PREFIX, content='{}'),
        ]
    )

    assert type(loaded[0].parts[0]) is ToolCallPart
    assert type(loaded[1].parts[0]) is ToolReturnPart


@pytest.mark.parametrize('ag_ui_version', ['0.1.10', '0.1.13'])
async def test_run_stream_load_capability_tool_kind_encrypted_value(
    ag_ui_version: Literal['0.1.10', '0.1.13'],
) -> None:
    """Streamed `load_capability` calls carry `tool_kind` via `REASONING_ENCRYPTED_VALUE`.

    Clients build their `ToolCall` history from streamed events, echoing this back as
    `encrypted_value` — without it, streaming-built histories reload as plain parts.
    The event doesn't exist before 0.1.13, so it's skipped there.
    """

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='load_capability', json_args='{"id": "refunds"}', tool_call_id='load-1')}
        else:
            yield 'done'

    agent = Agent(
        model=FunctionModel(stream_function=stream_function),
        capabilities=[
            Capability[object](
                id='refunds',
                description='Refund tools.',
                instructions='Refund instructions.',
                defer_loading=True,
            )
        ],
    )

    run_input = create_input(UserMessage(id='msg_1', content='Help me with a refund'))
    events = await run_and_collect_events(agent, run_input, ag_ui_version=ag_ui_version)

    tool_events = [e for e in events if e['type'].startswith('TOOL_CALL') or e['type'] == 'REASONING_ENCRYPTED_VALUE']
    encrypted_value_event = {
        'type': 'REASONING_ENCRYPTED_VALUE',
        'timestamp': IsInt(),
        'subtype': 'tool-call',
        'entityId': 'load-1',
        'encryptedValue': '{"pydantic_ai": {"tool_kind": "capability-load"}}',
    }
    expected: list[dict[str, Any]] = [
        {
            'type': 'TOOL_CALL_START',
            'timestamp': IsInt(),
            'toolCallId': 'load-1',
            'toolCallName': 'load_capability',
            'parentMessageId': IsStr(),
        },
        *([encrypted_value_event] if ag_ui_version == '0.1.13' else []),
        {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': 'load-1', 'delta': '{"id": "refunds"}'},
        {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': 'load-1'},
        {
            'type': 'TOOL_CALL_RESULT',
            'timestamp': IsInt(),
            'messageId': IsStr(),
            'toolCallId': 'load-1',
            'content': '{"instructions":"Refund instructions."}',
            'role': 'tool',
        },
    ]
    assert tool_events == expected


@pytest.mark.parametrize('ag_ui_version', ['0.1.10', '0.1.13'])
async def test_run_stream_native_tool_search_tool_kind_encrypted_value(
    ag_ui_version: Literal['0.1.10', '0.1.13'],
) -> None:
    """Streamed native `tool_search` calls carry `tool_kind` via `REASONING_ENCRYPTED_VALUE`.

    Mirrors `test_run_stream_load_capability_tool_kind_encrypted_value`, but for the builtin
    (`provider_executed`) streaming path, which is a distinct code path. Clients build their
    `ToolCall` history from streamed events, echoing this back as `encrypted_value` — without
    it, streaming-built histories reload as plain parts and `parse_discovered_tools()` is empty
    on resume. The event doesn't exist before 0.1.13, so it's skipped there.
    """

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: NativeToolSearchCallPart(
                    tool_call_id='search-1', args='{"queries": ["refund"]}', provider_name='function'
                )
            }
            yield {
                1: NativeToolSearchReturnPart(
                    tool_call_id='search-1',
                    content={'discovered_tools': [{'name': 'refund_tool'}]},
                    provider_name='function',
                )
            }
        else:
            yield 'done'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_input = create_input(UserMessage(id='msg_1', content='Find me a refund tool'))
    events = await run_and_collect_events(agent, run_input, ag_ui_version=ag_ui_version)

    builtin_id = 'pyd_ai_builtin|function|search-1'
    tool_events = [e for e in events if e['type'].startswith('TOOL_CALL') or e['type'] == 'REASONING_ENCRYPTED_VALUE']
    encrypted_value_event = {
        'type': 'REASONING_ENCRYPTED_VALUE',
        'timestamp': IsInt(),
        'subtype': 'tool-call',
        'entityId': builtin_id,
        'encryptedValue': '{"pydantic_ai": {"tool_kind": "tool-search"}}',
    }
    expected: list[dict[str, Any]] = [
        {
            'type': 'TOOL_CALL_START',
            'timestamp': IsInt(),
            'toolCallId': builtin_id,
            'toolCallName': 'tool_search',
            'parentMessageId': IsStr(),
        },
        *([encrypted_value_event] if ag_ui_version == '0.1.13' else []),
        {'type': 'TOOL_CALL_ARGS', 'timestamp': IsInt(), 'toolCallId': builtin_id, 'delta': '{"queries": ["refund"]}'},
        {'type': 'TOOL_CALL_END', 'timestamp': IsInt(), 'toolCallId': builtin_id},
        {
            'type': 'TOOL_CALL_RESULT',
            'timestamp': IsInt(),
            'messageId': IsStr(),
            'toolCallId': builtin_id,
            'content': '{"discovered_tools":[{"name":"refund_tool"}]}',
            'role': 'tool',
        },
    ]
    assert tool_events == expected


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

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
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


@pytest.mark.parametrize(
    'original',
    [
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart(content='Generate an image')]),
                ModelResponse(
                    parts=[
                        FilePart(
                            content=BinaryImage(data=b'generated file content', media_type='image/png'),
                            id='file-001',
                            provider_name='openai',
                            provider_details={'model': 'gpt-image'},
                        ),
                        TextPart(content='Here is your generated image.'),
                    ]
                ),
            ],
            id='full-attrs',
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart(content='Generate')]),
                ModelResponse(
                    parts=[
                        FilePart(content=BinaryImage(data=b'minimal file', media_type='image/png')),
                        TextPart(content='Done'),
                    ]
                ),
            ],
            id='minimal-attrs',
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart(content='Generate image only')]),
                ModelResponse(parts=[FilePart(content=BinaryImage(data=b'only file', media_type='image/png'))]),
            ],
            id='file-only',
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart(content='Generate an image')]),
                ModelResponse(
                    parts=[
                        FilePart(
                            content=BinaryImage(
                                data=b'generated file content',
                                media_type='image/png',
                                vendor_metadata={'detail': 'high'},
                            ),
                        ),
                    ]
                ),
            ],
            id='vendor-metadata',
        ),
    ],
)
def test_dump_load_roundtrip_file_part(original: list[ModelMessage]) -> None:
    """Test round-trip for FilePart variants: full attributes, minimal, and file-only response.

    Note: BinaryImage is used because from_data_uri() returns BinaryImage for image/* media types.
    """
    ag_ui_msgs = AGUIAdapter.dump_messages(original, preserve_file_data=True)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs, preserve_file_data=True)
    _sync_timestamps(original, reloaded)

    assert reloaded == original


def test_dump_load_roundtrip_builtin_tool_return() -> None:
    """Test round-trip for builtin tool calls with their return values.

    Note: The round-trip reorders parts within ModelResponse because AG-UI's AssistantMessage
    has separate content and tool_calls fields. TextPart comes first (from content), then
    NativeToolCallPart (from tool_calls), then NativeToolReturnPart (from subsequent ToolMessage).
    """
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search for info')]),
        ModelResponse(
            parts=[
                TextPart(content='Based on the search...'),
                NativeToolCallPart(
                    tool_name='web_search',
                    tool_call_id='call_123',
                    args='{"query": "test"}',
                    provider_name='anthropic',
                ),
                NativeToolReturnPart(
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


def test_dump_builtin_tool_call_without_return() -> None:
    """Test that NativeToolCallPart without a matching NativeToolReturnPart still dumps correctly."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search for info')]),
        ModelResponse(
            parts=[
                NativeToolCallPart(
                    tool_name='web_search',
                    tool_call_id='call_orphan',
                    args='{"query": "test"}',
                    provider_name='anthropic',
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages)

    assert len(ag_ui_msgs) == 2
    assistant_msg = ag_ui_msgs[1]
    assert isinstance(assistant_msg, AssistantMessage)
    assert assistant_msg.tool_calls is not None
    assert len(assistant_msg.tool_calls) == 1
    assert assistant_msg.tool_calls[0].id == 'pyd_ai_builtin|anthropic|call_orphan'


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


def test_dump_load_roundtrip_uploaded_file() -> None:
    """Test that UploadedFile is filtered out during round-trip (opaque provider file_id)."""
    original: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=['Hello', UploadedFile(file_id='file-abc123', provider_name='anthropic'), 'world']
                ),
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
    retry_part = message_part(reloaded, ToolReturnPart, message_index=2)
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
    retry_part = message_part(reloaded, UserPromptPart, message_index=2)
    assert 'Please try again' in str(retry_part.content)


def test_dump_messages_preserves_part_order() -> None:
    """Dumping a `ModelRequest` keeps `ToolReturnPart`s interleaved with user prompts (regression for #5964).

    User content was previously buffered and emitted as a single `UserMessage` at the end, so a
    `ToolReturnPart` following a `UserPromptPart` would be reordered after it. That produces a
    `tool_use` block without an immediately-following `tool_result`, which providers like Anthropic
    reject on the next request. The buffer must be flushed before each tool message so the original
    part order survives, including user prompts on both sides of a tool return.
    """
    original: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='suggest', args={'suggestions': ['Yes', 'No']}, tool_call_id='call_1'),
            ]
        ),
        ModelRequest(
            parts=[
                UserPromptPart(content='Before the tool return.'),
                ToolReturnPart(tool_name='suggest', tool_call_id='call_1', content='suggested'),
                UserPromptPart(content='After the tool return.'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)

    assert [type(msg).__name__ for msg in ag_ui_msgs] == snapshot(
        ['AssistantMessage', 'UserMessage', 'ToolMessage', 'UserMessage']
    )
    tool_msg = ag_ui_msgs[2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == 'call_1'
    assert [getattr(msg, 'content', None) for msg in ag_ui_msgs[1:]] == snapshot(
        ['Before the tool return.', 'suggested', 'After the tool return.']
    )


def test_dump_messages_preserves_uploaded_file_order() -> None:
    """Text straddling an `UploadedFile` keeps its order around the emitted `ActivityMessage`.

    An `UploadedFile` is emitted as an `ActivityMessage` directly, so buffered user text must be
    flushed before it, otherwise text that precedes the file would be reordered after it (the same
    reordering class as #5964). The text on either side is therefore split into separate
    `UserMessage`s rather than combined into one.
    """
    original: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Before the file.',
                        UploadedFile(file_id='file-abc123', provider_name='anthropic'),
                        'After the file.',
                    ]
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, preserve_file_data=True)

    assert [type(msg).__name__ for msg in ag_ui_msgs] == snapshot(['UserMessage', 'ActivityMessage', 'UserMessage'])
    before, activity, after = ag_ui_msgs
    assert before.content == snapshot('Before the file.')
    assert isinstance(activity, ActivityMessage)
    assert activity.content['file_id'] == 'file-abc123'
    assert after.content == snapshot('After the file.')


def test_file_part_dropped_by_default() -> None:
    """Test that FilePart is silently dropped when preserve_file_data=False (default).

    dump_messages drops FilePart from output, and load_messages ignores
    ActivityMessage(pydantic_ai_file) — both without raising errors.
    """
    messages_with_file: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Generate an image')]),
        ModelResponse(
            parts=[
                FilePart(content=BinaryImage(data=b'image data', media_type='image/png')),
                TextPart(content='Here is your image.'),
            ]
        ),
    ]

    # dump_messages drops FilePart by default
    ag_ui_msgs = AGUIAdapter.dump_messages(messages_with_file)
    assert not any(isinstance(m, ActivityMessage) and m.activity_type == 'pydantic_ai_file' for m in ag_ui_msgs)

    # load_messages ignores ActivityMessage(pydantic_ai_file) by default
    ag_ui_msgs_with_activity = AGUIAdapter.dump_messages(messages_with_file, preserve_file_data=True)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs_with_activity)
    assert not any(isinstance(part, FilePart) for msg in reloaded for part in msg.parts)


def test_dump_load_roundtrip_interleaved_text_and_tools() -> None:
    """Test round-trip for response with text interleaved around tool calls.

    When text appears after tool calls, the flush pattern splits them into
    separate AssistantMessages to preserve ordering on round-trip.
    """
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Do things')]),
        ModelResponse(
            parts=[
                TextPart(content='Before tools'),
                ToolCallPart(tool_name='search', args='{"q": "test"}', tool_call_id='call_1'),
                TextPart(content='After tools'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)

    # Text before tools shares an AssistantMessage with the tool call;
    # text after tools gets its own AssistantMessage.
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in ag_ui_msgs] == snapshot(
        [
            {'role': 'user', 'content': 'Do things'},
            {
                'role': 'assistant',
                'content': 'Before tools',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'search', 'arguments': '{"q": "test"}'},
                    },
                ],
            },
            {'role': 'assistant', 'content': 'After tools'},
        ]
    )

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    # Round-trip splits into two ModelResponses due to the two AssistantMessages
    assert reloaded == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Do things', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    TextPart(content='Before tools'),
                    ToolCallPart(tool_name='search', args='{"q": "test"}', tool_call_id='call_1'),
                    TextPart(content='After tools'),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_reasoning_events_empty_content_with_metadata() -> None:
    """Test REASONING_* events for ThinkingPart with no content but with metadata.

    This exercises the path in handle_thinking_end where _reasoning_started is False
    (no content was streamed) but encrypted metadata is present — e.g. redacted thinking.
    """
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    part = ThinkingPart(
        content='',
        id='think_redacted',
        signature='sig_redacted',
    )

    events: list[BaseEvent] = [e async for e in event_stream.handle_thinking_start(part)]
    async for e in event_stream.handle_thinking_end(part):
        events.append(e)

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'subtype': 'message',
                'entity_id': IsStr(),
                'encrypted_value': '{"id": "think_redacted", "signature": "sig_redacted"}',
            },
            {'type': 'REASONING_END', 'message_id': IsStr()},
        ]
    )


@pytest.mark.vcr()
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_thinking_roundtrip_anthropic(allow_model_requests: None, anthropic_api_key: str) -> None:
    """Test that pydantic -> AG-UI -> pydantic round-trip preserves thinking metadata with real Anthropic responses."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings: AnthropicModelSettings = {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024}}
    agent = Agent(m, model_settings=settings)

    result = await agent.run('What is 1+1? Reply in one word.')
    original = result.all_messages()

    ag_ui_msgs = AGUIAdapter.dump_messages(original, ag_ui_version='0.1.13')
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(original, reloaded)

    assert reloaded == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What is 1+1? Reply in one word.', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking what 1+1 equals and wants a one-word reply. The answer is 2, which is one word.',
                        signature='EooCCkYICxgCKkDYW6Ka+Mo73ZE34HVijmFbdV6QH/iRdv+3WuisH3pR8D5aSFASMBsF1F1bZRQFQXuM0+G4H83czthKvHqdqWriEgwB0eJaWoXZWU18NKoaDMH4nN8ZwJ6W9DnYLyIwrdTWmfc5QTqDr8gye3/yrPpV2YPeZnUBoHBLOGl8MUaC6SuGmxcm8rGqf2s+P+ZtKnJPJJzQiTrvPcEkF3ij22w3bXC9yoyZCyJVPcibR2ZZpLYF/UOoZ+BRBs0FCdm/QFXUUe8W1tcQ/ZQgBaW44LTcdzwOSP5hJb25UrPiGWuTytGMxIr7QyG7INpVbmm8JRBIIEzj3gs2zlxdbl17yZ/yZXcYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(content='Two'),
                ],
                timestamp=IsDatetime(),
            ),
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
        capabilities=[PrepareTools(store_state)],
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

        adapter = AGUIAdapter(agent=agent, run_input=run_input)
        async for event in adapter.encode_stream(adapter.run_stream(deps=deps, on_complete=on_complete)):
            events.append(json.loads(event.removeprefix('data: ')))

        assert events == simple_result(outcome={'type': 'success'})
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
        adapter = AGUIAdapter(agent=agent, run_input=run_input)
        async for event in adapter.encode_stream(adapter.run_stream()):
            events.append(json.loads(event.removeprefix('data: ')))

    assert events == simple_result(outcome={'type': 'success'})


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
    adapter = AGUIAdapter(agent=agent, run_input=run_input)
    async for event in adapter.encode_stream(adapter.run_stream()):
        events.append(json.loads(event.removeprefix('data: ')))

    assert events == simple_result(outcome={'type': 'success'})


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
        capabilities=[PrepareTools(store_state)],
    )

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello, how are you?',
        ),
        state={'value': 42},
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)
    async for _ in adapter.encode_stream(adapter.run_stream(deps=CustomStateDeps(state={'value': 0}))):
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
    req = message(messages, ModelRequest)
    assert req.run_id == run_result.run_id

    # Verify events were still streamed normally
    assert len(events) > 0
    assert events[0]['type'] == 'RUN_STARTED'
    assert events[-1]['type'] == 'RUN_FINISHED'


async def test_adapter_sets_current_run_id_on_trailing_mapped_request() -> None:
    """The adapter sets `run_id` on the current run's mapped request, not older history."""
    captured_results: list[AgentRunResult[Any]] = []

    def sync_callback(run_result: AgentRunResult[Any]) -> None:
        captured_results.append(run_result)

    agent = Agent(TestModel())
    run_input = create_input(
        UserMessage(id='msg0', content='Previous question'),
        AssistantMessage(id='msg1', content='Previous response'),
        UserMessage(id='msg2', content='Hello!'),
    )

    await run_and_collect_events(agent, run_input, on_complete=sync_callback)

    assert len(captured_results) == 1
    run_result = captured_results[0]
    messages = run_result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())],
            ),
            ModelResponse(
                parts=[TextPart(content='Previous response')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello!', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=(run_id := IsSameStr()),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=run_id,
                conversation_id=IsStr(),
            ),
        ]
    )
    assert messages[0].run_id is None
    assert messages[1].run_id is None
    assert messages[2].run_id == run_result.run_id
    assert messages[3].run_id == run_result.run_id
    assert run_result.new_messages() == messages[-1:]


async def test_adapter_uses_run_input_thread_id_as_conversation_id() -> None:
    """`RunAgentInput.threadId` is wired through to `gen_ai.conversation.id`."""
    captured_results: list[AgentRunResult[Any]] = []

    agent = Agent(TestModel())
    run_input = create_input(UserMessage(id='msg0', content='Hello!'), thread_id='thread-abc')

    await run_and_collect_events(agent, run_input, on_complete=captured_results.append)

    assert captured_results[0].conversation_id == 'thread-abc'
    assert captured_results[0].all_messages()[-1].conversation_id == 'thread-abc'


async def test_adapter_explicit_conversation_id_overrides_thread_id() -> None:
    """Passing `conversation_id` explicitly to `run_stream_native` overrides `RunAgentInput.threadId`."""
    captured_results: list[AgentRunResult[Any]] = []

    agent = Agent(TestModel())
    run_input = create_input(UserMessage(id='msg0', content='Hello!'), thread_id='thread-abc')
    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=None)

    async for _ in adapter.transform_stream(
        adapter.run_stream_native(conversation_id='explicit-conv-id'),
        on_complete=captured_results.append,
    ):
        pass

    assert captured_results[0].conversation_id == 'explicit-conv-id'


async def test_adapter_run_stream_native_capabilities_kwarg_merged_into_run() -> None:
    """`AGUIAdapter.run_stream_native(capabilities=[...])` extends the run-level capability
    list passed through to `Agent.run_stream_events`."""
    seen_tool_defs: list[ToolDefinition] = []

    async def prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen_tool_defs.extend(tool_defs)
        return tool_defs

    def my_tool() -> str:
        return 'ok'

    agent = Agent(TestModel(), tools=[my_tool])
    run_input = create_input(UserMessage(id='msg0', content='Hello!'))
    adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=None)

    async for _ in adapter.transform_stream(adapter.run_stream_native(capabilities=[PrepareTools(prep)])):
        pass

    assert seen_tool_defs, 'PrepareTools capability passed via run_stream_native(capabilities=...) should fire'


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
                        content=[ImageUrl(url='http://example.com/image.png', _media_type='image/png')],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[VideoUrl(url='http://example.com/video.mp4', _media_type='video/mp4')],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[AudioUrl(url='http://example.com/audio.mp3', _media_type='audio/mpeg')],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[DocumentUrl(url='http://example.com/doc.pdf', _media_type='application/pdf')],
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
                    NativeToolCallPart(
                        tool_name='web_search',
                        args='{"query": "Hello, world!"}',
                        tool_call_id='search_1',
                        provider_name='function',
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content={
                            'results': [
                                {'title': 'Hello, world!', 'url': 'https://en.wikipedia.org/wiki/Hello,_world!'}
                            ]
                        },
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


async def test_builtin_tool_return_json_string_content_parsed() -> None:
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/4623.

    AG-UI ToolMessage.content is always a string. For built-in tools the original
    dict content gets JSON-serialized on the way out. The adapter must parse it
    back so downstream model code (which checks isinstance(content, dict)) doesn't
    silently drop the tool result.
    """
    messages: list[Message] = [
        AssistantMessage(
            id='msg_1',
            tool_calls=[
                ToolCall(
                    id='pyd_ai_builtin|anthropic|srvtoolu_abc123',
                    function=FunctionCall(
                        name='web_fetch',
                        arguments='{"url": "https://example.com"}',
                    ),
                ),
            ],
        ),
        ToolMessage(
            id='msg_2',
            content='{"type": "web_fetch_result", "url": "https://example.com", "page_content": "hello"}',
            tool_call_id='pyd_ai_builtin|anthropic|srvtoolu_abc123',
        ),
    ]

    result = AGUIAdapter.load_messages(messages)
    return_part = message_part(result, NativeToolReturnPart, part_index=1)
    assert return_part.tool_name == 'web_fetch'
    assert return_part.tool_call_id == 'srvtoolu_abc123'
    assert return_part.provider_name == 'anthropic'
    content = return_part.content
    assert content == {'type': 'web_fetch_result', 'url': 'https://example.com', 'page_content': 'hello'}


async def test_builtin_tool_return_plain_string_content_preserved() -> None:
    """Plain string content that isn't valid JSON stays as-is."""
    messages: list[Message] = [
        AssistantMessage(
            id='msg_1',
            tool_calls=[
                ToolCall(
                    id='pyd_ai_builtin|anthropic|srvtoolu_abc456',
                    function=FunctionCall(
                        name='web_fetch',
                        arguments='{"url": "https://example.com"}',
                    ),
                ),
            ],
        ),
        ToolMessage(
            id='msg_2',
            content='just a plain string, not JSON',
            tool_call_id='pyd_ai_builtin|anthropic|srvtoolu_abc456',
        ),
    ]

    result = AGUIAdapter.load_messages(messages)
    return_part = message_part(result, NativeToolReturnPart, part_index=1)
    assert return_part.content == 'just a plain string, not JSON'


async def test_builtin_tool_return_non_string_content_passthrough() -> None:
    """When ToolMessage.content is already a non-string (e.g. dict), it passes through without JSON parsing."""
    tool_msg = ToolMessage.model_construct(
        id='msg_2',
        content={'type': 'web_fetch_result', 'url': 'https://example.com'},
        tool_call_id='pyd_ai_builtin|anthropic|srvtoolu_abc789',
    )
    messages: list[Message] = [
        AssistantMessage(
            id='msg_1',
            tool_calls=[
                ToolCall(
                    id='pyd_ai_builtin|anthropic|srvtoolu_abc789',
                    function=FunctionCall(
                        name='web_fetch',
                        arguments='{"url": "https://example.com"}',
                    ),
                ),
            ],
        ),
        tool_msg,
    ]

    result = AGUIAdapter.load_messages(messages)
    return_part = message_part(result, NativeToolReturnPart, part_index=1)
    assert return_part.content == {'type': 'web_fetch_result', 'url': 'https://example.com'}


async def test_builtin_tool_return_non_string_scalar_content_passthrough() -> None:
    """A non-string, non-mapping/sequence `content` (a scalar) passes through untouched.

    Only mappings/sequences can nest multimodal items, so the discriminator is skipped for a scalar and
    the value is returned as-is rather than coerced.
    """
    tool_msg = ToolMessage.model_construct(
        id='msg_2',
        content=42,
        tool_call_id='pyd_ai_builtin|anthropic|srvtoolu_scalar',
    )
    messages: list[Message] = [
        AssistantMessage(
            id='msg_1',
            tool_calls=[
                ToolCall(
                    id='pyd_ai_builtin|anthropic|srvtoolu_scalar',
                    function=FunctionCall(name='web_fetch', arguments='{"url": "https://example.com"}'),
                ),
            ],
        ),
        tool_msg,
    ]

    result = AGUIAdapter.load_messages(messages)
    return_part = message_part(result, NativeToolReturnPart, part_index=1)
    assert return_part.content == 42


async def test_user_message_empty_content_list_skipped() -> None:
    """A UserMessage with an empty content list produces no UserPromptPart."""
    messages: list[Message] = [
        UserMessage(id='msg_1', content=[]),
    ]

    result = AGUIAdapter.load_messages(messages)
    assert result == []


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
            0: NativeToolCallPart(
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
            1: NativeToolReturnPart(
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
            2: NativeToolCallPart(
                tool_name=WebSearchTool.kind,
                args='{"query": "Hello world history"}',
                tool_call_id='search_2',
                provider_name='function',
            )
        }
        yield {
            3: NativeToolReturnPart(
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
                'outcome': {'type': 'success'},
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
            part=ToolCallPart(tool_name='tool_call_1', args='{"query": "Hello world"}', tool_call_id='tool_call_1'),
            args_valid=True,
        )
        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_2', args='{"query": "Goodbye world"}', tool_call_id='tool_call_2'),
            args_valid=True,
        )

        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='tool_call_1', content='Hi!', tool_call_id='tool_call_1')
        )
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='tool_call_2', content='Bye!', tool_call_id='tool_call_2')
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
            part=ToolCallPart(tool_name='tool_call_3', args='{"query": "Hello world"}', tool_call_id='tool_call_3'),
            args_valid=True,
        )
        yield FunctionToolCallEvent(
            part=ToolCallPart(tool_name='tool_call_4', args='{"query": "Goodbye world"}', tool_call_id='tool_call_4'),
            args_valid=True,
        )

        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='tool_call_3', content='Hi!', tool_call_id='tool_call_3')
        )
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='tool_call_4', content='Bye!', tool_call_id='tool_call_4')
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
                'outcome': {'type': 'success'},
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
            part=ToolReturnPart(
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


async def test_dispatch_request():
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

    response = await AGUIAdapter.dispatch_request(starlette_request, agent=agent)

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
                    'outcome': {'type': 'success'},
                },
                'more_body': True,
            },
            {'type': 'http.response.body', 'body': b'', 'more_body': False},
        ]
    )


def test_dump_load_roundtrip_uploaded_file_preserved() -> None:
    """Test UploadedFile round-trips via ActivityMessage when preserve_file_data=True."""
    original: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Describe this file',
                        UploadedFile(
                            file_id='file-abc123',
                            provider_name='anthropic',
                            media_type='application/pdf',
                            vendor_metadata={'source': 'upload'},
                            identifier='my-doc.pdf',
                        ),
                    ]
                ),
            ]
        ),
        ModelResponse(parts=[TextPart(content='I see a PDF.')]),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original, preserve_file_data=True)

    # Verify ActivityMessage was emitted
    activity_msgs = [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)]
    assert len(activity_msgs) == 1
    assert activity_msgs[0].activity_type == 'pydantic_ai_uploaded_file'
    assert activity_msgs[0].content['file_id'] == 'file-abc123'

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs, preserve_file_data=True)

    # The text and UploadedFile come back as separate UserPromptParts
    request_parts = [p for msg in reloaded if isinstance(msg, ModelRequest) for p in msg.parts]
    user_parts = [p for p in request_parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 2

    # First UserPromptPart has the text
    assert user_parts[0].content == 'Describe this file'

    # Second UserPromptPart has the UploadedFile
    assert isinstance(user_parts[1].content, list)
    uploaded = user_parts[1].content[0]
    assert isinstance(uploaded, UploadedFile)
    assert uploaded.file_id == 'file-abc123'
    assert uploaded.provider_name == 'anthropic'
    assert uploaded.media_type == 'application/pdf'
    assert uploaded.vendor_metadata == {'source': 'upload'}
    assert uploaded.identifier == 'my-doc.pdf'


@pytest.mark.parametrize(
    'case_id',
    [
        'plain-string',
        'structured-dict',
        'structured-list',
        'single-image',
        'text-then-audio',
        'image-and-video',
        'document-url',
        'dict-with-nested-image',
    ],
)
def test_dump_load_roundtrip_tool_return_multimodal(
    case_id: str,
    tiny_image: BinaryImage,
    tiny_audio: BinaryContent,
    tiny_video: BinaryContent,
) -> None:
    """Multimodal and structured `ToolReturnPart.content` ride inline in `ToolMessage.content` and round-trip with no flag.

    `ag_ui.core.ToolMessage.content` is a plain `str`, but it already carries JSON for structured returns,
    so the full content — files serialized as base64/URL dicts included — is written inline and rehydrated
    on load via the `ToolReturnContent` discriminator. No sidecar `ActivityMessage` and no `preserve_file_data`
    flag are involved: inline content round-trips verbatim through any frontend, whereas a custom sidecar
    only round-trips if the frontend echoes it back. A file nested in a mapping (unreachable by
    `BaseToolReturnPart.files`) round-trips too.
    """
    contents: dict[str, Any] = {
        'plain-string': 'just some text',
        'structured-dict': {'temperature': 21, 'unit': 'C'},
        'structured-list': [1, 2, 3],
        'single-image': tiny_image,
        'text-then-audio': ['the audio narration says...', tiny_audio],
        'image-and-video': [tiny_image, tiny_video],
        'document-url': DocumentUrl(url='https://example.com/doc.pdf', media_type='application/pdf'),
        'dict-with-nested-image': {'caption': 'see image', 'attachment': tiny_image},
    }
    content = contents[case_id]
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Call tool')]),
        ModelResponse(parts=[ToolCallPart(tool_name='get_files', tool_call_id='tc-1', args='{}')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_files', tool_call_id='tc-1', content=content)]),
        ModelResponse(parts=[TextPart(content='Done')]),
    ]

    # No flag: tool-return files always ride inline in the tool message, never a sidecar.
    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    assert [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)] == []
    tool_msgs = [m for m in ag_ui_msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert isinstance(tool_msgs[0].content, str)

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    tool_returns = [
        p for m in reloaded if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == snapshot(
        [ToolReturnPart(tool_name='get_files', tool_call_id='tc-1', content=content, timestamp=IsDatetime())]
    )


def test_dump_tool_return_none_content_becomes_empty_string() -> None:
    """A `None` tool-return content dumps to an empty string, since AG-UI `ToolMessage.content` is text-only.

    On reload the empty string is not valid JSON, so it stays `''` rather than round-tripping back to `None`.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='noop', tool_call_id='tc-1', args='{}')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='noop', tool_call_id='tc-1', content=None)]),
    ]
    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    tool_msgs = [m for m in ag_ui_msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == ''

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    tool_returns = [
        p for m in reloaded if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == snapshot(
        [ToolReturnPart(tool_name='noop', tool_call_id='tc-1', content='', timestamp=IsDatetime())]
    )


@pytest.mark.parametrize('content', ['123', 'true', 'null'])
def test_tool_return_json_scalar_string_stays_string(content: str) -> None:
    """A string return that happens to be a valid JSON *scalar* must not change type on the round-trip.

    `ToolMessage.content` is text-only on the AG-UI wire, so a string return is dumped verbatim. Re-parsing it
    through the discriminator would turn `'123'` into `123`, `'true'` into `True`, etc. The rehydrator only runs the
    discriminator on a parsed mapping/sequence (where nested multimodal items can live), leaving scalars as strings.
    A container-shaped string (`'[1, 2]'`) is wire-indistinguishable from a real list return, so it does rehydrate —
    that ambiguity is inherent to the text-only wire and only the scalar coercion is recoverable.
    """
    original: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='get_value', tool_call_id='tc-1', args='{}')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_value', tool_call_id='tc-1', content=content)]),
    ]
    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    tool_returns = [
        p for m in reloaded if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == snapshot(
        [ToolReturnPart(tool_name='get_value', tool_call_id='tc-1', content=content, timestamp=IsDatetime())]
    )


def test_dump_load_roundtrip_builtin_tool_return_multimodal(tiny_image: BinaryImage) -> None:
    """Multimodal `NativeToolReturnPart.content` rides inline in `ToolMessage.content` and round-trips with no flag."""
    original: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search')]),
        ModelResponse(
            parts=[
                NativeToolCallPart(
                    tool_name='web_search',
                    tool_call_id='call_1',
                    args='{"q": "test"}',
                    provider_name='anthropic',
                ),
                NativeToolReturnPart(
                    tool_name='web_search',
                    tool_call_id='call_1',
                    content=['Search results', tiny_image],
                    provider_name='anthropic',
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(original)
    assert [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)] == []

    reloaded = AGUIAdapter.load_messages(ag_ui_msgs)
    returns = [
        p for m in reloaded if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, NativeToolReturnPart)
    ]
    assert returns == snapshot(
        [
            NativeToolReturnPart(
                tool_name='web_search',
                tool_call_id='call_1',
                content=['Search results', tiny_image],
                timestamp=IsDatetime(),
                provider_name='anthropic',
            )
        ]
    )


def test_load_messages_builtin_tool_return_json_content_rehydrates() -> None:
    """A builtin tool's JSON `ToolMessage.content` rehydrates to structured content (inline, no sidecar)."""
    prefixed_id = 'pyd_ai_builtin|anthropic|call_1'
    raw = [
        AssistantMessage(
            id='msg-1',
            tool_calls=[
                ToolCall(
                    id=prefixed_id,
                    type='function',
                    function=FunctionCall(name='web_search', arguments='{"q": "test"}'),
                ),
            ],
        ),
        ToolMessage(
            id='msg-2',
            content='[{"a": 1}, {"b": 2}]',
            tool_call_id=prefixed_id,
        ),
    ]

    reloaded = AGUIAdapter.load_messages(raw)
    returns = [
        p for m in reloaded if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, NativeToolReturnPart)
    ]
    assert returns == snapshot(
        [
            NativeToolReturnPart(
                tool_name='web_search',
                tool_call_id='call_1',
                content=[{'a': 1}, {'b': 2}],
                timestamp=IsDatetime(),
                provider_name='anthropic',
            )
        ]
    )


@pytest.mark.parametrize(
    'version,expected_reasoning',
    [
        pytest.param('0.1.10', snapshot([]), id='v010-drops-thinking'),
        pytest.param(
            '0.1.13',
            snapshot(
                [{'content': 'Deep thoughts...', 'encrypted_value': '{"signature": "sig_xyz"}', 'role': 'reasoning'}]
            ),
            id='v013-includes-reasoning',
        ),
    ],
)
def test_dump_messages_thinking_version_gated(version: str, expected_reasoning: list[Any]) -> None:
    """Test that dump_messages drops ThinkingPart at <0.1.13 and emits ReasoningMessage at >=0.1.13."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Think about this')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='Deep thoughts...', signature='sig_xyz'),
                TextPart(content='Conclusion'),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, ag_ui_version=version)
    reasoning_msgs = [m for m in ag_ui_msgs if isinstance(m, ReasoningMessage)]
    assert [m.model_dump(exclude={'id'}) for m in reasoning_msgs] == expected_reasoning
    assert any(isinstance(m, AssistantMessage) and m.content == 'Conclusion' for m in ag_ui_msgs)


async def test_tool_return_with_files():
    """A tool return carrying files streams its full content inline in `ToolCallResultEvent.content`.

    Files are serialized (base64 for `BinaryContent`, URL for `ImageUrl`) using the same dump as history
    serialization, not collapsed to a `[File: ...]` placeholder, so a streaming frontend can echo the
    content back and have the files rehydrated and re-sent to the model on the next step.
    """

    async def event_generator():
        # Content with text and file - files property extracts BinaryContent from the list
        yield FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name='get_image',
                content=['Image analysis result', BinaryContent(data=b'img', media_type='image/png')],
                tool_call_id='call_1',
            )
        )
        # Content with only a FileUrl - files property returns [ImageUrl]
        yield FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name='get_url',
                content=ImageUrl(url='https://example.com/image.jpg'),
                tool_call_id='call_2',
            )
        )

    run_input = create_input(UserMessage(id='msg_1', content='Analyze images'))
    event_stream = AGUIEventStream(run_input=run_input)
    events = [
        json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    tool_results = [e for e in events if e.get('type') == 'TOOL_CALL_RESULT']
    assert tool_results == snapshot(
        [
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'call_1',
                'content': '["Image analysis result",{"data":"aW1n","media_type":"image/png","vendor_metadata":null,"kind":"binary","identifier":"978ea7"}]',
                'role': 'tool',
            },
            {
                'type': 'TOOL_CALL_RESULT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'toolCallId': 'call_2',
                'content': '{"url":"https://example.com/image.jpg","force_download":false,"vendor_metadata":null,"kind":"image-url","media_type":"image/jpeg","identifier":"39cfc4"}',
                'role': 'tool',
            },
        ]
    )


async def test_stream_tool_return_files_roundtrip_to_history() -> None:
    """The content a tool return streams can be replayed as history and rehydrates to the original file.

    This is the round-trip that matters for a streaming frontend: the file a tool produced during the run
    is streamed inline in `ToolCallResultEvent.content`, and when the frontend echoes that content back on
    the next request it is recovered as a `BinaryImage` — so it can be sent to the model again (preserving
    prompt-cache prefixes) instead of degrading to a text placeholder.
    """
    image = BinaryImage(data=b'img', media_type='image/png')

    async def event_generator():
        yield FunctionToolResultEvent(
            part=ToolReturnPart(tool_name='get_image', content=['here it is', image], tool_call_id='call_1')
        )

    run_input = create_input(UserMessage(id='msg_1', content='Analyze'))
    event_stream = AGUIEventStream(run_input=run_input)
    events = [
        json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]
    result_content = next(e['content'] for e in events if e.get('type') == 'TOOL_CALL_RESULT')

    # Replay the streamed content back as client-submitted history, paired with its tool call.
    reloaded = AGUIAdapter.load_messages(
        [
            AssistantMessage(
                id='msg_2',
                tool_calls=[
                    ToolCall(id='call_1', type='function', function=FunctionCall(name='get_image', arguments='{}')),
                ],
            ),
            ToolMessage(id='msg_3', content=result_content, tool_call_id='call_1'),
        ]
    )
    tool_returns = [
        p for m in reloaded if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == snapshot(
        [
            ToolReturnPart(
                tool_name='get_image', content=['here it is', image], tool_call_id='call_1', timestamp=IsDatetime()
            )
        ]
    )


# region: Coverage — event_stream thinking version branches


async def test_thinking_events_v010_with_content() -> None:
    """Test v0.1.10 THINKING_* events for ThinkingPart with content."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.10')

    part = ThinkingPart(content='Some thoughts', signature='sig_abc')

    events: list[BaseEvent] = []
    async for e in event_stream.handle_thinking_start(part):
        events.append(e)
    async for e in event_stream.handle_thinking_end(part):
        events.append(e)

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'THINKING_START'},
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'Some thoughts'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'THINKING_END'},
        ]
    )


async def test_thinking_events_v010_empty_content() -> None:
    """Test v0.1.10 early return when ThinkingPart has no content."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.10')

    part = ThinkingPart(content='', signature='sig_abc')

    events = [e async for e in event_stream.handle_thinking_start(part)]
    events.extend([e async for e in event_stream.handle_thinking_end(part)])

    assert events == []


async def test_thinking_delta_v013() -> None:
    """Test v0.1.13 REASONING_* events emitted via handle_thinking_delta."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    start_part = ThinkingPart(content='')
    events: list[BaseEvent] = [e async for e in event_stream.handle_thinking_start(start_part)]

    delta = ThinkingPartDelta(content_delta='chunk1')
    async for e in event_stream.handle_thinking_delta(delta):
        events.append(e)

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {'type': 'REASONING_MESSAGE_START', 'message_id': IsStr(), 'role': 'reasoning'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'chunk1'},
        ]
    )


async def test_thinking_end_v013_no_content_no_metadata() -> None:
    """Test v0.1.13 early return when ThinkingPart has no content and no encrypted metadata."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    part = ThinkingPart(content='')

    events = [e async for e in event_stream.handle_thinking_start(part)]
    events.extend([e async for e in event_stream.handle_thinking_end(part)])

    assert events == []


async def test_thinking_delta_v013_after_content_start() -> None:
    """Test v0.1.13 delta skips START/MESSAGE_START when reasoning already started."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    start_part = ThinkingPart(content='initial')
    events = [e async for e in event_stream.handle_thinking_start(start_part)]

    delta = ThinkingPartDelta(content_delta='more')
    events.extend([e async for e in event_stream.handle_thinking_delta(delta)])

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {'type': 'REASONING_MESSAGE_START', 'message_id': IsStr(), 'role': 'reasoning'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'initial'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'more'},
        ]
    )


async def test_thinking_end_v010_with_content() -> None:
    """Test v0.1.10 end emits TextMessageEnd when content was streamed, and ThinkingStart when not started."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))

    # Case 1: start with content → _reasoning_started=True, _reasoning_text=True
    # end should emit TextMessageEnd + ThinkingEnd
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.10')
    part = ThinkingPart(content='text')
    events = [e async for e in event_stream.handle_thinking_start(part)]
    events.extend([e async for e in event_stream.handle_thinking_end(part)])

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'THINKING_START'},
            {'type': 'THINKING_TEXT_MESSAGE_START'},
            {'type': 'THINKING_TEXT_MESSAGE_CONTENT', 'delta': 'text'},
            {'type': 'THINKING_TEXT_MESSAGE_END'},
            {'type': 'THINKING_END'},
        ]
    )

    # Case 2: start with empty content → _reasoning_started=False
    # end with content → hits ThinkingStartEvent at line 246
    event_stream2 = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.10')
    empty_part = ThinkingPart(content='')
    events2 = [e async for e in event_stream2.handle_thinking_start(empty_part)]

    full_part = ThinkingPart(content='non-empty')
    events2.extend([e async for e in event_stream2.handle_thinking_end(full_part)])

    assert [e.model_dump(exclude_none=True) for e in events2] == snapshot(
        [
            {'type': 'THINKING_START'},
            {'type': 'THINKING_END'},
        ]
    )


async def test_thinking_end_v013_no_encrypted_metadata() -> None:
    """Test v0.1.13 end skips encrypted_value event when part has no signature or metadata."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    part = ThinkingPart(content='text')
    events = [e async for e in event_stream.handle_thinking_start(part)]
    events.extend([e async for e in event_stream.handle_thinking_end(part)])

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {'type': 'REASONING_MESSAGE_START', 'message_id': IsStr(), 'role': 'reasoning'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'text'},
            {'type': 'REASONING_MESSAGE_END', 'message_id': IsStr()},
            {'type': 'REASONING_END', 'message_id': IsStr()},
        ]
    )


# endregion

# region: Coverage — encrypted_metadata branch gap


async def test_thinking_encrypted_metadata_partial_fields() -> None:
    """Test thinking_encrypted_metadata with signature but no provider_name."""
    run_input = create_input(UserMessage(id='msg_1', content='test'))
    event_stream = AGUIEventStream(run_input, accept=SSE_CONTENT_TYPE, ag_ui_version='0.1.13')

    part = ThinkingPart(content='Thoughts', signature='sig_only')

    events: list[BaseEvent] = []
    async for e in event_stream.handle_thinking_start(part):
        events.append(e)
    async for e in event_stream.handle_thinking_end(part):
        events.append(e)

    assert [e.model_dump(exclude_none=True) for e in events] == snapshot(
        [
            {'type': 'REASONING_START', 'message_id': IsStr()},
            {'type': 'REASONING_MESSAGE_START', 'message_id': IsStr(), 'role': 'reasoning'},
            {'type': 'REASONING_MESSAGE_CONTENT', 'message_id': IsStr(), 'delta': 'Thoughts'},
            {'type': 'REASONING_MESSAGE_END', 'message_id': IsStr()},
            {
                'type': 'REASONING_ENCRYPTED_VALUE',
                'subtype': 'message',
                'entity_id': IsStr(),
                'encrypted_value': '{"signature": "sig_only"}',
            },
            {'type': 'REASONING_END', 'message_id': IsStr()},
        ]
    )


# endregion

# region: Coverage — adapter uploaded file edge cases


def test_load_messages_uploaded_file_missing_fields() -> None:
    """Test load_messages raises ValueError for malformed pydantic_ai_uploaded_file ActivityMessage."""
    with pytest.raises(ValueError, match='must have non-empty file_id and provider_name'):
        AGUIAdapter.load_messages(
            [ActivityMessage(id='msg_1', activity_type='pydantic_ai_uploaded_file', content={})],
            preserve_file_data=True,
        )


def test_load_messages_uploaded_file_dropped_by_default() -> None:
    """AG-UI is default-safe: a client `pydantic_ai_uploaded_file` activity is ignored unless
    `preserve_file_data=True`, so a client-supplied `file_id` is never honored by default."""
    activity = ActivityMessage(
        id='msg_1',
        activity_type='pydantic_ai_uploaded_file',
        content={'file_id': 's3://private-bucket/payroll.pdf', 'provider_name': 'bedrock'},
    )

    # Default (preserve_file_data=False): the activity is ignored, no UploadedFile is produced.
    assert AGUIAdapter.load_messages([activity]) == []

    # Opt-in (preserve_file_data=True): the UploadedFile is reconstructed.
    reloaded = AGUIAdapter.load_messages([activity], preserve_file_data=True)
    uploaded = [
        item
        for msg in reloaded
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(item, UploadedFile)
    ]
    assert len(uploaded) == 1
    assert uploaded[0].file_id == 's3://private-bucket/payroll.pdf'


def test_agui_adapter_allow_uploaded_files_end_to_end() -> None:
    """End-to-end: with `preserve_file_data=True` the adapter reconstructs a client `UploadedFile`, and
    with `allow_uploaded_files=True` it survives `sanitize_messages` into the agent-visible messages.

    `preserve_file_data` controls the AG-UI wire representation (reconstructing the `UploadedFile` from a
    `pydantic_ai_uploaded_file` activity message), while `allow_uploaded_files` is the inbound security
    gate; both must be set for a client-submitted uploaded file to reach the agent.
    """
    activity = ActivityMessage(
        id='activity_1',
        activity_type='pydantic_ai_uploaded_file',
        content={'file_id': 's3://private-bucket/payroll.pdf', 'provider_name': 'bedrock'},
    )
    run_input = create_input(activity)
    agent: Agent[None, str] = Agent(model=TestModel())

    adapter = AGUIAdapter(agent=agent, run_input=run_input, preserve_file_data=True, allow_uploaded_files=True)

    # `preserve_file_data=True` reconstructs the `UploadedFile` from the activity message.
    loaded_uploaded = [
        item
        for msg in adapter.messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(item, UploadedFile)
    ]
    assert len(loaded_uploaded) == 1
    assert loaded_uploaded[0].file_id == 's3://private-bucket/payroll.pdf'

    # `allow_uploaded_files=True` lets it survive sanitization with no warning.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sanitized = adapter.sanitize_messages(adapter.messages)
    sanitized_uploaded = [
        item
        for msg in sanitized
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(item, UploadedFile)
    ]
    assert len(sanitized_uploaded) == 1
    assert sanitized_uploaded[0].file_id == 's3://private-bucket/payroll.pdf'


def test_agui_preserve_file_data_and_allow_uploaded_files_independent() -> None:
    """`preserve_file_data` (representation) and `allow_uploaded_files` (security) are independent.

    With `preserve_file_data=True, allow_uploaded_files=False`, `load_messages` still reconstructs the
    `UploadedFile` from the activity message, but `sanitize_messages` DROPS it with the uploaded-file
    warning because the security gate is closed.
    """
    activity = ActivityMessage(
        id='activity_1',
        activity_type='pydantic_ai_uploaded_file',
        content={'file_id': 's3://private-bucket/payroll.pdf', 'provider_name': 'bedrock'},
    )
    run_input = create_input(activity)
    agent: Agent[None, str] = Agent(model=TestModel())

    adapter = AGUIAdapter(agent=agent, run_input=run_input, preserve_file_data=True, allow_uploaded_files=False)

    # Representation opt-in still reconstructs the `UploadedFile`.
    loaded_uploaded = [
        item
        for msg in adapter.messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(item, UploadedFile)
    ]
    assert len(loaded_uploaded) == 1

    # But the closed security gate drops it during sanitization with a warning.
    with pytest.warns(UserWarning, match=r"uploaded file\(s\) for provider\(s\) \['bedrock'\]"):
        sanitized = adapter.sanitize_messages(adapter.messages)
    sanitized_uploaded = [
        item
        for msg in sanitized
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
        for item in (part.content if isinstance(part.content, list) else [part.content])
        if isinstance(item, UploadedFile)
    ]
    assert sanitized_uploaded == []


def test_dump_messages_uploaded_file_with_vendor_metadata() -> None:
    """Test dump_messages includes vendor_metadata in ActivityMessage when present on UploadedFile."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        UploadedFile(
                            file_id='file-xyz',
                            provider_name='openai',
                            media_type='text/plain',
                            vendor_metadata={'custom': 'data'},
                        ),
                    ]
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, preserve_file_data=True)
    activity_msgs = [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)]
    assert [m.model_dump() for m in activity_msgs] == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'activity',
                'activity_type': 'pydantic_ai_uploaded_file',
                'content': {
                    'file_id': 'file-xyz',
                    'provider_name': 'openai',
                    'media_type': 'text/plain',
                    'identifier': '6f0bbc',
                    'vendor_metadata': {'custom': 'data'},
                },
            }
        ]
    )


def test_dump_messages_uploaded_file_without_vendor_metadata() -> None:
    """Test dump_messages omits vendor_metadata from ActivityMessage when None on UploadedFile."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        UploadedFile(
                            file_id='file-xyz',
                            provider_name='openai',
                            media_type='text/plain',
                        ),
                    ]
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, preserve_file_data=True)
    activity_msgs = [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)]
    assert [m.model_dump() for m in activity_msgs] == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'activity',
                'activity_type': 'pydantic_ai_uploaded_file',
                'content': {
                    'file_id': 'file-xyz',
                    'provider_name': 'openai',
                    'media_type': 'text/plain',
                    'identifier': '6f0bbc',
                },
            }
        ]
    )


def test_dump_messages_file_part_with_vendor_metadata() -> None:
    """Test dump_messages includes vendor_metadata in the file ActivityMessage when present on a FilePart."""
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                FilePart(
                    content=BinaryImage(
                        data=b'generated file content',
                        media_type='image/png',
                        vendor_metadata={'detail': 'high'},
                    ),
                ),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, preserve_file_data=True)
    activity_msgs = [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)]
    assert len(activity_msgs) == 1
    assert activity_msgs[0].activity_type == 'pydantic_ai_file'
    assert activity_msgs[0].content.get('vendor_metadata') == {'detail': 'high'}


def test_dump_messages_file_part_without_vendor_metadata() -> None:
    """Test dump_messages omits vendor_metadata from the file ActivityMessage when None on a FilePart."""
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                FilePart(content=BinaryImage(data=b'generated file content', media_type='image/png')),
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, preserve_file_data=True)
    activity_msgs = [m for m in ag_ui_msgs if isinstance(m, ActivityMessage)]
    assert len(activity_msgs) == 1
    assert 'vendor_metadata' not in activity_msgs[0].content


def test_load_messages_file_part_ignores_non_dict_vendor_metadata() -> None:
    """Test load_messages ignores client-supplied vendor_metadata that isn't a dict.

    `vendor_metadata` is typed `Any` in the wire schema and set on a `BinaryContent` that
    doesn't validate on assignment, so a malformed value must not reach the model.
    """
    data_uri = BinaryImage(data=b'generated file content', media_type='image/png').data_uri
    content: dict[str, Any] = {'url': data_uri, 'media_type': 'image/png', 'vendor_metadata': 'not-a-dict'}

    reloaded = AGUIAdapter.load_messages(
        [ActivityMessage(id='activity-1', activity_type='pydantic_ai_file', content=content)],
        preserve_file_data=True,
    )
    file_parts = [part for message in reloaded for part in message.parts if isinstance(part, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].content.vendor_metadata is None


# endregion


# region: Coverage — parse_ag_ui_version validation + TextContent + detect fallback


def test_parse_ag_ui_version_invalid() -> None:
    """Test that parse_ag_ui_version raises UserError for malformed input."""
    with pytest.raises(UserError, match="Invalid AG-UI version 'latest'"):
        parse_ag_ui_version('latest')

    with pytest.raises(UserError, match="Invalid AG-UI version ''"):
        parse_ag_ui_version('')


def test_parse_ag_ui_version_prerelease() -> None:
    """Test that parse_ag_ui_version strips pre-release suffixes."""
    assert parse_ag_ui_version('0.1.13a1') == snapshot((0, 1, 13))
    assert parse_ag_ui_version('0.1.13b2') == snapshot((0, 1, 13))
    assert parse_ag_ui_version('0.1.13rc1') == snapshot((0, 1, 13))
    assert parse_ag_ui_version('0.1.13.dev0') == snapshot((0, 1, 13))
    assert parse_ag_ui_version('0.1.x') == snapshot((0, 1))


def test_detect_ag_ui_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that detect_ag_ui_version returns '0.1.10' when package is not found."""

    def _raise_not_found(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError()

    monkeypatch.setattr('pydantic_ai.ui.ag_ui._utils.importlib.metadata.version', _raise_not_found)
    assert detect_ag_ui_version() == snapshot('0.1.10')


def test_detect_ag_ui_version_old(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that detect_ag_ui_version returns the raw installed version string."""

    def _return_old_version(_name: str) -> str:
        return '0.1.10'

    monkeypatch.setattr('pydantic_ai.ui.ag_ui._utils.importlib.metadata.version', _return_old_version)
    assert detect_ag_ui_version() == snapshot('0.1.10')


def test_dump_messages_text_content() -> None:
    """Test that TextContent in UserPromptPart is converted to TextInputContent."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=[TextContent(content='hello')])]),
    ]

    result = AGUIAdapter.dump_messages(messages)
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in result] == snapshot(
        [{'role': 'user', 'content': 'hello'}]
    )


# region multimodal and coverage tests


@pytest.mark.parametrize(
    'input_content,expected_type',
    [
        pytest.param(
            ImageInputContent(
                source=InputContentUrlSource(type='url', value='https://example.com/photo.jpg', mime_type='image/jpeg')
            ),
            ImageUrl(url='https://example.com/photo.jpg', media_type='image/jpeg'),
            id='image-url',
        ),
        pytest.param(
            AudioInputContent(
                source=InputContentUrlSource(type='url', value='https://example.com/clip.mp3', mime_type='audio/mpeg')
            ),
            AudioUrl(url='https://example.com/clip.mp3', media_type='audio/mpeg'),
            id='audio-url',
        ),
        pytest.param(
            VideoInputContent(
                source=InputContentUrlSource(type='url', value='https://example.com/vid.mp4', mime_type='video/mp4')
            ),
            VideoUrl(url='https://example.com/vid.mp4', media_type='video/mp4'),
            id='video-url',
        ),
        pytest.param(
            DocumentInputContent(
                source=InputContentUrlSource(
                    type='url', value='https://example.com/doc.pdf', mime_type='application/pdf'
                )
            ),
            DocumentUrl(url='https://example.com/doc.pdf', media_type='application/pdf'),
            id='document-url',
        ),
    ]
    if imports_successful()
    else [],
)
def test_load_multimodal_url_sources(
    input_content: ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent,
    expected_type: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
) -> None:
    """Test that typed multimodal URL input content is converted to the correct Pydantic AI URL type."""
    messages = AGUIAdapter.load_messages([UserMessage(id='msg-1', content=[input_content])])
    assert len(messages) == 1
    request = message(messages, ModelRequest)
    assert len(request.parts) == 1
    part = message_part(messages, UserPromptPart)
    assert isinstance(part.content, list)
    assert len(part.content) == 1
    assert part.content[0] == expected_type


def test_load_multimodal_data_source() -> None:
    """Test that multimodal data source input content is converted to BinaryContent."""
    messages = AGUIAdapter.load_messages(
        [
            UserMessage(
                id='msg-1',
                content=[
                    ImageInputContent(
                        source=InputContentDataSource(type='data', value='aGVsbG8=', mime_type='image/png')
                    )
                ],
            )
        ]
    )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[BinaryContent(data=b'hello', media_type='image/png')], timestamp=IsDatetime()
                    ),
                ]
            )
        ]
    )


def test_dump_messages_multimodal_url() -> None:
    """Test that media URLs are dumped as typed multimodal content with ag_ui_version >= 0.1.15."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=[ImageUrl(url='https://example.com/img.png', media_type='image/png')])]
        ),
    ]
    result = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.15')
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in result] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'source': {
                            'type': 'url',
                            'value': 'https://example.com/img.png',
                            'mime_type': 'image/png',
                        },
                        'type': 'image',
                    }
                ],
            }
        ]
    )


def test_dump_messages_legacy_binary_content() -> None:
    """Test that media URLs and BinaryContent are dumped as BinaryInputContent with ag_ui_version < 0.1.15."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/img.png', media_type='image/png'),
                        BinaryContent(data=b'raw data', media_type='image/jpeg'),
                    ]
                )
            ]
        ),
    ]
    result = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.10')
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in result] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'type': 'binary', 'url': 'https://example.com/img.png', 'mime_type': 'image/png'},
                    {'type': 'binary', 'data': 'cmF3IGRhdGE=', 'mime_type': 'image/jpeg'},
                ],
            }
        ]
    )


def test_multimodal_roundtrip_preserves_file_vendor_metadata() -> None:
    """`vendor_metadata` on `FileUrl`/`BinaryContent` survives a dump -> load round-trip (ag-ui >= 0.1.15).

    Regression test for #5764: the AG-UI adapter dropped `vendor_metadata`
    (e.g. OpenAI/xAI image `detail`, Google `video_metadata`) for every
    `ImageUrl`/`AudioUrl`/`VideoUrl`/`DocumentUrl`/`BinaryContent`, even though the adjacent
    `UploadedFile` branch already round-tripped it. Multimodal input content carries it under
    a `vendor_metadata` key in the typed part's `metadata` field.
    """
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(
                            url='https://example.com/image.png',
                            media_type='image/png',
                            vendor_metadata={'detail': 'high'},
                        ),
                        AudioUrl(
                            url='https://example.com/audio.mp3',
                            media_type='audio/mpeg',
                            vendor_metadata={'foo': 'bar'},
                        ),
                        VideoUrl(
                            url='https://example.com/video.mp4',
                            media_type='video/mp4',
                            vendor_metadata={'fps': 5},
                        ),
                        DocumentUrl(
                            url='https://example.com/doc.pdf',
                            media_type='application/pdf',
                            vendor_metadata={'foo': 'baz'},
                        ),
                        BinaryContent(
                            data=b'fake_doc',
                            media_type='application/pdf',
                            vendor_metadata={'detail': 'low'},
                        ),
                    ]
                )
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.15')
    # The dumped `metadata` is the external contract a frontend persists and re-sends.
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in ag_ui_msgs] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {'type': 'url', 'value': 'https://example.com/image.png', 'mime_type': 'image/png'},
                        'metadata': {'vendor_metadata': {'detail': 'high'}},
                    },
                    {
                        'type': 'audio',
                        'source': {'type': 'url', 'value': 'https://example.com/audio.mp3', 'mime_type': 'audio/mpeg'},
                        'metadata': {'vendor_metadata': {'foo': 'bar'}},
                    },
                    {
                        'type': 'video',
                        'source': {'type': 'url', 'value': 'https://example.com/video.mp4', 'mime_type': 'video/mp4'},
                        'metadata': {'vendor_metadata': {'fps': 5}},
                    },
                    {
                        'type': 'document',
                        'source': {
                            'type': 'url',
                            'value': 'https://example.com/doc.pdf',
                            'mime_type': 'application/pdf',
                        },
                        'metadata': {'vendor_metadata': {'foo': 'baz'}},
                    },
                    {
                        'type': 'document',
                        'source': {'type': 'data', 'value': 'ZmFrZV9kb2M=', 'mime_type': 'application/pdf'},
                        'metadata': {'vendor_metadata': {'detail': 'low'}},
                    },
                ],
            }
        ]
    )

    loaded = AGUIAdapter.load_messages(ag_ui_msgs)
    assert loaded == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                identifier='01a7df',
                                vendor_metadata={'detail': 'high'},
                            ),
                            AudioUrl(
                                url='https://example.com/audio.mp3',
                                vendor_metadata={'foo': 'bar'},
                                _media_type='audio/mpeg',
                            ),
                            VideoUrl(
                                url='https://example.com/video.mp4',
                                media_type='video/mp4',
                                identifier='8cb95e',
                                vendor_metadata={'fps': 5},
                            ),
                            DocumentUrl(
                                url='https://example.com/doc.pdf',
                                media_type='application/pdf',
                                identifier='e3337d',
                                vendor_metadata={'foo': 'baz'},
                            ),
                            BinaryContent(
                                data=b'fake_doc',
                                media_type='application/pdf',
                                identifier='42a9bb',
                                vendor_metadata={'detail': 'low'},
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ]
            )
        ]
    )


@pytest.mark.parametrize(
    'content',
    [
        pytest.param(
            ImageUrl(url='https://example.com/image.png', media_type='image/png', force_download=True),
            id='image-true',
        ),
        pytest.param(
            AudioUrl(url='https://example.com/audio.mp3', media_type='audio/mpeg', force_download='allow-local'),
            id='audio-allow-local',
        ),
        pytest.param(
            VideoUrl(url='https://example.com/video.mp4', media_type='video/mp4', force_download=True),
            id='video-true',
        ),
        pytest.param(
            DocumentUrl(url='https://example.com/doc.pdf', media_type='application/pdf', force_download='allow-local'),
            id='document-allow-local',
        ),
    ],
)
def test_multimodal_roundtrip_preserves_file_url_force_download(
    content: ImageUrl | AudioUrl | VideoUrl | DocumentUrl,
) -> None:
    """`FileUrl.force_download` survives an AG-UI multimodal dump -> load round-trip."""
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[content])])]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.15')
    user_msg = ag_ui_msgs[0]
    assert isinstance(user_msg, UserMessage)
    assert isinstance(user_msg.content, list)
    dumped_content = user_msg.content[0]
    assert isinstance(dumped_content, ImageInputContent | AudioInputContent | VideoInputContent | DocumentInputContent)
    assert dumped_content.metadata == {'force_download': content.force_download}

    loaded = AGUIAdapter.load_messages(ag_ui_msgs)
    _sync_timestamps(messages, loaded)
    assert loaded == messages


def test_multimodal_roundtrip_drops_file_url_force_download_before_0_1_15() -> None:
    """`FileUrl.force_download` is dropped when dumping to AG-UI versions before `0.1.15`.

    Legacy `BinaryInputContent` (emitted for `ag_ui_version < '0.1.15'`) has no `metadata`
    carrier, so there is nowhere to stash `force_download` and it silently resets to `False`
    on reload — the security-conservative default, matching the other legacy-version drops.
    """
    content = ImageUrl(url='https://example.com/image.png', media_type='image/png', force_download=True)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=[content])])]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.14')
    user_msg = ag_ui_msgs[0]
    assert isinstance(user_msg, UserMessage)
    assert isinstance(user_msg.content, list)
    dumped_content = user_msg.content[0]
    assert isinstance(dumped_content, BinaryInputContent)
    assert not hasattr(dumped_content, 'metadata')

    loaded = AGUIAdapter.load_messages(ag_ui_msgs)
    user_part = message_part(loaded, UserPromptPart)
    assert isinstance(user_part.content, list)
    loaded_content = user_part.content[0]
    assert isinstance(loaded_content, ImageUrl)
    assert loaded_content.force_download is False


def test_multimodal_roundtrip_file_without_vendor_metadata_stays_none() -> None:
    """A file with no `vendor_metadata` round-trips to `None` (no spurious `metadata`)."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        ImageUrl(url='https://example.com/image.png', media_type='image/png'),
                        BinaryContent(data=b'fake_image', media_type='image/png'),
                    ]
                )
            ]
        ),
    ]

    ag_ui_msgs = AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.15')
    # `exclude_none` drops the `metadata` key entirely when no vendor_metadata is present.
    assert [m.model_dump(exclude={'id'}, exclude_none=True) for m in ag_ui_msgs] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'source': {'type': 'url', 'value': 'https://example.com/image.png', 'mime_type': 'image/png'},
                    },
                    {
                        'type': 'image',
                        'source': {'type': 'data', 'value': 'ZmFrZV9pbWFnZQ==', 'mime_type': 'image/png'},
                    },
                ],
            }
        ]
    )

    loaded = AGUIAdapter.load_messages(ag_ui_msgs)
    user_part = message_part(loaded, UserPromptPart)
    assert isinstance(user_part.content, list)
    for item in user_part.content:
        assert getattr(item, 'vendor_metadata', None) is None


def test_load_multimodal_rejects_invalid_vendor_metadata() -> None:
    """A malformed `vendor_metadata` on multimodal input content is rejected on load.

    The `metadata` field is typed as `Any`, so a non-`dict` client value is passed to the file
    constructor which raises `ValidationError` here (matching the Vercel adapter), instead of
    being stored unvalidated and crashing a provider model later.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AGUIAdapter.load_messages(
            [
                UserMessage(
                    id='msg-1',
                    content=[
                        ImageInputContent(
                            source=InputContentUrlSource(
                                type='url', value='https://example.com/image.png', mime_type='image/png'
                            ),
                            metadata={'vendor_metadata': 'not-a-dict'},
                        )
                    ],
                )
            ]
        )


def test_load_messages_unknown_type_warns() -> None:
    """Test that an unknown AG-UI message type emits a warning and is skipped."""

    class UnknownMessage(BaseModel):
        id: str
        role: str = 'unknown'

    with pytest.warns(UserWarning, match='AG-UI message type UnknownMessage is not yet implemented; skipping.'):
        messages = AGUIAdapter.load_messages([UnknownMessage(id='msg-1')])  # pyright: ignore[reportArgumentType]

    assert messages == []


# endregion


# region: System prompt tests


async def test_system_prompt_with_ag_ui_adapter():
    """Test that system prompts are included when using AGUIAdapter on first message."""

    system_prompt = 'You are a helpful assistant'
    agent = Agent(model=TestModel(), system_prompt=system_prompt)

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    with capture_run_messages() as messages:
        adapter = AGUIAdapter(agent=agent, run_input=run_input)
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful assistant', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=56, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_dynamic_system_prompt_with_ag_ui_adapter():
    """Test that dynamic system prompts are included when using AGUIAdapter on first message."""

    agent = Agent(model=TestModel())

    @agent.system_prompt
    def dynamic_prompt(ctx: RunContext) -> str:
        return 'Dynamic system prompt'

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    with capture_run_messages() as messages:
        adapter = AGUIAdapter(agent=agent, run_input=run_input)
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Dynamic system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_frontend_system_prompt_stripped_by_default():
    """Test that frontend system prompts are stripped and a warning emitted when `manage_system_prompt='server'`."""

    agent = Agent(model=TestModel(), system_prompt='Agent system prompt')

    run_input = create_input(
        SystemMessage(
            id='msg_sys',
            content='Frontend system prompt',
        ),
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with capture_run_messages() as messages:
        with pytest.warns(UserWarning, match='manage_system_prompt'):
            async for _ in adapter.encode_stream(adapter.run_stream()):
                pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Agent system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_frontend_system_prompt_stripped_no_agent_prompt():
    """Test that frontend system prompts are stripped even when there's no agent system prompt."""

    agent = Agent(model=TestModel())

    run_input = create_input(
        SystemMessage(
            id='msg_sys',
            content='Frontend system prompt',
        ),
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with capture_run_messages() as messages:
        with pytest.warns(UserWarning, match='manage_system_prompt'):
            async for _ in adapter.encode_stream(adapter.run_stream()):
                pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_frontend_system_prompt_only_request_dropped():
    """Test that a `ModelRequest` containing only `SystemPromptParts` is dropped entirely when filtering."""

    agent = Agent(model=TestModel())

    run_input = create_input(
        SystemMessage(
            id='msg_sys',
            content='Frontend system prompt',
        ),
        AssistantMessage(
            id='msg_assistant',
            content='Previous response',
        ),
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with capture_run_messages() as messages:
        with pytest.warns(UserWarning, match='manage_system_prompt'):
            async for _ in adapter.encode_stream(adapter.run_stream()):
                pass

    assert messages == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='Previous response')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_client_mode_keeps_frontend_system_prompt():
    """Test that frontend system prompts are kept and agent prompt skipped when `manage_system_prompt='client'`."""

    agent = Agent(model=TestModel(), system_prompt='Agent system prompt')

    run_input = create_input(
        SystemMessage(
            id='msg_sys',
            content='Frontend system prompt',
        ),
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, manage_system_prompt='client')

    with capture_run_messages() as messages:
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Frontend system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_client_mode_keeps_frontend_system_prompt_no_agent_prompt():
    """Test that frontend system prompts are used when `manage_system_prompt='client'` and agent has no system_prompt."""

    agent = Agent(model=TestModel())

    run_input = create_input(
        SystemMessage(
            id='msg_sys',
            content='Frontend system prompt',
        ),
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, manage_system_prompt='client')

    with capture_run_messages() as messages:
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Frontend system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_client_mode_keeps_frontend_system_prompt_multi_turn():
    """Test that client-managed frontend system prompts are preserved across multi-turn conversations."""

    agent = Agent(model=TestModel(), system_prompt='Agent system prompt')

    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        messages=[
            SystemMessage(
                id='msg_sys',
                content='Frontend system prompt',
            ),
            UserMessage(
                id='msg_1',
                content='First message',
            ),
            AssistantMessage(
                id='msg_2',
                content='First response',
            ),
            UserMessage(
                id='msg_3',
                content='Second message',
            ),
        ],
        state=None,
        context=[],
        tools=[],
        forwarded_props=None,
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, manage_system_prompt='client')

    with capture_run_messages() as messages:
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Frontend system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='First message', timestamp=IsDatetime()),
                ],
            ),
            ModelResponse(parts=[TextPart(content='First response')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Second message', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=57, output_tokens=6),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_client_mode_does_not_reinject_agent_system_prompt():
    """In `manage_system_prompt='client'`, the agent's configured prompt is not injected when
    the frontend sends none — frontend ownership means the frontend is responsible for any
    system prompt. To get fallback-to-configured behavior anyway, callers can add the
    [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability to the
    agent.
    """

    agent = Agent(model=TestModel(), system_prompt='Agent system prompt')

    run_input = create_input(
        UserMessage(
            id='msg_1',
            content='Hello',
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input, manage_system_prompt='client')

    with capture_run_messages() as messages:
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_system_prompt_reinjected_with_ag_ui_history():
    """Test that system prompts ARE reinjected on followup messages via UI adapters."""

    system_prompt = 'You are a helpful assistant'
    agent = Agent(model=TestModel(), system_prompt=system_prompt)

    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        messages=[
            UserMessage(
                id='msg_1',
                content='First message',
            ),
            AssistantMessage(
                id='msg_2',
                content='First response',
            ),
            UserMessage(
                id='msg_3',
                content='Second message',
            ),
        ],
        state=None,
        context=[],
        tools=[],
        forwarded_props=None,
    )

    with capture_run_messages() as messages:
        adapter = AGUIAdapter(agent=agent, run_input=run_input)
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful assistant', timestamp=IsDatetime()),
                    UserPromptPart(content='First message', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(parts=[TextPart(content='First response')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='Second message', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=59, output_tokens=6),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_client_submitted_dangling_tool_calls_not_executed() -> None:
    """A client-submitted history ending with an unresolved tool call has that tool call
    stripped before the agent sees the history, so the agent never has the chance to
    execute it.
    """
    captured: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        captured.append(list(messages))
        yield 'done'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    run_input = create_input(
        UserMessage(id='msg_1', content='Hi'),
        AssistantMessage(
            id='msg_2',
            tool_calls=[
                ToolCall(
                    id='client-call-1',
                    type='function',
                    function=FunctionCall(name='refresh_cache', arguments='{"key": "prod"}'),
                )
            ],
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with pytest.warns(UserWarning, match=r'unresolved tool call.*refresh_cache'):
        async for _ in adapter.encode_stream(adapter.run_stream()):
            pass

    assert len(captured) == 1
    history_seen_by_model = captured[0]
    assert not any(
        isinstance(message, ModelResponse) and any(isinstance(part, ToolCallPart) for part in message.parts)
        for message in history_seen_by_model
    ), 'dangling client-submitted tool call leaked into the agent run'


async def test_client_submitted_tool_call_resolved_by_deferred_results_runs() -> None:
    """Tool calls matched by caller-supplied `deferred_tool_results` survive sanitization,
    so human-in-the-loop resumption still works.
    """
    executed: list[dict[str, Any]] = []

    agent = Agent(model=TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def refresh_cache(key: str) -> str:
        executed.append({'key': key})
        return 'refreshed'

    run_input = create_input(
        UserMessage(id='msg_1', content='Hi'),
        AssistantMessage(
            id='msg_2',
            tool_calls=[
                ToolCall(
                    id='approved-call-1',
                    type='function',
                    function=FunctionCall(name='refresh_cache', arguments='{"key": "prod"}'),
                )
            ],
        ),
    )

    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        async for _ in adapter.encode_stream(
            adapter.run_stream(deferred_tool_results=DeferredToolResults(approvals={'approved-call-1': True}))
        ):
            pass

    assert executed == [{'key': 'prod'}], 'approval-resumed tool call must execute'


async def test_client_submitted_file_url_disallowed_scheme_stripped() -> None:
    """An AG-UI `AGUIAdapter.sanitize_messages` call drops `FileUrl` parts whose URL
    scheme isn't in `allowed_file_url_schemes`, matching the base `UIAdapter` contract.
    """
    agent = Agent(model=TestModel())
    adapter = AGUIAdapter(
        agent=agent,
        run_input=create_input(UserMessage(id='msg_1', content='Hi')),
    )

    crafted: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'See attached',
                        ImageUrl(url='s3://some-bucket/internal.png'),
                        ImageUrl(url='https://example.com/ok.png'),
                    ]
                )
            ]
        )
    ]

    with pytest.warns(UserWarning, match=r"scheme\(s\).*'s3'"):
        sanitized = adapter.sanitize_messages(crafted)

    assert len(sanitized) == 1
    user_part = message_part(sanitized, UserPromptPart)
    assert user_part.content == ['See attached', ImageUrl(url='https://example.com/ok.png')]


# endregion


# region: Interrupts — AG-UI RunFinished outcome and resume[] translation


pytestmark_interrupts = pytest.mark.skipif(
    not interrupts_imports_successful(),
    reason='ag-ui-protocol < 0.1.19 — interrupt types unavailable',
)


_INTERRUPTS_AG_UI_VERSION = '0.1.19'


async def _collect_adapter_events(
    agent: Agent[Any, Any],
    run_input: RunAgentInput,
    *,
    ag_ui_version: str = _INTERRUPTS_AG_UI_VERSION,
    deferred_tool_results: DeferredToolResults | None = None,
) -> list[dict[str, Any]]:
    """Drive `AGUIAdapter` directly so we can pin `ag_ui_version` to the interrupt-aware release."""
    adapter = AGUIAdapter(agent=agent, run_input=run_input, ag_ui_version=ag_ui_version)
    events: list[dict[str, Any]] = []
    async for encoded in adapter.encode_stream(adapter.run_stream(deferred_tool_results=deferred_tool_results)):
        events.append(json.loads(encoded.removeprefix('data: ')))
    return events


@pytestmark_interrupts
async def test_run_finished_success_outcome_on_modern_version() -> None:
    """On ag-ui-protocol >= 0.1.19, a normal run ends with `outcome.type == 'success'`."""
    agent = Agent(model=FunctionModel(stream_function=simple_stream))
    events = await _collect_adapter_events(agent, create_input(UserMessage(id='m1', content='hi')))

    run_finished = next(e for e in events if e['type'] == 'RUN_FINISHED')
    assert run_finished['outcome'] == snapshot({'type': 'success'})


@pytestmark_interrupts
async def test_run_finished_no_outcome_on_legacy_version() -> None:
    """On ag-ui-protocol < 0.1.19, no `outcome` field is emitted — matches today's behavior."""
    agent = Agent(model=FunctionModel(stream_function=simple_stream))
    events = await _collect_adapter_events(
        agent, create_input(UserMessage(id='m1', content='hi')), ag_ui_version='0.1.10'
    )

    run_finished = next(e for e in events if e['type'] == 'RUN_FINISHED')
    assert 'outcome' not in run_finished


@pytestmark_interrupts
async def test_run_finished_no_outcome_when_sdk_lacks_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the installed ag-ui-protocol SDK predates interrupts, `after_stream` emits a bare
    `RUN_FINISHED` with no `outcome` field — even on a modern negotiated version. This is the
    import-gated path (`HAS_INTERRUPTS` is False), distinct from the version-gated path.
    """
    monkeypatch.setattr('pydantic_ai.ui.ag_ui._event_stream.HAS_INTERRUPTS', False)
    agent = Agent(model=FunctionModel(stream_function=simple_stream))
    events = await _collect_adapter_events(agent, create_input(UserMessage(id='m1', content='hi')))

    run_finished = next(e for e in events if e['type'] == 'RUN_FINISHED')
    assert 'outcome' not in run_finished


@pytestmark_interrupts
async def test_run_finished_interrupt_outcome_for_pending_approval() -> None:
    """When the run ends with `DeferredToolRequests.approvals`, the adapter emits an
    interrupt outcome carrying one `Interrupt` per pending approval, with `reason='tool_call'`
    and the original `tool_call_id` bound for resume.
    """

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {0: DeltaToolCall(name='delete_file', json_args='{"path": ".env"}')}

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:  # pragma: no cover
        # Body never runs: the tool is deferred for approval and this test asserts the
        # pre-execution interrupt outcome. The roundtrip test covers an approved execution.
        return f'deleted {path}'

    events = await _collect_adapter_events(agent, create_input(UserMessage(id='m1', content='delete .env')))

    run_finished = next(e for e in events if e['type'] == 'RUN_FINISHED')
    assert run_finished['outcome'] == snapshot(
        {
            'type': 'interrupt',
            'interrupts': [
                {
                    'id': IsStr(),
                    'reason': 'tool_call',
                    'message': IsStr(),
                    'toolCallId': IsStr(),
                    'responseSchema': {
                        'type': 'object',
                        'properties': {
                            'approved': {'type': 'boolean'},
                            'editedArgs': {'type': 'object'},
                            'reason': {'type': 'string'},
                        },
                        'required': ['approved'],
                    },
                }
            ],
        }
    )
    interrupt = run_finished['outcome']['interrupts'][0]
    # The `id` must let the client round-trip back to the original tool_call_id.
    assert interrupt['id'].endswith(interrupt['toolCallId'])


@pytestmark_interrupts
async def test_run_finished_interrupt_outcome_carries_metadata() -> None:
    """Per-call `DeferredToolRequests.metadata` surfaces on `Interrupt.metadata`.

    A tool raising `ApprovalRequired(metadata=...)` attaches metadata keyed by `tool_call_id`;
    the adapter must forward it so a frontend can render context-specific approval UI.
    """

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {0: DeltaToolCall(name='delete_file', json_args='{"path": ".env"}')}

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def delete_file(path: str) -> str:
        raise ApprovalRequired(metadata={'risk': 'high', 'scope': 'filesystem'})

    events = await _collect_adapter_events(agent, create_input(UserMessage(id='m1', content='delete .env')))

    run_finished = next(e for e in events if e['type'] == 'RUN_FINISHED')
    interrupt = run_finished['outcome']['interrupts'][0]
    assert interrupt['metadata'] == snapshot({'risk': 'high', 'scope': 'filesystem'})


@pytestmark_interrupts
async def test_resume_resolved_approves_tool() -> None:
    """`status='resolved'` + `payload.approved=True` → `ToolApproved()` for the bound tool call."""
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id='int-tc-001', status='resolved', payload={'approved': True})],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(DeferredToolResults(approvals={'tc-001': ToolApproved()}))


@pytestmark_interrupts
async def test_resume_resolved_with_edited_args_passes_override_args() -> None:
    """`payload.editedArgs` on an approval translates to `ToolApproved(override_args=...)`."""
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[
            ResumeEntry(
                interrupt_id='int-tc-001',
                status='resolved',
                payload={'approved': True, 'editedArgs': {'path': '/tmp/safer.env'}},
            )
        ],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(
        DeferredToolResults(approvals={'tc-001': ToolApproved(override_args={'path': '/tmp/safer.env'})})
    )


@pytestmark_interrupts
async def test_resume_resolved_with_approved_false_denies_tool() -> None:
    """`payload.approved=False` with a `reason` → `ToolDenied(reason)`."""
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[
            ResumeEntry(
                interrupt_id='int-tc-001',
                status='resolved',
                payload={'approved': False, 'reason': 'too destructive'},
            )
        ],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(
        DeferredToolResults(approvals={'tc-001': ToolDenied(message='too destructive')})
    )


@pytestmark_interrupts
async def test_resume_cancelled_denies_tool_regardless_of_payload() -> None:
    """`status='cancelled'` → `ToolDenied('Cancelled by user.')` regardless of payload contents."""
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[
            # Even an apparently-approving payload is overridden by `status='cancelled'`.
            ResumeEntry(interrupt_id='int-tc-001', status='cancelled', payload={'approved': True}),
        ],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(
        DeferredToolResults(approvals={'tc-001': ToolDenied(message='Cancelled by user.')})
    )


@pytestmark_interrupts
@pytest.mark.parametrize(
    'payload',
    [
        pytest.param({}, id='missing-approved'),
        pytest.param({'approved': None}, id='approved-null'),
        pytest.param({'approved': 'true'}, id='approved-string-true'),
        pytest.param({'approved': 1}, id='approved-truthy-int'),
        pytest.param('approved', id='payload-is-string'),
        pytest.param([{'approved': True}], id='payload-is-list'),
        pytest.param(None, id='payload-null'),
    ],
)
async def test_resume_deny_by_default_for_ambiguous_payload(payload: Any) -> None:
    """Approval requires an explicit `payload.approved == True`.

    Anything else (missing, null, non-bool, non-dict payload) must deny so a malformed or
    hostile client cannot bypass the `requires_approval=True` gate by omitting the field.
    """
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id='int-tc-001', status='resolved', payload=payload)],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(DeferredToolResults(approvals={'tc-001': ToolDenied()}))


@pytestmark_interrupts
async def test_resume_unknown_interrupt_id_prefix_raises() -> None:
    """An `interrupt_id` that doesn't carry the adapter's prefix is a protocol error
    we surface as `UserError` rather than silently mapping to a wrong tool call.
    """
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id='bogus-123', status='resolved', payload={'approved': True})],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    with pytest.raises(UserError, match=r'does not start with the expected'):
        _ = adapter.deferred_tool_results


@pytestmark_interrupts
async def test_deferred_tool_results_none_when_sdk_lacks_interrupts(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the installed ag-ui-protocol SDK predates interrupts, `resume[]` is ignored entirely
    (`deferred_tool_results` returns `None`) so the old SDK path stays byte-for-byte unchanged.
    """
    monkeypatch.setattr('pydantic_ai.ui.ag_ui._adapter.HAS_INTERRUPTS', False)
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id='int-tc-001', status='resolved', payload={'approved': True})],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results is None


@pytestmark_interrupts
async def test_resume_well_formed_but_nonexistent_interrupt_id_is_noop() -> None:
    """A well-formed `int-` id that binds to no pending approval is accepted, not an error.

    Unlike a missing prefix (which raises `UserError`), a correctly-prefixed id that matches no
    pending call maps to its own stripped `tool_call_id` — never to a different pending call — so
    the agent simply finds no call to resolve and the entry is a harmless no-op.
    """
    agent = Agent(model=TestModel())
    run_input = RunAgentInput(
        thread_id=uuid_str(),
        run_id=uuid_str(),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id='int-tc-does-not-exist', status='resolved', payload={'approved': True})],
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input)

    assert adapter.deferred_tool_results == snapshot(
        DeferredToolResults(approvals={'tc-does-not-exist': ToolApproved()})
    )


@pytestmark_interrupts
async def test_interrupt_resume_roundtrip_executes_approved_tool() -> None:
    """End-to-end: turn 1 ends with an interrupt outcome; turn 2 supplies `resume[]` and the
    tool actually runs. The resumed turn must NOT re-emit `TOOL_CALL_START` for the same
    `tool_call_id` (per AG-UI spec) — the agent loops back into execution with the original id
    and we only see `TOOL_CALL_RESULT`.
    """
    executed: list[dict[str, Any]] = []

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        # Whenever the agent has just a user prompt + system prompts, propose the call;
        # after the tool result is in history, emit a final text response.
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, 'parts', [])):
            yield {0: DeltaToolCall(name='delete_file', json_args='{"path": ".env"}')}
        else:
            yield 'done'

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        executed.append({'path': path})
        return f'deleted {path}'

    # Turn 1: collect the interrupt outcome.
    turn1_input = create_input(UserMessage(id='m1', content='delete .env'))
    turn1_events = await _collect_adapter_events(agent, turn1_input)

    run_finished_1 = next(e for e in turn1_events if e['type'] == 'RUN_FINISHED')
    interrupt = run_finished_1['outcome']['interrupts'][0]
    interrupt_id = interrupt['id']
    tool_call_id = interrupt['toolCallId']
    assert executed == [], 'turn 1 must not execute the deferred tool'

    # Turn 2: resume with approval; the message history must include the proposed call so the
    # agent's tool manager can match the resume back to it.
    turn2_input = RunAgentInput(
        thread_id=turn1_input.thread_id,
        run_id=uuid_str(),
        state={},
        messages=[
            UserMessage(id='m1', content='delete .env'),
            AssistantMessage(
                id='m2',
                tool_calls=[
                    ToolCall(
                        id=tool_call_id,
                        type='function',
                        function=FunctionCall(name='delete_file', arguments='{"path": ".env"}'),
                    )
                ],
            ),
        ],
        tools=[],
        context=[],
        forwarded_props=None,
        resume=[ResumeEntry(interrupt_id=interrupt_id, status='resolved', payload={'approved': True})],
    )
    turn2_events = await _collect_adapter_events(agent, turn2_input)

    assert executed == [{'path': '.env'}], 'turn 2 must execute the approved tool'

    tool_starts_for_call = [
        e for e in turn2_events if e['type'] == 'TOOL_CALL_START' and e.get('toolCallId') == tool_call_id
    ]
    tool_results_for_call = [
        e for e in turn2_events if e['type'] == 'TOOL_CALL_RESULT' and e.get('toolCallId') == tool_call_id
    ]
    assert tool_starts_for_call == [], 'spec: resumed turn must not re-emit TOOL_CALL_START for the same tool_call_id'
    assert len(tool_results_for_call) == 1, 'resumed turn must emit exactly one TOOL_CALL_RESULT'

    run_finished_2 = next(e for e in turn2_events if e['type'] == 'RUN_FINISHED')
    assert run_finished_2['outcome'] == snapshot({'type': 'success'})


# endregion
