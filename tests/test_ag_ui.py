"""Tests for AG-UI implementation."""

# pyright: reportPossiblyUnboundVariable=none
from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Final, Literal

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel, TestNode, TestTextPart, TestThinkingPart, TestToolCallPart

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

    from pydantic_ai.ag_ui import (
        SSE_CONTENT_TYPE,
        Role,
        StateDeps,
        _Adapter,  # type: ignore[reportPrivateUsage]
    )

    has_ag_ui = True


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not has_ag_ui, reason='ag-ui-protocol not installed'),
]


# Type aliases.
_MockUUID = Callable[[], str]

# Constants.
THREAD_ID_PREFIX: Final[str] = 'thread_'
RUN_ID_PREFIX: Final[str] = 'run_'
EXPECTED_EVENTS: Final[list[str]] = [
    '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
    '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000003","role":"assistant"}',
    '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"success "}',
    '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"(no "}',
    '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"tool "}',
    '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"calls)"}',
    '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000003"}',
    '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
]
UUID_PATTERN: Final[re.Pattern[str]] = re.compile(r'\d{8}-\d{4}-\d{4}-\d{4}-\d{12}')


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


@pytest.fixture
async def adapter() -> _Adapter[StateDeps[StateInt], str]:
    """Fixture to create an Adapter instance for testing.

    Returns:
        An Adapter instance configured for testing.
    """
    return await create_adapter([])


async def create_adapter(
    call_tools: list[str] | Literal['all'],
) -> _Adapter[StateDeps[StateInt], str]:
    """Create an Adapter instance for testing.

    Args:
        call_tools: List of tool names to enable, or 'all' for all tools.

    Returns:
        An Adapter instance configured with the specified tools.
    """
    return _Adapter(
        agent=Agent(
            model=TestModel(
                call_tools=call_tools,
                tool_call_deltas={'get_weather_parts', 'current_time'},
            ),
            deps_type=StateDeps[StateInt],  # type: ignore[reportUnknownArgumentType]
            tools=[send_snapshot, send_custom, current_time],
        ),
    )


@pytest.fixture
def mock_uuid(monkeypatch: pytest.MonkeyPatch) -> _MockUUID:
    """Mock UUID generation for consistent test results.

    This fixture replaces the uuid.uuid4 function with a mock that generates
    sequential UUIDs for testing purposes. This ensures that UUIDs are
    predictable and consistent across test runs.

    Args:
        monkeypatch: The pytest monkeypatch fixture to modify uuid.uuid4.

    Returns:
        A function that generates a mock UUID.
    """
    counter = count(1)

    def _fake_uuid() -> str:
        """Generate a fake UUID string with sequential numbering.

        Returns:
            A fake UUID string in the format '00000000-0000-0000-0000-{counter:012d}'.
        """
        return f'00000000-0000-0000-0000-{next(counter):012d}'

    def _fake_uuid4() -> uuid.UUID:
        """Generate a fake UUID object using the fake UUID string.

        Returns:
            A UUID object created from the fake UUID string.
        """
        return uuid.UUID(_fake_uuid())

    # Due to how ToolCallPart uses generate_tool_call_id with field default_factory,
    # we have to patch uuid.uuid4 directly instead of the generate function. This
    # also covers how we generate messages IDs.
    monkeypatch.setattr('uuid.uuid4', _fake_uuid4)

    return _fake_uuid


def assert_events(events: list[str], expected_events: list[str], *, loose: bool = False) -> None:
    for event, expected in zip(events, expected_events):
        if loose:
            expected = normalize_uuids(expected)
            event = normalize_uuids(event)
        assert event == f'data: {expected}\n\n'
    assert len(events) == len(expected_events)


def normalize_uuids(text: str) -> str:
    """Normalize UUIDs in the given text to a fixed format.

    Args:
        text: The input text containing UUIDs.

    Returns:
        The text with UUIDs replaced by a fixed UUID.
    """
    return UUID_PATTERN.sub('00000000-0000-0000-0000-000000000001', text)


def current_time() -> str:
    """Get the current time in ISO format.

    Returns:
        The current UTC time in ISO format string.
    """
    return '21T12:08:45.485981+00:00'


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


@dataclass(frozen=True)
class Run:
    """Test parameter class for Adapter.run method tests.

    Args:
        messages: List of messages for the run input.
        state: State object for the run input.
        context: Context list for the run input.
        tools: List of tools for the run input.
        forwarded_props: Forwarded properties for the run input.
        nodes: List of TestNode instances for the run input.
    """

    messages: list[Message]
    state: Any = None
    context: list[Any] = field(default_factory=lambda: list[Any]())
    tools: list[Tool] = field(default_factory=lambda: list[Tool]())
    nodes: list[TestNode] | None = None
    forwarded_props: Any = None

    def run_input(self, *, thread_id: str, run_id: str) -> RunAgentInput:
        """Create a RunAgentInput instance for the test case.

        Args:
            thread_id: The thread ID for the run.
            run_id: The run ID for the run.

        Returns:
            A RunAgentInput instance with the test case parameters.
        """
        return RunAgentInput(
            thread_id=thread_id,
            run_id=run_id,
            messages=self.messages,
            state=self.state,
            context=self.context,
            tools=self.tools,
            forwarded_props=self.forwarded_props,
        )


@dataclass(frozen=True)
class AdapterRunTest:
    """Test parameter class for Adapter.run method tests.

    Args:
        id: Name of the test case.
        runs: List of Run instances for the test case.
    """

    id: str
    runs: list[Run]
    call_tools: list[str] = field(default_factory=lambda: list[str]())
    expected_events: list[str] = field(default_factory=lambda: list(EXPECTED_EVENTS))
    expected_state: int | None = None


# Test parameter data
def tc_parameters() -> list[AdapterRunTest]:
    if not has_ag_ui:  # pragma: no branch
        return [AdapterRunTest(id='skipped', runs=[])]

    return [
        AdapterRunTest(
            id='basic_user_message',
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Hello, how are you?',
                        ),
                    ],
                ),
            ],
        ),
        AdapterRunTest(
            id='empty_messages',
            runs=[
                Run(messages=[]),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"RUN_ERROR","message":"no messages found in the input","code":"no_messages"}',
            ],
        ),
        AdapterRunTest(
            id='multiple_messages',
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='First message',
                        ),
                        AssistantMessage(
                            id='msg_2',
                            role=Role.ASSISTANT.value,
                            content='Assistant response',
                        ),
                        SystemMessage(
                            id='msg_3',
                            role=Role.SYSTEM.value,
                            content='System message',
                        ),
                        DeveloperMessage(
                            id='msg_4',
                            role=Role.DEVELOPER.value,
                            content='Developer note',
                        ),
                        UserMessage(
                            id='msg_5',
                            role=Role.USER.value,
                            content='Second message',
                        ),
                    ],
                ),
            ],
        ),
        AdapterRunTest(
            id='messages_with_history',
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='First message',
                        ),
                        UserMessage(
                            id='msg_2',
                            role=Role.USER.value,
                            content='Second message',
                        ),
                    ],
                ),
            ],
        ),
        AdapterRunTest(
            id='tool_ag_ui',
            call_tools=['get_weather'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather for Paris',
                        ),
                    ],
                    tools=[get_weather()],
                ),
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather for Paris',
                        ),
                        AssistantMessage(
                            id='msg_2',
                            role=Role.ASSISTANT.value,
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
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000003',
                        ),
                    ],
                    tools=[get_weather()],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"get_weather"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000004"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000005","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000005","delta":"{\\"get_weather\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000005","delta":"result\\"}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000005"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000004"}',
            ],
        ),
        AdapterRunTest(
            id='tool_ag_ui_multiple',
            call_tools=['get_weather', 'get_weather_parts'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather and get_weather_parts for Paris',
                        ),
                    ],
                    tools=[get_weather(), get_weather('get_weather_parts')],
                ),
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather for Paris',
                        ),
                        AssistantMessage(
                            id='msg_2',
                            role=Role.ASSISTANT.value,
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
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000003',
                        ),
                        AssistantMessage(
                            id='msg_4',
                            role=Role.ASSISTANT.value,
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
                            id='msg_5',
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000003',
                        ),
                    ],
                    tools=[get_weather(), get_weather('get_weather_parts')],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"get_weather"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000004","toolCallName":"get_weather_parts"}',
                '{"type":"TOOL_CALL_ARGS","toolCallId":"pyd_ai_00000000000000000000000000000004","delta":"{\\"location\\":\\"a\\"}"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000004"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000005"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000006","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"{\\"get_weather\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"result\\",\\"get_weather_parts\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"result\\"}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000006"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000005"}',
            ],
        ),
        AdapterRunTest(
            id='tool_ag_ui_parts',
            call_tools=['get_weather_parts'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather_parts for Paris',
                        ),
                    ],
                    tools=[get_weather('get_weather_parts')],
                ),
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather_parts for Paris',
                        ),
                        AssistantMessage(
                            id='msg_2',
                            role=Role.ASSISTANT.value,
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
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000003',
                        ),
                    ],
                    tools=[get_weather('get_weather_parts')],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"get_weather_parts"}',
                '{"type":"TOOL_CALL_ARGS","toolCallId":"pyd_ai_00000000000000000000000000000003","delta":"{\\"location\\":\\"a\\"}"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000004"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000005","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000005","delta":"{\\"get_weather_parts\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000005","delta":"result\\"}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000005"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000004"}',
            ],
        ),
        AdapterRunTest(
            id='tool_local_single_event',
            call_tools=['send_snapshot'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call send_snapshot',
                        ),
                    ],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"send_snapshot"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"TOOL_CALL_RESULT","messageId":"msg_1","toolCallId":"pyd_ai_00000000000000000000000000000003","content":"{\\"type\\":\\"STATE_SNAPSHOT\\",\\"timestamp\\":null,\\"raw_event\\":null,\\"snapshot\\":{\\"key\\":\\"value\\"}}","role":"tool"}',
                '{"type":"STATE_SNAPSHOT","snapshot":{"key":"value"}}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000004","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"{\\"send_snapshot\\":{\\"type\\":\\"STATE_SNAPSHOT\\",\\"timestam"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"p\\":null,\\"rawEvent\\":null,\\"snapshot\\":{\\"key\\":\\"value\\"}}}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000004"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
            ],
        ),
        AdapterRunTest(
            id='tool_local_multiple_events',
            call_tools=['send_custom'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call send_custom',
                        ),
                    ],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"send_custom"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"TOOL_CALL_RESULT","messageId":"msg_1","toolCallId":"pyd_ai_00000000000000000000000000000003","content":"[{\\"type\\":\\"CUSTOM\\",\\"timestamp\\":null,\\"raw_event\\":null,\\"name\\":\\"custom_event1\\",\\"value\\":{\\"key1\\":\\"value1\\"}},{\\"type\\":\\"CUSTOM\\",\\"timestamp\\":null,\\"raw_event\\":null,\\"name\\":\\"custom_event2\\",\\"value\\":{\\"key2\\":\\"value2\\"}}]","role":"tool"}',
                '{"type":"CUSTOM","name":"custom_event1","value":{"key1":"value1"}}',
                '{"type":"CUSTOM","name":"custom_event2","value":{"key2":"value2"}}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000004","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"{\\"send_custom\\":[{\\"type\\":\\"CUSTOM\\",\\"timestamp\\":null,\\"rawEvent\\":null,\\"name\\":\\"custom_event1\\",\\"value\\":{\\"key1\\":\\"va"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"lue1\\"}},{\\"type\\":\\"CUSTOM\\",\\"timestamp\\":null,\\"rawEvent\\":null,\\"name\\":\\"custom_event2\\",\\"value\\":{\\"key2\\":\\"value2\\"}}]}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000004"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
            ],
        ),
        AdapterRunTest(
            id='tool_local_parts',
            call_tools=['current_time'],
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call current_time',
                        ),
                    ],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"current_time"}',
                '{"type":"TOOL_CALL_ARGS","toolCallId":"pyd_ai_00000000000000000000000000000003","delta":"{}"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"TOOL_CALL_RESULT","messageId":"msg_1","toolCallId":"pyd_ai_00000000000000000000000000000003","content":"21T12:08:45.485981+00:00","role":"tool"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000004","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"{\\"current_time\\":\\"21T1"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000004","delta":"2:08:45.485981+00:00\\"}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000004"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
            ],
        ),
        AdapterRunTest(
            id='thinking',
            runs=[
                Run(
                    nodes=[
                        TestNode(
                            parts=[
                                TestThinkingPart(content='Thinking'),
                                TestThinkingPart(content='Thinking about the weather'),
                                TestTextPart('Thought about the weather'),
                            ],
                        ),
                    ],
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Hello',
                        ),
                    ],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"THINKING_TEXT_MESSAGE_START"}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"\\"Thinking\\""}',
                '{"type":"THINKING_TEXT_MESSAGE_END"}',
                '{"type":"THINKING_TEXT_MESSAGE_START"}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"\\"Thinking "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"about "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"the "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"weather\\""}',
                '{"type":"THINKING_TEXT_MESSAGE_END"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000003","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"Thought "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"about "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"the "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000003","delta":"weather"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000003"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
            ],
        ),
        AdapterRunTest(
            id='tool_local_then_ag_ui',
            call_tools=['current_time', 'get_weather'],
            runs=[
                Run(
                    nodes=[
                        TestNode(
                            parts=[
                                TestToolCallPart(call_tools=['current_time']),
                                TestThinkingPart(content='Thinking about the weather'),
                            ],
                        ),
                        TestNode(
                            parts=[TestToolCallPart(call_tools=['get_weather'])],
                        ),
                    ],
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please tell me the time and then call get_weather for Paris',
                        ),
                    ],
                    tools=[get_weather()],
                ),
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Please call get_weather for Paris',
                        ),
                        AssistantMessage(
                            id='msg_2',
                            role=Role.ASSISTANT.value,
                            tool_calls=[
                                ToolCall(
                                    id='pyd_ai_00000000000000000000000000000003',
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
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000003',
                        ),
                        AssistantMessage(
                            id='msg_4',
                            role=Role.ASSISTANT.value,
                            tool_calls=[
                                ToolCall(
                                    id='pyd_ai_00000000000000000000000000000004',
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
                            role=Role.TOOL.value,
                            content='Tool result',
                            tool_call_id='pyd_ai_00000000000000000000000000000004',
                        ),
                    ],
                    tools=[get_weather()],
                ),
            ],
            expected_events=[
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000003","toolCallName":"current_time"}',
                '{"type":"TOOL_CALL_ARGS","toolCallId":"pyd_ai_00000000000000000000000000000003","delta":"{}"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000003"}',
                '{"type":"THINKING_TEXT_MESSAGE_START"}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"\\"Thinking "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"about "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"the "}',
                '{"type":"THINKING_TEXT_MESSAGE_CONTENT","delta":"weather\\""}',
                '{"type":"THINKING_TEXT_MESSAGE_END"}',
                '{"type":"TOOL_CALL_RESULT","messageId":"msg_1","toolCallId":"pyd_ai_00000000000000000000000000000003","content":"21T12:08:45.485981+00:00","role":"tool"}',
                '{"type":"TOOL_CALL_START","toolCallId":"pyd_ai_00000000000000000000000000000004","toolCallName":"get_weather"}',
                '{"type":"TOOL_CALL_END","toolCallId":"pyd_ai_00000000000000000000000000000004"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000002"}',
                '{"type":"RUN_STARTED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000005"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000006","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"{\\"current_time\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"result\\",\\"get_weather\\":\\"Tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000006","delta":"result\\"}"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000006"}',
                '{"type":"RUN_FINISHED","threadId":"thread_00000000-0000-0000-0000-000000000001","runId":"run_00000000-0000-0000-0000-000000000005"}',
            ],
        ),
        AdapterRunTest(
            id='request_with_state',
            runs=[
                Run(
                    messages=[  # pyright: ignore[reportArgumentType]
                        UserMessage(
                            id='msg_1',
                            role=Role.USER.value,
                            content='Hello, how are you?',
                        ),
                    ],
                    state={'value': 42},
                ),
            ],
            expected_state=42,
        ),
    ]


@pytest.mark.parametrize('tc', tc_parameters(), ids=lambda tc: tc.id)
async def test_run_method(mock_uuid: _MockUUID, tc: AdapterRunTest) -> None:
    """Test the Adapter.run method with various scenarios.

    Args:
        mock_uuid: The mock UUID generator fixture.
        tc: The test case parameters.
    """

    run: Run
    events: list[str] = []
    thread_id: str = f'{THREAD_ID_PREFIX}{mock_uuid()}'
    adapter: _Adapter[StateDeps[StateInt], str] = await create_adapter(tc.call_tools)
    deps: StateDeps[StateInt] = StateDeps(StateInt())
    for run in tc.runs:
        if run.nodes is not None:
            assert isinstance(adapter.agent.model, TestModel), (
                'Agent model is not TestModel'
                'data: {"type":"TOOL_CALL_RESULT","messageId":"msg_1","toolCallId":"pyd_ai_00000000000000000000000000000003","content":"21T12:08:45.485981+00:00","role":"tool"}\n\n'
            )
            adapter.agent.model.custom_response_nodes = run.nodes

        run_input: RunAgentInput = run.run_input(
            thread_id=thread_id,
            run_id=f'{RUN_ID_PREFIX}{mock_uuid()}',
        )

        events.extend([event async for event in adapter.run(run_input, deps=deps)])

    assert_events(events, tc.expected_events)
    if tc.expected_state is not None:
        assert deps.state.value == tc.expected_state


async def test_concurrent_runs(mock_uuid: _MockUUID, adapter: _Adapter[None, str]) -> None:
    """Test concurrent execution of multiple runs."""

    async def collect_events(run_input: RunAgentInput) -> list[str]:
        """Collect all events from an adapter run.

        Args:
            run_input: The input configuration for the adapter run.

        Returns:
            List of all events generated by the adapter run.
        """
        return [event async for event in adapter.run(run_input)]

    concurrent_tasks: list[asyncio.Task[list[str]]] = []

    for i in range(20):
        run_input: RunAgentInput = RunAgentInput(
            thread_id=f'{THREAD_ID_PREFIX}{mock_uuid()}',
            run_id=f'{RUN_ID_PREFIX}{mock_uuid()}',
            messages=[  # pyright: ignore[reportArgumentType]
                UserMessage(
                    id=f'msg_{i}',
                    role=Role.USER.value,
                    content=f'Message {i}',
                ),
            ],
            state=None,
            context=[],
            tools=[],
            forwarded_props=None,
        )

        task = asyncio.create_task(collect_events(run_input))
        concurrent_tasks.append(task)

    results = await asyncio.gather(*concurrent_tasks)

    for events in results:
        assert_events(events, EXPECTED_EVENTS, loose=True)
        assert len(events) == len(EXPECTED_EVENTS)


@pytest.mark.anyio
async def test_to_ag_ui(mock_uuid: _MockUUID) -> None:
    """Test the agent.to_ag_ui method."""

    agent: Agent[None, str] = Agent(model=TestModel())
    app = agent.to_ag_ui()
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as client:
            client.base_url = 'http://localhost:8000'
            run_input: RunAgentInput = RunAgentInput(
                state=None,
                thread_id=f'{THREAD_ID_PREFIX}test_thread',
                run_id=f'{RUN_ID_PREFIX}test_run',
                messages=[  # pyright: ignore[reportArgumentType]
                    UserMessage(
                        id='msg_1',
                        role=Role.USER.value,
                        content='Hello, world!',
                    ),
                ],
                tools=[],
                context=[],
                forwarded_props=None,
            )
            events: list[str]
            async with client.stream(
                'POST',
                '/',
                content=run_input.model_dump_json(),
                headers={'Content-Type': 'application/json', 'Accept': SSE_CONTENT_TYPE},
            ) as response:
                assert response.status_code == 200, f'Unexpected status code: {response.status_code}'
                events = [line + '\n\n' async for line in response.aiter_lines() if line.startswith('data: ')]

            assert events, 'No parts received from the server'
            expected: list[str] = [
                '{"type":"RUN_STARTED","threadId":"thread_test_thread","runId":"run_test_run"}',
                '{"type":"TEXT_MESSAGE_START","messageId":"00000000-0000-0000-0000-000000000001","role":"assistant"}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000001","delta":"success "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000001","delta":"(no "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000001","delta":"tool "}',
                '{"type":"TEXT_MESSAGE_CONTENT","messageId":"00000000-0000-0000-0000-000000000001","delta":"calls)"}',
                '{"type":"TEXT_MESSAGE_END","messageId":"00000000-0000-0000-0000-000000000001"}',
                '{"type":"RUN_FINISHED","threadId":"thread_test_thread","runId":"run_test_run"}',
            ]
            assert_events(events, expected)
