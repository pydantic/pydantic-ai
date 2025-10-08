"""AG-UI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the AG-UI protocol,
enabling streaming event-based communication for interactive AI applications.
"""

# pyright: reportIncompatibleMethodOverride=false, reportUnusedClass=false, reportGeneralTypeIssues=false, reportInvalidTypeArguments=false

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import Field, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Generic, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    pass  # Agent type is not actually used in this module

from pydantic import BaseModel, ValidationError

from ...messages import (
    BuiltinToolCallEvent,  # type: ignore[reportDeprecated]
    BuiltinToolCallPart,
    BuiltinToolResultEvent,  # type: ignore[reportDeprecated]
    BuiltinToolReturnPart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...run import AgentRunResultEvent
from ...tools import AgentDepsT, ToolDefinition
from ...toolsets.external import ExternalToolset
from .. import BaseEventStream

try:
    from ag_ui.core import (
        AssistantMessage,
        BaseEvent,
        DeveloperMessage,
        EventType,
        Message,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        SystemMessage,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        Tool as AGUITool,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
        ToolMessage,
        UserMessage,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

__all__ = [
    'AGUIEventStream',
    'StateHandler',
    'StateDeps',
    'protocol_messages_to_pai_messages',
    '_AGUIFrontendToolset',
    '_NoMessagesError',
    '_InvalidStateError',
    '_RunError',
    'RunAgentInput',
    'RunStartedEvent',
    'RunFinishedEvent',
]

_BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'


# State management types

StateT = TypeVar('StateT', bound=BaseModel)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> Any:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: Any) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.

        Raises:
            InvalidStateError: If `state` does not match the expected model.
        """
        ...


@dataclass
class StateDeps(Generic[StateT]):
    """Provides AG-UI state management.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT


# Error types


@dataclass
class _RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:  # pragma: no cover
        return self.message


@dataclass
class _NoMessagesError(_RunError):
    """Exception raised when no messages are found in the input."""

    message: str = 'no messages found in the input'
    code: str = 'no_messages'


@dataclass
class _InvalidStateError(_RunError, ValidationError):
    """Exception raised when an invalid state is provided."""

    message: str = 'invalid state provided'
    code: str = 'invalid_state'


class _ToolCallNotFoundError(_RunError, ValueError):
    """Exception raised when an tool result is present without a matching call."""

    def __init__(self, tool_call_id: str) -> None:
        """Initialize the exception with the tool call ID."""
        super().__init__(  # pragma: no cover
            message=f'Tool call with ID {tool_call_id} not found in the history.',
            code='tool_call_not_found',
        )


# Frontend toolset


class _AGUIFrontendToolset(ExternalToolset[AgentDepsT]):
    """Toolset for AG-UI frontend tools."""

    def __init__(self, tools: list[AGUITool]):
        """Initialize the toolset with AG-UI tools.

        Args:
            tools: List of AG-UI tool definitions.
        """
        super().__init__(
            [
                ToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.parameters,
                )
                for tool in tools
            ]
        )

    @property
    def label(self) -> str:
        """Return the label for this toolset."""
        return 'the AG-UI frontend tools'  # pragma: no cover


class AGUIEventStream(BaseEventStream[BaseEvent, AgentDepsT]):
    """Transforms Pydantic AI agent events into AG-UI protocol events.

    This class handles the stateful transformation of streaming agent events
    into the AG-UI protocol format, managing message IDs, thinking mode state,
    and tool call ID mappings for builtin tools.

    Example:
        ```python
        event_stream = AGUIEventStream()
        async for ag_ui_event in event_stream.agent_event_to_events(pai_event):
            print(ag_ui_event)
        ```
    """

    def __init__(self) -> None:
        """Initialize AG-UI event stream state."""
        super().__init__()
        self.part_end: BaseEvent | None = None
        self.thinking: bool = False
        self.builtin_tool_call_ids: dict[str, str] = {}

    # Granular handlers implementation

    async def handle_text_start(self, part: TextPart) -> AsyncIterator[BaseEvent]:
        """Handle a TextPart at start."""
        if self.part_end:
            yield self.part_end
            self.part_end = None

        if self.thinking:
            yield ThinkingEndEvent(type=EventType.THINKING_END)
            self.thinking = False

        message_id = self.new_message_id()
        yield TextMessageStartEvent(message_id=message_id)
        if part.content:  # pragma: no branch
            yield TextMessageContentEvent(message_id=message_id, delta=part.content)
        self.part_end = TextMessageEndEvent(message_id=message_id)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a TextPartDelta."""
        if delta.content_delta:  # pragma: no branch
            yield TextMessageContentEvent(message_id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_start(self, part: ThinkingPart) -> AsyncIterator[BaseEvent]:
        """Handle a ThinkingPart at start."""
        if self.part_end:
            yield self.part_end
            self.part_end = None

        if not self.thinking:
            yield ThinkingStartEvent(type=EventType.THINKING_START)
            self.thinking = True

        if part.content:
            yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
            yield ThinkingTextMessageContentEvent(type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=part.content)
            self.part_end = ThinkingTextMessageEndEvent(type=EventType.THINKING_TEXT_MESSAGE_END)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a ThinkingPartDelta."""
        if delta.content_delta:  # pragma: no branch
            if not isinstance(self.part_end, ThinkingTextMessageEndEvent):
                yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
                self.part_end = ThinkingTextMessageEndEvent(type=EventType.THINKING_TEXT_MESSAGE_END)

            yield ThinkingTextMessageContentEvent(
                type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=delta.content_delta
            )

    async def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        if self.part_end:
            yield self.part_end
            self.part_end = None

        if self.thinking:
            yield ThinkingEndEvent(type=EventType.THINKING_END)
            self.thinking = False

        tool_call_id = part.tool_call_id
        if isinstance(part, BuiltinToolCallPart):
            builtin_tool_call_id = '|'.join([_BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
            self.builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
            tool_call_id = builtin_tool_call_id

        message_id = self.message_id or self.new_message_id()
        yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=message_id)
        if part.args:
            yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=part.args_as_json_str())
        self.part_end = ToolCallEndEvent(tool_call_id=tool_call_id)

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPartDelta."""
        tool_call_id = delta.tool_call_id
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        if tool_call_id in self.builtin_tool_call_ids:
            tool_call_id = self.builtin_tool_call_ids[tool_call_id]
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
        )

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseEvent]:
        """Handle a BuiltinToolReturnPart."""
        # Emit any pending part_end event (e.g., TOOL_CALL_END) before the result
        if self.part_end:
            yield self.part_end
            self.part_end = None

        tool_call_id = self.builtin_tool_call_ids[part.tool_call_id]
        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=tool_call_id,
            content=part.model_response_str(),
        )

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[BaseEvent]:
        """Handle a FunctionToolCallEvent.

        This event is emitted when a function tool is called, but no AG-UI events
        are needed at this stage since tool calls are handled in PartStartEvent.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        if not isinstance(result, ToolReturnPart):
            return

        # Emit any pending part_end event (e.g., TOOL_CALL_END) before the result
        if self.part_end:
            yield self.part_end
            self.part_end = None

        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=result.model_response_str(),
        )

        # Check for AG-UI events returned by tool calls.
        possible_event = result.metadata or result.content
        if isinstance(possible_event, BaseEvent):
            yield possible_event
        elif isinstance(possible_event, str | bytes):  # pragma: no branch
            # Avoid iterable check for strings and bytes.
            pass
        elif isinstance(possible_event, Iterable):  # pragma: no branch
            for item in possible_event:  # type: ignore[reportUnknownMemberType]
                if isinstance(item, BaseEvent):  # pragma: no branch
                    yield item

    async def handle_builtin_tool_call(self, event: BuiltinToolCallEvent) -> AsyncIterator[BaseEvent]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolCallEvent.

        This event is emitted when a builtin tool is called, but no AG-UI events
        are needed at this stage since builtin tool calls are handled in PartStartEvent.
        """
        return
        yield  # Make this an async generator

    async def handle_builtin_tool_result(self, event: BuiltinToolResultEvent) -> AsyncIterator[BaseEvent]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolResultEvent.

        This event is emitted when a builtin tool returns. We need to emit any pending
        part_end event (TOOL_CALL_END) before the result is shown in handle_builtin_tool_return.
        """
        # Emit any pending part_end event (e.g., TOOL_CALL_END) before the result
        if self.part_end:
            yield self.part_end
            self.part_end = None

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[BaseEvent]:
        """Handle a FinalResultEvent.

        This event is emitted when the agent produces a final result, but no AG-UI events
        are needed at this stage.
        """
        return
        yield  # Make this an async generator

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[BaseEvent]:
        """Handle an AgentRunResultEvent, cleaning up any pending state."""
        # Emit any pending part end event
        if self.part_end:  # pragma: no branch
            yield self.part_end
            self.part_end = None

        # End thinking mode if still active
        if self.thinking:
            yield ThinkingEndEvent(
                type=EventType.THINKING_END,
            )
            self.thinking = False

    async def on_validation_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        """Handle validation errors before stream starts."""
        # Try to get code from exception if it has one, otherwise use class name
        code = getattr(error, 'code', error.__class__.__name__)
        yield RunErrorEvent(message=str(error), code=code)

    async def on_stream_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        """Handle errors during streaming."""
        # Try to get code from exception if it has one, otherwise use class name
        code = getattr(error, 'code', error.__class__.__name__)
        yield RunErrorEvent(message=str(error), code=code)

    def encode_event(self, event: BaseEvent, accept: str) -> str:
        """Encode an AG-UI event as SSE.

        Args:
            event: The AG-UI event to encode.
            accept: The accept header value for encoding format.

        Returns:
            The SSE-formatted string.
        """
        from ag_ui.encoder import EventEncoder

        encoder = EventEncoder(accept=accept)
        return encoder.encode(event)


def protocol_messages_to_pai_messages(messages: list[Message]) -> list[ModelMessage]:
    """Convert AG-UI messages to Pydantic AI messages.

    Args:
        messages: List of AG-UI messages.

    Returns:
        List of Pydantic AI ModelMessage objects.
    """
    from ...messages import (
        ModelRequest,
        ModelRequestPart,
        ModelResponse,
        ModelResponsePart,
        SystemPromptPart,
        UserPromptPart,
    )

    result: list[ModelMessage] = []
    tool_calls: dict[str, str] = {}  # Tool call ID to tool name mapping.
    request_parts: list[ModelRequestPart] | None = None
    response_parts: list[ModelResponsePart] | None = None

    for msg in messages:
        if isinstance(msg, UserMessage | SystemMessage | DeveloperMessage) or (
            isinstance(msg, ToolMessage) and not msg.tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if request_parts is None:
                request_parts = []
                result.append(ModelRequest(parts=request_parts))
                response_parts = None

            if isinstance(msg, UserMessage):
                request_parts.append(UserPromptPart(content=msg.content))
            elif isinstance(msg, SystemMessage | DeveloperMessage):
                request_parts.append(SystemPromptPart(content=msg.content))
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')

                request_parts.append(
                    ToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                    )
                )

        elif isinstance(msg, AssistantMessage) or (  # pragma: no branch
            isinstance(msg, ToolMessage) and msg.tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX)
        ):
            if response_parts is None:
                response_parts = []
                result.append(ModelResponse(parts=response_parts))
                request_parts = None

            if isinstance(msg, AssistantMessage):
                if msg.content:
                    response_parts.append(TextPart(content=msg.content))

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_call_id = tool_call.id
                        tool_name = tool_call.function.name
                        tool_calls[tool_call_id] = tool_name

                        if tool_call_id.startswith(_BUILTIN_TOOL_CALL_ID_PREFIX):
                            _, provider_name, tool_call_id = tool_call_id.split('|', 2)
                            response_parts.append(
                                BuiltinToolCallPart(
                                    tool_name=tool_name,
                                    args=tool_call.function.arguments,
                                    tool_call_id=tool_call_id,
                                    provider_name=provider_name,
                                )
                            )
                        else:
                            response_parts.append(
                                ToolCallPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    args=tool_call.function.arguments,
                                )
                            )
            else:
                tool_call_id = msg.tool_call_id
                tool_name = tool_calls.get(tool_call_id)
                if tool_name is None:  # pragma: no cover
                    raise ValueError(f'Tool call with ID {tool_call_id} not found in the history.')
                _, provider_name, tool_call_id = tool_call_id.split('|', 2)

                response_parts.append(
                    BuiltinToolReturnPart(
                        tool_name=tool_name,
                        content=msg.content,
                        tool_call_id=tool_call_id,
                        provider_name=provider_name,
                    )
                )

    return result
