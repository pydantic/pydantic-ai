"""AG-UI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the AG-UI protocol,
enabling streaming event-based communication for interactive AI applications.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Final

from ...messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...tools import AgentDepsT
from .. import BaseEventStream

if TYPE_CHECKING:
    pass  # Agent type is not actually used in this module

try:
    from ag_ui.core import (
        BaseEvent,
        EventType,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ThinkingEndEvent,
        ThinkingStartEvent,
        ThinkingTextMessageContentEvent,
        ThinkingTextMessageEndEvent,
        ThinkingTextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

__all__ = [
    'AGUIEventStream',
    'RunAgentInput',
    'RunStartedEvent',
    'RunFinishedEvent',
]

BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'


class AGUIEventStream(BaseEventStream[RunAgentInput, BaseEvent, AgentDepsT]):
    """Transforms Pydantic AI agent events into AG-UI protocol events.

    This class handles the stateful transformation of streaming agent events
    into the AG-UI protocol format, managing message IDs, thinking mode state,
    and tool call ID mappings for builtin tools.

    Example:
        ```python
        event_stream = AGUIEventStream()
        async for ag_ui_event in event_stream.handle_event(pai_event):
            print(ag_ui_event)
        ```
    """

    def __init__(self, request: RunAgentInput) -> None:
        """Initialize AG-UI event stream state."""
        super().__init__(request)
        self.part_end: BaseEvent | None = None
        self.thinking: bool = False
        self.builtin_tool_call_ids: dict[str, str] = {}

    async def before_stream(self) -> AsyncIterator[BaseEvent]:
        """Yield events before agent streaming starts."""
        yield RunStartedEvent(
            thread_id=self.request.thread_id,
            run_id=self.request.run_id,
        )

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
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

        # Emit finish event
        yield RunFinishedEvent(
            thread_id=self.request.thread_id,
            run_id=self.request.run_id,
        )

    async def on_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        """Handle errors during streaming."""
        # Try to get code from exception if it has one, otherwise use class name
        code = getattr(error, 'code', error.__class__.__name__)
        yield RunErrorEvent(message=str(error), code=code)

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
            builtin_tool_call_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
            self.builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
            tool_call_id = builtin_tool_call_id

        message_id = self.message_id or self.new_message_id()
        yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=message_id)
        if part.args:
            yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=part.args_as_json_str())
        self.part_end = ToolCallEndEvent(tool_call_id=tool_call_id)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a BuiltinToolCallPart at start."""
        return self.handle_tool_call_start(part)

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

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[BaseEvent]:
        """Handle a FinalResultEvent.

        This event is emitted when the agent produces a final result, but no AG-UI events
        are needed at this stage.
        """
        return
        yield  # Make this an async generator
