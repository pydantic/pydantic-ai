"""AG-UI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the AG-UI protocol,
enabling streaming event-based communication for interactive AI applications.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from typing import Final

from ...messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FunctionToolResultEvent,
    RetryPromptPart,
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
    from ag_ui.encoder import EventEncoder

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

SSE_CONTENT_TYPE: Final[str] = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""


BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'


class AGUIEventStream(BaseEventStream[RunAgentInput, BaseEvent, AgentDepsT]):
    """TODO (DouwM): Docstring."""

    def __init__(self, request: RunAgentInput) -> None:
        """Initialize AG-UI event stream state."""
        super().__init__(request)
        self._thinking_text = False
        self._builtin_tool_call_ids: dict[str, str] = {}

    def encode_event(self, event: BaseEvent, accept: str | None = None) -> str:
        """Encode an AG-UI event as SSE.

        Args:
            event: The AG-UI event to encode.
            accept: The accept header value for encoding format.

        Returns:
            The SSE-formatted string.
        """
        encoder = EventEncoder(accept=accept or SSE_CONTENT_TYPE)
        return encoder.encode(event)

    async def before_stream(self) -> AsyncIterator[BaseEvent]:
        """Yield events before agent streaming starts."""
        yield RunStartedEvent(
            thread_id=self.request.thread_id,
            run_id=self.request.run_id,
        )

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        """Handle an AgentRunResultEvent, cleaning up any pending state."""
        yield RunFinishedEvent(
            thread_id=self.request.thread_id,
            run_id=self.request.run_id,
        )

    async def on_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        """Handle errors during streaming."""
        yield RunErrorEvent(message=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseEvent]:
        """Handle a TextPart at start."""
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextMessageStartEvent(message_id=message_id)

        if part.content:  # pragma: no branch
            yield TextMessageContentEvent(message_id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a TextPartDelta."""
        if delta.content_delta:  # pragma: no branch
            yield TextMessageContentEvent(message_id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseEvent]:
        """Handle a TextPart at end."""
        if not followed_by_text:
            yield TextMessageEndEvent(message_id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        """Handle a ThinkingPart at start."""
        if not follows_thinking:
            yield ThinkingStartEvent(type=EventType.THINKING_START)

        if part.content:
            yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
            yield ThinkingTextMessageContentEvent(type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=part.content)
            self._thinking_text = True

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a ThinkingPartDelta."""
        if not delta.content_delta:
            return

        if not self._thinking_text:
            yield ThinkingTextMessageStartEvent(type=EventType.THINKING_TEXT_MESSAGE_START)
            self._thinking_text = True

        yield ThinkingTextMessageContentEvent(type=EventType.THINKING_TEXT_MESSAGE_CONTENT, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        """Handle a ThinkingPart at end."""
        if self._thinking_text:
            yield ThinkingTextMessageEndEvent(type=EventType.THINKING_TEXT_MESSAGE_END)
            self._thinking_text = False

        if not followed_by_thinking:
            yield ThinkingEndEvent(type=EventType.THINKING_END)

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a BuiltinToolCallPart at start."""
        tool_call_id = part.tool_call_id
        builtin_tool_call_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
        self._builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
        tool_call_id = builtin_tool_call_id

        return self._handle_tool_call_start(part, tool_call_id)

    async def _handle_tool_call_start(
        self, part: ToolCallPart | BuiltinToolCallPart, tool_call_id: str | None = None
    ) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        tool_call_id = tool_call_id or part.tool_call_id
        message_id = self.message_id or self.new_message_id()

        yield ToolCallStartEvent(tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=message_id)
        if part.args:
            yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPartDelta."""
        tool_call_id = delta.tool_call_id
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        if tool_call_id in self._builtin_tool_call_ids:
            tool_call_id = self._builtin_tool_call_ids[tool_call_id]
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a ToolCallPart at end."""
        yield ToolCallEndEvent(tool_call_id=part.tool_call_id)

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        """Handle a BuiltinToolCallPart at end."""
        yield ToolCallEndEvent(tool_call_id=self._builtin_tool_call_ids[part.tool_call_id])

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseEvent]:
        """Handle a BuiltinToolReturnPart."""
        tool_call_id = self._builtin_tool_call_ids[part.tool_call_id]
        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=tool_call_id,
            content=part.model_response_str(),
        )

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        output = result.model_response() if isinstance(result, RetryPromptPart) else result.model_response_str()

        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=output,
        )

        if isinstance(result, ToolReturnPart):
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

        # TODO (DouweM): Stream ToolCallResultEvent.content as user parts?
