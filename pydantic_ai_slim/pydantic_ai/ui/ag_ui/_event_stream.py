"""AG-UI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the AG-UI protocol,
enabling streaming event-based communication for interactive AI applications.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Final
from uuid import uuid4

from ..._utils import now_utc
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
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import SSE_CONTENT_TYPE, NativeEvent, UIEventStream

try:
    from ag_ui.core import (
        BaseEvent,
        EventType,
        ReasoningEncryptedValueEvent,
        ReasoningEndEvent,
        ReasoningMessageContentEvent,
        ReasoningMessageEndEvent,
        ReasoningMessageStartEvent,
        ReasoningStartEvent,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
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

BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'

_THINKING_METADATA_ATTRS: Final[tuple[str, ...]] = ('id', 'signature', 'provider_name', 'provider_details')


def thinking_encrypted_metadata(part: ThinkingPart) -> dict[str, Any]:
    """Collect non-None metadata fields from a ThinkingPart for AG-UI encrypted_value."""
    encrypted: dict[str, Any] = {
        attr: value for attr in _THINKING_METADATA_ATTRS if (value := getattr(part, attr)) is not None
    }
    return encrypted


@dataclass
class AGUIEventStream(UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Agent-User Interaction (AG-UI) protocol."""

    _reasoning_message_id: str | None = None
    _reasoning_started: bool = False
    _reasoning_text: bool = False
    _builtin_tool_call_ids: dict[str, str] = field(default_factory=dict[str, str])
    _error: bool = False

    @property
    def _event_encoder(self) -> EventEncoder:
        return EventEncoder(accept=self.accept or SSE_CONTENT_TYPE)

    @property
    def content_type(self) -> str:
        return self._event_encoder.get_content_type()

    def encode_event(self, event: BaseEvent) -> str:
        return self._event_encoder.encode(event)

    @staticmethod
    def _get_timestamp() -> int:
        return int(now_utc().timestamp() * 1_000)

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[BaseEvent]:
        """Override to set timestamps on all AG-UI events."""
        async for agui_event in super().handle_event(event):
            if agui_event.timestamp is None:
                agui_event.timestamp = self._get_timestamp()
            yield agui_event

    async def before_stream(self) -> AsyncIterator[BaseEvent]:
        yield RunStartedEvent(
            thread_id=self.run_input.thread_id,
            run_id=self.run_input.run_id,
            timestamp=self._get_timestamp(),
        )

    async def before_response(self) -> AsyncIterator[BaseEvent]:
        # Prevent parts from a subsequent response being tied to parts from an earlier response.
        # See https://github.com/pydantic/pydantic-ai/issues/3316
        self.new_message_id()
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        if not self._error:
            yield RunFinishedEvent(
                thread_id=self.run_input.thread_id,
                run_id=self.run_input.run_id,
                timestamp=self._get_timestamp(),
            )

    async def on_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        self._error = True
        yield RunErrorEvent(message=str(error), timestamp=self._get_timestamp())

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseEvent]:
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextMessageStartEvent(message_id=message_id)

        if part.content:  # pragma: no branch
            yield TextMessageContentEvent(message_id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        if delta.content_delta:  # pragma: no branch
            yield TextMessageContentEvent(message_id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseEvent]:
        if not followed_by_text:
            yield TextMessageEndEvent(message_id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        self._reasoning_message_id = str(uuid4())
        self._reasoning_started = False

        if part.content:
            yield ReasoningStartEvent(message_id=self._reasoning_message_id)
            self._reasoning_started = True
            yield ReasoningMessageStartEvent(message_id=self._reasoning_message_id, role='assistant')
            yield ReasoningMessageContentEvent(message_id=self._reasoning_message_id, delta=part.content)
            self._reasoning_text = True

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseEvent]:
        if not delta.content_delta:
            return  # pragma: no cover

        message_id = self._reasoning_message_id or ''

        if not self._reasoning_started:
            yield ReasoningStartEvent(message_id=message_id)
            self._reasoning_started = True

        if not self._reasoning_text:
            yield ReasoningMessageStartEvent(message_id=message_id, role='assistant')
            self._reasoning_text = True

        yield ReasoningMessageContentEvent(message_id=message_id, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        message_id = self._reasoning_message_id or ''

        encrypted = thinking_encrypted_metadata(part)

        if not self._reasoning_started and not encrypted:
            self._reasoning_message_id = None
            return

        if not self._reasoning_started:
            yield ReasoningStartEvent(message_id=message_id)

        if self._reasoning_text:
            yield ReasoningMessageEndEvent(message_id=message_id)
            self._reasoning_text = False

        if encrypted:
            yield ReasoningEncryptedValueEvent(
                subtype='message',
                entity_id=message_id,
                encrypted_value=json.dumps(encrypted),
            )

        yield ReasoningEndEvent(message_id=message_id)
        self._reasoning_message_id = None

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = part.tool_call_id
        builtin_tool_call_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
        self._builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
        tool_call_id = builtin_tool_call_id

        return self._handle_tool_call_start(part, tool_call_id)

    async def _handle_tool_call_start(
        self, part: ToolCallPart | BuiltinToolCallPart, tool_call_id: str | None = None
    ) -> AsyncIterator[BaseEvent]:
        tool_call_id = tool_call_id or part.tool_call_id
        parent_message_id = self.message_id

        yield ToolCallStartEvent(
            tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=parent_message_id
        )
        if part.args:
            yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseEvent]:
        tool_call_id = delta.tool_call_id
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        if tool_call_id in self._builtin_tool_call_ids:
            tool_call_id = self._builtin_tool_call_ids[tool_call_id]
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseEvent]:
        yield ToolCallEndEvent(tool_call_id=part.tool_call_id)

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseEvent]:
        yield ToolCallEndEvent(tool_call_id=self._builtin_tool_call_ids[part.tool_call_id])

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = self._builtin_tool_call_ids[part.tool_call_id]
        # Use a one-off message ID instead of `self.new_message_id()` to avoid
        # mutating `self.message_id`, which is used as `parent_message_id` for
        # subsequent tool calls in the same response.
        yield ToolCallResultEvent(
            message_id=str(uuid4()),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=tool_call_id,
            content=part.model_response_str(),
        )

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        result = event.result
        output = result.model_response() if isinstance(result, RetryPromptPart) else result.model_response_str()

        yield ToolCallResultEvent(
            message_id=self.new_message_id(),
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=output,
        )

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that AG-UI does not currently have events for

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
