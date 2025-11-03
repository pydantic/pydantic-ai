"""OpenAI Chat Completions protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the OpenAI Chat Completions protocol.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from openai.types.chat import ChatCompletionChunk, CompletionCreateParams
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

from ...messages import (
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import SSE_CONTENT_TYPE, UIEventStream

__all__ = [
    'ChatCompletionsEventStream',
]


@dataclass
class ChatCompletionsEventStream(UIEventStream[CompletionCreateParams, ChatCompletionChunk, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the OpenAI Chat Completions protocol."""

    _role_sent: bool = False
    _tool_call_part_started: ToolCallPart | None = None
    _tool_call_index: int = 0
    _run_id: str = ''

    @property
    def content_type(self) -> str:
        return SSE_CONTENT_TYPE

    def encode_event(self, event: ChatCompletionChunk) -> str:
        """Encode a ChatCompletionChunk as a Server-Sent Event."""
        return f'data: {event.model_dump_json()}\n\n'

    async def encode_stream(self, stream: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[str]:
        """Encode a stream of ChatCompletionChunk events, adding [DONE] at the end."""
        async for event in stream:
            yield self.encode_event(event)
        # Add the [DONE] marker at the end
        yield 'data: [DONE]\n\n'

    async def before_stream(self) -> AsyncIterator[ChatCompletionChunk]:
        """Initialize streaming by generating a unique run ID."""
        self._run_id = self.new_message_id()
        # No initial events needed for Chat Completions
        return
        yield  # type: ignore[unreachable]

    async def after_stream(self) -> AsyncIterator[ChatCompletionChunk]:
        """Send the final chunk with finish_reason."""
        yield ChatCompletionChunk(
            id=self._run_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(),
                    index=0,
                    finish_reason='stop',
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=self._get_model_name(),
            object='chat.completion.chunk',
        )

    async def on_error(self, error: Exception) -> AsyncIterator[ChatCompletionChunk]:
        """Handle errors by sending a final chunk with error finish_reason."""
        # OpenAI doesn't have a specific error event, just end the stream
        return
        yield  # type: ignore[unreachable]

    def _get_model_name(self) -> str:
        """Extract model name from run input."""
        if isinstance(self.run_input, dict):
            return str(self.run_input.get('model', 'unknown'))
        return 'unknown'

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[ChatCompletionChunk]:
        """Handle the start of a text part."""
        if not self._role_sent:
            delta = ChoiceDelta(role='assistant')
            self._role_sent = True
            yield self._create_chunk(delta)

        if part.content:
            delta = ChoiceDelta(content=part.content)
            yield self._create_chunk(delta)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[ChatCompletionChunk]:
        """Handle a text part delta."""
        if not self._role_sent:
            choice_delta = ChoiceDelta(role='assistant', content=delta.content_delta or '')
            self._role_sent = True
        else:
            choice_delta = ChoiceDelta(content=delta.content_delta or '')

        if choice_delta.role or choice_delta.content:
            yield self._create_chunk(choice_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[
        ChatCompletionChunk]:
        """Handle the end of a text part."""
        # No specific chunk needed for text end
        return
        yield  # type: ignore[unreachable]

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[ChatCompletionChunk]:
        """Handle the start of a tool call."""
        if not self._role_sent:
            delta = ChoiceDelta(role='assistant')
            self._role_sent = True
            yield self._create_chunk(delta)

        # Store the tool call part for the first delta
        self._tool_call_part_started = part

        # Send initial tool call with id, type, and name
        if part.args:
            args_str = part.args_as_json_str()
            delta = ChoiceDelta(
                tool_calls=[
                    ChoiceDelta.ToolCall(
                        index=self._tool_call_index,
                        id=part.tool_call_id,
                        type='function',
                        function=ChoiceDelta.ToolCall.Function(
                            name=part.tool_name,
                            arguments=args_str,
                        ),
                    )
                ]
            )
            self._tool_call_part_started = None
            yield self._create_chunk(delta)

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[ChatCompletionChunk]:
        """Handle a tool call delta."""
        if self._tool_call_part_started:
            # First delta for a new tool call
            args_delta = delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta)
            choice_delta = ChoiceDelta(
                tool_calls=[
                    ChoiceDelta.ToolCall(
                        index=self._tool_call_index,
                        id=self._tool_call_part_started.tool_call_id,
                        type='function',
                        function=ChoiceDelta.ToolCall.Function(
                            name=self._tool_call_part_started.tool_name,
                            arguments=args_delta,
                        ),
                    )
                ]
            )
            self._tool_call_part_started = None
        else:
            # Subsequent delta for the same tool call
            args_delta = delta.args_delta if isinstance(delta.args_delta, str) else json.dumps(delta.args_delta)
            choice_delta = ChoiceDelta(
                tool_calls=[
                    ChoiceDelta.ToolCall(
                        index=self._tool_call_index,
                        function=ChoiceDelta.ToolCall.Function(
                            arguments=args_delta,
                        ),
                    )
                ]
            )

        yield self._create_chunk(choice_delta)

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[ChatCompletionChunk]:
        """Handle the end of a tool call."""
        # Move to next tool call index
        self._tool_call_index += 1
        # No specific chunk needed for tool call end
        return
        yield  # type: ignore

    def _create_chunk(self, delta: ChoiceDelta) -> ChatCompletionChunk:
        """Create a ChatCompletionChunk with the given delta."""
        return ChatCompletionChunk(
            id=self._run_id,
            choices=[
                ChunkChoice(
                    delta=delta,
                    index=0,
                    finish_reason=None,
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=self._get_model_name(),
            object='chat.completion.chunk',
        )
