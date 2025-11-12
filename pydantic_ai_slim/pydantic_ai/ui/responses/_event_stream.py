"""OpenAI Responses protocol event stream transformer for Pydantic AI agents."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai.types.responses import (
    Response as ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from ...messages import (
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import UIEventStream

__all__ = [
    'ResponsesEventStream',
]


@dataclass
class ResponsesEventStream(UIEventStream[ResponseCreateParamsStreaming, Any, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the OpenAI Responses protocol."""

    _response_id: str = ''
    _item_id: str = ''
    _output_index: int = 0
    _content_index: int = 0
    _sequence_number: int = 0
    _message_added: bool = False
    _content_part_added: bool = False

    @property
    def content_type(self) -> str:
        return 'text/event-stream'

    def encode_event(self, event: Any) -> str:
        """Encode a Responses event as a Server-Sent Event."""
        if hasattr(event, 'type'):
            event_type = event.type
            event_data = event.model_dump_json(exclude_unset=True)
            return f'event: {event_type}\ndata: {event_data}\n\n'
        return f'data: {json.dumps(event)}\n\n'

    async def encode_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[str]:
        """Encode a stream of Responses events, adding [DONE] at the end."""
        async for event in stream:
            yield self.encode_event(event)
        # Add the [DONE] marker at the end
        yield 'event: done\ndata: [DONE]\n\n'

    async def before_stream(self) -> AsyncIterator[Any]:
        """Initialize streaming by generating response.created event."""
        from openai.types.responses import ResponseCreatedEvent
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

        self._response_id = self.new_message_id()
        self._item_id = str(uuid.uuid4())

        # Get model name from run input
        model = 'unknown'
        instructions = None
        if isinstance(self.run_input, dict):
            model = str(self.run_input.get('model', 'unknown'))
            instructions = self.run_input.get('instructions')

        # Send initial response.created event
        initial_response = ResponseObject(
            id=self._response_id,
            object='response',
            created_at=time.time(),
            model=model,
            output=[],
            status='in_progress',
            usage=ResponseUsage(
                input_tokens=0,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
            instructions=instructions,
            parallel_tool_calls=True,
            tool_choice='auto',
            tools=[],
        )
        created_event = ResponseCreatedEvent(
            type='response.created',
            response=initial_response,
            sequence_number=self._sequence_number,
        )
        self._sequence_number += 1
        yield created_event

    async def after_stream(self) -> AsyncIterator[Any]:
        """Send the final response.completed event."""
        from openai.types.responses import ResponseCompletedEvent
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

        # Get model name and instructions from run input
        model = 'unknown'
        instructions = None
        if isinstance(self.run_input, dict):
            model = str(self.run_input.get('model', 'unknown'))
            instructions = self.run_input.get('instructions')

        # Get usage from result if available
        usage = ResponseUsage(
            input_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        )

        final_response = ResponseObject(
            id=self._response_id,
            object='response',
            created_at=time.time(),
            model=model,
            output=[],
            status='completed',
            usage=usage,
            instructions=instructions,
            parallel_tool_calls=True,
            tool_choice='auto',
            tools=[],
        )
        completed_event = ResponseCompletedEvent(
            type='response.completed',
            response=final_response,
            sequence_number=self._sequence_number,
        )
        self._sequence_number += 1
        yield completed_event

    async def on_error(self, error: Exception) -> AsyncIterator[Any]:
        """Handle errors by sending error event."""
        # For Responses protocol, we can send an error event or just end the stream
        return
        yield  # type: ignore[unreachable]

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[Any]:
        """Handle the start of a text part."""
        from openai.types.responses import (
            ResponseOutputItemAddedEvent,
            ResponseContentPartAddedEvent,
        )

        events: list[Any] = []

        # Add output item (message) if not already added
        if not self._message_added:
            message_item = ResponseOutputMessage(
                type='message',
                id=self._item_id,
                role='assistant',
                status='in_progress',
                content=[],
            )
            events.append(
                ResponseOutputItemAddedEvent(
                    type='response.output_item.added',
                    item=message_item,
                    output_index=self._output_index,
                    sequence_number=self._sequence_number,
                )
            )
            self._sequence_number += 1
            self._message_added = True

        # Add the content part
        if not self._content_part_added:
            content_part = ResponseOutputText(
                type='output_text',
                text='',
                annotations=[],
            )
            events.append(
                ResponseContentPartAddedEvent(
                    type='response.content_part.added',
                    item_id=self._item_id,
                    output_index=self._output_index,
                    content_index=self._content_index,
                    part=content_part,
                    sequence_number=self._sequence_number,
                )
            )
            self._sequence_number += 1
            self._content_part_added = True

        for event in events:
            yield event

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[Any]:
        """Handle a text part delta."""
        from openai.types.responses import ResponseTextDeltaEvent

        if delta.content_delta:
            yield ResponseTextDeltaEvent(
                type='response.output_text.delta',
                item_id=self._item_id,
                output_index=self._output_index,
                content_index=self._content_index,
                delta=delta.content_delta,
                logprobs=[],
                sequence_number=self._sequence_number,
            )
            self._sequence_number += 1

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[Any]:
        """Handle the end of a text part."""
        # No specific event needed for text end in Responses protocol
        return
        yield  # type: ignore[unreachable]

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[Any]:
        """Handle the start of a tool call."""
        from openai.types.responses import (
            ResponseOutputItemAddedEvent,
            ResponseFunctionToolCall,
        )

        # Tool calls are separate output items in Responses protocol
        tool_call_item = ResponseFunctionToolCall(
            type='function_tool_call',
            call_id=part.tool_call_id,
            name=part.tool_name,
            arguments=part.args_as_json_str(),
        )

        yield ResponseOutputItemAddedEvent(
            type='response.output_item.added',
            item=tool_call_item,
            output_index=self._output_index,
            sequence_number=self._sequence_number,
        )
        self._sequence_number += 1
        self._output_index += 1

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[Any]:
        """Handle a tool call delta."""
        # For simplicity, we send the complete tool call at start
        # Streaming tool call arguments could be implemented if needed
        return
        yield  # type: ignore[unreachable]

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[Any]:
        """Handle the end of a tool call."""
        # No specific event needed for tool call end
        return
        yield  # type: ignore[unreachable]
