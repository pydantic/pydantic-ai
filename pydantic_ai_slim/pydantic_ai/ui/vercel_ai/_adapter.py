"""Vercel AI adapter for handling requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from ...agent import AgentDepsT
from ...messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from ..adapter import BaseAdapter
from ..event_stream import BaseEventStream
from ._event_stream import VercelAIEventStream
from ._request_types import RequestData, TextUIPart, UIMessage, request_data_ta
from ._response_types import BaseChunk

try:
    from starlette.requests import Request
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e

# See https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
VERCEL_AI_DSP_HEADERS = {'x-vercel-ai-ui-message-stream': 'v1'}


__all__ = ['VercelAIAdapter']


@dataclass
class VercelAIAdapter(BaseAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT]):
    """TODO (DouwM): Docstring."""

    @classmethod
    async def validate_request(cls, request: Request) -> RequestData:
        """Validate a Vercel AI request."""
        return request_data_ta.validate_json(await request.body())

    def dump_messages(self, messages: Sequence[ModelMessage]) -> list[UIMessage]:
        """Dump messages to the request and return the dumped messages."""
        # TODO (DouweM): implement
        raise NotImplementedError

    @cached_property
    def event_stream(self) -> BaseEventStream[RequestData, BaseChunk, AgentDepsT]:
        return VercelAIEventStream(self.request)

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Get the response headers for the adapter."""
        return VERCEL_AI_DSP_HEADERS

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Convert Vercel AI protocol messages to Pydantic AI messages.

        Args:
            messages: List of Vercel AI UIMessage objects.

        Returns:
            List of Pydantic AI ModelMessage objects.

        Raises:
            ValueError: If message format is not supported.
        """
        return self.load_messages(self.request.messages)

    # TODO (DouweM): model, builtin_tools?

    @classmethod
    def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:
        """Load messages from the request and return the loaded messages."""
        pai_messages: list[ModelMessage] = []

        for msg in messages:
            if msg.role == 'user':
                # User message - extract text from parts
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)
                    else:
                        raise ValueError(f'Only text parts are supported for user messages, got {type(part).__name__}')

                if texts:
                    pai_messages.append(ModelRequest(parts=[UserPromptPart(content='\n'.join(texts))]))

            elif msg.role == 'assistant':
                # Assistant message - for now, just extract text
                # Full reconstruction of ModelResponse with tool calls would require more complex logic
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)
                    # TODO: Handle ToolOutputAvailablePart for full message history reconstruction

                if texts:
                    pai_messages.append(ModelResponse(parts=[TextPart(content='\n'.join(texts))]))

            elif msg.role == 'system':
                # System message - not in standard Vercel AI protocol but might be custom
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)

                if texts:
                    pai_messages.append(ModelRequest(parts=[SystemPromptPart(content='\n'.join(texts))]))

        return pai_messages
