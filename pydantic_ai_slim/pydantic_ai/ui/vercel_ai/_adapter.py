"""Vercel AI adapter for handling requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from ...agent import AgentDepsT
from ...messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ...output import OutputDataT
from ..adapter import BaseAdapter
from ..event_stream import BaseEventStream
from ._event_stream import VercelAIEventStream
from ._request_types import (
    DataUIPart,
    DynamicToolUIPart,
    FileUIPart,
    ReasoningUIPart,
    RequestData,
    TextUIPart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    ToolUIPart,
    UIMessage,
    request_data_ta,
)
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
class VercelAIAdapter(BaseAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    """TODO (DouwM): Docstring."""

    @classmethod
    async def validate_request(cls, request: Request) -> RequestData:
        """Validate a Vercel AI request."""
        return request_data_ta.validate_json(await request.body())

    @property
    def event_stream(self) -> BaseEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]:
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

    @classmethod
    def load_messages(cls, messages: Sequence[UIMessage]) -> list[ModelMessage]:  # noqa: C901
        """Load messages from the request and return the loaded messages."""
        result: list[ModelMessage] = []
        request_parts: list[ModelRequestPart] | None = None
        response_parts: list[ModelResponsePart] | None = None

        for msg in messages:
            if msg.role in ('system', 'user'):
                if request_parts is None:  # TODO (DouweM): coverage branch
                    request_parts = []
                    result.append(ModelRequest(parts=request_parts))
                    response_parts = None

                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        if msg.role == 'system':
                            request_parts.append(SystemPromptPart(content=part.text))  # TODO (DouweM): coverage
                        else:
                            request_parts.append(UserPromptPart(content=part.text))
                    elif isinstance(part, FileUIPart):  # TODO (DouweM): coverage
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError:
                            media_type_prefix = part.media_type.split('/', 1)[0]
                            match media_type_prefix:
                                case 'image':
                                    file = ImageUrl(url=part.url, media_type=part.media_type)
                                case 'video':
                                    file = VideoUrl(url=part.url, media_type=part.media_type)
                                case 'audio':
                                    file = AudioUrl(url=part.url, media_type=part.media_type)
                                case _:
                                    file = DocumentUrl(url=part.url, media_type=part.media_type)
                        request_parts.append(UserPromptPart(content=[file]))

            elif msg.role == 'assistant':  # TODO (DouweM): coverage branch
                for part in msg.parts:
                    if response_parts is None:
                        response_parts = []
                        result.append(ModelResponse(parts=response_parts))
                        request_parts = None

                    if isinstance(part, TextUIPart):
                        response_parts.append(TextPart(content=part.text))
                    elif isinstance(part, ReasoningUIPart):
                        response_parts.append(ThinkingPart(content=part.text))  # TODO (DouweM): coverage
                    elif isinstance(part, FileUIPart):  # TODO (DouweM): coverage
                        try:
                            file = BinaryContent.from_data_uri(part.url)
                        except ValueError as e:
                            # We don't yet handle non-data-URI file URLs returned by assistants, as no Pydantic AI models do this.
                            raise ValueError(
                                'Vercel AI integration can currently only handle assistant file parts with data URIs.'
                            ) from e
                        response_parts.append(FilePart(content=file))
                    elif isinstance(part, DataUIPart):
                        # Not currently supported
                        pass
                    elif isinstance(part, ToolUIPart | DynamicToolUIPart):  # TODO (DouweM): coverage branch
                        if isinstance(part, DynamicToolUIPart):  # TODO (DouweM): coverage
                            tool_name = part.tool_name
                            builtin_tool = False
                        else:
                            tool_name = part.type.removeprefix('tool-')
                            builtin_tool = part.provider_executed

                        tool_call_id = part.tool_call_id
                        args = part.input

                        if builtin_tool:  # TODO (DouweM): coverage
                            call_part = BuiltinToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                            response_parts.append(call_part)

                            if isinstance(part, ToolOutputAvailablePart | ToolOutputErrorPart):
                                if part.state == 'output-available':
                                    output = part.output
                                else:
                                    output = part.error_text

                                provider_name = (
                                    (part.call_provider_metadata or {}).get('pydantic_ai', {}).get('provider_name')
                                )
                                call_part.provider_name = provider_name

                                return_part = BuiltinToolReturnPart(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call_id,
                                    content=output,
                                    provider_name=provider_name,
                                )
                                response_parts.append(return_part)
                        else:
                            call_part = ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)
                            response_parts.append(call_part)

                            result_part: ToolReturnPart | RetryPromptPart | None = None
                            if part.state == 'output-available':
                                result_part = ToolReturnPart(
                                    tool_name=tool_name, tool_call_id=tool_call_id, content=part.output
                                )
                            elif part.state == 'output-error':  # TODO (DouweM): coverage
                                result_part = RetryPromptPart(
                                    tool_name=tool_name, tool_call_id=tool_call_id, content=part.error_text
                                )

                            if result_part:  # TODO (DouweM): coverage branch
                                request_parts = [result_part]
                                result.append(ModelRequest(parts=request_parts))
                                response_parts = None

        return result
