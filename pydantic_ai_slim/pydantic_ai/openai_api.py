"""OpenAI API compatibility module for Pydantic AI agents.

This module provides OpenAI API-compatible endpoints for Pydantic AI agents, allowing them to be used
as drop-in replacements for OpenAI's Chat Completions API and Responses API. The module includes
functionality for handling both streaming and non-streaming requests, tool calls, multimodal inputs,
and conversation history.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    get_args, TypeVar,
)

from openai.types.chat.chat_completion_content_part_param import File
from pydantic import TypeAdapter, ValidationError

from .agent import AbstractAgent
from .exceptions import (
    AgentRunError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from .messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    DocumentFormat,
    AudioFormat,
    ImageFormat,
    VideoFormat,
)
from .models import KnownModelName, Model
from .output import OutputDataT, OutputSpec
from .run import AgentRunResult
from .settings import ModelSettings
from .tools import AgentDepsT
from .toolsets import AbstractToolset
from .ui.openai._adapter import ChatCompletionsAdapter
from .usage import RunUsage, UsageLimits

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse
    from starlette.routing import BaseRoute
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_openai_api()` method, '
        'you can use the `openai` & `starlette` optional group — `pip install "pydantic-ai-slim[openai,starlette]"`'
    ) from e

try:
    from openai import (
        APIError,
        APIStatusError,
    )
    from openai.types import CompletionUsage
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        CompletionCreateParams as CompletionCreateParamsT,
        ChatCompletionContentPartTextParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
    )
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )
    from openai.types.chat.completion_create_params import (
        CompletionCreateParamsNonStreaming as CompletionCreateParamsNonStreamingT,
        CompletionCreateParamsStreaming as CompletionCreateParamsStreamingT,
    )
    from openai.types.responses import (
        Response as ResponseObject,
        ResponseCreateParams as ResponseCreateParamsT,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseUsage,
    )
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsNonStreaming as ResponseCreateParamsNonStreamingT,
        ResponseCreateParamsStreaming as ResponseCreateParamsStreamingT,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class OpenAIApp(Generic[AgentDepsT, OutputDataT], Starlette):
    """ASGI application for running Pydantic AI agents with OpenAI API-compatible endpoints.

    This class provides a Starlette-based ASGI application that exposes Pydantic AI agents through
    OpenAI-compatible API endpoints, specifically `/v1/chat/completions` and `/v1/responses`.
    The application handles both streaming and non-streaming requests, tool calls, multimodal inputs,
    and conversation history, making it a drop-in replacement for OpenAI's API.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.openai_api import OpenAIApp

        agent = Agent('openai:gpt-4o')
        app = OpenAIApp(agent)
        ```

        To run the application:

        ```bash
        uvicorn app:app --host 0.0.0.0 --port 8000
        ```

    The application automatically sets up the following routes:
    - `POST /v1/chat/completions` - OpenAI Chat Completions API compatible endpoint
    - `POST /v1/responses` - OpenAI Responses API compatible endpoint
    """

    def __init__(
            self,
            agent: AbstractAgent[AgentDepsT, OutputDataT],
            *,
            # Agent.iter parameters.
            output_type: OutputSpec[Any] | None = None,
            model: Model | KnownModelName | str | None = None,
            deps: AgentDepsT | None = None,
            model_settings: ModelSettings | None = None,
            usage_limits: UsageLimits | None = None,
            usage: RunUsage | None = None,
            infer_name: bool = True,
            toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
            # Starlette parameters.
            debug: bool = False,
            routes: Sequence[BaseRoute] | None = None,
            middleware: Sequence[Middleware] | None = None,
            exception_handlers: Mapping[Any, ExceptionHandler] | None = None,
            on_startup: Sequence[Callable[[], Any]] | None = None,
            on_shutdown: Sequence[Callable[[], Any]] | None = None,
            lifespan: Lifespan[OpenAIApp[AgentDepsT, OutputDataT]] | None = None,
    ):
        """Initialize the OpenAI API-compatible ASGI application.

        Args:
            agent: The Pydantic AI agent to expose via the OpenAI API endpoints.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
                no output validators since output validators would expect an argument that matches the agent's
                output type.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.

            debug: Boolean indicating if debug tracebacks should be returned on errors.
            routes: A list of routes to serve incoming HTTP and WebSocket requests.
            middleware: A list of middleware to run for every request. A starlette application will always
                automatically include two middleware classes. `ServerErrorMiddleware` is added as the very
                outermost middleware, to handle any uncaught errors occurring anywhere in the entire stack.
                `ExceptionMiddleware` is added as the very innermost middleware, to deal with handled
                exception cases occurring in the routing or endpoints.
            exception_handlers: A mapping of either integer status codes, or exception class types onto
                callables which handle the exceptions. Exception handler callables should be of the form
                `handler(request, exc) -> response` and may be either standard functions, or async functions.
            on_startup: A list of callables to run on application startup. Startup handler callables do not
                take any arguments, and may be either standard functions, or async functions.
            on_shutdown: A list of callables to run on application shutdown. Shutdown handler callables do
                not take any arguments, and may be either standard functions, or async functions.
            lifespan: A lifespan context function, which can be used to perform startup and shutdown tasks.
                This is a newer style that replaces the `on_startup` and `on_shutdown` handlers. Use one or
                the other, not both.
        """
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            on_startup=on_startup,
            on_shutdown=on_shutdown,
            lifespan=lifespan,
        )

        async def chat_completions_endpoint(request: Request) -> Response:
            return await handle_chat_completions_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        async def responses_endpoint(request: Request) -> Response:
            return await handle_responses_request(
                agent,
                request,
                output_type=output_type,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            )

        self.router.add_route(
            '/v1/chat/completions', chat_completions_endpoint, methods=['POST'], name='chat_completions'
        )
        self.router.add_route('/v1/responses', responses_endpoint, methods=['POST'], name='chat_completions')


CompletionCreateParams: TypeAdapter[CompletionCreateParamsT] = TypeAdapter(CompletionCreateParamsT)
CompletionCreateParamsNonStreaming: TypeAdapter[CompletionCreateParamsNonStreamingT] = TypeAdapter(
    CompletionCreateParamsNonStreamingT)
CompletionCreateParamsStreaming: TypeAdapter[CompletionCreateParamsStreamingT] = TypeAdapter(
    CompletionCreateParamsStreamingT)

ResponseCreateParams: TypeAdapter[ResponseCreateParamsT] = TypeAdapter(ResponseCreateParamsT)
ResponseCreateParamsNonStreaming: TypeAdapter[ResponseCreateParamsNonStreamingT] = TypeAdapter(
    ResponseCreateParamsNonStreamingT)
ResponseCreateParamsStreaming: TypeAdapter[ResponseCreateParamsStreamingT] = TypeAdapter(ResponseCreateParamsStreamingT)

OpenAIRole: TypeAlias = Literal['developer', 'system', 'assistant', 'user', 'tool', 'function']


def _map_exception_to_openai_error(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Map internal exceptions to OpenAI-compatible error payload and HTTP status.

    Returns a tuple of (status_code, error_object) where error_object matches OpenAI's
    error schema: {"error": {"message": str, "type": str, "param": None, "code": None}}.
    """
    # Default values
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    error_type = 'api_error'

    if isinstance(exc, UserError):
        status_code = HTTPStatus.BAD_REQUEST
        error_type = 'invalid_request_error'
    elif isinstance(exc, UsageLimitExceeded):
        status_code = HTTPStatus.TOO_MANY_REQUESTS
        error_type = 'rate_limit_exceeded'
    elif isinstance(exc, ModelHTTPError):
        status_code = exc.status_code or HTTPStatus.BAD_GATEWAY
        if 400 <= int(status_code) < 500:
            error_type = 'invalid_request_error'
        else:
            error_type = 'api_error'
    elif isinstance(exc, UnexpectedModelBehavior):
        status_code = HTTPStatus.BAD_GATEWAY
        error_type = 'api_error'
    elif isinstance(exc, AgentRunError):
        status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        error_type = 'api_error'

    error_payload = {
        'error': {
            'message': str(exc),
            'type': error_type,
            'param': None,
            'code': None,
        }
    }
    return int(status_code), error_payload


def _openai_error_response(exc: Exception) -> Response:
    status_code, payload = _map_exception_to_openai_error(exc)
    return Response(
        content=json.dumps(payload),
        media_type='application/json',
        status_code=status_code,
    )


@dataclass
class _ChatCompletionStreamContext:
    """Internal context for tracking streaming Chat Completion state.

    This dataclass maintains state information during streaming responses to ensure
    proper OpenAI-compatible chunk generation, particularly for managing role
    information and tool call streaming.

    Attributes:
        role_sent: Whether the assistant role has been sent in the stream yet.
        tool_call_part_started: The current tool call part being streamed, if any.
        tool_call_index: Index counter for tool calls in the current response.
        got_tool_calls: Whether any tool calls have been encountered in this response.
    """

    role_sent: bool = False
    tool_call_part_started: ToolCallPart | None = None
    tool_call_index: int = 0
    got_tool_calls: bool = False


@dataclass
class _ResponsesStreamContext:
    """Internal context for tracking streaming Responses API state.

    This dataclass maintains state information during streaming responses to ensure
    proper OpenAI Responses API-compatible event generation.

    Attributes:
        sequence_number: Sequential counter for events in the stream.
        item_id: The current output item ID being streamed.
        output_index: The index of the output item in the response.
        content_index: The index of the content part within the output item.
        message_added: Whether the output message item has been added yet.
        content_part_added: Whether the content part has been added yet.
    """

    sequence_number: int = 0
    item_id: str = ''
    output_index: int = 0
    content_index: int = 0
    message_added: bool = False
    content_part_added: bool = False


_document_format_inverse_lookup: dict[DocumentFormat | str, str] = {
    'pdf': 'application/pdf',
    'txt': 'text/plain',
    'csv': 'text/csv',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'html': 'text/html',
    'md': 'text/markdown',
    'xls': 'application/vnd.ms-excel',
}
_audio_format_inverse_lookup: dict[AudioFormat | str, str] = {
    'mp3': 'audio/mpeg',
    'wav': 'audio/wav',
    'flac': 'audio/flac',
    'oga': 'audio/ogg',
    'aiff': 'audio/aiff',
    'aac': 'audio/aac',
}
_image_format_inverse_lookup: dict[ImageFormat | str, str] = {
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'gif': 'image/gif',
    'webp': 'image/webp',
}
_video_format_inverse_lookup: dict[VideoFormat | str, str] = {
    'mkv': 'video/x-matroska',
    'mov': 'video/quicktime',
    'mp4': 'video/mp4',
    'webm': 'video/webm',
    'flv': 'video/x-flv',
    'mpeg': 'video/mpeg',
    'wmv': 'video/x-ms-wmv',
    'three_gp': 'video/3gpp',
}


def _is_text_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartTextParam]:
    """Check if the part is a text content part."""
    return isinstance(part, dict) and part.get('type') == 'text'


def _is_image_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartImageParam]:
    """Check if the part is an image content part."""
    return isinstance(part, dict) and part.get('type') == 'image_url'


def _is_audio_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartInputAudioParam]:
    """Check if the part is an input audio content part."""
    return isinstance(part, dict) and part.get('type') == 'input_audio'


def _is_file_part(part: ChatCompletionContentPartParam) -> TypeGuard[File]:
    """Check if the part is an input audio content part."""
    return isinstance(part, dict) and part.get('type') == 'file'


def _parse_user_content_part(part: ChatCompletionContentPartParam) -> UserContent:
    """Parse a single user content part from OpenAI format.

    Args:
        part: A content part which can be text, image, audio, or file.

    Returns:
        A UserContent object or None if the part type is not recognized.
    """
    if _is_file_part(part):
        # part is now narrowed to File
        # Encode base64 string as bytes
        b_data = base64.b64decode(part['file']['file_data'].encode())
        filename = part['file']['filename']
        file_ext = Path(filename).suffix
        try:
            media_type = _document_format_inverse_lookup[file_ext]
        except KeyError as e:
            raise ValueError(f'Unknown media type: {file_ext}') from e

        return BinaryContent(data=b_data, media_type=media_type)

        raise ValueError(f'Unknown media type: {file_ext}')

    if _is_text_part(part):
        # part is now narrowed to ChatCompletionContentPartTextParam
        return part.get('text', '')

    elif _is_image_part(part):
        # part is now narrowed to ChatCompletionContentPartImageParam
        image_url_data = part.get('image_url')
        if image_url_data:
            url = image_url_data.get('url')
            return ImageUrl(url=url)

    elif _is_audio_part(part):
        # part is now narrowed to ChatCompletionContentPartInputAudioParam
        input_audio = part.get('input_audio')
        data = input_audio.get('data')
        format_value = input_audio.get('format')

        try:
            media_type = _audio_format_inverse_lookup[format_value]
        except KeyError as e:
            raise ValueError(f'Unknown media type: {format_value}') from e
        b_data = base64.b64decode(data.encode())
        return BinaryContent(data=b_data, media_type=media_type)

    raise ValueError(f'Unknown part type: {part}')


def _handle_assistant_message(message: ChatCompletionMessageParam) -> ModelResponse:
    """Handle an assistant role message and convert to ModelResponse.

    Args:
        message: OpenAI assistant message.

    Returns:
        A ModelResponse with text and tool call parts.
    """
    response_parts: list[TextPart | ToolCallPart] = []

    if content := message.get('content'):
        # TODO: Handle 'Iterable[ContentArrayOfContentPart]'
        #  This is now simply stringified
        response_parts.append(TextPart(content=str(content)))

    if tool_calls := message.get('tool_calls'):
        for tc in tool_calls:
            response_parts.append(
                # All fields are Required[] so indexing is safe
                ToolCallPart(
                    tool_call_id=tc['id'],
                    tool_name=tc['function']['name'],
                    # Store as json string
                    args=tc['function']['arguments'],
                )
            )

    return ModelResponse(parts=response_parts)


def _handle_user_message(message: ChatCompletionMessageParam, request_parts: list[ModelRequestPart]) -> None:
    """Handle a user role message and append to request_parts.

    Args:
        message: OpenAI user message.
        request_parts: List to append the parsed user prompt part to.
    """
    content = message.get('content')
    if isinstance(content, str):
        request_parts.append(UserPromptPart(content=content))
    elif content:  # should be a list
        parts: list[UserContent] = []
        for part in content:
            parsed_part = _parse_user_content_part(part)
            if parsed_part is not None:
                parts.append(parsed_part)

        if parts:
            request_parts.append(UserPromptPart(content=parts))


def _handle_system_message(message: ChatCompletionMessageParam, request_parts: list[ModelRequestPart]) -> None:
    """Handle a system role message and append to request_parts.

    Args:
        message: OpenAI system message.
        request_parts: List to append the system prompt part to.
    """
    if content := message.get('content'):
        request_parts.append(SystemPromptPart(content=str(content)))


def _handle_tool_message(message: ChatCompletionMessageParam, request_parts: list[ModelRequestPart]) -> None:
    """Handle a tool role message and append to request_parts.

    Args:
        message: OpenAI tool message.
        request_parts: List to append the tool return part to.
    """
    request_parts.append(
        ToolReturnPart(tool_call_id=message.get('tool_call_id', ''), content=str(message.get('content')))
    )


def _handle_function_message(message: ChatCompletionMessageParam, request_parts: list[ModelRequestPart]) -> None:
    """Handle a function role message and append to request_parts.

    'function' role is deprecated in OpenAI API but still supported for backward compatibility.

    Args:
        message: OpenAI function message.
        request_parts: List to append the tool return part to.
    """
    request_parts.append(
        ToolReturnPart(
            tool_call_id=message.get('name', ''),  # function role uses 'name' field
            content=str(message.get('content')),
        )
    )


def _from_openai_messages(messages: Iterable[ChatCompletionMessageParam]) -> list[ModelMessage]:
    """Converts OpenAI chat completion messages to Pydantic AI's internal message format.

    This function transforms OpenAI API message formats into Pydantic AI's structured
    message types, handling different roles (system, user, assistant, tool), content types
    (text, images), and tool calls. It maintains conversation flow by grouping related
    messages into ModelRequest and ModelResponse objects.

    Args:
        messages: An iterable of OpenAI chat completion message parameters.

    Returns:
        A list of ModelMessage objects (ModelRequest and ModelResponse) that represent
        the conversation history in Pydantic AI's internal format.

    Example:
        ```python
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        pydantic_messages = _from_openai_messages(openai_messages)
        ```
    """
    history: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] = []

    for message in messages:
        role: OpenAIRole | None = message.get('role')

        if role is None:
            raise RuntimeError('No role found in message.')

        if role == 'assistant':
            if request_parts:
                history.append(ModelRequest(parts=request_parts))
                request_parts = []
            history.append(_handle_assistant_message(message))
        elif role == 'system':
            _handle_system_message(message, request_parts)
        elif role == 'user':
            _handle_user_message(message, request_parts)
        elif role == 'tool':
            _handle_tool_message(message, request_parts)
        elif role == 'function':
            _handle_function_message(message, request_parts)

    if request_parts:
        history.append(ModelRequest(parts=request_parts))

    return history


def _from_responses_input(input_data: Any, instructions: str | None = None) -> list[ModelMessage]:
    """Converts OpenAI Responses API input to Pydantic AI's internal message format.

    The Responses API uses a different input format than Chat Completions:
    - `input` can be a string (simple user message) or a list of various item types
    - `instructions` (if provided) acts as a system prompt

    Args:
        input_data: The input field from ResponseCreateParams - can be str or list of items.
        instructions: Optional instructions to prepend as a system message.

    Returns:
        A list of ModelMessage objects representing the conversation in Pydantic AI format.

    Example:
        ```python
        messages = _from_responses_input("Hello!", "You are helpful")
        # Results in system prompt + user message
        ```
    """

    def _parse_response_user_content_part(part: dict[str, Any]) -> UserContent | None:
        """Parse a single Responses API user content part.

        Supports:
        - input_text: {"type":"input_text","text":...}
        - input_image: {"type":"input_image","image_url":...,"detail":...}
        - input_file: {"type":"input_file","file_data":data_uri,"filename":...}
        """
        ptype = part.get('type')
        if ptype == 'input_text':
            return part.get('text', '')
        if ptype == 'input_image':
            image_url = part.get('image_url')
            if not image_url:
                return None
            # If it's a data URI, convert to BinaryContent; otherwise ImageUrl
            if isinstance(image_url, str) and image_url.startswith('data:'):
                try:
                    bc = BinaryContent.from_data_uri(image_url)
                    # carry detail if present
                    detail = part.get('detail')
                    if detail is not None:
                        bc.vendor_metadata = {**(bc.vendor_metadata or {}), 'detail': detail}
                    return bc
                except Exception:
                    return ImageUrl(url=str(image_url))
            else:
                vendor_metadata = {}
                if (detail := part.get('detail')) is not None:
                    vendor_metadata['detail'] = detail
                return ImageUrl(url=str(image_url), vendor_metadata=vendor_metadata or None)
        if ptype == 'input_file':
            data_uri = part.get('file_data')
            if isinstance(data_uri, str) and data_uri.startswith('data:'):
                try:
                    return BinaryContent.from_data_uri(data_uri)
                except Exception:
                    return None
            # If file_data missing or not a data URI, ignore for now
            return None
        # Fallback to legacy parser
        return _parse_user_content_part(part)

    history: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] = []

    # Handle instructions as system prompt
    if instructions:
        request_parts.append(SystemPromptPart(content=instructions))

    # Handle input - can be string or list
    if isinstance(input_data, str):
        # Simple string input becomes a user message
        request_parts.append(UserPromptPart(content=input_data))
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type')

            # Easy message param may omit type
            if item_type == 'message' or (item_type is None and 'role' in item and 'content' in item):
                role = item.get('role', 'user')
                content = item.get('content')
                # content can be str or list
                if role in ('system', 'developer'):
                    # System/developer both map to system prompt
                    if isinstance(content, str):
                        request_parts.append(SystemPromptPart(content=content))
                    elif isinstance(content, list):
                        # Concatenate text parts into one system string; ignore non-text
                        texts: list[str] = []
                        for c in content:
                            if isinstance(c, dict) and c.get('type') in ('input_text', 'text'):
                                txt = c.get('text') or c.get('content') or ''
                                if isinstance(txt, str):
                                    texts.append(txt)
                        if texts:
                            request_parts.append(SystemPromptPart(content=''.join(texts)))
                elif role == 'user':
                    if isinstance(content, str):
                        request_parts.append(UserPromptPart(content=content))
                    elif isinstance(content, list):
                        parts: list[UserContent] = []
                        for c in content:
                            if isinstance(c, dict):
                                parsed = _parse_response_user_content_part(c)
                                if parsed is not None:
                                    parts.append(parsed)
                        if parts:
                            request_parts.append(UserPromptPart(content=parts))
                elif role == 'assistant':
                    # Finish any accumulated request parts first
                    if request_parts:
                        history.append(ModelRequest(parts=request_parts))
                        request_parts = []
                    # Assistant content to ModelResponse
                    response_parts: list[TextPart] = []
                    if isinstance(content, str):
                        response_parts.append(TextPart(content=content))
                    elif isinstance(content, list):
                        txt = ''.join(
                            str(c.get('text', ''))
                            for c in content
                            if isinstance(c, dict) and c.get('type') in ('output_text', 'input_text', 'text')
                        )
                        if txt:
                            response_parts.append(TextPart(content=txt))
                    if response_parts:
                        history.append(ModelResponse(parts=response_parts))

            elif item_type == 'function_call_output':
                call_id = item.get('call_id', '')
                output = item.get('output', '')
                request_parts.append(ToolReturnPart(tool_call_id=call_id, content=str(output)))
            else:
                # Unknown type: ignore
                pass

    # Add any remaining request parts
    if request_parts:
        history.append(ModelRequest(parts=request_parts))

    return history


def _to_openai_chat_completion(run: AgentRunResult[OutputDataT], model: str) -> ChatCompletion:
    """Converts a Pydantic AI agent run result to an OpenAI ChatCompletion object.

    This function transforms the result of a Pydantic AI agent run into an OpenAI-compatible
    ChatCompletion response format, including message content, tool calls, usage statistics,
    and appropriate finish reasons.

    Args:
        run: The completed agent run result containing the conversation and response data.
        model: The model name to include in the response metadata.

    Returns:
        An OpenAI ChatCompletion object with the agent's response formatted according to
        the OpenAI API specification.

    Example:
        ```python
        completion = _to_openai_chat_completion(agent_run, "gpt-4o")
        print(completion.choices[0].message.content)
        ```
    """
    last_response = next((m for m in reversed(run.all_messages()) if isinstance(m, ModelResponse)), None)

    content_parts: list[str] = []
    tool_calls = []
    if last_response:
        for part in last_response.parts:
            if isinstance(part, TextPart):
                content_parts.append(part.content)
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        id=part.tool_call_id,
                        function=Function(name=part.tool_name, arguments=json.dumps(part.args)),
                        type='function',
                    )
                )

    content = ''.join(content_parts) if content_parts else None
    finish_reason: Literal['tool_calls', 'stop'] = 'tool_calls' if tool_calls else 'stop'

    run_usage = run.usage()
    completion_usage = CompletionUsage(
        completion_tokens=run_usage.output_tokens,
        prompt_tokens=run_usage.input_tokens,
        total_tokens=run_usage.total_tokens,
    )

    return ChatCompletion(
        id=str(uuid.uuid4()),
        choices=[
            Choice(
                finish_reason=finish_reason,
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=content,
                    role='assistant',
                    tool_calls=tool_calls if tool_calls else None,
                ),
            )
        ],
        created=int(time.time()),
        model=model,
        object='chat.completion',
        usage=completion_usage,
    )


def _to_openai_response(
        run: AgentRunResult[OutputDataT], model: str, input_data: Any, instructions: str | None = None
) -> ResponseObject:
    """Converts a Pydantic AI agent run result to an OpenAI Response object.

    This function transforms the result of a Pydantic AI agent run into a Responses API-compatible
    Response format. Unlike ChatCompletion which uses `choices`, Response uses an `output` list
    containing messages and other items.

    Args:
        run: The completed agent run result containing the conversation and response data.
        model: The model name to include in the response metadata.
        input_data: The original input data from the request.
        instructions: Optional instructions from the request.

    Returns:
        An OpenAI Response object with the agent's response formatted according to
        the Responses API specification.

    Example:
        ```python
        response = _to_openai_response(agent_run, "gpt-4o", "Hello", None)
        print(response.output[0])
        ```
    """
    from openai.types.responses import ResponseFunctionToolCall

    last_response = next((m for m in reversed(run.all_messages()) if isinstance(m, ModelResponse)), None)

    output: list[Any] = []

    if last_response:
        # Build output message with text and tool calls
        message_content: list[ResponseOutputText] = []

        for part in last_response.parts:
            if isinstance(part, TextPart):
                message_content.append(
                    ResponseOutputText(
                        type='output_text',
                        text=part.content,
                        annotations=[],
                    )
                )
            elif isinstance(part, ToolCallPart):
                # Tool calls are separate items in the output list
                output.append(
                    ResponseFunctionToolCall(
                        type='function_tool_call',
                        call_id=part.tool_call_id,
                        name=part.tool_name,
                        arguments=part.args_as_json_str(),
                    )
                )

        # Add message to output if there's any text content
        if message_content:
            output.insert(
                0,
                ResponseOutputMessage(
                    type='message',
                    id=str(uuid.uuid4()),
                    role='assistant',
                    status='completed',
                    content=message_content,
                ),
            )

    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

    run_usage = run.usage()
    response_usage = ResponseUsage(
        input_tokens=run_usage.input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=run_usage.output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=run_usage.total_tokens,
    )

    return ResponseObject(
        id=str(uuid.uuid4()),
        object='response',
        created_at=time.time(),
        model=model,
        output=output,
        status='completed',
        usage=response_usage,
        instructions=instructions,
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )


def _to_openai_response_stream_event(
        event: ModelResponseStreamEvent,
        model: str,
        response_id: str,
        context: _ResponsesStreamContext,
) -> Any | None:
    r"""Converts a Pydantic AI agent stream event to an OpenAI Response API stream event.

    Args:
        event: Agent stream event (PartStartEvent or PartDeltaEvent)
        model: The model name to include in event metadata.
        response_id: Unique identifier for the streaming response.
        context: Stream context for maintaining state across events.

    Returns:
        An OpenAI ResponseStreamEvent object if the event contains streamable content,
        or None if the event should be skipped in the stream.

    Example:
        ```python
        context = _ResponseStreamContext()
        stream_event = _to_openai_response_stream_event(event, "gpt-4o", response_id, context)
        if stream_event:
            yield f'event: {stream_event.type}\ndata: {stream_event.model_dump_json()}\n\n'
        ```
    """
    from openai.types.responses import (
        ResponseContentPartAddedEvent,
        ResponseOutputItemAddedEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
    )

    # Initialize item_id if not set
    if not context.item_id:
        context.item_id = str(uuid.uuid4())

    events: list[Any] = []

    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart):
            # When we start a text part, we need to:
            # 1. Add output item (message) if not already added
            # 2. Add content part (output_text)

            if not context.message_added:
                # Add the message output item
                message_item = ResponseOutputMessage(
                    type='message',
                    id=context.item_id,
                    role='assistant',
                    status='in_progress',
                    content=[],
                )
                events.append(
                    ResponseOutputItemAddedEvent(
                        type='response.output_item.added',
                        item=message_item,
                        output_index=context.output_index,
                        sequence_number=context.sequence_number,
                    )
                )
                context.sequence_number += 1
                context.message_added = True

            # Add the content part
            if not context.content_part_added:
                content_part = ResponseOutputText(
                    type='output_text',
                    text='',
                    annotations=[],
                )
                events.append(
                    ResponseContentPartAddedEvent(
                        type='response.content_part.added',
                        item_id=context.item_id,
                        output_index=context.output_index,
                        content_index=context.content_index,
                        part=content_part,
                        sequence_number=context.sequence_number,
                    )
                )
                context.sequence_number += 1
                context.content_part_added = True

    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            # Send text delta
            events.append(
                ResponseTextDeltaEvent(
                    type='response.output_text.delta',
                    item_id=context.item_id,
                    output_index=context.output_index,
                    content_index=context.content_index,
                    delta=event.delta.content_delta,
                    logprobs=[],
                    sequence_number=context.sequence_number,
                )
            )
            context.sequence_number += 1

    # Return the first event if any, or None
    # For simplicity, we return one event at a time
    return events[0] if events else None


def _to_openai_chat_completion_chunk(
        event: ModelResponseStreamEvent, model: str, run_id: str, context: _ChatCompletionStreamContext
) -> ChatCompletionChunk | None:
    r"""Converts a Pydantic AI agent stream event to an OpenAI ChatCompletionChunk object.

    Args:
        event: Agent stream event (PartStartEvent or PartDeltaEvent)
        model: The model name to include in the chunk metadata.
        run_id: Unique identifier for the streaming run.
        context: Stream context for maintaining state across chunks.

    Returns:
        An OpenAI ChatCompletionChunk object if the event contains streamable content,
        or None if the event should be skipped in the stream.

    Example:
        ```python
        context = _StreamContext()
        chunk = _to_openai_chat_completion_chunk(event, "gpt-4o", run_id, context)
        if chunk:
            yield f'data: {chunk.model_dump_json()}\n\n'
        ```
    """
    delta = ChoiceDelta()

    if isinstance(event, PartStartEvent):
        if isinstance(event.part, ToolCallPart):
            context.tool_call_part_started = event.part
            context.got_tool_calls = True

    elif isinstance(event, PartDeltaEvent):
        if not context.role_sent:
            delta.role = 'assistant'
            context.role_sent = True

        if isinstance(event.delta, TextPartDelta):
            delta.content = event.delta.content_delta
        elif isinstance(event.delta, ToolCallPartDelta):
            if context.tool_call_part_started:
                # First delta for a new tool call
                delta.tool_calls = [
                    ChoiceDelta.ToolCall(
                        index=context.tool_call_index,
                        id=context.tool_call_part_started.tool_call_id,
                        type='function',
                        function=ChoiceDelta.ToolCall.Function(
                            name=context.tool_call_part_started.tool_name,
                            arguments=event.delta.args_delta
                            if isinstance(event.delta.args_delta, str)
                            else json.dumps(event.delta.args_delta),
                        ),
                    )
                ]
                context.tool_call_part_started = None  # Consume it
            else:
                # Subsequent delta for the same tool call
                delta.tool_calls = [
                    ChoiceDelta.ToolCall(
                        index=context.tool_call_index,
                        function=ChoiceDelta.ToolCall.Function(
                            arguments=event.delta.args_delta
                            if isinstance(event.delta.args_delta, str)
                            else json.dumps(event.delta.args_delta)
                        ),
                    )
                ]

    if not delta.role and not delta.content and not delta.tool_calls:
        return None

    return ChatCompletionChunk(
        id=run_id,
        choices=[ChunkChoice(delta=delta, index=0, finish_reason=None, logprobs=None)],
        created=int(time.time()),
        model=model,
        object='chat.completion.chunk',
    )


def _is_streaming_chat_completions_request(value: CompletionCreateParamsT) -> TypeGuard[CompletionCreateParamsStreamingT]:
    return value["stream"]


async def handle_chat_completions_request(
        agent: AbstractAgent[AgentDepsT, Any],
        request: Request,
        *,
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
) -> Response:
    """Handles OpenAI Chat Completions API requests.

    This function processes HTTP requests to the `/v1/chat/completions` endpoint,
    providing OpenAI API compatibility for Pydantic AI agents. It supports both
    streaming and non-streaming responses, tool calls, and multimodal inputs.

    Args:
        agent: The Pydantic AI agent to run for generating responses.
        request: The HTTP request containing the chat completion parameters.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.

    Returns:
        A Starlette Response object containing either:
        - A StreamingResponse with server-sent events for streaming requests
        - A JSON Response with the complete chat completion for non-streaming requests
        - An error response for invalid requests

    Example:
        ```python
        response = await handle_chat_completions_request(agent, request)
        ```
    """
    try:
        params: CompletionCreateParamsT = TypeAdapter(CompletionCreateParamsT).validate_python(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=e.json(),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    messages = ChatCompletionsAdapter.load_messages(params["messages"])
    agent_kwargs = dict(
        output_type=output_type,
        model=model or params.get('model'),
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
    )

    if _is_streaming_chat_completions_request(params):
        run_id = str(uuid.uuid4())
        context = _ChatCompletionStreamContext()

        async def stream_generator() -> AsyncIterator[str]:
            from pydantic_graph import End

            from ._agent_graph import ModelRequestNode

            try:
                async with agent.iter(message_history=messages, **agent_kwargs) as run:
                    async for node in run:
                        if isinstance(node, End):
                            finish_reason: Literal['tool_calls', 'stop'] = (
                                'tool_calls' if context.got_tool_calls else 'stop'
                            )
                            final_chunk = ChatCompletionChunk(
                                id=run_id,
                                choices=[
                                    ChunkChoice(
                                        delta=ChoiceDelta(), index=0, finish_reason=finish_reason, logprobs=None
                                    )
                                ],
                                created=int(time.time()),
                                model=params["model"],
                                object='chat.completion.chunk',
                            )
                            yield f'data: {final_chunk.model_dump_json(exclude_unset=True)}\n\n'
                            yield 'data: [DONE]\n\n'
                        elif isinstance(node, ModelRequestNode):
                            async with node.stream(run.ctx) as request_stream:
                                async for agent_event in request_stream:
                                    chunk = _to_openai_chat_completion_chunk(
                                        agent_event, params.get('model'), run_id, context
                                    )
                                    if chunk:
                                        yield f'data: {chunk.model_dump_json(exclude_unset=True)}\n\n'
            except Exception as exc:  # pragma: no cover
                status_code, payload = _map_exception_to_openai_error(exc)
                # Emit error payload in stream as a data line, then DONE
                yield f'data: {json.dumps(payload)}\n\n'
                yield 'data: [DONE]\n\n'

        return StreamingResponse(
            stream_generator(),
            media_type='text/event-stream',
        )
    else:
        try:
            run_result = await agent.run(message_history=messages, **agent_kwargs)
            completion = _to_openai_chat_completion(run_result, model=params.get('model'))
            return Response(
                content=completion.model_dump_json(exclude_unset=True),
                media_type='application/json',
            )
        except Exception as exc:
            return _openai_error_response(exc)


async def handle_responses_request(
        agent: AbstractAgent[AgentDepsT, Any],
        request: Request,
        *,
        output_type: OutputSpec[Any] | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
) -> Response:
    """Handles OpenAI Responses API requests.

    This function processes HTTP requests to the `/v1/responses` endpoint,
    providing OpenAI Responses API compatibility for Pydantic AI agents. Unlike
    Chat Completions, the Responses API uses different request/response formats
    with `input` instead of `messages` and `output` instead of `choices`.

    Args:
        agent: The Pydantic AI agent to run for generating responses.
        request: The HTTP request containing the response parameters.

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has
            no output validators since output validators would expect an argument that matches the agent's
            output type.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.

    Returns:
        A Starlette Response object containing either:
        - A StreamingResponse with server-sent events for streaming requests (not yet implemented)
        - A JSON Response with the complete Response object for non-streaming requests
        - An error response for invalid requests

    Example:
        ```python
        response = await handle_responses_request(agent, request)
        ```
    """
    try:
        params: ResponseCreateParamsT = ResponseCreateParams.validate_python(await request.json())
    except ValidationError as e:  # pragma: no cover
        return Response(
            content=e.json(),
            media_type='application/json',
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    # Extract input and instructions from Responses API format
    input_data = params.get('input')
    instructions = params.get('instructions')

    # Convert Responses API input to internal message format
    messages = _from_responses_input(input_data, instructions)

    agent_kwargs = dict(
        output_type=output_type,
        model=model or params.get('model'),
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=infer_name,
        toolsets=toolsets,
    )

    if params.get('stream'):
        # Streaming support for Responses API
        from openai.types.responses import (
            Response as ResponseObject,
            ResponseCompletedEvent,
            ResponseCreatedEvent,
        )
        from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

        response_id = str(uuid.uuid4())
        context = _ResponsesStreamContext()

        async def stream_generator() -> AsyncIterator[str]:
            from pydantic_graph import End

            from ._agent_graph import ModelRequestNode

            # Send initial response.created event
            initial_response = ResponseObject(
                id=response_id,
                object='response',
                created_at=time.time(),
                model=params.get('model'),
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
                sequence_number=context.sequence_number,
            )
            context.sequence_number += 1
            yield f'event: {created_event.type}\ndata: {created_event.model_dump_json(exclude_unset=True)}\n\n'

            # Stream the agent's response
            try:
                async with agent.iter(message_history=messages, **agent_kwargs) as run:
                    async for node in run:
                        if isinstance(node, End):
                            # Send final response.completed event
                            run_usage = run.usage()
                            final_response = ResponseObject(
                                id=response_id,
                                object='response',
                                created_at=initial_response.created_at,
                                model=params.get('model'),
                                output=[],
                                status='completed',
                                usage=ResponseUsage(
                                    input_tokens=run_usage.input_tokens,
                                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                                    output_tokens=run_usage.output_tokens,
                                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                                    total_tokens=run_usage.total_tokens,
                                ),
                                instructions=instructions,
                                parallel_tool_calls=True,
                                tool_choice='auto',
                                tools=[],
                            )
                            completed_event = ResponseCompletedEvent(
                                type='response.completed',
                                response=final_response,
                                sequence_number=context.sequence_number,
                            )
                            yield f'event: {completed_event.type}\ndata: {completed_event.model_dump_json(exclude_unset=True)}\n\n'
                            yield 'event: done\ndata: [DONE]\n\n'
                        elif isinstance(node, ModelRequestNode):
                            async with node.stream(run.ctx) as request_stream:
                                async for agent_event in request_stream:
                                    stream_event = _to_openai_response_stream_event(
                                        agent_event, params.get('model'), response_id, context
                                    )
                                    if stream_event:
                                        yield f'event: {stream_event.type}\ndata: {stream_event.model_dump_json(exclude_unset=True)}\n\n'
            except Exception as exc:  # pragma: no cover
                # Emit an error payload as a data line, then done
                _, payload = _map_exception_to_openai_error(exc)
                yield f'data: {json.dumps(payload)}\n\n'
                yield 'event: done\ndata: [DONE]\n\n'

        return StreamingResponse(
            stream_generator(),
            media_type='text/event-stream',
        )
    else:
        # Run the agent and convert to Response format
        try:
            run_result = await agent.run(message_history=messages, **agent_kwargs)
            response_obj = _to_openai_response(
                run_result, model=params.get('model'), input_data=input_data, instructions=instructions
            )

            return Response(
                content=response_obj.model_dump_json(exclude_unset=True),
                media_type='application/json',
            )
        except Exception as exc:
            return _openai_error_response(exc)
