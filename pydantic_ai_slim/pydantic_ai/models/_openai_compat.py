from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Any, cast

from typing_extensions import assert_never

from pydantic_ai import UnexpectedModelBehavior, usage
from pydantic_ai._utils import PeekableAsyncStream, Unset, guard_tool_call_id
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

try:
    from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
    from openai.types import ChatModel, chat
    from openai.types.chat import ChatCompletionChunk
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use OpenAI-compatible models, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenRouterModelName = ChatModel | str


def format_openrouter_url(raw: str) -> str:
    """Format OpenRouter URL to ensure proper API endpoint structure."""
    from urllib.parse import urlparse, urlunparse

    p = urlparse(raw, scheme='https')

    if not p.netloc:
        p = p._replace(netloc=p.path, path='')

    parts = [seg for seg in p.path.split('/') if seg]

    if not parts or parts[0] != 'api':
        parts.insert(0, 'api')
    if parts[-1] != 'v1':
        parts.append('v1')
    new_path = '/' + '/'.join(parts) + '/'
    return urlunparse(p._replace(path=new_path))


def build_reasoning_param(model_settings: ModelSettings) -> dict[str, Any] | None:
    """Build the reasoning parameter for OpenRouter API from model settings."""
    reasoning_config: dict[str, Any] = {}

    if 'openrouter_reasoning_effort' in model_settings:
        reasoning_config['effort'] = model_settings['openrouter_reasoning_effort']
    elif 'openrouter_reasoning_max_tokens' in model_settings:
        reasoning_config['max_tokens'] = model_settings['openrouter_reasoning_max_tokens']
    elif 'openrouter_reasoning_enabled' in model_settings:
        reasoning_config['enabled'] = model_settings['openrouter_reasoning_enabled']

    if 'openrouter_reasoning_exclude' in model_settings:
        reasoning_config['exclude'] = model_settings['openrouter_reasoning_exclude']

    return reasoning_config if reasoning_config else None


def map_tool_definition(tool_def: ToolDefinition) -> chat.ChatCompletionToolParam:
    """Map a ToolDefinition to OpenAI-compatible tool param."""
    tool_param = {
        'type': 'function',
        'function': {
            'name': tool_def.name,
            'description': tool_def.description or '',
            'parameters': tool_def.parameters_json_schema,
        },
    }
    if tool_def.strict:
        function = cast(dict[str, Any], tool_param['function'])
        function['strict'] = tool_def.strict
    return cast(chat.ChatCompletionToolParam, tool_param)


def get_tools(model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
    """Get tools from model request parameters (OpenAI-compatible)."""
    tools: list[chat.ChatCompletionToolParam] = []
    for tool_def in model_request_parameters.function_tools:
        tools.append(map_tool_definition(tool_def))
    for tool_def in model_request_parameters.output_tools:
        tools.append(map_tool_definition(tool_def))
    return tools


def map_message(message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:  # noqa: C901
    """Maps a ModelMessage to OpenAI-compatible ChatCompletionMessageParam (basic version)."""
    if isinstance(message, ModelRequest):
        # System and user prompts
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    content_parts: list[str] = [item for item in part.content if isinstance(item, str)]
                    yield chat.ChatCompletionUserMessageParam(role='user', content=' '.join(content_parts))
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)
    elif isinstance(message, ModelResponse):
        texts: list[str] = []
        tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
        for item in message.parts:
            if isinstance(item, TextPart):
                texts.append(item.content)
            elif isinstance(item, ToolCallPart):
                tool_calls.append(_map_tool_call(item))
            elif isinstance(item, ThinkingPart):
                pass  # Ignore thinking for basic mapping
            elif isinstance(item, BuiltinToolCallPart) or isinstance(item, BuiltinToolReturnPart):
                pass  # Ignore builtin parts for compat mapping
            else:
                assert_never(item)
        message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
        if texts:
            message_param['content'] = '\n\n'.join(texts)
        if tool_calls:
            message_param['tool_calls'] = tool_calls
        yield message_param
    else:
        assert_never(message)


def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    from pydantic_ai._utils import guard_tool_call_id as _guard_tool_call_id

    return chat.ChatCompletionMessageToolCallParam(
        id=_guard_tool_call_id(t=t),
        type='function',
        function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
    )


def map_messages(messages: list[ModelMessage]) -> list[chat.ChatCompletionMessageParam]:
    """Map list of ModelMessage to OpenAI-compatible messages (basic)."""
    return list(chain(*(map_message(m) for m in messages)))


async def completions_create(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: ModelSettings,
    model_request_parameters: ModelRequestParameters,
    extra_body: dict[str, Any] | None = None,
    base_url: str | None = None,
    **openai_kwargs: Any,
) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
    """Generic OpenAI-compatible completions create call."""
    tools = get_tools(model_request_parameters)

    openai_messages = map_messages(messages)

    model_settings = model_settings or {}

    raw_extra_body: object | None = model_settings.get('extra_body')
    if isinstance(raw_extra_body, Mapping):
        typed_mapping = cast(Mapping[str, Any], raw_extra_body)
        extra_body = dict(typed_mapping)
    else:
        extra_body = extra_body or {}

    return await client.chat.completions.create(
        model=model_name,
        messages=openai_messages,
        tools=tools or NOT_GIVEN,
        stream=stream,
        stream_options={'include_usage': True} if stream else NOT_GIVEN,
        extra_body=extra_body or None,
        **openai_kwargs,
    )


def process_response(
    response: chat.ChatCompletion,
    model_name: str,
    handle_reasoning: bool = True,
) -> ModelResponse:
    """Process non-streamed OpenAI-compatible response."""
    if response.created > 10000000000:
        timestamp = datetime.fromtimestamp(response.created / 1000, tz=timezone.utc)
    else:
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
    choice = response.choices[0]
    items: list[ModelResponsePart] = []

    if handle_reasoning:
        reasoning_content = getattr(choice.message, 'reasoning', None)
        if reasoning_content:
            items.append(ThinkingPart(content=reasoning_content))

    if choice.message.content is not None:
        items.append(TextPart(choice.message.content))

    if choice.message.tool_calls is not None:
        for c in choice.message.tool_calls:
            if c.type == 'function':
                items.append(ToolCallPart(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id))
            else:
                raise NotImplementedError(f'Tool call type {c.type} not supported in compat mode')

    return ModelResponse(items, timestamp=timestamp, model_name=model_name, usage=_map_usage(response))


async def process_streamed_response(
    response: AsyncStream[ChatCompletionChunk],
    model_request_parameters: ModelRequestParameters,
    model_name: str,
    timestamp: datetime,
    handle_reasoning: bool = True,
) -> tuple[OpenAICompatStreamedResponse, Any, usage.RequestUsage]:
    """Process streamed OpenAI-compatible response."""
    peekable_response = PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, Unset):
        raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

    streamed = OpenAICompatStreamedResponse(
        model_request_parameters=model_request_parameters,
        _model_name=model_name,
        _response=peekable_response,
        _timestamp=timestamp,
        _handle_reasoning=handle_reasoning,
    )
    return streamed, streamed._parts_manager, streamed._usage  # pyright: ignore[reportPrivateUsage]


@dataclass
class OpenAICompatStreamedResponse(StreamedResponse):
    """Generic streamed response for OpenAI-compatible models."""

    _model_name: str
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime
    _handle_reasoning: bool = True
    _provider_name: str = field(default='openai-compat', repr=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of ModelResponseStreamEvent objects."""
        async for chunk in self._response:
            self._usage += _map_usage(chunk)

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            content = choice.delta.content
            if content is not None:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=content,
                )
                if maybe_event is not None:
                    yield maybe_event

            if self._handle_reasoning:
                reasoning_content = getattr(choice.delta, 'reasoning', None)
                if reasoning_content is not None:
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id='reasoning',
                        content=reasoning_content,
                    )

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def provider_name(self) -> str:
        return self._provider_name


def _map_usage(response: chat.ChatCompletion | ChatCompletionChunk) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()
    details = {
        key: value
        for key, value in response_usage.model_dump(
            exclude_none=True,
            exclude={'prompt_tokens', 'completion_tokens', 'total_tokens'},
        ).items()
        if isinstance(value, int)
    }
    u = usage.RequestUsage(
        input_tokens=response_usage.prompt_tokens or 0,
        output_tokens=response_usage.completion_tokens or 0,
        details=details,
    )
    if response_usage.completion_tokens_details is not None:
        completion_details = response_usage.completion_tokens_details.model_dump(exclude_none=True)
        details.update({k: v for k, v in completion_details.items() if isinstance(v, int)})
        u.output_audio_tokens = response_usage.completion_tokens_details.audio_tokens or 0
    if response_usage.prompt_tokens_details is not None:
        u.input_audio_tokens = response_usage.prompt_tokens_details.audio_tokens or 0
        u.cache_read_tokens = response_usage.prompt_tokens_details.cached_tokens or 0
    return u
