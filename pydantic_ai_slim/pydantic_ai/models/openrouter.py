"""OpenRouter model implementation for pydantic-ai.

This module provides native OpenRouter support that handles OpenRouter-specific
response formats and API quirks without relying on monkey-patching the OpenAI model.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Literal, cast, overload
from urllib.parse import urlparse, urlunparse

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from pydantic_ai import UnexpectedModelBehavior, usage
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import PeekableAsyncStream, Unset, guard_tool_call_id as _guard_tool_call_id
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
    Model,
    ModelRequestParameters,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

try:
    from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
    from openai.types import ChatModel, chat
    from openai.types.chat import ChatCompletionChunk

except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenRouter model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenRouterModelName = ChatModel | str
"""
Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with various OpenRouter model types.
"""


def format_openrouter_url(raw: str) -> str:
    """Format OpenRouter URL to ensure proper API endpoint structure."""
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


__all__ = ['OpenRouterModel', 'format_openrouter_url']


@dataclass(init=False)
class OpenRouterModel(Model):
    """A model that uses the OpenRouter API with proper compatibility handling.

    This model handles OpenRouter-specific response formats and API quirks
    without relying on monkey-patching the OpenAI model.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)
    _model_name: OpenRouterModelName = field(repr=False)
    _system: str = field(default='openrouter', repr=False)

    def __init__(
        self,
        model_name: OpenRouterModelName,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an OpenRouter model.

        Args:
            model_name: The name of the OpenRouter model to use.
            base_url: The base url for OpenRouter requests. Defaults to 'https://openrouter.ai/api/v1'.
            api_key: The API key to use for authentication, if not provided, the `OPENROUTER_API_KEY`
                environment variable will be used if available.
            openai_client: An existing AsyncOpenAI client to use. If provided, `base_url`, `api_key`,
                and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self._model_name = model_name

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self.client = openai_client
        else:
            if base_url is None:
                base_url = 'https://openrouter.ai/api/v1'
            else:
                base_url = format_openrouter_url(base_url)

            if api_key is None:
                api_key = os.environ.get('OPENROUTER_API_KEY')

            if http_client is not None:
                self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
            else:
                self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=cached_async_http_client())

        super().__init__()

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(messages, False, model_settings, model_request_parameters)
        model_response = self._process_response(response)
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(messages, True, model_settings, model_request_parameters)
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openrouter_messages = list(chain(*(self._map_message(m) for m in messages)))

        model_settings = model_settings or {}

        reasoning_param = self._build_reasoning_param(model_settings)

        raw_extra_body: object | None = model_settings.get('extra_body')

        if isinstance(raw_extra_body, Mapping):
            typed_mapping = cast(Mapping[str, Any], raw_extra_body)
            extra_body: dict[str, Any] = dict(typed_mapping)
        else:
            extra_body = {}
        if reasoning_param:
            extra_body['reasoning'] = reasoning_param

        return await self.client.chat.completions.create(
            model=self._model_name,
            messages=openrouter_messages,
            n=1,
            parallel_tool_calls=True if tools else NOT_GIVEN,
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={'include_usage': True} if stream else NOT_GIVEN,
            max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
            extra_body=extra_body or None,
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        """Get tools from model request parameters."""
        tools: list[chat.ChatCompletionToolParam] = []
        for tool_def in model_request_parameters.function_tools:
            tools.append(self._map_tool_definition(tool_def))
        for tool_def in model_request_parameters.output_tools:
            tools.append(self._map_tool_definition(tool_def))
        return tools

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }

    def _build_reasoning_param(self, model_settings: ModelSettings) -> dict[str, Any] | None:
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

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []

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
                    raise NotImplementedError(f'Tool call type {c.type} not supported')

        return ModelResponse(items, timestamp=timestamp, model_name=self._model_name, usage=_map_usage(response))

    async def _process_streamed_response(
        self, response: AsyncStream[ChatCompletionChunk], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response with OpenRouter compatibility handling."""
        peekable_response = PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return OpenRouterStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
        """Maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if isinstance(message, ModelRequest):
            yield from cls._map_user_message(message)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(_map_tool_call(item))
                elif isinstance(item, ThinkingPart):
                    pass
                elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):
                    pass
                else:
                    assert_never(item)
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            if texts:
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param['content'] = '\n\n'.join(texts)
            if tool_calls:  # pragma: no branch
                message_param['tool_calls'] = tool_calls
            yield message_param
        else:
            assert_never(message)

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        """Map a UserPromptPart to a ChatCompletionUserMessageParam."""
        if isinstance(part.content, str):
            return chat.ChatCompletionUserMessageParam(role='user', content=part.content)
        else:
            content_parts: list[str] = []
            for item in part.content:
                if isinstance(item, str):
                    content_parts.append(item)
            return chat.ChatCompletionUserMessageParam(role='user', content=' '.join(content_parts))

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part),
                        content=part.model_response(),
                    )
            else:
                assert_never(part)


@dataclass
class OpenRouterStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenRouter models."""

    _model_name: OpenRouterModelName
    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime

    @property
    def provider_name(self) -> str | None:
        """Get the provider name."""
        return 'openrouter'

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

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


def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
        id=_guard_tool_call_id(t=t),
        type='function',
        function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
    )


def _map_usage(response: chat.ChatCompletion | ChatCompletionChunk) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()
    else:
        details: dict[str, int] = {}
        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))
        if response_usage.prompt_tokens_details is not None:
            details.update(response_usage.prompt_tokens_details.model_dump(exclude_none=True))
        return usage.RequestUsage(
            input_tokens=response_usage.prompt_tokens,
            output_tokens=response_usage.completion_tokens,
            details=details,
        )
