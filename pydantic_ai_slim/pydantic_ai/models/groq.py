from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Literal, overload

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, usage
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    AgentModel,
    Model,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from groq import NOT_GIVEN, AsyncGroq, AsyncStream
    from groq.types import chat
    from groq.types.chat import ChatCompletion, ChatCompletionChunk
except ImportError as _import_error:
    raise ImportError(
        'Please install `groq` to use the Groq model, '
        "you can use the `groq` optional group — `pip install 'pydantic-ai-slim[groq]'`"
    ) from _import_error

GroqModelName = Literal[
    'llama-3.3-70b-versatile',
    'llama-3.1-70b-versatile',
    'llama3-groq-70b-8192-tool-use-preview',
    'llama3-groq-8b-8192-tool-use-preview',
    'llama-3.1-70b-specdec',
    'llama-3.1-8b-instant',
    'llama-3.2-1b-preview',
    'llama-3.2-3b-preview',
    'llama-3.2-11b-vision-preview',
    'llama-3.2-90b-vision-preview',
    'llama3-70b-8192',
    'llama3-8b-8192',
    'mixtral-8x7b-32768',
    'gemma2-9b-it',
    'gemma-7b-it',
]
"""Named Groq models.

See [the Groq docs](https://console.groq.com/docs/models) for a full list.
"""


@dataclass(init=False)
class GroqModel(Model):
    """A model that uses the Groq API.

    Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: GroqModelName
    client: AsyncGroq = field(repr=False)

    def __init__(
        self,
        model_name: GroqModelName,
        *,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize a Groq model.

        Args:
            model_name: The name of the Groq model to use. List of model names available
                [here](https://console.groq.com/docs/models).
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self.client = groq_client
        elif http_client is not None:
            self.client = AsyncGroq(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncGroq(api_key=api_key, http_client=cached_async_http_client())

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return GroqAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'groq:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
        }


@dataclass
class GroqAgentModel(AgentModel):
    """Implementation of `AgentModel` for Groq models."""

    client: AsyncGroq
    model_name: str
    allow_text_result: bool
    tools: list[chat.ChatCompletionToolParam]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, usage.Usage]:
        response = await self._completions_create(messages, False, model_settings)
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        response = await self._completions_create(messages, True, model_settings)
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[True], model_settings: ModelSettings | None
    ) -> AsyncStream[ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self, messages: list[ModelMessage], stream: Literal[False], model_settings: ModelSettings | None
    ) -> chat.ChatCompletion:
        pass

    async def _completions_create(
        self, messages: list[ModelMessage], stream: bool, model_settings: ModelSettings | None
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        groq_messages = list(chain(*(self._map_message(m) for m in messages)))

        model_settings = model_settings or {}

        return await self.client.chat.completions.create(
            model=str(self.model_name),
            messages=groq_messages,
            n=1,
            parallel_tool_calls=model_settings.get('parallel_tool_calls', True if self.tools else NOT_GIVEN),
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
        )

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.content is not None:
            items.append(TextPart(content=choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(
                    ToolCallPart.from_raw_args(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id)
                )
        return ModelResponse(items, model_name=self.model_name, timestamp=timestamp)

    async def _process_streamed_response(self, response: AsyncStream[ChatCompletionChunk]) -> GroqStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return GroqStreamedResponse(
            _response=peekable_response,
            _model_name=self.model_name,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `groq.types.ChatCompletionMessageParam`."""
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
                else:
                    assert_never(item)
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            if texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param['content'] = '\n\n'.join(texts)
            if tool_calls:
                message_param['tool_calls'] = tool_calls
            yield message_param
        else:
            assert_never(message)

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
            elif isinstance(part, ToolReturnPart):
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part, model_source='Groq'),
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
                else:
                    yield chat.ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=_guard_tool_call_id(t=part, model_source='Groq'),
                        content=part.model_response(),
                    )


@dataclass
class GroqStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Groq models."""

    _response: AsyncIterable[ChatCompletionChunk]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _map_usage(chunk)

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # Handle the text part of the response
            content = choice.delta.content
            if content is not None:
                yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=content)

            # Handle the tool calls
            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    def timestamp(self) -> datetime:
        return self._timestamp


def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
        id=_guard_tool_call_id(t=t, model_source='Groq'),
        type='function',
        function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
    )


def _map_usage(completion: ChatCompletionChunk | ChatCompletion) -> usage.Usage:
    response_usage = None
    if isinstance(completion, ChatCompletion):
        response_usage = completion.usage
    elif completion.x_groq is not None:
        response_usage = completion.x_groq.usage

    if response_usage is None:
        return usage.Usage()

    return usage.Usage(
        request_tokens=response_usage.prompt_tokens,
        response_tokens=response_usage.completion_tokens,
        total_tokens=response_usage.total_tokens,
    )
