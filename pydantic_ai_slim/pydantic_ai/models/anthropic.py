from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, overload

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, result
from ..messages import (
    ArgsJson,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolCall,
    ToolReturn,
)
from ..result import Cost
from ..tools import ToolDefinition
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.types import (
        ContentBlock,
        Message as AnthropicMessage,
        MessageParam,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStreamEvent,
        ToolChoiceParam,
        ToolParam,
        ToolUseBlock,
        ToolUseBlockParam,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        "you can use the `anthropic` optional group — `pip install 'pydantic-ai[anthropic]'`"
    ) from _import_error

# TODO: should we do the named anthropic model thing here, or only support models that aren't deprecated?
AnthropicModelName = Literal[
    'claude-3-5-haiku-latest',
    'claude-3-5-haiku-20241022',
    'claude-3-5-sonnet-latest',
    'claude-3-5-sonnet-20241022',
    'claude-3-5-sonnet-20240620',
    'claude-3-opus-latest',
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-2.1',
    'claude-2.0',
]
"""Named Anthropic models.

See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: AnthropicModelName
    client: AsyncAnthropic = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        api_key: str | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            api_key: The API key to use for authentication, if not provided, the `ANTHROPIC_API_KEY` environment variable
                will be used if available.
            anthropic_client: An existing
                [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name
        if anthropic_client is not None:
            assert http_client is None, 'Cannot provide both `anthropic_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `anthropic_client` and `api_key`'
            self.client = anthropic_client
        elif http_client is not None:
            self.client = AsyncAnthropic(api_key=api_key, http_client=http_client)
        else:
            self.client = AsyncAnthropic(api_key=api_key, http_client=cached_async_http_client())

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
        return AnthropicAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return self.model_name

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolParam:
        return {
            'name': f.name,
            'description': f.description,
            'input_schema': f.parameters_json_schema,
        }


@dataclass
class AnthropicAgentModel(AgentModel):
    """Implementation of `AgentModel` for Anthropic models."""

    client: AsyncAnthropic
    model_name: str
    allow_text_result: bool
    tools: list[ToolParam]

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._messages_create(messages, False)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._messages_create(messages, True)
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _messages_create(
        self, messages: list[Message], stream: Literal[True]
    ) -> AsyncStream[RawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(self, messages: list[Message], stream: Literal[False]) -> AnthropicMessage:
        pass

    async def _messages_create(
        self, messages: list[Message], stream: bool
    ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: ToolChoiceParam | None = None
        elif not self.allow_text_result:
            tool_choice = {'type': 'any'}
        else:
            tool_choice = {'type': 'auto'}

        anthropic_messages = [self._map_message(m) for m in messages]
        return await self.client.messages.create(
            # TODO: might want to change max tokens (or make configurable for user), and same with other models...
            max_tokens=1024,
            messages=anthropic_messages,
            model=self.model_name,
            temperature=0.0,
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
        )

    @staticmethod
    def _process_response(response: AnthropicMessage) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        choice: ContentBlock = response.content[0]
        if isinstance(choice, ToolUseBlock):
            # TODO: fix this - different structure than groq and others
            return ModelStructuredResponse(
                [ToolCall.from_json(c.function.name, c.function.arguments, c.id) for c in choice.message.tool_calls],
            )
        else:
            assert choice.text is not None, choice
            return ModelTextResponse(choice.text)

    @staticmethod
    async def _process_streamed_response(response: AsyncStream[RawMessageStreamEvent]) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        timestamp: datetime | None = None
        start_cost = Cost()
        # the first chunk may contain enough information so we iterate until we get either `tool_calls` or `content`
        while True:
            try:
                chunk = await response.__anext__()
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e
            timestamp = timestamp or _utils.now_utc()
            start_cost += _map_cost(chunk)

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta.content is not None:
                    return AnthropicStreamTextResponse(delta.content, response, timestamp, start_cost)
                elif delta.tool_calls is not None:
                    return AnthropicStreamStructuredResponse(
                        response,
                        {c.index: c for c in delta.tool_calls},
                        timestamp,
                        start_cost,
                    )

    @staticmethod
    def _map_message(message: Message) -> MessageParam:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        # TODO: MessageParam only has 2 roles on the Anthropic end of things?
        if message.role == 'system':
            return MessageParam(role='assistant', content=message.content)
        elif message.role == 'user':
            return MessageParam(role='user', content=message.content)
        elif message.role == 'tool-return':
            return MessageParam(role='assistant', content=message.content)
        elif message.role == 'retry-prompt':
            return MessageParam(role='user', content=message.model_response())
        elif message.role == 'model-text-response':
            return MessageParam(role='assistant', content=message.content)
        elif message.role == 'model-structured-response':
            return MessageParam(role='assistant', content=[_map_tool_call(t) for t in message.calls])
        else:
            assert_never(message)


@dataclass
class AnthropicStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for Anthropic models."""

    _first: str | None
    _response: AsyncStream[RawMessageStreamEvent]
    _timestamp: datetime
    _cost: result.Cost
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._first is not None:
            self._buffer.append(self._first)
            self._first = None
            return None

        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk)

        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        # we don't raise StopAsyncIteration on the last chunk because usage comes after this
        if choice.finish_reason is None:
            assert choice.delta.content is not None, f'Expected delta with content, invalid chunk: {chunk!r}'
        if choice.delta.content is not None:
            self._buffer.append(choice.delta.content)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class AnthropicStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for Anthropic models."""

    _response: AsyncStream[RawMessageStreamEvent]
    _delta_tool_calls: dict[int, ChoiceDeltaToolCall]
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk)

        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

        assert choice.delta.content is None, f'Expected tool calls, got content instead, invalid chunk: {chunk!r}'

        for new in choice.delta.tool_calls or []:
            if current := self._delta_tool_calls.get(new.index):
                if current.function is None:
                    current.function = new.function
                elif new.function is not None:
                    current.function.name = _utils.add_optional(current.function.name, new.function.name)
                    current.function.arguments = _utils.add_optional(current.function.arguments, new.function.arguments)
            else:
                self._delta_tool_calls[new.index] = new

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []
        for c in self._delta_tool_calls.values():
            if f := c.function:
                if f.name is not None and f.arguments is not None:
                    calls.append(ToolCall.from_json(f.name, f.arguments, c.id))

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


def _guard_tool_id(t: ToolCall | ToolReturn | RetryPrompt) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f'Anthropic requires `tool_id` to be set: {t}'
    return t.tool_id


def _map_tool_call(t: ToolCall) -> ToolUseBlockParam:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsJson, got {t.args}'
    return ToolUseBlockParam(
        id=_guard_tool_id(t),
        type='tool_use',
        name=t.tool_name,
        # TODO: input shouldn't be none here?
        input=None,
    )


def _map_cost(message: AnthropicMessage | RawMessageStreamEvent) -> result.Cost:
    usage = None
    if isinstance(message, AnthropicMessage):
        usage = message.usage
    else:
        # TODO: I'm not sure if this usage counting is correct...
        if isinstance(message, RawMessageDeltaEvent):
            usage = message.usage
        elif isinstance(message, RawMessageStartEvent):
            usage = message.message.usage

    if usage is None:
        return result.Cost()

    return result.Cost(
        request_tokens=getattr(usage, 'input_tokens', None),
        response_tokens=usage.output_tokens,
    )
