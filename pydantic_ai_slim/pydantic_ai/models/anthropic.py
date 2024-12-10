from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union, cast, overload

from anthropic.types import ToolResultBlockParam
from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, result
from ..messages import (
    ArgsDict,
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
        Message as AnthropicMessage,
        MessageParam,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
        RawMessageStreamEvent,
        TextBlock,
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

LatestAnthropicModelNames = Literal[
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
]
"""Latest named Anthropic models."""

AnthropicModelName = Union[str, LatestAnthropicModelNames]
"""Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
Since [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
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
        content = response.content
        first_response = content[0]
        if len(content) == 1 and isinstance(first_response, TextBlock):
            assert first_response.text is not None, first_response
            return ModelTextResponse(first_response.text)

        return ModelStructuredResponse(
            [
                ToolCall.from_dict(
                    c.name,
                    cast(dict[str, Any], c.input),
                    c.id,
                )
                # TODO: note, we're sort of ignoring any textual messages here, do we need to save them in some way?
                for c in [tub for tub in content if isinstance(tub, ToolUseBlock)]
            ],
        )

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

            # TODO: should be returning some sort of AnthropicStreamTextResponse or AnthropicStreamStructuredResponse
            # depending on the type of chunk we get
            if isinstance(chunk, RawMessageStartEvent):
                pass
            elif isinstance(chunk, RawMessageDeltaEvent):
                pass
            elif isinstance(chunk, RawMessageStopEvent):
                pass
            elif isinstance(chunk, RawContentBlockStartEvent):
                pass
            elif isinstance(chunk, RawContentBlockDeltaEvent):
                pass
            elif isinstance(chunk, RawContentBlockDeltaEvent):
                pass

    @staticmethod
    def _map_message(message: Message) -> MessageParam:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        # TODO: confirm the below roles are appropriate, make sure the roles are actually correct...
        if message.role == 'system':
            return MessageParam(role='user', content=message.content)
        elif message.role == 'user':
            return MessageParam(role='user', content=message.content)
        elif message.role == 'tool-return':
            return MessageParam(
                role='assistant',
                content=[
                    ToolResultBlockParam(
                        tool_use_id=_guard_tool_id(message),
                        type='tool_result',
                        content=message.model_response_str(),
                        is_error=False,
                    )
                ],
            )
        elif message.role == 'retry-prompt':
            if message.tool_name is None:
                return MessageParam(role='user', content=message.model_response())
            else:
                return MessageParam(
                    role='user',
                    content=[
                        ToolUseBlockParam(
                            id=_guard_tool_id(message),
                            input=message.model_response(),
                            name=message.tool_name,
                            type='tool_use',
                        ),
                    ],
                )
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

        # TODO: implement support here, need to figure out how to handle RawMessageStreamEvent chunks

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
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk)

        # TODO: implement support here, need to figure out how to handle RawMessageStreamEvent chunks

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []

        # TODO: implement support here, need to figure out how to handle RawMessageStreamEvent chunks

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
    assert isinstance(t.args, ArgsDict), f'Expected ArgsDict, got {t.args}'
    return ToolUseBlockParam(
        id=_guard_tool_id(t),
        type='tool_use',
        name=t.tool_name,
        input=t.args.args_dict,
    )


def _map_cost(message: AnthropicMessage | RawMessageStreamEvent) -> result.Cost:
    if isinstance(message, AnthropicMessage):
        usage = message.usage
    else:
        # TODO: I'm not sure if this usage counting is correct...
        # But I've extracted usage info from the various types of messages that can be sent.
        # Usage coming from the RawMessageDeltaEvent doesn't have input token data (see getattr below),
        # does this mean that we're double counting output token data from the previous RawMessageStartEvent?
        if isinstance(message, RawMessageStartEvent):
            usage = message.message.usage
        elif isinstance(message, RawMessageDeltaEvent):
            usage = message.usage
        else:
            # No usage information provided in:
            # - RawMessageStopEvent
            # - RawContentBlockStartEvent
            # - RawContentBlockDeltaEvent
            # - RawContentBlockStopEvent
            usage = None

    if usage is None:
        return result.Cost()

    return result.Cost(
        request_tokens=getattr(usage, 'input_tokens', None),
        response_tokens=usage.output_tokens,
    )
