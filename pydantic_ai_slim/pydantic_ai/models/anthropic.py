from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError, loads as json_loads
from typing import Any, Literal, Union, cast, overload

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, usage
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    ArgsDict,
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
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.types import (
        Message as AnthropicMessage,
        MessageParam,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
        RawMessageStreamEvent,
        TextBlock,
        TextBlockParam,
        TextDelta,
        ToolChoiceParam,
        ToolParam,
        ToolResultBlockParam,
        ToolUseBlock,
        ToolUseBlockParam,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        "you can use the `anthropic` optional group — `pip install 'pydantic-ai-slim[anthropic]'`"
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

    !!! note
        The `AnthropicModel` class does not yet support streaming responses.
        We anticipate adding support for streaming responses in a near-term future release.
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
        return f'anthropic:{self.model_name}'

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
    model_name: AnthropicModelName
    allow_text_result: bool
    tools: list[ToolParam]

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, usage.Usage]:
        response = await self._messages_create(messages, False, model_settings)
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        response = await self._messages_create(messages, True, model_settings)
        async with response:
            yield await self._process_streamed_response(response)

    @overload
    async def _messages_create(
        self, messages: list[ModelMessage], stream: Literal[True], model_settings: ModelSettings | None
    ) -> AsyncStream[RawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self, messages: list[ModelMessage], stream: Literal[False], model_settings: ModelSettings | None
    ) -> AnthropicMessage:
        pass

    async def _messages_create(
        self, messages: list[ModelMessage], stream: bool, model_settings: ModelSettings | None
    ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
        # standalone function to make it easier to override
        model_settings = model_settings or {}

        tool_choice: ToolChoiceParam | None

        if not self.tools:
            tool_choice = None
        else:
            if not self.allow_text_result:
                tool_choice = {'type': 'any'}
            else:
                tool_choice = {'type': 'auto'}

            if (allow_parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
                tool_choice['disable_parallel_tool_use'] = not allow_parallel_tool_calls

        system_prompt, anthropic_messages = self._map_message(messages)

        return await self.client.messages.create(
            max_tokens=model_settings.get('max_tokens', 1024),
            system=system_prompt or NOT_GIVEN,
            messages=anthropic_messages,
            model=self.model_name,
            tools=self.tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            temperature=model_settings.get('temperature', NOT_GIVEN),
            top_p=model_settings.get('top_p', NOT_GIVEN),
            timeout=model_settings.get('timeout', NOT_GIVEN),
        )

    def _process_response(self, response: AnthropicMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.content:
            if isinstance(item, TextBlock):
                items.append(TextPart(content=item.text))
            else:
                assert isinstance(item, ToolUseBlock), 'unexpected item type'
                items.append(
                    ToolCallPart.from_raw_args(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        return ModelResponse(items, model_name=self.model_name)

    async def _process_streamed_response(self, response: AsyncStream[RawMessageStreamEvent]) -> StreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        # Since Anthropic doesn't provide a timestamp in the message, we'll use the current time
        timestamp = datetime.now(tz=timezone.utc)
        return AnthropicStreamedResponse(_model_name=self.model_name, _response=peekable_response, _timestamp=timestamp)

    @staticmethod
    def _map_message(messages: list[ModelMessage]) -> tuple[str, list[MessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ''
        anthropic_messages: list[MessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt += part.content
                    elif isinstance(part, UserPromptPart):
                        anthropic_messages.append(MessageParam(role='user', content=part.content))
                    elif isinstance(part, ToolReturnPart):
                        anthropic_messages.append(
                            MessageParam(
                                role='user',
                                content=[
                                    ToolResultBlockParam(
                                        tool_use_id=_guard_tool_call_id(t=part, model_source='Anthropic'),
                                        type='tool_result',
                                        content=part.model_response_str(),
                                        is_error=False,
                                    )
                                ],
                            )
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            anthropic_messages.append(MessageParam(role='user', content=part.model_response()))
                        else:
                            anthropic_messages.append(
                                MessageParam(
                                    role='user',
                                    content=[
                                        ToolResultBlockParam(
                                            tool_use_id=_guard_tool_call_id(t=part, model_source='Anthropic'),
                                            type='tool_result',
                                            content=part.model_response(),
                                            is_error=True,
                                        ),
                                    ],
                                )
                            )
            elif isinstance(m, ModelResponse):
                content: list[TextBlockParam | ToolUseBlockParam] = []
                for item in m.parts:
                    if isinstance(item, TextPart):
                        content.append(TextBlockParam(text=item.content, type='text'))
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(_map_tool_call(item))
                anthropic_messages.append(MessageParam(role='assistant', content=content))
            else:
                assert_never(m)
        return system_prompt, anthropic_messages


def _map_tool_call(t: ToolCallPart) -> ToolUseBlockParam:
    assert isinstance(t.args, ArgsDict), f'Expected ArgsDict, got {t.args}'
    return ToolUseBlockParam(
        id=_guard_tool_call_id(t=t, model_source='Anthropic'),
        type='tool_use',
        name=t.tool_name,
        input=t.args_as_dict(),
    )


def _map_usage(message: AnthropicMessage | RawMessageStreamEvent) -> usage.Usage:
    if isinstance(message, AnthropicMessage):
        response_usage = message.usage
    else:
        if isinstance(message, RawMessageStartEvent):
            response_usage = message.message.usage
        elif isinstance(message, RawMessageDeltaEvent):
            response_usage = message.usage
        else:
            # No usage information provided in:
            # - RawMessageStopEvent
            # - RawContentBlockStartEvent
            # - RawContentBlockDeltaEvent
            # - RawContentBlockStopEvent
            response_usage = None

    if response_usage is None:
        return usage.Usage()

    request_tokens = getattr(response_usage, 'input_tokens', None)

    return usage.Usage(
        # Usage coming from the RawMessageDeltaEvent doesn't have input token data, hence this getattr
        request_tokens=request_tokens,
        response_tokens=response_usage.output_tokens,
        total_tokens=(request_tokens or 0) + response_usage.output_tokens,
    )


@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _response: AsyncIterable[RawMessageStreamEvent]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        current_block: TextBlock | ToolUseBlock | None = None
        current_json: str = ''

        async for event in self._response:
            self._usage += _map_usage(event)

            if isinstance(event, RawContentBlockStartEvent):
                current_block = event.content_block
                if isinstance(current_block, TextBlock) and current_block.text:
                    yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=current_block.text)
                elif isinstance(current_block, ToolUseBlock):
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=current_block.id,
                        tool_name=current_block.name,
                        args=cast(dict[str, Any], current_block.input),
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event

            elif isinstance(event, RawContentBlockDeltaEvent):
                if isinstance(event.delta, TextDelta):
                    yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=event.delta.text)
                elif (
                    current_block and event.delta.type == 'input_json_delta' and isinstance(current_block, ToolUseBlock)
                ):
                    # Try to parse the JSON immediately, otherwise cache the value for later. This handles
                    # cases where the JSON is not currently valid but will be valid once we stream more tokens.
                    try:
                        parsed_args = json_loads(current_json + event.delta.partial_json)
                        current_json = ''
                    except JSONDecodeError:
                        current_json += event.delta.partial_json
                        continue

                    # For tool calls, we need to handle partial JSON updates
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=current_block.id,
                        tool_name='',
                        args=parsed_args,
                        tool_call_id=current_block.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event

            elif isinstance(event, (RawContentBlockStopEvent, RawMessageStopEvent)):
                current_block = None

    def timestamp(self) -> datetime:
        return self._timestamp
