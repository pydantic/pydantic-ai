from __future__ import annotations as _annotations

import io
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from json import JSONDecodeError, loads as json_loads
from typing import Any, Literal, Union, cast, overload

from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
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
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
    get_user_agent,
)

try:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropic, AsyncStream
    from anthropic.types import (
        Base64PDFSourceParam,
        ContentBlock,
        DocumentBlockParam,
        ImageBlockParam,
        Message as AnthropicMessage,
        MessageParam,
        MetadataParam,
        PlainTextSourceParam,
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
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error

LatestAnthropicModelNames = Literal[
    'claude-3-7-sonnet-latest',
    'claude-3-5-haiku-latest',
    'claude-3-5-sonnet-latest',
    'claude-3-opus-latest',
]
"""Latest Anthropic models."""

AnthropicModelName = Union[str, LatestAnthropicModelNames]
"""Possible Anthropic model names.

Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""


class AnthropicModelSettings(ModelSettings):
    """Settings used for an Anthropic model request.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_metadata: MetadataParam
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request."""


@dataclass(init=False)
class AnthropicModel(Model):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.

    !!! note
        The `AnthropicModel` class does not yet support streaming responses.
        We anticipate adding support for streaming responses in a near-term future release.
    """

    client: AsyncAnthropic = field(repr=False)

    _model_name: AnthropicModelName = field(repr=False)
    _system: str = field(default='anthropic', repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['anthropic'] | Provider[AsyncAnthropic] = 'anthropic',
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
                instance of `Provider[AsyncAnthropic]`. If not provided, the other parameters will be used.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        check_allow_model_requests()
        response = await self._messages_create(
            messages, False, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._messages_create(
            messages, True, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response)

    @property
    def model_name(self) -> AnthropicModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[RawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AnthropicMessage:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
        # standalone function to make it easier to override
        tools = self._get_tools(model_request_parameters)
        tool_choice: ToolChoiceParam | None

        if not tools:
            tool_choice = None
        else:
            if not model_request_parameters.allow_text_result:
                tool_choice = {'type': 'any'}
            else:
                tool_choice = {'type': 'auto'}

            if (allow_parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
                tool_choice['disable_parallel_tool_use'] = not allow_parallel_tool_calls

        system_prompt, anthropic_messages = await self._map_message(messages)

        try:
            return await self.client.messages.create(
                max_tokens=model_settings.get('max_tokens', 1024),
                system=system_prompt or NOT_GIVEN,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                stop_sequences=model_settings.get('stop_sequences', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                metadata=model_settings.get('anthropic_metadata', NOT_GIVEN),
                extra_headers={'User-Agent': get_user_agent()},
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _process_response(self, response: AnthropicMessage) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.content:
            if isinstance(item, TextBlock):
                items.append(TextPart(content=item.text))
            else:
                assert isinstance(item, ToolUseBlock), 'unexpected item type'
                items.append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        return ModelResponse(items, model_name=response.model)

    async def _process_streamed_response(self, response: AsyncStream[RawMessageStreamEvent]) -> StreamedResponse:
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        # Since Anthropic doesn't provide a timestamp in the message, we'll use the current time
        timestamp = datetime.now(tz=timezone.utc)
        return AnthropicStreamedResponse(
            _model_name=self._model_name, _response=peekable_response, _timestamp=timestamp
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    async def _map_message(self, messages: list[ModelMessage]) -> tuple[str, list[MessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        system_prompt: str = ''
        anthropic_messages: list[MessageParam] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                user_content_params: list[
                    ToolResultBlockParam | TextBlockParam | ImageBlockParam | DocumentBlockParam
                ] = []
                for request_part in m.parts:
                    if isinstance(request_part, SystemPromptPart):
                        system_prompt += request_part.content
                    elif isinstance(request_part, UserPromptPart):
                        async for content in self._map_user_prompt(request_part):
                            user_content_params.append(content)
                    elif isinstance(request_part, ToolReturnPart):
                        tool_result_block_param = ToolResultBlockParam(
                            tool_use_id=_guard_tool_call_id(t=request_part),
                            type='tool_result',
                            content=request_part.model_response_str(),
                            is_error=False,
                        )
                        user_content_params.append(tool_result_block_param)
                    elif isinstance(request_part, RetryPromptPart):
                        if request_part.tool_name is None:
                            retry_param = TextBlockParam(type='text', text=request_part.model_response())
                        else:
                            retry_param = ToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=request_part.model_response(),
                                is_error=True,
                            )
                        user_content_params.append(retry_param)
                anthropic_messages.append(MessageParam(role='user', content=user_content_params))
            elif isinstance(m, ModelResponse):
                assistant_content_params: list[TextBlockParam | ToolUseBlockParam] = []
                for response_part in m.parts:
                    if isinstance(response_part, TextPart):
                        assistant_content_params.append(TextBlockParam(text=response_part.content, type='text'))
                    else:
                        tool_use_block_param = ToolUseBlockParam(
                            id=_guard_tool_call_id(t=response_part),
                            type='tool_use',
                            name=response_part.tool_name,
                            input=response_part.args_as_dict(),
                        )
                        assistant_content_params.append(tool_use_block_param)
                anthropic_messages.append(MessageParam(role='assistant', content=assistant_content_params))
            else:
                assert_never(m)
        return system_prompt, anthropic_messages

    @staticmethod
    async def _map_user_prompt(
        part: UserPromptPart,
    ) -> AsyncGenerator[ImageBlockParam | TextBlockParam | DocumentBlockParam]:
        if isinstance(part.content, str):
            yield TextBlockParam(text=part.content, type='text')
        else:
            for item in part.content:
                if isinstance(item, str):
                    yield TextBlockParam(text=item, type='text')
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        yield ImageBlockParam(
                            source={'data': io.BytesIO(item.data), 'media_type': item.media_type, 'type': 'base64'},  # type: ignore
                            type='image',
                        )
                    elif item.media_type == 'application/pdf':
                        yield DocumentBlockParam(
                            source=Base64PDFSourceParam(
                                data=io.BytesIO(item.data),
                                media_type='application/pdf',
                                type='base64',
                            ),
                            type='document',
                        )
                    else:
                        raise RuntimeError('Only images and PDFs are supported for binary content')
                elif isinstance(item, ImageUrl):
                    yield ImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')
                elif isinstance(item, DocumentUrl):
                    if item.media_type == 'application/pdf':
                        yield DocumentBlockParam(source={'url': item.url, 'type': 'url'}, type='document')
                    elif item.media_type == 'text/plain':
                        response = await cached_async_http_client().get(item.url)
                        response.raise_for_status()
                        yield DocumentBlockParam(
                            source=PlainTextSourceParam(data=response.text, media_type=item.media_type, type='text'),
                            type='document',
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported media type: {item.media_type}')
                else:
                    raise RuntimeError(f'Unsupported content type: {type(item)}')

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolParam:
        return {
            'name': f.name,
            'description': f.description,
            'input_schema': f.parameters_json_schema,
        }


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

    _model_name: AnthropicModelName
    _response: AsyncIterable[RawMessageStreamEvent]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        current_block: ContentBlock | None = None
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

    @property
    def model_name(self) -> AnthropicModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
