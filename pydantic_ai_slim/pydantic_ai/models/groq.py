from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Literal, Union, cast, overload

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
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, get_user_agent

try:
    from groq import NOT_GIVEN, APIStatusError, AsyncGroq, AsyncStream
    from groq.types import chat
    from groq.types.chat.chat_completion_content_part_image_param import ImageURL
except ImportError as _import_error:
    raise ImportError(
        'Please install `groq` to use the Groq model, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error

ProductionGroqModelNames = Literal[
    'distil-whisper-large-v3-en',
    'gemma2-9b-it',
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'llama-guard-3-8b',
    'llama3-70b-8192',
    'llama3-8b-8192',
    'whisper-large-v3',
    'whisper-large-v3-turbo',
]
"""Production Groq models from <https://console.groq.com/docs/models#production-models>."""

PreviewGroqModelNames = Literal[
    'playai-tts',
    'playai-tts-arabic',
    'qwen-qwq-32b',
    'mistral-saba-24b',
    'qwen-2.5-coder-32b',
    'qwen-2.5-32b',
    'deepseek-r1-distill-qwen-32b',
    'deepseek-r1-distill-llama-70b',
    'llama-3.3-70b-specdec',
    'llama-3.2-1b-preview',
    'llama-3.2-3b-preview',
    'llama-3.2-11b-vision-preview',
    'llama-3.2-90b-vision-preview',
]
"""Preview Groq models from <https://console.groq.com/docs/models#preview-models>."""

GroqModelName = Union[str, ProductionGroqModelNames, PreviewGroqModelNames]
"""Possible Groq model names.

Since Groq supports a variety of models and the list changes frequencly, we explicitly list the named models as of 2025-03-31
but allow any name in the type hints.

See <https://console.groq.com/docs/models> for an up to date date list of models and more details.
"""


class GroqModelSettings(ModelSettings):
    """Settings used for a Groq model request.

    ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    # This class is a placeholder for any future groq-specific settings


@dataclass(init=False)
class GroqModel(Model):
    """A model that uses the Groq API.

    Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncGroq = field(repr=False)

    _model_name: GroqModelName = field(repr=False)
    _system: str = field(default='groq', repr=False)

    def __init__(self, model_name: GroqModelName, *, provider: Literal['groq'] | Provider[AsyncGroq] = 'groq'):
        """Initialize a Groq model.

        Args:
            model_name: The name of the Groq model to use. List of model names available
                [here](https://console.groq.com/docs/models).
            provider: The provider to use for authentication and API access. Can be either the string
                'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
                created using the other parameters.
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
        response = await self._completions_create(
            messages, False, cast(GroqModelSettings, model_settings or {}), model_request_parameters
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
        response = await self._completions_create(
            messages, True, cast(GroqModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(response)

    @property
    def model_name(self) -> GroqModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[chat.ChatCompletionChunk]:
        pass

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion:
        pass

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GroqModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[chat.ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)
        # standalone function to make it easier to override
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        groq_messages = list(chain(*(self._map_message(m) for m in messages)))

        try:
            return await self.client.chat.completions.create(
                model=str(self._model_name),
                messages=groq_messages,
                n=1,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stop=model_settings.get('stop_sequences', NOT_GIVEN),
                stream=stream,
                max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
                temperature=model_settings.get('temperature', NOT_GIVEN),
                top_p=model_settings.get('top_p', NOT_GIVEN),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                seed=model_settings.get('seed', NOT_GIVEN),
                presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
                frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
                logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
                extra_headers={'User-Agent': get_user_agent()},
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.content is not None:
            items.append(TextPart(content=choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id))
        return ModelResponse(items, model_name=response.model, timestamp=timestamp)

    async def _process_streamed_response(self, response: AsyncStream[chat.ChatCompletionChunk]) -> GroqStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return GroqStreamedResponse(
            _response=peekable_response,
            _model_name=self._model_name,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
        return tools

    def _map_message(self, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `groq.types.ChatCompletionMessageParam`."""
        if isinstance(message, ModelRequest):
            yield from self._map_user_message(message)
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(self._map_tool_call(item))
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

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
        return chat.ChatCompletionMessageToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

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

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[chat.ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(chat.ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url = ImageURL(url=item.url)
                    content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        image_url = ImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
                        content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    else:
                        raise RuntimeError('Only images are supported for binary content in Groq.')
                elif isinstance(item, DocumentUrl):  # pragma: no cover
                    raise RuntimeError('DocumentUrl is not supported in Groq.')
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported content type: {type(item)}')

        return chat.ChatCompletionUserMessageParam(role='user', content=content)


@dataclass
class GroqStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Groq models."""

    _model_name: GroqModelName
    _response: AsyncIterable[chat.ChatCompletionChunk]
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

    @property
    def model_name(self) -> GroqModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_usage(completion: chat.ChatCompletionChunk | chat.ChatCompletion) -> usage.Usage:
    response_usage = None
    if isinstance(completion, chat.ChatCompletion):
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
