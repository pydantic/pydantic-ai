from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Union, cast, overload

from typing_extensions import assert_never

from pydantic_ai._thinking_part import split_content_into_text_and_thinking
from pydantic_ai.providers import Provider, infer_provider

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc
from ..messages import (
    AudioUrl,
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
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests

try:
    import aiohttp
    from huggingface_hub import (
        AsyncInferenceClient,
        ChatCompletionInputMessage,
        ChatCompletionInputMessageChunk,
        ChatCompletionInputTool,
        ChatCompletionInputToolCall,
        ChatCompletionInputURL,
        ChatCompletionOutput,
        ChatCompletionOutputMessage,
        ChatCompletionStreamOutput,
    )
    from huggingface_hub.errors import HfHubHTTPError

except ImportError as _import_error:
    raise ImportError(
        'Please install `huggingface_hub` to use Hugging Face Inference Providers, '
        'you can use the `huggingface` optional group — `pip install "pydantic-ai-slim[huggingface]"`'
    ) from _import_error

__all__ = (
    'HuggingFaceModel',
    'HuggingFaceModelSettings',
)


HFSystemPromptRole = Literal['system', 'user']

LatestHuggingFaceModelNames = Literal[
    'deepseek-ai/DeepSeek-R1',
    'meta-llama/Llama-3.3-70B-Instruct',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'Qwen/QwQ-32B',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen3-235B-A22B',
    'Qwen/Qwen3-32B',
]
"""Latest Hugging Face models."""


HuggingFaceModelName = Union[str, LatestHuggingFaceModelNames]
"""Possible Hugging Face model names.

You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
"""


class HuggingFaceModelSettings(ModelSettings, total=False):
    """Settings used for a Hugging Face model request."""

    # ALL FIELDS MUST BE `huggingface_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future huggingface-specific settings


@dataclass(init=False)
class HuggingFaceModel(Model):
    """A model that uses Hugging Face Inference Providers.

    Internally, this uses the [HF Python client](https://github.com/huggingface/huggingface_hub) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncInferenceClient = field(repr=False)

    _model_name: str = field(repr=False)
    _system: str = field(default='huggingface', repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
    ):
        """Initialize a Hugging Face model.

        Args:
            model_name: The name of the Model to use. You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
            provider: The provider to use for Hugging Face Inference Providers. Can be either the string 'huggingface' or an
                instance of `Provider[AsyncInferenceClient]`. If not provided, the other parameters will be used.
        """
        self._model_name = model_name
        self._provider = provider
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = provider.client

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, False, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
        )
        model_response = self._process_response(response)
        model_response.usage.requests = 1
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        response = await self._completions_create(
            messages, True, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
        )
        yield await self._process_streamed_response(response)

    @property
    def model_name(self) -> HuggingFaceModelName:
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
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterable[ChatCompletionStreamOutput]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletionOutput: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: HuggingFaceModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletionOutput | AsyncIterable[ChatCompletionStreamOutput]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        hf_messages = await self._map_messages(messages)

        try:
            return await self.client.chat.completions.create(  # type: ignore
                model=self._model_name,
                messages=hf_messages,  # type: ignore
                tools=tools,
                tool_choice=tool_choice or None,
                stream=stream,
                stop=model_settings.get('stop_sequences', None),
                temperature=model_settings.get('temperature', None),
                top_p=model_settings.get('top_p', None),
                seed=model_settings.get('seed', None),
                presence_penalty=model_settings.get('presence_penalty', None),
                frequency_penalty=model_settings.get('frequency_penalty', None),
                logit_bias=model_settings.get('logit_bias', None),  # type: ignore
                logprobs=model_settings.get('logprobs', None),
                top_logprobs=model_settings.get('top_logprobs', None),
                extra_body=model_settings.get('extra_body'),  # type: ignore
            )
        except aiohttp.ClientResponseError as e:
            raise ModelHTTPError(
                status_code=e.status,
                model_name=self.model_name,
                body=e.response_error_payload,  # type: ignore
            ) from e
        except HfHubHTTPError as e:
            raise ModelHTTPError(
                status_code=e.response.status_code,
                model_name=self.model_name,
                body=e.response.content,
            ) from e

    def _process_response(self, response: ChatCompletionOutput) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

        choice = response.choices[0]
        content = choice.message.content
        tool_calls = choice.message.tool_calls

        items: list[ModelResponsePart] = []

        if content is not None:
            items.extend(split_content_into_text_and_thinking(content))
        if tool_calls is not None:
            for c in tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))
        return ModelResponse(
            items,
            usage=_map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            vendor_id=response.id,
        )

    async def _process_streamed_response(self, response: AsyncIterable[ChatCompletionStreamOutput]) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return HuggingFaceStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ChatCompletionInputTool]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.output_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.output_tools]
        return tools

    async def _map_messages(
        self, messages: list[ModelMessage]
    ) -> list[ChatCompletionInputMessage | ChatCompletionOutputMessage]:
        """Just maps a `pydantic_ai.Message` to a `huggingface_hub.ChatCompletionInputMessage`."""
        hf_messages: list[ChatCompletionInputMessage | ChatCompletionOutputMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    hf_messages.append(item)
            elif isinstance(message, ModelResponse):
                texts: list[str] = []
                tool_calls: list[ChatCompletionInputToolCall] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        texts.append(item.content)
                    elif isinstance(item, ToolCallPart):
                        tool_calls.append(self._map_tool_call(item))
                    elif isinstance(item, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet. If you are unsatisfied with this,
                        # please open an issue. The below code is the code to send thinking to the provider.
                        # texts.append(f'<think>\n{item.content}\n</think>')
                        pass
                    else:
                        assert_never(item)
                message_param = ChatCompletionInputMessage(role='assistant')  # type: ignore
                if texts:
                    # Note: model responses from this model should only have one text item, so the following
                    # shouldn't merge multiple texts into one unless you switch models between runs:
                    message_param['content'] = '\n\n'.join(texts)
                if tool_calls:
                    message_param['tool_calls'] = tool_calls
                hf_messages.append(message_param)
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages):
            hf_messages.insert(0, ChatCompletionInputMessage(content=instructions, role='system'))  # type: ignore
        return hf_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ChatCompletionInputToolCall:
        return ChatCompletionInputToolCall.parse_obj_as_instance(  # type: ignore
            {
                'id': _guard_tool_call_id(t=t),
                'type': 'function',
                'function': {
                    'name': t.tool_name,
                    'arguments': t.args_as_json_str(),
                },
            }
        )

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ChatCompletionInputTool:
        tool_param: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj_as_instance(  # type: ignore
            {
                'type': 'function',
                'function': {
                    'name': f.name,
                    'description': f.description,
                    'parameters': f.parameters_json_schema,
                },
            }
        )
        if f.strict is not None:
            tool_param['function']['strict'] = f.strict
        return tool_param

    async def _map_user_message(
        self, message: ModelRequest
    ) -> AsyncIterable[ChatCompletionInputMessage | ChatCompletionOutputMessage]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield ChatCompletionInputMessage.parse_obj_as_instance({'role': 'system', 'content': part.content})  # type: ignore
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield ChatCompletionOutputMessage.parse_obj_as_instance(  # type: ignore
                    {
                        'role': 'tool',
                        'tool_call_id': _guard_tool_call_id(t=part),
                        'content': part.model_response_str(),
                    }
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield ChatCompletionInputMessage.parse_obj_as_instance(  # type: ignore
                        {'role': 'user', 'content': part.model_response()}
                    )
                else:
                    yield ChatCompletionInputMessage.parse_obj_as_instance(  # type: ignore
                        {
                            'role': 'tool',
                            'tool_call_id': _guard_tool_call_id(t=part),
                            'content': part.model_response(),
                        }
                    )
            else:
                assert_never(part)

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> ChatCompletionInputMessage:
        content: str | list[ChatCompletionInputMessage]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionInputMessageChunk(type='text', text=item))  # type: ignore
                elif isinstance(item, ImageUrl):
                    url = ChatCompletionInputURL(url=item.url)  # type: ignore
                    content.append(ChatCompletionInputMessageChunk(type='image_url', image_url=url))  # type: ignore
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        url = ChatCompletionInputURL(url=f'data:{item.media_type};base64,{base64_encoded}')  # type: ignore
                        content.append(ChatCompletionInputMessageChunk(type='image_url', image_url=url))  # type: ignore
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    raise NotImplementedError('AudioUrl is not supported for Hugging Face')
                elif isinstance(item, DocumentUrl):
                    raise NotImplementedError('DocumentUrl is not supported for Hugging Face')
                elif isinstance(item, VideoUrl):
                    raise NotImplementedError('VideoUrl is not supported for Hugging Face')
                else:
                    assert_never(item)
        return ChatCompletionInputMessage(role='user', content=content)  # type: ignore


@dataclass
class HuggingFaceStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Hugging Face models."""

    _model_name: str
    _response: AsyncIterable[ChatCompletionStreamOutput]
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
            if content:
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id='content', content=content, extract_think_tags=True
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            for dtc in choice.delta.tool_calls or []:
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=dtc.index,
                    tool_name=dtc.function and dtc.function.name,  # type: ignore
                    args=dtc.function and dtc.function.arguments,
                    tool_call_id=dtc.id,
                )
                if maybe_event is not None:
                    yield maybe_event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_usage(response: ChatCompletionOutput | ChatCompletionStreamOutput) -> usage.Usage:
    response_usage = response.usage
    if response_usage is None:
        return usage.Usage()

    return usage.Usage(
        request_tokens=response_usage.prompt_tokens,
        response_tokens=response_usage.completion_tokens,
        total_tokens=response_usage.total_tokens,
        details=None,
    )
