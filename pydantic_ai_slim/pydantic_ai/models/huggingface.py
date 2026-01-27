from __future__ import annotations as _annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal, cast, overload

from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id as _guard_tool_call_id
from ..messages import (
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FilePart,
    FinishReason,
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
from ..profiles import ModelProfile, ModelProfileSpec
from ..providers import Provider, infer_provider
from ..providers.huggingface import HfRouterProvider, get_router_info, select_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests

try:
    import aiohttp
    from huggingface_hub import (
        AsyncInferenceClient,
        ChatCompletionInputGrammarType,
        ChatCompletionInputMessage,
        ChatCompletionInputMessageChunk,
        ChatCompletionInputResponseFormatJSONObject,
        ChatCompletionInputResponseFormatJSONSchema,
        ChatCompletionInputTool,
        ChatCompletionInputToolCall,
        ChatCompletionInputURL,
        ChatCompletionOutput,
        ChatCompletionOutputMessage,
        ChatCompletionStreamOutput,
        TextGenerationOutputFinishReason,
    )
    from huggingface_hub.errors import HfHubHTTPError

except ImportError as _import_error:
    raise ImportError(
        'Please install `huggingface_hub` to use Hugging Face Inference Providers, '
        'you can use the `huggingface` optional group — `pip install "pydantic-ai-slim[huggingface]"`'
    ) from _import_error

_logger = logging.getLogger(__name__)

__all__ = (
    'HuggingFaceModel',
    'HuggingFaceModelSettings',
)


HFSystemPromptRole = Literal['system', 'user']

LatestHuggingFaceModelNames = Literal[
    'deepseek-ai/DeepSeek-R1',
    'meta-llama/Llama-3.1-8B-Instruct',
    'MiniMaxAI/MiniMax-M2',
    'openai/gpt-oss-20b',
    'openai/gpt-oss-120b',
    'Qwen/Qwen3-Coder-30B-A3B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen3-235B-A22B',
    'Qwen/Qwen3-32B',
]
"""Latest Hugging Face models."""


HuggingFaceModelName = str | LatestHuggingFaceModelNames
"""Possible Hugging Face model names.

You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
"""

_FINISH_REASON_MAP: dict[TextGenerationOutputFinishReason, FinishReason] = {
    'length': 'length',
    'eos_token': 'stop',
    'stop_sequence': 'stop',
}


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
    _provider: Provider[AsyncInferenceClient] = field(repr=False)
    _router_initialized: bool = field(default=False, repr=False)
    _selected_provider_info: HfRouterProvider | None = field(default=None, repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['huggingface'] | Provider[AsyncInferenceClient] = 'huggingface',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Hugging Face model.

        Args:
            model_name: The name of the Model to use. You can browse available models [here](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending).
            provider: The provider to use for Hugging Face Inference Providers. Can be either the string 'huggingface' or an
                instance of `Provider[AsyncInferenceClient]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        provider_profile = provider.model_profile(model_name)
        self._model_name = model_name.rsplit(':', 1)[0]

        super().__init__(settings=settings, profile=profile or provider_profile)
        self.client = provider.client
        self._router_initialized = False
        self._selected_provider_info = None

    @property
    def base_url(self) -> str:
        """The base URL of the provider."""
        return self._provider.base_url

    @property
    def model_name(self) -> HuggingFaceModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    async def _ensure_router_initialized(self) -> None:
        """Lazily fetch router info and update client/profile on first request.

        This is called at the start of request() and request_stream()
        """
        if self._router_initialized:
            return

        self._router_initialized = True

        model_name_lower = self._model_name.lower()
        router_info = await _utils.run_in_executor(get_router_info, model_name_lower)

        if not router_info:
            return

        providers = router_info['providers']

        from ..providers.huggingface import HuggingFaceProvider

        user_provider_name: str | None = None
        if isinstance(self._provider, HuggingFaceProvider):
            user_provider_name = self._provider.provider_name

        # Select the best provider
        if user_provider_name:
            for p in providers:
                if p['provider'] == user_provider_name:
                    self._selected_provider_info = p
                    break
            if self._selected_provider_info is None:
                self._selected_provider_info = select_provider(providers)
        else:
            self._selected_provider_info = select_provider(providers)

        if self._selected_provider_info is None:
            return

        # Create a model specific client with the selected provider
        if isinstance(self._provider, HuggingFaceProvider):
            self.client = AsyncInferenceClient(
                token=self._provider.api_key,
                provider=self._selected_provider_info['provider'],  # type: ignore
            )

        selected_provider_name = self._selected_provider_info['provider']
        supports_tools = self._selected_provider_info['supports_tools']
        supports_structured_output = self._selected_provider_info['supports_structured_output']

        if not supports_structured_output:
            _logger.warning(
                f'Provider {selected_provider_name} does not support structured output (NativeOutput).',
            )
        if not supports_tools:
            _logger.warning(f"Provider '{selected_provider_name}' does not support tools.")

        current_profile = self._profile
        if callable(current_profile):
            current_profile = current_profile(self._model_name)
        if current_profile is None:
            current_profile = ModelProfile()

        self._profile = replace(
            current_profile,
            supports_tools=supports_tools,
            supports_json_schema_output=supports_structured_output,
            supports_json_object_output=supports_structured_output,
        )
        if 'profile' in self.__dict__:
            del self.__dict__['profile']

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        await self._ensure_router_initialized()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, False, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
        )
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
        await self._ensure_router_initialized()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, True, cast(HuggingFaceModelSettings, model_settings or {}), model_request_parameters
        )
        yield await self._process_streamed_response(response, model_request_parameters)

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

        hf_messages = await self._map_messages(messages, model_request_parameters)
        response_format: ChatCompletionInputGrammarType | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = ChatCompletionInputResponseFormatJSONObject.parse_obj_as_instance({'type': 'json_object'})  # type: ignore

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
                response_format=response_format or None,
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
                body=e.message,
            ) from e
        except HfHubHTTPError as e:
            raise ModelHTTPError(
                status_code=e.response.status_code,
                model_name=self.model_name,
                body=e.response.content,
            ) from e

    def _process_response(self, response: ChatCompletionOutput) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        choice = response.choices[0]
        content = choice.message.content
        tool_calls = choice.message.tool_calls

        items: list[ModelResponsePart] = []

        if content:
            items.extend(split_content_into_text_and_thinking(content, self.profile.thinking_tags))
        if tool_calls is not None:
            for c in tool_calls:
                items.append(ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id))

        raw_finish_reason = choice.finish_reason
        provider_details: dict[str, Any] = {'finish_reason': raw_finish_reason}
        if response.created:  # pragma: no branch
            provider_details['timestamp'] = datetime.fromtimestamp(response.created, tz=timezone.utc)
        finish_reason = _FINISH_REASON_MAP.get(cast(TextGenerationOutputFinishReason, raw_finish_reason), None)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self.base_url,
            finish_reason=finish_reason,
            provider_details=provider_details,
        )

    async def _process_streamed_response(
        self, response: AsyncIterable[ChatCompletionStreamOutput], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        return HuggingFaceStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.model,
            _model_profile=self.profile,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _provider_url=self.base_url,
            _provider_timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
        )

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ChatCompletionInputTool]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
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
                        start_tag, end_tag = self.profile.thinking_tags
                        texts.append('\n'.join([start_tag, item.content, end_tag]))
                    elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                        # This is currently never returned from huggingface
                        pass
                    elif isinstance(item, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
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
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt_count = sum(1 for m in hf_messages if getattr(m, 'role', None) == 'system')
            hf_messages.insert(system_prompt_count, ChatCompletionInputMessage(content=instructions, role='system'))  # type: ignore
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

    def _map_json_schema(self, o: OutputObjectDefinition) -> ChatCompletionInputGrammarType:
        response_format_param: ChatCompletionInputResponseFormatJSONSchema = {  # type: ignore
            'type': 'json_schema',
            'json_schema': {
                'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
                'schema': o.json_schema,
                'strict': o.strict,
            },
        }
        if o.description:  # pragma: no branch
            response_format_param['json_schema']['description'] = o.description
        return response_format_param

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
                    if item.is_image:
                        url = ChatCompletionInputURL(url=item.data_uri)  # type: ignore
                        content.append(ChatCompletionInputMessageChunk(type='image_url', image_url=url))  # type: ignore
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    raise NotImplementedError('AudioUrl is not supported for Hugging Face')
                elif isinstance(item, DocumentUrl):
                    raise NotImplementedError('DocumentUrl is not supported for Hugging Face')
                elif isinstance(item, VideoUrl):
                    raise NotImplementedError('VideoUrl is not supported for Hugging Face')
                elif isinstance(item, CachePoint):
                    # Hugging Face doesn't support prompt caching via CachePoint
                    pass
                else:
                    assert_never(item)
        return ChatCompletionInputMessage(role='user', content=content)  # type: ignore


@dataclass
class HuggingFaceStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Hugging Face models."""

    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionStreamOutput]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._provider_timestamp is not None:  # pragma: no branch
            self.provider_details = {'timestamp': self._provider_timestamp}
        async for chunk in self._response:
            self._usage += _map_usage(chunk)

            if chunk.id:  # pragma: no branch
                self.provider_response_id = chunk.id

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            if raw_finish_reason := choice.finish_reason:
                self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
                self.finish_reason = _FINISH_REASON_MAP.get(
                    cast(TextGenerationOutputFinishReason, raw_finish_reason), None
                )

            # Handle the text part of the response
            content = choice.delta.content
            if content:
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id='content',
                    content=content,
                    thinking_tags=self._model_profile.thinking_tags,
                    ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
                ):
                    yield event

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
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def provider_url(self) -> str:
        """Get the provider base URL."""
        return self._provider_url

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _map_usage(response: ChatCompletionOutput | ChatCompletionStreamOutput) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()

    return usage.RequestUsage(
        input_tokens=response_usage.prompt_tokens,
        output_tokens=response_usage.completion_tokens,
    )
