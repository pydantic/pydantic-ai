from __future__ import annotations as _annotations

import base64
import itertools
import json
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cached_property
from typing import Any, Literal, cast, overload

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import to_json
from typing_extensions import assert_never, deprecated

from .. import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_NAME, OutputObjectDefinition
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from .._utils import guard_tool_call_id as _guard_tool_call_id, now_utc as _now_utc, number_to_datetime
from ..builtin_tools import (
    AbstractBuiltinTool,
    CodeExecutionTool,
    FileSearchTool,
    ImageAspectRatio,
    ImageGenerationTool,
    MCPServerTool,
    WebSearchTool,
)
from ..exceptions import UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
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
    PartStartEvent,
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
from ..profiles.openai import SAMPLING_PARAMS, OpenAIModelProfile, OpenAISystemPromptRole
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    OpenAIChatCompatibleProvider,
    OpenAIResponsesCompatibleProvider,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)

try:
    from openai import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncOpenAI, AsyncStream, Omit, omit
    from openai.types import AllModels, chat, responses
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionContentPartImageParam,
        ChatCompletionContentPartInputAudioParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
        chat_completion,
        chat_completion_chunk,
        chat_completion_token_logprob,
    )
    from openai.types.chat.chat_completion_content_part_image_param import ImageURL
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
    from openai.types.chat.chat_completion_content_part_param import File, FileFile
    from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_function_tool_call_param import (
        ChatCompletionMessageFunctionToolCallParam,
    )
    from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import (
        WebSearchOptions,
        WebSearchOptionsUserLocation,
        WebSearchOptionsUserLocationApproximate,
    )
    from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
    from openai.types.responses.response_input_param import FunctionCallOutput, Message
    from openai.types.responses.response_reasoning_item_param import (
        Content as ReasoningContent,
        Summary as ReasoningSummary,
    )
    from openai.types.responses.response_status import ResponseStatus
    from openai.types.shared import ReasoningEffort
    from openai.types.shared_params import Reasoning

    OMIT = omit
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


__all__ = (
    'OpenAIModel',
    'OpenAIChatModel',
    'OpenAIResponsesModel',
    'OpenAIModelSettings',
    'OpenAIChatModelSettings',
    'OpenAIResponsesModelSettings',
    'OpenAIModelName',
)

OpenAIModelName = str | AllModels
"""
Possible OpenAI model names.

Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.

Using this more broad type for the model name instead of the ChatModel definition
allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
"""

MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME: Literal['x-openai-connector'] = 'x-openai-connector'
"""
Prefix for OpenAI connector IDs. OpenAI supports either a URL or a connector ID when passing MCP configuration to a model,
by using that prefix like `x-openai-connector:<connector-id>` in a URL, you can pass a connector ID to a model.
"""

_CHAT_FINISH_REASON_MAP: dict[
    Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call'], FinishReason
] = {
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_call',
    'content_filter': 'content_filter',
    'function_call': 'tool_call',
}

_RESPONSES_FINISH_REASON_MAP: dict[Literal['max_output_tokens', 'content_filter'] | ResponseStatus, FinishReason] = {
    'max_output_tokens': 'length',
    'content_filter': 'content_filter',
    'completed': 'stop',
    'cancelled': 'error',
    'failed': 'error',
}

_OPENAI_ASPECT_RATIO_TO_SIZE: dict[ImageAspectRatio, Literal['1024x1024', '1024x1536', '1536x1024']] = {
    '1:1': '1024x1024',
    '2:3': '1024x1536',
    '3:2': '1536x1024',
}

_OPENAI_IMAGE_SIZE = Literal['auto', '1024x1024', '1024x1536', '1536x1024']
_OPENAI_IMAGE_SIZES: tuple[_OPENAI_IMAGE_SIZE, ...] = _utils.get_args(_OPENAI_IMAGE_SIZE)


class _AzureContentFilterResultDetail(BaseModel):
    filtered: bool
    severity: str | None = None
    detected: bool | None = None


class _AzureContentFilterResult(BaseModel):
    hate: _AzureContentFilterResultDetail | None = None
    self_harm: _AzureContentFilterResultDetail | None = None
    sexual: _AzureContentFilterResultDetail | None = None
    violence: _AzureContentFilterResultDetail | None = None
    jailbreak: _AzureContentFilterResultDetail | None = None
    profanity: _AzureContentFilterResultDetail | None = None


class _AzureInnerError(BaseModel):
    code: str
    content_filter_result: _AzureContentFilterResult


class _AzureError(BaseModel):
    code: str
    message: str
    innererror: _AzureInnerError | None = None


class _AzureErrorResponse(BaseModel):
    error: _AzureError


def _resolve_openai_image_generation_size(
    tool: ImageGenerationTool,
) -> _OPENAI_IMAGE_SIZE:
    """Map `ImageGenerationTool.aspect_ratio` to an OpenAI size string when provided."""
    aspect_ratio = tool.aspect_ratio
    if aspect_ratio is None:
        if tool.size is None:
            return 'auto'  # default
        if tool.size not in _OPENAI_IMAGE_SIZES:
            raise UserError(
                f'OpenAI image generation only supports `size` values: {_OPENAI_IMAGE_SIZES}. '
                f'Got: {tool.size}. Omit `size` to use the default (auto).'
            )
        return tool.size

    mapped_size = _OPENAI_ASPECT_RATIO_TO_SIZE.get(aspect_ratio)
    if mapped_size is None:
        supported = ', '.join(_OPENAI_ASPECT_RATIO_TO_SIZE)
        raise UserError(
            f'OpenAI image generation only supports `aspect_ratio` values: {supported}. Specify one of those values or omit `aspect_ratio`.'
        )
    # When aspect_ratio is set, size must be None, 'auto', or match the mapped size
    if tool.size not in (None, 'auto', mapped_size):
        raise UserError(
            '`ImageGenerationTool` cannot combine `aspect_ratio` with a conflicting `size` when using OpenAI.'
        )

    return mapped_size


def _check_azure_content_filter(e: APIStatusError, system: str, model_name: str) -> ModelResponse | None:
    """Check if the error is an Azure content filter error."""
    # Assign to Any to avoid 'dict[Unknown, Unknown]' inference in strict mode
    body_any: Any = e.body

    if system == 'azure' and e.status_code == 400 and isinstance(body_any, dict):
        try:
            error_data = _AzureErrorResponse.model_validate(body_any)

            if error_data.error.code == 'content_filter':
                provider_details: dict[str, Any] = {'finish_reason': 'content_filter'}

                if error_data.error.innererror:
                    provider_details['content_filter_result'] = (
                        error_data.error.innererror.content_filter_result.model_dump(exclude_none=True)
                    )

                return ModelResponse(
                    parts=[],  # Empty parts to trigger content filter error in agent graph
                    model_name=model_name,
                    timestamp=_utils.now_utc(),
                    provider_name=system,
                    finish_reason='content_filter',
                    provider_details=provider_details,
                )
        except ValidationError:
            pass
    return None


def _drop_sampling_params_for_reasoning(profile: OpenAIModelProfile, model_settings: OpenAIChatModelSettings) -> None:
    """Drop sampling params when reasoning is enabled on models that support it.

    Reasoning models (o-series, GPT-5, GPT-5.1+) don't support sampling parameters when
    reasoning is active. For models that support reasoning_effort='none' (GPT-5.1+),
    sampling params are allowed when reasoning is off.
    """
    if not profile.openai_supports_reasoning:
        return

    reasoning_effort = model_settings.get('openai_reasoning_effort')
    # On GPT-5.1+ models, 'none' is the default
    if profile.openai_supports_reasoning_effort_none and reasoning_effort in (None, 'none'):
        return

    if dropped := [k for k in SAMPLING_PARAMS if k in model_settings]:
        warnings.warn(
            f'Sampling parameters {dropped} are not supported when reasoning is enabled. '
            'These settings will be ignored.',
            UserWarning,
        )

    for k in SAMPLING_PARAMS:
        model_settings.pop(k, None)


def _drop_unsupported_params(profile: OpenAIModelProfile, model_settings: OpenAIChatModelSettings) -> None:
    """Drop unsupported parameters based on model profile.

    Used currently only by Cerebras
    """
    for setting in profile.openai_unsupported_model_settings:
        model_settings.pop(setting, None)


class OpenAIChatModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response.

    For Chat models, these will be included in `ModelResponse.provider_details['logprobs']`.
    For Responses models, these will be included in the response output parts `TextPart.provider_details['logprobs']`.
    """

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_service_tier: Literal['auto', 'default', 'flex', 'priority']
    """The service tier to use for the model request.

    Currently supported values are `auto`, `default`, `flex`, and `priority`.
    For more information, see [OpenAI's service tiers documentation](https://platform.openai.com/docs/api-reference/chat/object#chat/object-service_tier).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """

    openai_prompt_cache_key: str
    """Used by OpenAI to cache responses for similar requests to optimize your cache hit rates.

    See the [OpenAI Prompt Caching documentation](https://platform.openai.com/docs/guides/prompt-caching#how-it-works) for more information.
    """

    openai_prompt_cache_retention: Literal['in_memory', '24h']
    """The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours.

    See the [OpenAI Prompt Caching documentation](https://platform.openai.com/docs/guides/prompt-caching#how-it-works) for more information.
    """

    openai_continuous_usage_stats: bool
    """When True, enables continuous usage statistics in streaming responses.

    When enabled, the API returns cumulative usage data with each chunk rather than only at the end.
    This setting correctly handles the cumulative nature of these stats by using only the final
    usage values rather than summing all intermediate values.

    See [OpenAI's streaming documentation](https://platform.openai.com/docs/api-reference/chat/create#stream_options) for more information.
    """


@deprecated('Use `OpenAIChatModelSettings` instead.')
class OpenAIModelSettings(OpenAIChatModelSettings, total=False):
    """Deprecated alias for `OpenAIChatModelSettings`."""


class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_builtin_tools: Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_generate_summary: Literal['detailed', 'concise']
    """Deprecated alias for `openai_reasoning_summary`."""

    openai_reasoning_summary: Literal['detailed', 'concise', 'auto']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise`, `detailed`, or `auto`.

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses. Currently supported values are `low`,
    `medium`, and `high`.
    """

    openai_previous_response_id: Literal['auto'] | str
    """The ID of a previous response from the model to use as the starting point for a continued conversation.

    When set to `'auto'`, the request automatically uses the most recent
    `provider_response_id` from the message history and omits earlier messages.

    This enables the model to use server-side conversation state and faithfully reference previous reasoning.
    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """

    openai_include_file_search_results: bool
    """Whether to include the file search results in the response.

    Corresponds to the `file_search_call.results` value of the `include` parameter in the Responses API.
    """

    openai_include_raw_annotations: bool
    """Whether to include the raw annotations in `TextPart.provider_details`.

    When enabled, any annotations (e.g., citations from web search) will be available
    in the `provider_details['annotations']` field of text parts.
    This is opt-in since there may be overlap with native annotation support once
    added via https://github.com/pydantic/pydantic-ai/issues/3126.
    """


@dataclass(init=False)
class OpenAIChatModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIChatCompatibleProvider
        | Literal[
            'openai',
            'openai-chat',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    @overload
    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIChatCompatibleProvider
        | Literal[
            'openai',
            'openai-chat',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ) -> None: ...

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIChatCompatibleProvider
        | Literal[
            'openai',
            'openai-chat',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
                In the future, this may be inferred from the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

        if system_prompt_role is not None:
            self.profile = OpenAIModelProfile(openai_system_prompt_role=system_prompt_role).update(self.profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset({WebSearchTool})

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile.

        WebSearchTool is only supported if openai_chat_supports_web_search is True.
        """
        _profile = super().profile
        openai_profile = OpenAIModelProfile.from_profile(_profile)
        if not openai_profile.openai_chat_supports_web_search:
            new_tools = _profile.supported_builtin_tools - {WebSearchTool}
            _profile = replace(_profile, supported_builtin_tools=new_tools)
        return _profile

    @property
    @deprecated('Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
    def system_prompt_role(self) -> OpenAISystemPromptRole | None:
        return OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        # Check for WebSearchTool before base validation to provide a helpful error message
        if (
            any(isinstance(tool, WebSearchTool) for tool in model_request_parameters.builtin_tools)
            and not OpenAIModelProfile.from_profile(self.profile).openai_chat_supports_web_search
        ):
            raise UserError(
                f'WebSearchTool is not supported with `OpenAIChatModel` and model {self.model_name!r}. '
                f'Please use `OpenAIResponsesModel` instead.'
            )
        return super().prepare_request(model_settings, model_request_parameters)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._completions_create(
            messages, False, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )

        # Handle ModelResponse returned directly (for content filters)
        if isinstance(response, ModelResponse):
            return response

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
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings_cast = cast(OpenAIChatModelSettings, model_settings or {})
        response = await self._completions_create(messages, True, model_settings_cast, model_request_parameters)
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters, model_settings_cast)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | ModelResponse: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk] | ModelResponse:
        tools = self._get_tools(model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)

        profile = OpenAIModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.openai_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = await self._map_messages(messages, model_request_parameters)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        _drop_sampling_params_for_reasoning(profile, model_settings)

        _drop_unsupported_params(profile, model_settings)

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())

            # OpenAI SDK type stubs incorrectly use 'in-memory' but API requires 'in_memory', so we have to use `Any` to not hit type errors
            prompt_cache_retention: Any = model_settings.get('openai_prompt_cache_retention', OMIT)
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                stream=stream,
                stream_options=self._get_stream_options(model_settings) if stream else OMIT,
                stop=model_settings.get('stop_sequences', OMIT),
                max_completion_tokens=model_settings.get('max_tokens', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                response_format=response_format or OMIT,
                seed=model_settings.get('seed', OMIT),
                reasoning_effort=model_settings.get('openai_reasoning_effort', OMIT),
                user=model_settings.get('openai_user', OMIT),
                web_search_options=web_search_options or OMIT,
                service_tier=model_settings.get('openai_service_tier', OMIT),
                prediction=model_settings.get('openai_prediction', OMIT),
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                presence_penalty=model_settings.get('presence_penalty', OMIT),
                frequency_penalty=model_settings.get('frequency_penalty', OMIT),
                logit_bias=model_settings.get('logit_bias', OMIT),
                logprobs=model_settings.get('openai_logprobs', OMIT),
                top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
                prompt_cache_key=model_settings.get('openai_prompt_cache_key', OMIT),
                prompt_cache_retention=prompt_cache_retention,
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if model_response := _check_azure_content_filter(e, self.system, self.model_name):
                return model_response
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _validate_completion(self, response: chat.ChatCompletion) -> chat.ChatCompletion:
        """Hook that validates chat completions before processing.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom completion validations.
        """
        return chat.ChatCompletion.model_validate(response.model_dump())

    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        """Hook that response content to provider details.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        return _map_provider_details(response.choices[0])

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        # Although the OpenAI SDK claims to return a Pydantic model (`ChatCompletion`) from the chat completions function:
        # * it hasn't actually performed validation (presumably they're creating the model with `model_construct` or something?!)
        # * if the endpoint returns plain text, the return type is a string
        # Thus we validate it fully here.
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior(
                f'Invalid response from {self.system} chat completions endpoint, expected JSON data'
            )

        timestamp = _now_utc()
        if not response.created:
            response.created = int(timestamp.timestamp())

        # Workaround for local Ollama which sometimes returns a `None` finish reason.
        if response.choices and (choice := response.choices[0]) and choice.finish_reason is None:  # pyright: ignore[reportUnnecessaryComparison]
            choice.finish_reason = 'stop'

        try:
            response = self._validate_completion(response)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from {self.system} chat completions endpoint: {e}') from e

        choice = response.choices[0]

        items: list[ModelResponsePart] = []

        if thinking_parts := self._process_thinking(choice.message):
            items.extend(thinking_parts)

        if choice.message.content:
            items.extend(
                (replace(part, id='content', provider_name=self.system) if isinstance(part, ThinkingPart) else part)
                for part in split_content_into_text_and_thinking(choice.message.content, self.profile.thinking_tags)
            )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                if isinstance(c, ChatCompletionMessageFunctionToolCall):
                    part = ToolCallPart(c.function.name, c.function.arguments, tool_call_id=c.id)
                elif isinstance(c, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                    # NOTE: Custom tool calls are not supported.
                    # See <https://github.com/pydantic/pydantic-ai/issues/2513> for more details.
                    raise RuntimeError('Custom tool calls are not supported')
                else:
                    assert_never(c)
                part.tool_call_id = _guard_tool_call_id(part)
                items.append(part)

        provider_details = self._process_provider_details(response)
        if response.created:  # pragma: no branch
            if provider_details is None:
                provider_details = {}
            provider_details['timestamp'] = number_to_datetime(response.created)

        return ModelResponse(
            parts=items,
            usage=self._map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_details=provider_details or None,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=self._map_finish_reason(choice.finish_reason),
        )

    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        """Hook that maps reasoning tokens to thinking parts.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        profile = OpenAIModelProfile.from_profile(self.profile)
        custom_field = profile.openai_chat_thinking_field
        items: list[ThinkingPart] = []

        # Prefer the configured custom reasoning field, if present in profile.
        # Fall back to built-in fields if no custom field result was found.

        # The `reasoning_content` field is typically present in DeepSeek and Moonshot models.
        # https://api-docs.deepseek.com/guides/reasoning_model

        # The `reasoning` field is typically present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        for field_name in (custom_field, 'reasoning', 'reasoning_content'):
            if not field_name:
                continue
            reasoning: str | None = getattr(message, field_name, None)
            if reasoning:  # pragma: no branch
                items.append(ThinkingPart(id=field_name, content=reasoning, provider_name=self.system))
                return items

        return items or None

    async def _process_streamed_response(
        self,
        response: AsyncStream[ChatCompletionChunk],
        model_request_parameters: ModelRequestParameters,
        model_settings: OpenAIChatModelSettings | None = None,
    ) -> OpenAIStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        # When using Azure OpenAI and a content filter is enabled, the first chunk will contain a `''` model name,
        # so we set it from a later chunk in `OpenAIChatStreamedResponse`.
        model_name = first_chunk.model or self.model_name

        return self._streamed_response_cls(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
            _provider_timestamp=number_to_datetime(first_chunk.created) if first_chunk.created else None,
            _model_settings=model_settings,
        )

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Returns the `StreamedResponse` type that will be used for streamed responses.

        This method may be overridden by subclasses of `OpenAIChatModel` to provide their own `StreamedResponse` type.
        """
        return OpenAIStreamedResponse

    def _map_usage(self, response: chat.ChatCompletion) -> usage.RequestUsage:
        return _map_usage(response, self._provider.name, self._provider.base_url, self.model_name)

    def _get_stream_options(self, model_settings: OpenAIChatModelSettings) -> chat.ChatCompletionStreamOptionsParam:
        """Build stream_options for the API request.

        Returns a dict with include_usage=True and optionally continuous_usage_stats if configured.
        """
        options: dict[str, bool] = {'include_usage': True}
        if model_settings.get('openai_continuous_usage_stats'):
            options['continuous_usage_stats'] = True
        return cast(chat.ChatCompletionStreamOptionsParam, options)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_web_search_options(self, model_request_parameters: ModelRequestParameters) -> WebSearchOptions | None:
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):  # pragma: no branch
                if tool.user_location:
                    return WebSearchOptions(
                        search_context_size=tool.search_context_size,
                        user_location=WebSearchOptionsUserLocation(
                            type='approximate',
                            approximate=WebSearchOptionsUserLocationApproximate(**tool.user_location),
                        ),
                    )
                return WebSearchOptions(search_context_size=tool.search_context_size)
        return None

    @dataclass
    class _MapModelResponseContext:
        """Context object for mapping a `ModelResponse` to OpenAI chat completion parameters.

        This class is designed to be subclassed to add new fields for custom logic,
        collecting various parts of the model response (like text and tool calls)
        to form a single assistant message.
        """

        _model: OpenAIChatModel

        texts: list[str] = field(default_factory=list[str])
        thinkings: list[str] = field(default_factory=list[str])
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = field(
            default_factory=list[ChatCompletionMessageFunctionToolCallParam]
        )

        def map_assistant_message(self, message: ModelResponse) -> chat.ChatCompletionAssistantMessageParam:
            for item in message.parts:
                if isinstance(item, TextPart):
                    self._map_response_text_part(item)
                elif isinstance(item, ThinkingPart):
                    self._map_response_thinking_part(item)
                elif isinstance(item, ToolCallPart):
                    self._map_response_tool_call_part(item)
                elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
                    self._map_response_builtin_part(item)
                elif isinstance(item, FilePart):  # pragma: no cover
                    self._map_response_file_part(item)
                else:
                    assert_never(item)
            return self._into_message_param()

        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam:
            """Converts the collected texts and tool calls into a single OpenAI `ChatCompletionAssistantMessageParam`.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for how collected parts are transformed into the final message parameter.

            Returns:
                An OpenAI `ChatCompletionAssistantMessageParam` object representing the assistant's response.
            """
            profile = OpenAIModelProfile.from_profile(self._model.profile)
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            # Note: model responses from this model should only have one text item, so the following
            # shouldn't merge multiple texts into one unless you switch models between runs:
            if profile.openai_chat_send_back_thinking_parts == 'field' and self.thinkings:
                field = profile.openai_chat_thinking_field
                if field:  # pragma: no branch (handled by profile validation)
                    message_param[field] = '\n\n'.join(self.thinkings)
            if self.texts:
                message_param['content'] = '\n\n'.join(self.texts)
            else:
                message_param['content'] = None
            if self.tool_calls:
                message_param['tool_calls'] = self.tool_calls
            return message_param

        def _map_response_text_part(self, item: TextPart) -> None:
            """Maps a `TextPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling text parts.
            """
            self.texts.append(item.content)

        def _map_response_thinking_part(self, item: ThinkingPart) -> None:
            """Maps a `ThinkingPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling thinking parts.
            """
            profile = OpenAIModelProfile.from_profile(self._model.profile)
            include_method = profile.openai_chat_send_back_thinking_parts
            if include_method == 'tags':
                start_tag, end_tag = self._model.profile.thinking_tags
                self.texts.append('\n'.join([start_tag, item.content, end_tag]))
            elif include_method == 'field':
                self.thinkings.append(item.content)

        def _map_response_tool_call_part(self, item: ToolCallPart) -> None:
            """Maps a `ToolCallPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling tool call parts.
            """
            self.tool_calls.append(self._model._map_tool_call(item))

        def _map_response_builtin_part(self, item: BuiltinToolCallPart | BuiltinToolReturnPart) -> None:
            """Maps a built-in tool call or return part to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling built-in tool parts.
            """
            # OpenAI doesn't return built-in tool calls
            pass

        def _map_response_file_part(self, item: FilePart) -> None:
            """Maps a `FilePart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling file parts.
            """
            # Files generated by models are not sent back to models that don't themselves generate files.
            pass

    def _map_model_response(self, message: ModelResponse) -> chat.ChatCompletionMessageParam:
        """Hook that determines how `ModelResponse` is mapped into `ChatCompletionMessageParam` objects before sending.

        Subclasses of `OpenAIChatModel` may override this method to provide their own mapping logic.
        """
        return self._MapModelResponseContext(self).map_assistant_message(message)

    def _map_finish_reason(
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']
    ) -> FinishReason | None:
        """Hooks that maps a finish reason key to a [FinishReason][pydantic_ai.messages.FinishReason].

        This method may be overridden by subclasses of `OpenAIChatModel` to accommodate custom keys.
        """
        return _CHAT_FINISH_REASON_MAP.get(key)

    async def _map_messages(
        self, messages: Sequence[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                openai_messages.append(self._map_model_response(message))
            else:
                assert_never(message)
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_prompt_count = sum(1 for m in openai_messages if m.get('role') == 'system')
            openai_messages.insert(
                system_prompt_count, chat.ChatCompletionSystemMessageParam(content=instructions, role='system')
            )
        return openai_messages

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ChatCompletionMessageFunctionToolCallParam:
        return ChatCompletionMessageFunctionToolCallParam(
            id=_guard_tool_call_id(t=t),
            type='function',
            function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> chat.completion_create_params.ResponseFormat:
        response_format_param: chat.completion_create_params.ResponseFormatJSONSchema = {  # pyright: ignore[reportPrivateImportUsage]
            'type': 'json_schema',
            'json_schema': {'name': o.name or DEFAULT_OUTPUT_TOOL_NAME, 'schema': o.json_schema},
        }
        if o.description:
            response_format_param['json_schema']['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['json_schema']['strict'] = o.strict
        return response_format_param

    def _map_tool_definition(self, f: ToolDefinition) -> chat.ChatCompletionToolParam:
        tool_param: chat.ChatCompletionToolParam = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:
            tool_param['function']['strict'] = f.strict
        return tool_param

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                system_prompt_role = OpenAIModelProfile.from_profile(self.profile).openai_system_prompt_role
                if system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif system_prompt_role == 'user':
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
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

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:  # noqa: C901
        profile = OpenAIModelProfile.from_profile(self.profile)
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
                elif isinstance(item, ImageUrl):
                    image_url: ImageURL = {'url': item.url}
                    if metadata := item.vendor_metadata:
                        image_url['detail'] = metadata.get('detail', 'auto')
                    if item.force_download:
                        image_content = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url['url'] = image_content['data']
                    content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                elif isinstance(item, BinaryContent):
                    if self._is_text_like_media_type(item.media_type):
                        # Inline text-like binary content as a text block
                        content.append(
                            self._inline_text_file_part(
                                item.data.decode('utf-8'),
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    elif item.is_image:
                        image_url = ImageURL(url=item.data_uri)
                        if metadata := item.vendor_metadata:
                            image_url['detail'] = metadata.get('detail', 'auto')
                        content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
                    elif item.is_audio:
                        assert item.format in ('wav', 'mp3')
                        if profile.openai_chat_audio_input_encoding == 'uri':
                            audio = InputAudio(data=item.data_uri, format=item.format)
                        else:
                            audio = InputAudio(data=item.base64, format=item.format)
                        content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                    elif item.is_document:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=item.data_uri,
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, AudioUrl):
                    data_format = 'base64_uri' if profile.openai_chat_audio_input_encoding == 'uri' else 'base64'
                    downloaded_item = await download_item(item, data_format=data_format, type_format='extension')
                    assert downloaded_item['data_type'] in (
                        'wav',
                        'mp3',
                    ), f'Unsupported audio format: {downloaded_item["data_type"]}'
                    audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
                    content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
                elif isinstance(item, DocumentUrl):
                    # OpenAI Chat API's FileFile only supports base64-encoded data, not URLs.
                    # Some providers (e.g., OpenRouter) support URLs via the profile flag.
                    if not item.force_download and profile.openai_chat_supports_file_urls:
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=item.url,
                                    filename=f'filename.{item.format}',
                                ),
                                type='file',
                            )
                        )
                    elif self._is_text_like_media_type(item.media_type):
                        downloaded_text = await download_item(item, data_format='text')
                        content.append(
                            self._inline_text_file_part(
                                downloaded_text['data'],
                                media_type=item.media_type,
                                identifier=item.identifier,
                            )
                        )
                    else:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            File(
                                file=FileFile(
                                    file_data=downloaded_item['data'],
                                    filename=f'filename.{downloaded_item["data_type"]}',
                                ),
                                type='file',
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    @staticmethod
    def _is_text_like_media_type(media_type: str) -> bool:
        return (
            media_type.startswith('text/')
            or media_type == 'application/json'
            or media_type.endswith('+json')
            or media_type == 'application/xml'
            or media_type.endswith('+xml')
            or media_type in ('application/x-yaml', 'application/yaml')
        )

    @staticmethod
    def _inline_text_file_part(text: str, *, media_type: str, identifier: str) -> ChatCompletionContentPartTextParam:
        text = '\n'.join(
            [
                f'-----BEGIN FILE id="{identifier}" type="{media_type}"-----',
                text,
                f'-----END FILE id="{identifier}"-----',
            ]
        )
        return ChatCompletionContentPartTextParam(text=text, type='text')


@deprecated(
    '`OpenAIModel` was renamed to `OpenAIChatModel` to clearly distinguish it from `OpenAIResponsesModel` which '
    "uses OpenAI's newer Responses API. Use that unless you're using an OpenAI Chat Completions-compatible API, or "
    "require a feature that the Responses API doesn't support yet like audio."
)
@dataclass(init=False)
class OpenAIModel(OpenAIChatModel):
    """Deprecated alias for `OpenAIChatModel`."""


responses_output_text_annotations_ta = TypeAdapter(list[responses.response_output_text.Annotation])


@dataclass(init=False)
class OpenAIResponsesModel(Model):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    client: AsyncOpenAI = field(repr=False)

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIResponsesCompatibleProvider
        | Literal[
            'openai',
            'gateway',
        ]
        | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractBuiltinTool]]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset({WebSearchTool, CodeExecutionTool, FileSearchTool, MCPServerTool, ImageGenerationTool})

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, False, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )

        # Handle ModelResponse
        if isinstance(response, ModelResponse):
            return response

        return self._process_response(
            response, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        response = await self._responses_create(
            messages, True, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(
                response, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
            )

    def _process_response(  # noqa: C901
        self,
        response: responses.Response,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        for item in response.output:
            if isinstance(item, responses.ResponseReasoningItem):
                signature = item.encrypted_content
                # Handle raw CoT content from gpt-oss models
                provider_details: dict[str, Any] = {}
                raw_content: list[str] | None = [c.text for c in item.content] if item.content else None
                if raw_content:
                    provider_details['raw_content'] = raw_content

                if item.summary:
                    for summary in item.summary:
                        # We use the same id for all summaries so that we can merge them on the round trip.
                        items.append(
                            ThinkingPart(
                                content=summary.text,
                                id=item.id,
                                signature=signature,
                                provider_name=self.system,
                                provider_details=provider_details or None,
                            )
                        )
                        # We only need to store the signature and raw_content once.
                        signature = None
                        provider_details = {}
                elif signature or provider_details:
                    items.append(
                        ThinkingPart(
                            content='',
                            id=item.id,
                            signature=signature,
                            provider_name=self.system,
                            provider_details=provider_details or None,
                        )
                    )
            elif isinstance(item, responses.ResponseOutputMessage):
                for content in item.content:
                    if isinstance(content, responses.ResponseOutputText):  # pragma: no branch
                        part_provider_details: dict[str, Any] | None = None
                        if content.logprobs:
                            part_provider_details = {'logprobs': _map_logprobs(content.logprobs)}
                        if model_settings.get('openai_include_raw_annotations') and content.annotations:
                            part_provider_details = part_provider_details or {}
                            part_provider_details['annotations'] = responses_output_text_annotations_ta.dump_python(
                                list(content.annotations), warnings=False
                            )
                        items.append(
                            TextPart(
                                content.text,
                                id=item.id,
                                provider_name=self.system,
                                provider_details=part_provider_details,
                            )
                        )
            elif isinstance(item, responses.ResponseFunctionToolCall):
                items.append(
                    ToolCallPart(
                        item.name, item.arguments, tool_call_id=item.call_id, id=item.id, provider_name=self.system
                    )
                )
            elif isinstance(item, responses.ResponseCodeInterpreterToolCall):
                call_part, return_part, file_parts = _map_code_interpreter_tool_call(item, self.system)
                items.append(call_part)
                if file_parts:
                    items.extend(file_parts)
                items.append(return_part)
            elif isinstance(item, responses.ResponseFunctionWebSearch):
                call_part, return_part = _map_web_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.ImageGenerationCall):
                call_part, return_part, file_part = _map_image_generation_tool_call(item, self.system)
                items.append(call_part)
                if file_part:  # pragma: no branch
                    items.append(file_part)
                items.append(return_part)
            elif isinstance(item, responses.ResponseComputerToolCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the ComputerUse built-in tool
                pass
            elif isinstance(item, responses.ResponseCustomToolCall):  # pragma: no cover
                # Support is being implemented in https://github.com/pydantic/pydantic-ai/pull/2572
                pass
            elif isinstance(item, responses.response_output_item.LocalShellCall):  # pragma: no cover
                # Pydantic AI doesn't yet support the `codex-mini-latest` LocalShell built-in tool
                pass
            elif isinstance(item, responses.ResponseFileSearchToolCall):
                call_part, return_part = _map_file_search_tool_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpCall):
                call_part, return_part = _map_mcp_call(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpListTools):
                call_part, return_part = _map_mcp_list_tools(item, self.system)
                items.append(call_part)
                items.append(return_part)
            elif isinstance(item, responses.response_output_item.McpApprovalRequest):  # pragma: no cover
                # Pydantic AI doesn't yet support McpApprovalRequest (explicit tool usage approval)
                pass

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] = {}
        raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
        if raw_finish_reason:
            provider_details['finish_reason'] = raw_finish_reason
            finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)
        if response.created_at:  # pragma: no branch
            provider_details['timestamp'] = number_to_datetime(response.created_at)

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self.model_name),
            model_name=response.model,
            provider_response_id=response.id,
            timestamp=_now_utc(),
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=finish_reason,
            provider_details=provider_details or None,
        )

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, responses.ResponseCreatedEvent)
        return OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.response.model,
            _model_settings=model_settings,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
            _provider_timestamp=number_to_datetime(first_chunk.response.created_at)
            if first_chunk.response.created_at
            else None,
        )

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[False],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response: ...

    @overload
    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: Literal[True],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[responses.ResponseStreamEvent]: ...

    async def _responses_create(  # noqa: C901
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ModelResponse:
        tools = (
            self._get_builtin_tools(model_request_parameters)
            + list(model_settings.get('openai_builtin_tools', []))
            + self._get_tools(model_request_parameters)
        )
        profile = OpenAIModelProfile.from_profile(self.profile)
        if not tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not model_request_parameters.allow_text_output and profile.openai_supports_tool_choice_required:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        previous_response_id = model_settings.get('openai_previous_response_id')
        if previous_response_id == 'auto':
            previous_response_id, messages = self._get_previous_response_id_and_new_messages(messages)

        instructions, openai_messages = await self._map_messages(messages, model_settings, model_request_parameters)
        reasoning = self._get_reasoning(model_settings)

        text: responses.ResponseTextConfigParam | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {'format': self._map_json_schema(output_object)}
        elif (
            model_request_parameters.output_mode == 'prompted' and self.profile.supports_json_object_output
        ):  # pragma: no branch
            text = {'format': {'type': 'json_object'}}

            # Without this trick, we'd hit this error:
            # > Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.
            # Apparently they're only checking input messages for "JSON", not instructions.
            assert isinstance(instructions, str)
            system_prompt_count = sum(1 for m in openai_messages if m.get('role') == 'system')
            openai_messages.insert(
                system_prompt_count, responses.EasyInputMessageParam(role='system', content=instructions)
            )
            instructions = OMIT

        if verbosity := model_settings.get('openai_text_verbosity'):
            text = text or {}
            text['verbosity'] = verbosity

        _drop_sampling_params_for_reasoning(profile, model_settings)

        _drop_unsupported_params(profile, model_settings)

        include: list[responses.ResponseIncludable] = []
        if profile.openai_supports_encrypted_reasoning_content:
            include.append('reasoning.encrypted_content')
        if model_settings.get('openai_include_code_execution_outputs'):
            include.append('code_interpreter_call.outputs')
        if model_settings.get('openai_include_web_search_sources'):
            include.append('web_search_call.action.sources')
        if model_settings.get('openai_include_file_search_results'):
            include.append('file_search_call.results')
        if model_settings.get('openai_logprobs'):
            include.append('message.output_text.logprobs')

        # When there are no input messages and we're not reusing a previous response,
        # the OpenAI API will reject a request without any input,
        # even if there are instructions.
        # To avoid this provide an explicit empty user message.
        if not openai_messages and not previous_response_id:
            openai_messages.append(
                responses.EasyInputMessageParam(
                    role='user',
                    content='',
                )
            )

        try:
            extra_headers = model_settings.get('extra_headers', {})
            extra_headers.setdefault('User-Agent', get_user_agent())
            # OpenAI SDK type stubs incorrectly use 'in-memory' but API requires 'in_memory', so we have to use `Any` to not hit type errors
            prompt_cache_retention: Any = model_settings.get('openai_prompt_cache_retention', OMIT)
            return await self.client.responses.create(
                input=openai_messages,
                model=self.model_name,
                instructions=instructions,
                parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT),
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                max_output_tokens=model_settings.get('max_tokens', OMIT),
                stream=stream,
                temperature=model_settings.get('temperature', OMIT),
                top_p=model_settings.get('top_p', OMIT),
                truncation=model_settings.get('openai_truncation', OMIT),
                timeout=model_settings.get('timeout', NOT_GIVEN),
                service_tier=model_settings.get('openai_service_tier', OMIT),
                previous_response_id=previous_response_id or OMIT,
                top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
                reasoning=reasoning,
                user=model_settings.get('openai_user', OMIT),
                text=text or OMIT,
                include=include or OMIT,
                prompt_cache_key=model_settings.get('openai_prompt_cache_key', OMIT),
                prompt_cache_retention=prompt_cache_retention,
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )
        except APIStatusError as e:
            if model_response := _check_azure_content_filter(e, self.system, self.model_name):
                return model_response

            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

    def _get_reasoning(self, model_settings: OpenAIResponsesModelSettings) -> Reasoning | Omit:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_summary = model_settings.get('openai_reasoning_summary', None)
        reasoning_generate_summary = model_settings.get('openai_reasoning_generate_summary', None)

        if reasoning_summary and reasoning_generate_summary:  # pragma: no cover
            raise ValueError('`openai_reasoning_summary` and `openai_reasoning_generate_summary` cannot both be set.')

        if reasoning_generate_summary is not None:  # pragma: no cover
            warnings.warn(
                '`openai_reasoning_generate_summary` is deprecated, use `openai_reasoning_summary` instead',
                DeprecationWarning,
            )
            reasoning_summary = reasoning_generate_summary

        reasoning: Reasoning = {}
        if reasoning_effort:
            reasoning['effort'] = reasoning_effort
        if reasoning_summary:
            reasoning['summary'] = reasoning_summary
        return reasoning or OMIT

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.FunctionToolParam]:
        return [self._map_tool_definition(r) for r in model_request_parameters.tool_defs.values()]

    def _get_builtin_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        tools: list[responses.ToolParam] = []
        has_image_generating_tool = False
        for tool in model_request_parameters.builtin_tools:
            if isinstance(tool, WebSearchTool):
                web_search_tool = responses.WebSearchToolParam(
                    type='web_search', search_context_size=tool.search_context_size
                )
                if tool.user_location:
                    web_search_tool['user_location'] = responses.web_search_tool_param.UserLocation(
                        type='approximate', **tool.user_location
                    )
                if tool.allowed_domains:
                    web_search_tool['filters'] = responses.web_search_tool_param.Filters(
                        allowed_domains=tool.allowed_domains
                    )
                tools.append(web_search_tool)
            elif isinstance(tool, FileSearchTool):
                file_search_tool = cast(
                    responses.FileSearchToolParam,
                    {'type': 'file_search', 'vector_store_ids': list(tool.file_store_ids)},
                )
                tools.append(file_search_tool)
            elif isinstance(tool, CodeExecutionTool):
                has_image_generating_tool = True
                tools.append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
            elif isinstance(tool, MCPServerTool):
                mcp_tool = responses.tool_param.Mcp(
                    type='mcp',
                    server_label=tool.id,
                    require_approval='never',
                )

                if tool.authorization_token:  # pragma: no branch
                    mcp_tool['authorization'] = tool.authorization_token

                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_tool['allowed_tools'] = tool.allowed_tools

                if tool.description:  # pragma: no branch
                    mcp_tool['server_description'] = tool.description

                if tool.headers:  # pragma: no branch
                    mcp_tool['headers'] = tool.headers

                if tool.url.startswith(MCP_SERVER_TOOL_CONNECTOR_URI_SCHEME + ':'):
                    _, connector_id = tool.url.split(':', maxsplit=1)
                    mcp_tool['connector_id'] = connector_id  # pyright: ignore[reportGeneralTypeIssues]
                else:
                    mcp_tool['server_url'] = tool.url

                tools.append(mcp_tool)
            elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                has_image_generating_tool = True
                size = _resolve_openai_image_generation_size(tool)
                output_compression = tool.output_compression if tool.output_compression is not None else 100
                tools.append(
                    responses.tool_param.ImageGeneration(
                        type='image_generation',
                        background=tool.background,
                        input_fidelity=tool.input_fidelity,
                        moderation=tool.moderation,
                        output_compression=output_compression,
                        output_format=tool.output_format or 'png',
                        partial_images=tool.partial_images,
                        quality=tool.quality,
                        size=size,
                    )
                )
            else:
                raise UserError(  # pragma: no cover
                    f'`{tool.__class__.__name__}` is not supported by `OpenAIResponsesModel`. If it should be, please file an issue.'
                )

        if model_request_parameters.allow_image_output and not has_image_generating_tool:
            tools.append({'type': 'image_generation'})
        return tools

    def _map_tool_definition(self, f: ToolDefinition) -> responses.FunctionToolParam:
        return {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            'strict': bool(
                f.strict and OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition
            ),
        }

    def _get_previous_response_id_and_new_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        # When `openai_previous_response_id` is set to 'auto', the most recent
        # `provider_response_id` from the message history is selected and all
        # earlier messages are omitted. This allows the OpenAI SDK to reuse
        # server-side history for efficiency. The returned tuple contains the
        # `previous_response_id` (if found) and the trimmed list of messages.
        previous_response_id = None
        trimmed_messages: list[ModelMessage] = []
        for m in reversed(messages):
            if isinstance(m, ModelResponse) and m.provider_name == self.system:
                previous_response_id = m.provider_response_id
                break
            else:
                trimmed_messages.append(m)

        if previous_response_id and trimmed_messages:
            return previous_response_id, list(reversed(trimmed_messages))
        else:
            return None, messages

    async def _map_messages(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[str | Omit, list[responses.ResponseInputItemParam]]:
        """Maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam` i.e. the OpenAI Responses API input format.

        For `ThinkingParts`, this method:
        - Sends `signature` back as `encrypted_content` (for official OpenAI reasoning)
        - Sends `content` back as `summary` text
        - Sends `provider_details['raw_content']` back as `content` items (for gpt-oss raw CoT)

        Raw CoT is sent back to improve model performance in multi-turn conversations.
        """
        profile = OpenAIModelProfile.from_profile(self.profile)
        send_item_ids = model_settings.get(
            'openai_send_reasoning_ids', profile.openai_supports_encrypted_reasoning_content
        )

        openai_messages: list[responses.ResponseInputItemParam] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        openai_messages.append(responses.EasyInputMessageParam(role='system', content=part.content))
                    elif isinstance(part, UserPromptPart):
                        openai_messages.append(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        call_id = _guard_tool_call_id(t=part)
                        call_id, _ = _split_combined_tool_call_id(call_id)
                        item = FunctionCallOutput(
                            type='function_call_output',
                            call_id=call_id,
                            output=part.model_response_str(),
                        )
                        openai_messages.append(item)
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            openai_messages.append(
                                Message(role='user', content=[{'type': 'input_text', 'text': part.model_response()}])
                            )
                        else:
                            call_id = _guard_tool_call_id(t=part)
                            call_id, _ = _split_combined_tool_call_id(call_id)
                            item = FunctionCallOutput(
                                type='function_call_output',
                                call_id=call_id,
                                output=part.model_response(),
                            )
                            openai_messages.append(item)
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                message_item: responses.ResponseOutputMessageParam | None = None
                reasoning_item: responses.ResponseReasoningItemParam | None = None
                web_search_item: responses.ResponseFunctionWebSearchParam | None = None
                file_search_item: responses.ResponseFileSearchToolCallParam | None = None
                code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
                for item in message.parts:
                    should_send_item_id = send_item_ids and (
                        item.provider_name == self.system
                        or (item.provider_name is None and message.provider_name == self.system)
                    )

                    if isinstance(item, TextPart):
                        if item.id and should_send_item_id:
                            if message_item is None or message_item['id'] != item.id:  # pragma: no branch
                                message_item = responses.ResponseOutputMessageParam(
                                    role='assistant',
                                    id=item.id,
                                    content=[],
                                    type='message',
                                    status='completed',
                                )
                                openai_messages.append(message_item)

                            message_item['content'] = [
                                *message_item['content'],
                                responses.ResponseOutputTextParam(
                                    text=item.content, type='output_text', annotations=[]
                                ),
                            ]
                        else:
                            openai_messages.append(
                                responses.EasyInputMessageParam(role='assistant', content=item.content)
                            )
                    elif isinstance(item, ToolCallPart):
                        call_id = _guard_tool_call_id(t=item)
                        call_id, id = _split_combined_tool_call_id(call_id)
                        id = id or item.id

                        param = responses.ResponseFunctionToolCallParam(
                            name=item.tool_name,
                            arguments=item.args_as_json_str(),
                            call_id=call_id,
                            type='function_call',
                        )
                        if profile.openai_responses_requires_function_call_status_none:
                            param['status'] = None  # type: ignore[reportGeneralTypeIssues]
                        if id and should_send_item_id:  # pragma: no branch
                            param['id'] = id
                        openai_messages.append(param)
                    elif isinstance(item, BuiltinToolCallPart):
                        if should_send_item_id:  # pragma: no branch
                            if (
                                item.tool_name == CodeExecutionTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                                and (container_id := args.get('container_id'))
                            ):
                                code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                                    id=item.tool_call_id,
                                    code=args.get('code'),
                                    container_id=container_id,
                                    outputs=None,  # These can be read server-side
                                    status='completed',
                                    type='code_interpreter_call',
                                )
                                openai_messages.append(code_interpreter_item)
                            elif (
                                item.tool_name == WebSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
                                args = {k: v for k, v in args.items() if v is not None}
                                web_search_item = responses.ResponseFunctionWebSearchParam(
                                    id=item.tool_call_id,
                                    action=cast(responses.response_function_web_search_param.Action, args),
                                    status='completed',
                                    type='web_search_call',
                                )
                                openai_messages.append(web_search_item)
                            elif (  # pragma: no cover
                                item.tool_name == FileSearchTool.kind
                                and item.tool_call_id
                                and (args := item.args_as_dict())
                            ):
                                file_search_item = cast(
                                    responses.ResponseFileSearchToolCallParam,
                                    {
                                        'id': item.tool_call_id,
                                        'queries': args.get('queries', []),
                                        'status': 'completed',
                                        'type': 'file_search_call',
                                    },
                                )
                                openai_messages.append(file_search_item)
                            elif item.tool_name == ImageGenerationTool.kind and item.tool_call_id:
                                # The cast is necessary because of https://github.com/openai/openai-python/issues/2648
                                image_generation_item = cast(
                                    responses.response_input_item_param.ImageGenerationCall,
                                    {
                                        'id': item.tool_call_id,
                                        'type': 'image_generation_call',
                                    },
                                )
                                openai_messages.append(image_generation_item)
                            elif (  # pragma: no branch
                                item.tool_name.startswith(MCPServerTool.kind)
                                and item.tool_call_id
                                and (server_id := item.tool_name.split(':', 1)[1])
                                and (args := item.args_as_dict())
                                and (action := args.get('action'))
                            ):
                                if action == 'list_tools':
                                    mcp_list_tools_item = responses.response_input_item_param.McpListTools(
                                        id=item.tool_call_id,
                                        type='mcp_list_tools',
                                        server_label=server_id,
                                        tools=[],  # These can be read server-side
                                    )
                                    openai_messages.append(mcp_list_tools_item)
                                elif (  # pragma: no branch
                                    action == 'call_tool'
                                    and (tool_name := args.get('tool_name'))
                                    and (tool_args := args.get('tool_args'))
                                ):
                                    mcp_call_item = responses.response_input_item_param.McpCall(
                                        id=item.tool_call_id,
                                        server_label=server_id,
                                        name=tool_name,
                                        arguments=to_json(tool_args).decode(),
                                        error=None,  # These can be read server-side
                                        output=None,  # These can be read server-side
                                        type='mcp_call',
                                    )
                                    openai_messages.append(mcp_call_item)

                    elif isinstance(item, BuiltinToolReturnPart):
                        if should_send_item_id:  # pragma: no branch
                            content_is_dict = isinstance(item.content, dict)
                            status = cast(dict[str, Any], item.content).get('status') if content_is_dict else None
                            kind_to_item = {
                                CodeExecutionTool.kind: code_interpreter_item,
                                WebSearchTool.kind: web_search_item,
                                FileSearchTool.kind: file_search_item,
                            }
                            if status and (builtin_item := kind_to_item.get(item.tool_name)) is not None:
                                builtin_item['status'] = status
                            elif item.tool_name == ImageGenerationTool.kind:
                                # Image generation result does not need to be sent back, just the `id` off of `BuiltinToolCallPart`.
                                pass
                            elif item.tool_name.startswith(MCPServerTool.kind):  # pragma: no branch
                                # MCP call result does not need to be sent back, just the fields off of `BuiltinToolCallPart`.
                                pass
                    elif isinstance(item, FilePart):
                        # This was generated by the `ImageGenerationTool` or `CodeExecutionTool`,
                        # and does not need to be sent back separately from the corresponding `BuiltinToolReturnPart`.
                        # If `send_item_ids` is false, we won't send the `BuiltinToolReturnPart`, but OpenAI does not have a type for files from the assistant.
                        pass
                    elif isinstance(item, ThinkingPart):
                        # Get raw CoT content from provider_details if present and from this provider
                        raw_content: list[str] | None = None
                        if item.provider_name == self.system:
                            raw_content = (item.provider_details or {}).get('raw_content')

                        if item.id and (should_send_item_id or raw_content):
                            signature: str | None = None
                            if (
                                item.signature
                                and item.provider_name == self.system
                                and profile.openai_supports_encrypted_reasoning_content
                            ):
                                signature = item.signature

                            if (reasoning_item is None or reasoning_item['id'] != item.id) and (
                                signature or item.content or raw_content
                            ):  # pragma: no branch
                                reasoning_item = responses.ResponseReasoningItemParam(
                                    id=item.id,
                                    summary=[],
                                    encrypted_content=signature,
                                    type='reasoning',
                                )
                                openai_messages.append(reasoning_item)

                            if item.content:
                                # The check above guarantees that `reasoning_item` is not None
                                assert reasoning_item is not None
                                reasoning_item['summary'] = [
                                    *reasoning_item['summary'],
                                    ReasoningSummary(text=item.content, type='summary_text'),
                                ]

                            if raw_content:
                                # Send raw CoT back
                                assert reasoning_item is not None
                                reasoning_item['content'] = [
                                    ReasoningContent(text=text, type='reasoning_text') for text in raw_content
                                ]
                        else:
                            start_tag, end_tag = profile.thinking_tags
                            openai_messages.append(
                                responses.EasyInputMessageParam(
                                    role='assistant', content='\n'.join([start_tag, item.content, end_tag])
                                )
                            )
                    else:
                        assert_never(item)
            else:
                assert_never(message)
        instructions = self._get_instructions(messages, model_request_parameters) or OMIT
        return instructions, openai_messages

    def _map_json_schema(self, o: OutputObjectDefinition) -> responses.ResponseFormatTextJSONSchemaConfigParam:
        response_format_param: responses.ResponseFormatTextJSONSchemaConfigParam = {
            'type': 'json_schema',
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            response_format_param['description'] = o.description
        if OpenAIModelProfile.from_profile(self.profile).openai_supports_strict_tool_definition:  # pragma: no branch
            response_format_param['strict'] = o.strict
        return response_format_param

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> responses.EasyInputMessageParam:  # noqa: C901
        content: str | list[responses.ResponseInputContentParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(responses.ResponseInputTextParam(text=item, type='input_text'))
                elif isinstance(item, BinaryContent):
                    if item.is_image:
                        detail: Literal['auto', 'low', 'high'] = 'auto'
                        if metadata := item.vendor_metadata:
                            detail = cast(
                                Literal['auto', 'low', 'high'],
                                metadata.get('detail', 'auto'),
                            )
                        content.append(
                            responses.ResponseInputImageParam(
                                image_url=item.data_uri,
                                type='input_image',
                                detail=detail,
                            )
                        )
                    elif item.is_document:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=item.data_uri,
                                # NOTE: Type wise it's not necessary to include the filename, but it's required by the
                                # API itself. If we add empty string, the server sends a 500 error - which OpenAI needs
                                # to fix. In any case, we add a placeholder name.
                                filename=f'filename.{item.format}',
                            )
                        )
                    elif item.is_audio:
                        raise NotImplementedError('Audio as binary content is not supported for OpenAI Responses API.')
                    else:  # pragma: no cover
                        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
                elif isinstance(item, ImageUrl):
                    detail: Literal['auto', 'low', 'high'] = 'auto'
                    image_url = item.url
                    if metadata := item.vendor_metadata:
                        detail = cast(Literal['auto', 'low', 'high'], metadata.get('detail', 'auto'))
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        image_url = downloaded_item['data']

                    content.append(
                        responses.ResponseInputImageParam(
                            image_url=image_url,
                            type='input_image',
                            detail=detail,
                        )
                    )
                elif isinstance(item, AudioUrl | DocumentUrl):
                    if item.force_download:
                        downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_data=downloaded_item['data'],
                                filename=f'filename.{downloaded_item["data_type"]}',
                            )
                        )
                    else:
                        content.append(
                            responses.ResponseInputFileParam(
                                type='input_file',
                                file_url=item.url,
                            )
                        )
                elif isinstance(item, VideoUrl):  # pragma: no cover
                    raise NotImplementedError('VideoUrl is not supported for OpenAI.')
                elif isinstance(item, CachePoint):
                    # OpenAI doesn't support prompt caching via CachePoint, so we filter it out
                    pass
                else:
                    assert_never(item)
        return responses.EasyInputMessageParam(role='user', content=content)


@dataclass
class OpenAIStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI models."""

    _model_name: OpenAIModelName
    _model_profile: ModelProfile
    _response: AsyncIterable[ChatCompletionChunk]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_now_utc)
    _model_settings: OpenAIChatModelSettings | None = None

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._provider_timestamp is not None:  # pragma: no branch
            self.provider_details = {'timestamp': self._provider_timestamp}
        async for chunk in self._validate_response():
            chunk_usage = self._map_usage(chunk)
            if self._model_settings and self._model_settings.get('openai_continuous_usage_stats'):
                # When continuous_usage_stats is enabled, each chunk contains cumulative usage,
                # so we replace rather than increment to avoid double-counting.
                self._usage = chunk_usage
            else:
                self._usage += chunk_usage

            if chunk.id:  # pragma: no branch
                self.provider_response_id = chunk.id

            if chunk.model:
                self._model_name = chunk.model

            try:
                choice = chunk.choices[0]
            except IndexError:
                continue

            # When using Azure OpenAI and an async content filter is enabled, the openai SDK can return None deltas.
            if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue

            if raw_finish_reason := choice.finish_reason:
                self.finish_reason = self._map_finish_reason(raw_finish_reason)

            if provider_details := self._map_provider_details(chunk):  # pragma: no branch
                self.provider_details = {**(self.provider_details or {}), **provider_details}

            for event in self._map_part_delta(choice):
                yield event

    def _validate_response(self) -> AsyncIterable[ChatCompletionChunk]:
        """Hook that validates incoming chunks.

        This method may be overridden by subclasses of `OpenAIStreamedResponse` to apply custom chunk validations.

        By default, this is a no-op since `ChatCompletionChunk` is already validated.
        """
        return self._response

    def _map_part_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that determines the sequence of mappings that will be called to produce events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        return itertools.chain(
            self._map_thinking_delta(choice), self._map_text_delta(choice), self._map_tool_call_delta(choice)
        )

    def _map_thinking_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps thinking delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        profile = OpenAIModelProfile.from_profile(self._model_profile)
        custom_field = profile.openai_chat_thinking_field

        # Prefer the configured custom reasoning field, if present in profile.
        # Fall back to built-in fields if no custom field result was found.

        # The `reasoning_content` field is typically present in DeepSeek and Moonshot models.
        # https://api-docs.deepseek.com/guides/reasoning_model

        # The `reasoning` field is typically present in gpt-oss via Ollama and OpenRouter.
        # - https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot#chat-completions-api
        # - https://openrouter.ai/docs/use-cases/reasoning-tokens#basic-usage-with-reasoning-tokens
        for field_name in (custom_field, 'reasoning', 'reasoning_content'):
            if not field_name:
                continue
            reasoning: str | None = getattr(choice.delta, field_name, None)
            if reasoning:  # pragma: no branch
                yield from self._parts_manager.handle_thinking_delta(
                    vendor_part_id=field_name,
                    id=field_name,
                    content=reasoning,
                    provider_name=self.provider_name,
                )
                break

    def _map_text_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps text delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        # Handle the text part of the response
        content = choice.delta.content
        if content:
            for event in self._parts_manager.handle_text_delta(
                vendor_part_id='content',
                content=content,
                thinking_tags=self._model_profile.thinking_tags,
                ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
            ):
                if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
                    event.part.id = 'content'
                    event.part.provider_name = self.provider_name
                yield event

    def _map_tool_call_delta(self, choice: chat_completion_chunk.Choice) -> Iterable[ModelResponseStreamEvent]:
        """Hook that maps tool call delta content to events.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the mapping.
        """
        for dtc in choice.delta.tool_calls or []:
            maybe_event = self._parts_manager.handle_tool_call_delta(
                vendor_part_id=dtc.index,
                tool_name=dtc.function and dtc.function.name,
                args=dtc.function and dtc.function.arguments,
                tool_call_id=dtc.id,
            )
            if maybe_event is not None:
                yield maybe_event

    def _map_provider_details(self, chunk: ChatCompletionChunk) -> dict[str, Any] | None:
        """Hook that generates the provider details from chunk content.

        This method may be overridden by subclasses of `OpenAIStreamResponse` to customize the provider details.
        """
        return _map_provider_details(chunk.choices[0])

    def _map_usage(self, response: ChatCompletionChunk) -> usage.RequestUsage:
        return _map_usage(response, self._provider_name, self._provider_url, self.model_name)

    def _map_finish_reason(
        self, key: Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']
    ) -> FinishReason | None:
        """Hooks that maps a finish reason key to a [FinishReason](pydantic_ai.messages.FinishReason).

        This method may be overridden by subclasses of `OpenAIChatModel` to accommodate custom keys.
        """
        return _CHAT_FINISH_REASON_MAP.get(key)

    @property
    def model_name(self) -> OpenAIModelName:
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


@dataclass
class OpenAIResponsesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for OpenAI Responses API."""

    _model_name: OpenAIModelName
    _model_settings: OpenAIResponsesModelSettings
    _response: AsyncIterable[responses.ResponseStreamEvent]
    _provider_name: str
    _provider_url: str
    _provider_timestamp: datetime | None = None
    _timestamp: datetime = field(default_factory=_now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        # Track annotations by item_id and content_index
        _annotations_by_item: dict[str, list[Any]] = {}

        if self._provider_timestamp is not None:  # pragma: no branch
            self.provider_details = {'timestamp': self._provider_timestamp}

        async for chunk in self._response:
            # NOTE: You can inspect the builtin tools used checking the `ResponseCompletedEvent`.
            if isinstance(chunk, responses.ResponseCompletedEvent):
                self._usage += self._map_usage(chunk.response)

                raw_finish_reason = (
                    details.reason if (details := chunk.response.incomplete_details) else chunk.response.status
                )

                if raw_finish_reason:  # pragma: no branch
                    self.provider_details = {**(self.provider_details or {}), 'finish_reason': raw_finish_reason}
                    self.finish_reason = _RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)

            elif isinstance(chunk, responses.ResponseContentPartAddedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseContentPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCreatedEvent):
                if chunk.response.id:  # pragma: no branch
                    self.provider_response_id = chunk.response.id

            elif isinstance(chunk, responses.ResponseFailedEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=chunk.item_id,
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseFunctionCallArgumentsDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseIncompleteEvent):  # pragma: no cover
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseInProgressEvent):
                self._usage += self._map_usage(chunk.response)

            elif isinstance(chunk, responses.ResponseOutputItemAddedEvent):
                if isinstance(chunk.item, responses.ResponseFunctionToolCall):
                    yield self._parts_manager.handle_tool_call_part(
                        vendor_part_id=chunk.item.id,
                        tool_name=chunk.item.name,
                        args=chunk.item.arguments,
                        tool_call_id=chunk.item.call_id,
                        id=chunk.item.id,
                        provider_name=self.provider_name,
                    )
                elif isinstance(chunk.item, responses.ResponseReasoningItem):
                    pass
                elif isinstance(chunk.item, responses.ResponseOutputMessage):
                    pass
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, _ = _map_web_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseFileSearchToolCall):
                    call_part, _ = _map_file_search_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    call_part, _, _ = _map_code_interpreter_tool_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `"}` so that we can add code deltas
                    args_json_delta = args_json[:-2]
                    assert args_json_delta.endswith('"code":"'), f'Expected {args_json_delta!r} to end in `"code":"`'

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    call_part, _, _ = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    call_part, _ = _map_mcp_call(chunk.item, self.provider_name)

                    args_json = call_part.args_as_json_str()
                    # Drop the final `{}}` so that we can add tool args deltas
                    args_json_delta = args_json[:-3]
                    assert args_json_delta.endswith('"tool_args":'), (
                        f'Expected {args_json_delta!r} to end in `"tool_args":"`'
                    )

                    yield self._parts_manager.handle_part(
                        vendor_part_id=f'{chunk.item.id}-call', part=replace(call_part, args=None)
                    )
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=args_json_delta,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    call_part, _ = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-call', part=call_part)
                else:
                    warnings.warn(  # pragma: no cover
                        f'Handling of this item type is not yet implemented. Please report on our GitHub: {chunk}',
                        UserWarning,
                    )

            elif isinstance(chunk, responses.ResponseOutputItemDoneEvent):
                if isinstance(chunk.item, responses.ResponseReasoningItem):
                    if signature := chunk.item.encrypted_content:  # pragma: no branch
                        # Add the signature to the part corresponding to the first summary/raw CoT
                        for event in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=chunk.item.id,
                            id=chunk.item.id,
                            signature=signature,
                            provider_name=self.provider_name,
                        ):
                            yield event
                elif isinstance(chunk.item, responses.ResponseCodeInterpreterToolCall):
                    _, return_part, file_parts = _map_code_interpreter_tool_call(chunk.item, self.provider_name)
                    for i, file_part in enumerate(file_parts):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=f'{chunk.item.id}-file-{i}', part=file_part
                        )
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.ResponseFunctionWebSearch):
                    call_part, return_part = _map_web_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.ResponseFileSearchToolCall):
                    call_part, return_part = _map_file_search_tool_call(chunk.item, self.provider_name)

                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=f'{chunk.item.id}-call',
                        args=call_part.args,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.ImageGenerationCall):
                    _, return_part, file_part = _map_image_generation_tool_call(chunk.item, self.provider_name)
                    if file_part:  # pragma: no branch
                        yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-file', part=file_part)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

                elif isinstance(chunk.item, responses.response_output_item.McpCall):
                    _, return_part = _map_mcp_call(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)
                elif isinstance(chunk.item, responses.response_output_item.McpListTools):
                    _, return_part = _map_mcp_list_tools(chunk.item, self.provider_name)
                    yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item.id}-return', part=return_part)

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartAddedEvent):
                # Use same vendor_part_id as raw CoT for first summary (index 0) so they merge into one ThinkingPart
                vendor_id = chunk.item_id if chunk.summary_index == 0 else f'{chunk.item_id}-{chunk.summary_index}'
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_id,
                    content=chunk.part.text,
                    id=chunk.item_id,
                    provider_name=self.provider_name,
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningSummaryPartDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDoneEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseReasoningSummaryTextDeltaEvent):
                # Use same vendor_part_id as raw CoT for first summary (index 0) so they merge into one ThinkingPart
                vendor_id = chunk.item_id if chunk.summary_index == 0 else f'{chunk.item_id}-{chunk.summary_index}'
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=vendor_id,
                    content=chunk.delta,
                    id=chunk.item_id,
                    provider_name=self.provider_name,
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningTextDeltaEvent):
                # Handle raw CoT from gpt-oss models using callback pattern
                for event in self._parts_manager.handle_thinking_delta(
                    vendor_part_id=chunk.item_id,
                    id=chunk.item_id,
                    provider_name=self.provider_name,
                    provider_details=_make_raw_content_updater(chunk.delta, chunk.content_index),
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseReasoningTextDoneEvent):
                pass  # content already accumulated via delta events

            elif isinstance(chunk, responses.ResponseOutputTextAnnotationAddedEvent):
                # Collect annotations if the setting is enabled
                if self._model_settings.get('openai_include_raw_annotations'):
                    _annotations_by_item.setdefault(chunk.item_id, []).append(chunk.annotation)

            elif isinstance(chunk, responses.ResponseTextDeltaEvent):
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=chunk.item_id,
                    content=chunk.delta,
                    id=chunk.item_id,
                    provider_name=self.provider_name,
                ):
                    yield event

            elif isinstance(chunk, responses.ResponseTextDoneEvent):
                # Add annotations to provider_details if available
                provider_details: dict[str, Any] = {}
                annotations = _annotations_by_item.get(chunk.item_id)
                if annotations:
                    provider_details['annotations'] = responses_output_text_annotations_ta.dump_python(
                        list(annotations), warnings=False
                    )

                if provider_details:
                    for event in self._parts_manager.handle_text_delta(
                        vendor_part_id=chunk.item_id,
                        content='',
                        provider_name=self.provider_name,
                        provider_details=provider_details,
                    ):
                        yield event

            elif isinstance(chunk, responses.ResponseWebSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseWebSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseAudioDeltaEvent):  # pragma: lax no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDeltaEvent):
                json_args_delta = to_json(chunk.delta).decode()[1:-1]  # Drop the surrounding `"`
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=json_args_delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCodeDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='"}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseCodeInterpreterCallInterpretingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallCompletedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallGeneratingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseImageGenCallPartialImageEvent):
                # Not present on the type, but present on the actual object.
                # See https://github.com/openai/openai-python/issues/2649
                output_format = getattr(chunk, 'output_format', 'png')
                file_part = FilePart(
                    content=BinaryImage(
                        data=base64.b64decode(chunk.partial_image_b64),
                        media_type=f'image/{output_format}',
                    ),
                    id=chunk.item_id,
                )
                yield self._parts_manager.handle_part(vendor_part_id=f'{chunk.item_id}-file', part=file_part)

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDoneEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args='}',
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpCallArgumentsDeltaEvent):
                maybe_event = self._parts_manager.handle_tool_call_delta(
                    vendor_part_id=f'{chunk.item_id}-call',
                    args=chunk.delta,
                )
                if maybe_event is not None:  # pragma: no branch
                    yield maybe_event

            elif isinstance(chunk, responses.ResponseMcpListToolsInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpListToolsFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallInProgressEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallFailedEvent):  # pragma: no cover
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseMcpCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallCompletedEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallSearchingEvent):
                pass  # there's nothing we need to do here

            elif isinstance(chunk, responses.ResponseFileSearchCallInProgressEvent):
                pass  # there's nothing we need to do here

            else:  # pragma: no cover
                warnings.warn(
                    f'Handling of this event type is not yet implemented. Please report on our GitHub: {chunk}',
                    UserWarning,
                )

    def _map_usage(self, response: responses.Response) -> usage.RequestUsage:
        return _map_usage(response, self._provider_name, self._provider_url, self.model_name)

    @property
    def model_name(self) -> OpenAIModelName:
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


def _make_raw_content_updater(delta: str, index: int) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
    """Create a callback that updates `provider_details['raw_content']`.

    This is used for streaming raw CoT from gpt-oss models. The callback pattern keeps
    `raw_content` logic in OpenAI code while the parts manager stays provider-agnostic.
    """

    def update_provider_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = {**(existing or {})}
        raw_list: list[str] = list(details.get('raw_content', []))
        while len(raw_list) <= index:
            raw_list.append('')
        raw_list[index] += delta
        details['raw_content'] = raw_list
        return details

    return update_provider_details


# Convert logprobs to a serializable format
def _map_logprobs(
    logprobs: list[chat_completion_token_logprob.ChatCompletionTokenLogprob]
    | list[responses.response_output_text.Logprob],
) -> list[dict[str, Any]]:
    return [
        {
            'token': lp.token,
            'bytes': lp.bytes,
            'logprob': lp.logprob,
            'top_logprobs': [
                {'token': tlp.token, 'bytes': tlp.bytes, 'logprob': tlp.logprob} for tlp in lp.top_logprobs
            ],
        }
        for lp in logprobs
    ]


def _map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response,
    provider: str,
    provider_url: str,
    model: str,
) -> usage.RequestUsage:
    response_usage = response.usage
    if response_usage is None:
        return usage.RequestUsage()

    usage_data = response_usage.model_dump(exclude_none=True)
    details = {
        k: v
        for k, v in usage_data.items()
        if k not in {'prompt_tokens', 'completion_tokens', 'input_tokens', 'output_tokens', 'total_tokens'}
        if isinstance(v, int)
    }
    response_data = dict(model=model, usage=usage_data)
    if isinstance(response_usage, responses.ResponseUsage):
        api_flavor = 'responses'

        if getattr(response_usage, 'output_tokens_details', None) is not None:
            details['reasoning_tokens'] = response_usage.output_tokens_details.reasoning_tokens
        else:
            details['reasoning_tokens'] = 0
    else:
        api_flavor = 'chat'

        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))

    return usage.RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor=api_flavor,
        details=details,
    )


def _map_provider_details(
    choice: chat_completion_chunk.Choice | chat_completion.Choice,
) -> dict[str, Any] | None:
    provider_details: dict[str, Any] = {}

    # Add logprobs to vendor_details if available
    if choice.logprobs is not None and choice.logprobs.content:
        provider_details['logprobs'] = _map_logprobs(choice.logprobs.content)
    if raw_finish_reason := choice.finish_reason:
        provider_details['finish_reason'] = raw_finish_reason

    return provider_details or None


def _split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    # When reasoning, the Responses API requires the `ResponseFunctionToolCall` to be returned with both the `call_id` and `id` fields.
    # Before our `ToolCallPart` gained the `id` field alongside `tool_call_id` field, we combined the two fields into a single string stored on `tool_call_id`.
    if '|' in combined_id:
        call_id, id = combined_id.split('|', 1)
        return call_id, id
    else:
        return combined_id, None


def _map_code_interpreter_tool_call(
    item: responses.ResponseCodeInterpreterToolCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, list[FilePart]]:
    result: dict[str, Any] = {
        'status': item.status,
    }

    file_parts: list[FilePart] = []
    logs: list[str] = []
    if item.outputs:
        for output in item.outputs:
            if isinstance(output, responses.response_code_interpreter_tool_call.OutputImage):
                file_parts.append(
                    FilePart(
                        content=BinaryImage.from_data_uri(output.url),
                        id=item.id,
                    )
                )
            elif isinstance(output, responses.response_code_interpreter_tool_call.OutputLogs):
                logs.append(output.logs)
            else:
                assert_never(output)

    if logs:
        result['logs'] = logs

    return (
        BuiltinToolCallPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            args={
                'container_id': item.container_id,
                'code': item.code or '',
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=CodeExecutionTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_parts,
    )


def _map_web_search_tool_call(
    item: responses.ResponseFunctionWebSearch, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    args: dict[str, Any] | None = None

    result = {
        'status': item.status,
    }

    if action := item.action:
        # We need to exclude None values because of https://github.com/pydantic/pydantic-ai/issues/3653
        args = action.model_dump(mode='json', exclude_none=True)

        # To prevent `Unknown parameter: 'input[2].action.sources'` for `ActionSearch`
        if sources := args.pop('sources', None):
            result['sources'] = sources

    return (
        BuiltinToolCallPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=WebSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def _map_file_search_tool_call(
    item: responses.ResponseFileSearchToolCall,
    provider_name: str,
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    args = {'queries': item.queries}

    result: dict[str, Any] = {
        'status': item.status,
    }
    if item.results is not None:
        result['results'] = [r.model_dump(mode='json') for r in item.results]

    return (
        BuiltinToolCallPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            args=args,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=FileSearchTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
    )


def _map_image_generation_tool_call(
    item: responses.response_output_item.ImageGenerationCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart, FilePart | None]:
    result = {
        'status': item.status,
    }

    # Not present on the type, but present on the actual object.
    # See https://github.com/openai/openai-python/issues/2649
    if background := getattr(item, 'background', None):
        result['background'] = background
    if quality := getattr(item, 'quality', None):
        result['quality'] = quality
    if size := getattr(item, 'size', None):
        result['size'] = size
    if revised_prompt := getattr(item, 'revised_prompt', None):
        result['revised_prompt'] = revised_prompt
    output_format = getattr(item, 'output_format', 'png')

    file_part: FilePart | None = None
    if item.result:
        file_part = FilePart(
            content=BinaryImage(
                data=base64.b64decode(item.result),
                media_type=f'image/{output_format}',
            ),
            id=item.id,
        )

        # For some reason, the streaming API leaves `status` as `generating` even though generation has completed.
        result['status'] = 'completed'

    return (
        BuiltinToolCallPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=ImageGenerationTool.kind,
            tool_call_id=item.id,
            content=result,
            provider_name=provider_name,
        ),
        file_part,
    )


def _map_mcp_list_tools(
    item: responses.response_output_item.McpListTools, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            provider_name=provider_name,
            args={'action': 'list_tools'},
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content=item.model_dump(mode='json', include={'tools', 'error'}),
            provider_name=provider_name,
        ),
    )


def _map_mcp_call(
    item: responses.response_output_item.McpCall, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart]:
    tool_name = ':'.join([MCPServerTool.kind, item.server_label])
    return (
        BuiltinToolCallPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            args={
                'action': 'call_tool',
                'tool_name': item.name,
                'tool_args': json.loads(item.arguments) if item.arguments else {},
            },
            provider_name=provider_name,
        ),
        BuiltinToolReturnPart(
            tool_name=tool_name,
            tool_call_id=item.id,
            content={
                'output': item.output,
                'error': item.error,
            },
            provider_name=provider_name,
        ),
    )
