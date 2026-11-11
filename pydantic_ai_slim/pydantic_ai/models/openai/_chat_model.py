"""`OpenAIChatModel`, the Chat Completions API model."""

from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncGenerator, AsyncIterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Never, assert_never, cast, overload

from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, AsyncStream, Omit, omit as OMIT
from openai.types import chat
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    chat_completion,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_content_part_param import File, FileFile
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import ChatCompletionToolChoiceOptionParam
from openai.types.chat.completion_create_params import (
    PromptCacheOptions as ChatPromptCacheOptions,
    WebSearchOptions,
    WebSearchOptionsUserLocation,
    WebSearchOptionsUserLocationApproximate,
)
from openai.types.shared import ReasoningEffort
from pydantic import JsonValue, ValidationError

from ... import UnexpectedModelBehavior, _utils, usage
from ..._file_input import KNOWN_DOCUMENT_MEDIA_TYPES, TEXT_DOCUMENT_MEDIA_TYPES
from ..._framing import neutralize_voice_tag
from ..._http import to_httpx2_timeout
from ..._output import DEFAULT_OUTPUT_TOOL_NAME
from ..._run_context import RunContext
from ..._thinking_part import render_replayed_thinking
from ..._utils import (
    guard_tool_call_id as _guard_tool_call_id,
    is_text_like_media_type as _is_text_like_media_type,
    now_utc as _now_utc,
    number_to_datetime,
)
from ...exceptions import UserError
from ...messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from ...native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool, WebSearchTool
from ...output import OutputObjectDefinition
from ...profiles import ModelProfile, ModelProfileSpec, merge_profile
from ...profiles.openai import OpenAIModelProfile, validate_openai_profile
from ...providers import Provider, infer_provider
from ...settings import CacheSetting, ModelSettings
from ...tools import ToolDefinition
from .. import (
    Model,
    ModelRequestParameters,
    OpenAIChatCompatibleProvider,
    StreamedResponse,
    _sendable_request_parts,  # pyright: ignore[reportPrivateUsage]
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unprepared_part_error,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
    _unprojected_instruction_delta_error,  # pyright: ignore[reportPrivateUsage]
    _with_default_user_agent,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
    download_item,
)
from .._decode_errors import map_decode_errors
from .._sdk_retries import with_max_retries
from .._tool_choice import resolve_tool_choice
from ._caching import (
    CACHE_SETTINGS_KEYS,
    add_openai_prompt_cache_breakpoint,
    cacheable_instruction_count,
    has_dynamic_system_prompt,
    keep_implicit_cache_without_breakpoints,
    leading_system_message_count,
    move_leading_cache_breakpoints,
    openai_effective_cache_settings,
    resolve_legacy_cache_retention,
    translate_openai_cache,
)
from ._chat_completions import (
    CHAT_FINISH_REASON_MAP,
    LenientChatCompletion,
    map_provider_details,
    map_tool_call_provider_details,
    raw_non_text_content_parts,
    unmapped_choice_data,
)
from ._chat_streaming import OpenAIStreamedResponse
from ._citations import map_chat_content
from ._errors import check_azure_content_filter, is_azure, map_api_errors
from ._model_names import OpenAIModelName
from ._model_response_streaming import ModelResponseStreamedResponse
from ._reasoning import drop_sampling_params_for_reasoning, reasoning_active, resolve_openai_thinking_effort
from ._response_fields import map_usage
from ._settings import (
    OpenAIChatModelSettings,
    drop_unsupported_params,
    resolve_openai_service_tier,
    support_openai_tool_forcing,
    with_profile_unsupported_settings,
)

if TYPE_CHECKING:
    from ._responses_model import OpenAIResponsesModel


def preload_openai_sdk_resource_modules(model: OpenAIChatModel | OpenAIResponsesModel, client: AsyncOpenAI) -> None:
    """Load deferred OpenAI SDK modules before request handling.

    The provider client only triggers process-wide module loading; it need not be the exact client later used by
    an OpenAI-compatible model subclass.
    """
    if isinstance(model, OpenAIChatModel):
        _ = client.chat.completions
    else:
        _ = client.responses


class _ChatCompletionMessageFunctionToolCallParam(ChatCompletionMessageFunctionToolCallParam, total=False):
    """A Chat Completions tool call carrying the `extra_content` vendor extension, see `map_tool_call_provider_details`."""

    extra_content: JsonValue


def merge_leading_system_messages(
    openai_messages: list[chat.ChatCompletionMessageParam], system_prompt_role: str
) -> list[chat.ChatCompletionMessageParam]:
    """Merge consecutive messages with `system_prompt_role` at the start of the list into a single message.

    Required by strict OpenAI-compatible backends (some LiteLLM/vLLM deployments) that reject more than one
    initial system message with `System message must be at the beginning.`

    Only applies when system prompts are sent as `'system'` or `'developer'` role; with `'user'` role we
    can't distinguish a system prompt cast as user from an actual user turn.
    """
    if system_prompt_role not in ('system', 'developer'):
        return openai_messages

    leading_count = next(
        (i for i, m in enumerate(openai_messages) if m.get('role') != system_prompt_role),
        len(openai_messages),
    )
    if leading_count < 2:
        return openai_messages

    # Content is always `str` here: it originates from `SystemPromptPart.content` or instruction
    # `TextPart.content`, both of which are typed as `str`.
    merged_content = '\n\n'.join(m['content'] for m in openai_messages[:leading_count])  # type: ignore[misc]
    merged: chat.ChatCompletionMessageParam = {**openai_messages[0], 'content': merged_content}  # type: ignore[typeddict-item]
    return [merged, *openai_messages[leading_count:]]


@dataclass(init=False)
class OpenAIChatModel(Model[AsyncOpenAI]):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    # Chat Completions has no video content part.
    _unsupported_file_inputs = frozenset({'video'})

    unsupported_settings = frozenset({'top_k'})

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

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
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names available
                [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
                (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider

        super().__init__(settings=settings, profile=profile)

        validate_openai_profile(self.profile)

        preload_openai_sdk_resource_modules(self, self._provider.client)

    @property
    def client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        """Resolve the prompt cache retention requested by OpenAI settings or the unified `cache` setting."""
        return resolve_legacy_cache_retention(
            self.profile, self.settings, model_settings
        ) or super().resolve_cache_retention(model_settings)

    def _no_batch_api_message(self) -> str:
        # The profile's Batch API flag, not the provider name, so renamed and compatible providers get the hint too.
        if self.profile.get('openai_supports_batch_api', False):
            return (
                f'`{type(self).__name__}` {self.model_name!r} has no batch API: batch it through the Responses API, with '
                '`OpenAIResponsesModel` or the `openai:` prefix.'
            )
        return super()._no_batch_api_message()

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        """Turn off what Chat Completions doesn't serve, before the user's `profile=` applies.

        `supports_compaction_on_demand` is off: `OpenAIProvider` sets it for the Responses API's compaction endpoint,
        Chat Completions has none, and `OpenAIChatModel` doesn't implement `compact_messages`. The `'12h'` cache
        retention is left out of `supported_cache_retentions`: Chat Completions rejects `prompt_cache_options.ttl='12h'`
        with a 400 (verified live on every GPT-5.6, GPT-6 and GPT-6.1 model), while the Responses API on the same
        models accepts it, and the providers declare their tiers per model rather than per API.
        """
        profile = merge_profile(profile, ModelProfile(supports_compaction_on_demand=False))
        retentions = profile.get('supported_cache_retentions', ())
        if '12h' not in retentions:
            return profile
        return merge_profile(
            profile, ModelProfile(supported_cache_retentions=tuple(tier for tier in retentions if tier != '12h'))
        )

    def _has_provider_cache_settings(self, merged_settings: ModelSettings) -> bool:
        return any(key in merged_settings for key in CACHE_SETTINGS_KEYS)

    def _caching_not_enabled(self, model_settings: ModelSettings | None) -> bool:
        # OpenAI's models cache prompts implicitly even when nothing is configured, so unconfigured isn't uncached.
        # Subclasses serving other model families (OpenRouter's Anthropic routes) still report it.
        if self.profile.get('openai_supports_prompt_cache_breakpoints', False):
            return False
        return super()._caching_not_enabled(model_settings)

    def _effective_cache_settings(self, merged_settings: ModelSettings) -> tuple[CacheSetting | None, ...]:
        if self._has_provider_cache_settings(merged_settings) and self.profile.get('supports_cache', False):
            return openai_effective_cache_settings(cast(OpenAIChatModelSettings, merged_settings))
        return super()._effective_cache_settings(merged_settings)

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset({WebSearchTool})

    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        """Return all of `native_tool`'s restricting settings, as Chat Completions' `web_search_options` takes none."""
        return native_tool.restricting_settings

    @cached_property
    def profile(self) -> OpenAIModelProfile:
        """The model profile.

        WebSearchTool is only supported if openai_chat_supports_web_search is True.
        """
        _profile = super().profile
        if (provider := self.provider) is not None and is_azure(provider.client, provider.name):
            user = self._profile
            if user is None or (not callable(user) and 'openai_chat_supports_document_input' not in user):
                _profile = merge_profile(_profile, OpenAIModelProfile(openai_chat_supports_document_input=False))
        if not _profile.get('openai_chat_supports_document_input', True):
            # Without a `file` content part, a text document is still inlined into the prompt as text,
            # so documents are narrowed to those rather than turned off.
            document_media_types = _profile.get('supported_document_media_types', KNOWN_DOCUMENT_MEDIA_TYPES)
            _profile = merge_profile(
                _profile,
                ModelProfile(supported_document_media_types=document_media_types & TEXT_DOCUMENT_MEDIA_TYPES),
            )
        if not _profile.get('openai_chat_supports_web_search', False):
            new_tools = _profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS) - {WebSearchTool}
            _profile = merge_profile(_profile, ModelProfile(supported_native_tools=new_tools))
        return cast(OpenAIModelProfile, _profile)

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        # Check for WebSearchTool before base validation to provide a helpful error message
        if (
            any(isinstance(tool, WebSearchTool) for tool in model_request_parameters.native_tools)
            and not self.profile.get('openai_chat_supports_web_search', False)
            and not any(t.unless_native == 'web_search' for t in model_request_parameters.function_tools)
        ):
            raise UserError(
                f'WebSearchTool is not supported with `OpenAIChatModel` and model {self.model_name!r}. '
                f'Please use `OpenAIResponsesModel` instead.'
            )
        model_settings, model_request_parameters = super().prepare_request(model_settings, model_request_parameters)
        return translate_openai_cache(self.profile, model_settings, model_request_parameters), model_request_parameters

    def _unsupported_settings(self) -> frozenset[str]:
        return with_profile_unsupported_settings(self.unsupported_settings, self.profile)

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

        model_response = self._process_response(
            response,
            include_raw_annotations=bool(model_settings and model_settings.get('openai_include_raw_annotations')),
            include_unmapped_response_data=bool(
                model_settings and model_settings.get('openai_include_unmapped_response_data')
            ),
        )
        return model_response

    def _translate_thinking(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ReasoningEffort | Omit:
        """Get reasoning effort, falling back to unified thinking when provider-specific setting is not set."""
        if effort := model_settings.get('openai_reasoning_effort'):
            return effort
        thinking = model_request_parameters.thinking
        if thinking is None:
            return OMIT
        if thinking is False and 'openai_chat_thinking_false_body' in self.profile:
            # The profile's body fields stand in for `reasoning_effort`; see `_thinking_false_extra_body`.
            return OMIT
        effort = resolve_openai_thinking_effort(
            thinking,
            self.profile,
            supports_max=self.profile.get('openai_chat_supports_max_reasoning_effort', False),
        )
        return OMIT if effort is None else effort

    def _thinking_false_extra_body(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> object | None:
        """The request's `extra_body`, plus the profile's `openai_chat_thinking_false_body` fields for `thinking=False`.

        Keys the caller set in `extra_body` win, and nothing is added when `openai_reasoning_effort` is set.
        """
        extra_body = model_settings.get('extra_body')
        thinking_false_body = self.profile.get('openai_chat_thinking_false_body')
        if (
            thinking_false_body is None
            or model_request_parameters.thinking is not False
            or model_settings.get('openai_reasoning_effort')
        ):
            return extra_body
        return {**thinking_false_body, **cast(dict[str, Any], extra_body or {})}

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings_cast = cast(OpenAIChatModelSettings, model_settings or {})
        response = await self._completions_create(messages, True, model_settings_cast, model_request_parameters)
        if isinstance(response, ModelResponse):
            # Azure rejected the prompt with its content filter before the stream opened.
            yield ModelResponseStreamedResponse(
                model_request_parameters=model_request_parameters, _model_response=response
            )
            return
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters, model_settings_cast)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk] | ModelResponse: ...

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
        tool_choice: ChatCompletionToolChoiceOptionParam | None

        tools, tool_choice = self._get_tool_choice(model_settings, model_request_parameters)
        web_search_options = self._get_web_search_options(model_request_parameters)
        profile = self.profile

        openai_messages = await self._map_messages(messages, model_request_parameters, model_settings=model_settings)
        self._finalize_cache_breakpoints(tools, openai_messages)

        response_format: chat.completion_create_params.ResponseFormat | None = None
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_format = self._map_json_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and self.profile.get(
            'supports_json_object_output', False
        ):  # pragma: no branch
            response_format = {'type': 'json_object'}

        # These helpers mutate the settings they receive.
        model_settings = OpenAIChatModelSettings(**model_settings)
        drop_sampling_params_for_reasoning(profile, model_settings, model_request_parameters)
        keep_implicit_cache_without_breakpoints(model_settings, openai_messages)

        drop_unsupported_params(profile, model_settings)

        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            try:
                extra_headers = _with_default_user_agent(model_settings.get('extra_headers', {}))

                # Most providers only accept one of `max_completion_tokens` (OpenAI, incl. o-series) or
                # `max_tokens` (e.g. OpenRouter), so the profile decides which field the `max_tokens` setting maps to.
                max_tokens = model_settings.get('max_tokens', OMIT)
                supports_max_completion_tokens = profile.get('openai_chat_supports_max_completion_tokens', True)
                return await with_max_retries(self.client, model_settings).chat.completions.create(
                    model=self.model_name,
                    messages=openai_messages,
                    parallel_tool_calls=model_settings.get('parallel_tool_calls', OMIT) if tools else OMIT,
                    tools=tools or OMIT,
                    tool_choice=tool_choice or OMIT,
                    stream=stream,
                    stream_options=self._get_stream_options(model_settings) if stream else OMIT,
                    stop=model_settings.get('stop_sequences', OMIT),
                    max_completion_tokens=max_tokens if supports_max_completion_tokens else OMIT,
                    max_tokens=OMIT if supports_max_completion_tokens else max_tokens,
                    timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                    response_format=response_format or OMIT,
                    seed=model_settings.get('seed', OMIT),
                    reasoning_effort=self._translate_thinking(model_settings, model_request_parameters),
                    user=model_settings.get('openai_user', OMIT),
                    web_search_options=web_search_options or OMIT,
                    # The SDK types `ultrafast` as Responses-only; rejecting it is left to the API.
                    service_tier=cast(Any, resolve_openai_service_tier(model_settings)),
                    prediction=model_settings.get('openai_prediction', OMIT),
                    temperature=model_settings.get('temperature', OMIT),
                    top_p=model_settings.get('top_p', OMIT),
                    presence_penalty=model_settings.get('presence_penalty', OMIT),
                    frequency_penalty=model_settings.get('frequency_penalty', OMIT),
                    logit_bias=model_settings.get('logit_bias', OMIT),
                    logprobs=model_settings.get('openai_logprobs', OMIT),
                    top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
                    store=model_settings.get('openai_store', OMIT),
                    moderation=model_settings.get('openai_moderation', OMIT),
                    verbosity=model_settings.get('openai_text_verbosity', OMIT),
                    prompt_cache_key=model_settings.get('openai_prompt_cache_key', OMIT),
                    prompt_cache_retention=model_settings.get('openai_prompt_cache_retention', OMIT),
                    # Passed through as given: Chat Completions rejects a `'12h'` TTL with a 400, and the unified
                    # `cache` setting never sends one here (see `_instance_profile_defaults`).
                    prompt_cache_options=cast(
                        'ChatPromptCacheOptions | Omit', model_settings.get('openai_prompt_cache_options', OMIT)
                    ),
                    extra_headers=extra_headers,
                    extra_body=self._thinking_false_extra_body(model_settings, model_request_parameters),
                )
            except APIStatusError as e:
                if model_response := check_azure_content_filter(e, self.client, self.system, self.model_name):
                    return model_response
                raise

    def _validate_completion(self, response: chat.ChatCompletion) -> LenientChatCompletion:
        """Hook that validates chat completions before processing.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom completion validations.
        """
        return LenientChatCompletion.model_validate(response.model_dump(warnings=False))

    def _missing_finish_reason(
        self, choice: chat_completion.Choice
    ) -> Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']:
        """Hook that picks the finish reason of a response that came without one.

        This method may be overridden by subclasses of `OpenAIChatModel` whose model picks another one, along with
        `OpenAIStreamedResponse._missing_finish_reason` for a stream that ends without one.
        """
        return 'stop'

    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        """Hook that response content to provider details.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        return map_provider_details(response.choices[0])

    def _unmapped_provider_details(
        self, response: chat.ChatCompletion, content_parts: list[object], *, include: bool
    ) -> dict[str, Any]:
        """`provider_details['unmapped']` for `openai_include_unmapped_response_data`, from the validated response."""
        if not include:
            return {}
        choice = response.choices[0]
        data = unmapped_choice_data(
            response,
            choice,
            choice.message,
            content_parts,
            message_key='message',
            thinking_field=self.profile.get('openai_chat_thinking_field', None),
        )
        return {'unmapped': data} if data else {}

    def _process_response(  # noqa: C901
        self,
        response: chat.ChatCompletion | str,
        *,
        include_raw_annotations: bool = False,
        include_unmapped_response_data: bool = False,
    ) -> ModelResponse:
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
        # Some OpenAI-compatible providers send a `null` `created` (e.g. OpenRouter) or none at all (e.g. GitHub Copilot),
        # which fails validation. Like a zero one, it records no provider timestamp, as a stream without one doesn't.
        if response.created is None:  # pyright: ignore[reportUnnecessaryComparison]
            response.created = 0

        # Some OpenAI-compatible providers omit the finish reason (e.g. local Ollama) or send an empty one (e.g. Snowflake
        # Cortex), which fails validation. The response is treated as `_missing_finish_reason` says, like a stream that
        # ends without one (subclasses may also handle an empty one in `_validate_completion`), but that's not reported
        # as the provider's value.
        missing_finish_reason = False
        if response.choices and not (choice := response.choices[0]).finish_reason:
            missing_finish_reason = True
            if choice.finish_reason is None:
                choice.finish_reason = self._missing_finish_reason(choice)

        content_parts = raw_non_text_content_parts(response)

        try:
            response = self._validate_completion(response)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from {self.system} chat completions endpoint: {e}') from e

        choice = response.choices[0]
        finish_reason = self._map_finish_reason(choice.finish_reason)
        if missing_finish_reason:
            # Restore the missing value after validation, so it's not reported as the provider's.
            choice.finish_reason = None  # pyright: ignore[reportAttributeAccessIssue]

        # Moderation and service tier are top-level fields, so they're read here rather than in the choice-scoped
        # `_process_provider_details` hook that subclasses may override.
        provider_details = self._process_provider_details(response) or {}
        if response.moderation:
            provider_details['moderation'] = response.moderation.model_dump()
        if response.service_tier:
            provider_details['service_tier'] = response.service_tier
        provider_details.update(
            self._unmapped_provider_details(response, content_parts, include=include_unmapped_response_data)
        )

        # Handle refusal responses (structured output safety filter)
        if choice.message.refusal:
            provider_details.pop('finish_reason', None)
            provider_details['refusal'] = choice.message.refusal
            if response.created:  # pragma: no branch
                provider_details['timestamp'] = number_to_datetime(response.created)
            return ModelResponse(
                parts=[],
                usage=self._map_usage(response),
                model_name=response.model,
                timestamp=_now_utc(),
                provider_details=provider_details or None,
                provider_response_id=response.id or None,
                provider_name=self._provider.name,
                provider_url=self._provider.base_url,
                canonical_system=self._canonical_system_stamp,
                finish_reason='content_filter',
            )

        items: list[ModelResponsePart] = []

        if thinking_parts := self._process_thinking(choice.message):
            items.extend(thinking_parts)

        items.extend(
            map_chat_content(
                choice.message,
                self.profile,
                self.system,
                include_raw_annotations=include_raw_annotations,
            )
        )
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                if isinstance(c, ChatCompletionMessageFunctionToolCall):
                    part_provider_details = map_tool_call_provider_details(c)
                    part = ToolCallPart(
                        c.function.name,
                        c.function.arguments,
                        tool_call_id=c.id,
                        provider_name=self.system if part_provider_details else None,
                        provider_details=part_provider_details,
                    )
                elif isinstance(c, ChatCompletionMessageCustomToolCall):  # pragma: no cover
                    # NOTE: Custom tool calls are not supported.
                    # See <https://github.com/pydantic/pydantic-ai/issues/2513> for more details.
                    raise RuntimeError('Custom tool calls are not supported')
                else:
                    assert_never(c)
                part.tool_call_id = _guard_tool_call_id(part)
                items.append(part)

        if response.created:
            provider_details['timestamp'] = number_to_datetime(response.created)

        return ModelResponse(
            parts=items,
            usage=self._map_usage(response),
            model_name=response.model,
            timestamp=timestamp,
            provider_details=provider_details or None,
            provider_response_id=response.id or None,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            canonical_system=self._canonical_system_stamp,
            finish_reason=finish_reason,
        )

    def _process_thinking(self, message: chat.ChatCompletionMessage) -> list[ThinkingPart] | None:
        """Hook that maps reasoning tokens to thinking parts.

        This method may be overridden by subclasses of `OpenAIChatModel` to apply custom mappings.
        """
        profile = self.profile
        custom_field = profile.get('openai_chat_thinking_field', None)
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
            reasoning: object = getattr(message, field_name, None)
            if not reasoning:
                continue
            if not isinstance(reasoning, str):
                warnings.warn(
                    f'Unexpected non-string value for {field_name!r}: {type(reasoning).__name__}. '
                    'Please open an issue at https://github.com/pydantic/pydantic-ai/issues.',
                    UserWarning,
                )
                continue
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
        peekable_response: _utils.PeekableAsyncStream[ChatCompletionChunk, AsyncStream[ChatCompletionChunk]] = (
            _utils.PeekableAsyncStream(response)
        )
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Streamed response ended without content or tool calls'
            )

        # When using Azure OpenAI and a content filter is enabled, the first chunk will contain a `''` model name,
        # so we set it from a later chunk in `OpenAIChatStreamedResponse`.
        model_name = first_chunk.model or self.model_name

        streamed_response = self._streamed_response_cls(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _model_id_namespace=self._provider.model_id_namespace,
            _provider_url=self._provider.base_url,
            _model_settings=model_settings,
        )
        streamed_response.canonical_system = self._canonical_system_stamp
        return streamed_response

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        """Returns the `StreamedResponse` type that will be used for streamed responses.

        This method may be overridden by subclasses of `OpenAIChatModel` to provide their own `StreamedResponse` type.
        """
        return OpenAIStreamedResponse

    def _map_usage(self, response: chat.ChatCompletion) -> usage.RequestUsage:
        return map_usage(response, self._provider.name, self._provider.base_url, self.model_name)

    def _get_tool_choice(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[chat.ChatCompletionToolParam], ChatCompletionToolChoiceOptionParam | None]:
        """Determine which tools to send and the API tool_choice value.

        Returns:
            A tuple of (tools, tool_choice). The tools are filtered down to the allowed subset only when
            `allowed_tools` is unavailable; otherwise they are the full list and `tool_choice` restricts the call.
        """
        resolved_tool_choice = resolve_tool_choice(model_settings, model_request_parameters)
        tool_defs = model_request_parameters.declared_tool_defs

        tool_choice: ChatCompletionToolChoiceOptionParam
        if resolved_tool_choice in ('auto', 'none'):
            tool_choice = resolved_tool_choice
        elif resolved_tool_choice == 'required':
            supports = self._supports_tool_forcing(model_settings, model_request_parameters)
            tool_choice = 'required' if supports else 'auto'
        elif isinstance(resolved_tool_choice, tuple):
            tool_choice_mode, tool_names = resolved_tool_choice
            supports = self._supports_tool_forcing(model_settings, model_request_parameters)
            effective_mode = 'auto' if tool_choice_mode == 'auto' or not supports else tool_choice_mode
            if tool_choice_mode == 'required' and len(tool_names) == 1 and supports:
                tool_choice = {'type': 'function', 'function': {'name': next(iter(tool_names))}}
            elif self.profile.get('openai_chat_supports_allowed_tools', False):
                # `allowed_tools` restricts the choice via the API rather than the tools list, so caching is
                # preserved. Used for the multi-tool case and as the cache-preserving fallback when forcing
                # isn't supported.
                tool_choice = {
                    'type': 'allowed_tools',
                    'allowed_tools': {
                        'mode': effective_mode,
                        # Ignore names that are not currently available, keeping declaration order.
                        'tools': [{'type': 'function', 'function': {'name': n}} for n in tool_defs if n in tool_names],
                    },
                }
            else:
                # Breaks caching, but this API isn't known to support `allowed_tools`.
                tool_defs = {k: v for k, v in tool_defs.items() if k in tool_names}
                tool_choice = effective_mode
        else:
            assert_never(resolved_tool_choice)

        tools: list[chat.ChatCompletionToolParam] = [
            self._map_tool_definition(t, model_settings) for t in tool_defs.values()
        ]
        if not tools:
            return tools, None

        return tools, tool_choice

    def _request_thinks(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> bool:
        return reasoning_active(
            self.profile, cast(OpenAIChatModelSettings, model_settings or {}), model_request_parameters
        )

    def _settings_overriding_thinking(self, model_settings: Mapping[str, object]) -> tuple[str, ...]:
        # `_translate_thinking` sends an explicit effort instead of the one unified `thinking` maps to.
        return ('openai_reasoning_effort',) if model_settings.get('openai_reasoning_effort') else ()

    def _overrides_unified_effort(self, model_settings: Mapping[str, object]) -> bool:
        return bool(self._settings_overriding_thinking(model_settings))

    def _supports_tool_forcing(
        self, model_settings: OpenAIChatModelSettings, model_request_parameters: ModelRequestParameters
    ) -> bool:
        """Allow provider subclasses to express conditional forcing support.

        Overrides should raise `UserError` when the user explicitly requested forcing.
        """
        return support_openai_tool_forcing(
            self.model_name,
            self.profile,
            model_settings,
            thinking=self._request_thinks(model_settings, model_request_parameters),
        )

    def _get_stream_options(self, model_settings: OpenAIChatModelSettings) -> chat.ChatCompletionStreamOptionsParam:
        """Build stream_options for the API request.

        Returns a dict with include_usage=True and optionally continuous_usage_stats if configured.
        """
        options: dict[str, bool] = {'include_usage': True}
        if model_settings.get('openai_continuous_usage_stats'):
            options['continuous_usage_stats'] = True
        return cast(chat.ChatCompletionStreamOptionsParam, options)

    def _finalize_cache_breakpoints(
        self, tools: list[chat.ChatCompletionToolParam], openai_messages: list[chat.ChatCompletionMessageParam]
    ) -> None:
        """Hook for subclasses to adjust cache breakpoints once the tools and messages are both mapped."""

    def _get_web_search_options(self, model_request_parameters: ModelRequestParameters) -> WebSearchOptions | None:
        for tool in model_request_parameters.native_tools:
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
        thinkings: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
        carried_thinkings: list[str] = field(default_factory=list[str])
        """Reasoning rendered for a user message ahead of the assistant message, as `render_replayed_thinking` decides."""
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = field(
            default_factory=list[ChatCompletionMessageFunctionToolCallParam]
        )
        message: ModelResponse | None = field(default=None, init=False)
        """The response being mapped, set by `map_assistant_message`, for deciding which of its stateless data to send back."""

        def map_assistant_message(self, message: ModelResponse) -> chat.ChatCompletionAssistantMessageParam | None:
            self.message = message
            for item in message.parts:
                if isinstance(item, TextPart):
                    self._map_response_text_part(item)
                elif isinstance(item, ThinkingPart):
                    self._map_response_thinking_part(item)
                elif isinstance(item, ToolCallPart):
                    self._map_response_tool_call_part(item)
                elif isinstance(item, NativeToolCallPart | NativeToolReturnPart):  # pragma: no cover
                    self._map_response_builtin_part(item)
                elif isinstance(item, FilePart):  # pragma: no cover
                    self._map_response_file_part(item)
                # Reached from Harness tests that switch to Chat after another provider's compaction.
                elif isinstance(item, CompactionPart):  # pragma: lax no cover
                    # Provider-produced compaction parts are not sent back to the Chat Completions API.
                    pass
                elif isinstance(item, SpeechPart):  # pragma: no cover
                    # Unconverted realtime speech; `prepare_messages` turns these into `TextPart`s in `Model.prepare_messages`.
                    raise _unconverted_speech_part_error()
                else:
                    assert_never(item)
            return self._into_message_param()

        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam | None:
            """Converts the collected texts and tool calls into a single OpenAI `ChatCompletionAssistantMessageParam`.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for how collected parts are transformed into the final message parameter.

            Returns:
                An OpenAI `ChatCompletionAssistantMessageParam` representing the assistant's response,
                or `None` if there is nothing to send (e.g. the `ModelResponse` had no parts because
                the model returned an empty response). Returning `None` ensures we don't emit an
                assistant message with `content=None` and no `tool_calls`, which the Chat Completions
                API rejects with a 400 error.
            """
            if not self.texts and not self.tool_calls:
                return None
            message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
            # Chat Completions stores one `content` string per assistant message, so multiple
            # TextParts in a single response (text after a tool call in the same stream, or
            # history produced by a different model) are joined here:
            if self.thinkings:
                for field_name, contents in self.thinkings.items():
                    message_param[field_name] = '\n\n'.join(contents)
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
            profile = self._model.profile
            include_method = profile.get('openai_chat_send_back_thinking_parts', 'auto')

            # Auto-detect: if thinking came from a custom field and from the same provider, use field mode
            # id='content' means it came from tags in content, not a custom field
            if include_method == 'auto':
                # Check if thinking came from a custom field from the same provider
                custom_field = profile.get('openai_chat_thinking_field', None)
                matches_custom_field = (not custom_field) or (item.id == custom_field)

                if (
                    item.id
                    and item.id != 'content'
                    and item.provider_name == self._model.system
                    and matches_custom_field
                ):
                    # Store both content and field name for later use in _into_message_param
                    self.thinkings.setdefault(item.id, []).append(item.content)
                else:
                    # Fall back to tags mode
                    self._map_response_thinking_as_text(item)
            elif include_method == 'tags':
                self._map_response_thinking_as_text(item)
            elif include_method == 'field':
                field = profile.get('openai_chat_thinking_field', None)
                if field:  # pragma: no branch
                    self.thinkings.setdefault(field, []).append(item.content)

        def accepts_stateless_artifacts(self, item: ModelResponsePart) -> bool:
            """Whether the model sends back the stateless signed or encrypted data `item` carries.

            See [`Model.accepts_stateless_artifacts`][pydantic_ai.models.Model.accepts_stateless_artifacts].
            """
            assert self.message is not None, '`map_assistant_message` sets the response before mapping its parts'
            return self._model.accepts_stateless_artifacts(self.message, item)

        def _map_response_thinking_as_text(self, item: ThinkingPart) -> None:
            text, carry_in_user_turn = render_replayed_thinking(item.content, self._model.profile)
            if not carry_in_user_turn:
                self.texts.append(text)
            elif item.content:
                # An empty wrapper in the user turn would tell the model the assistant thought nothing.
                self.carried_thinkings.append(text)

        def _map_response_tool_call_part(self, item: ToolCallPart) -> None:
            """Maps a `ToolCallPart` to the response context.

            This method serves as a hook that can be overridden by subclasses
            to implement custom logic for handling tool call parts.
            """
            tool_call = self._model._map_tool_call(item)
            if (
                item.provider_details
                and (extra_content := item.provider_details.get('extra_content'))
                and self.accepts_stateless_artifacts(item)
            ):
                tool_call = _ChatCompletionMessageFunctionToolCallParam(**tool_call, extra_content=extra_content)
            self.tool_calls.append(tool_call)

        def _map_response_builtin_part(self, item: NativeToolCallPart | NativeToolReturnPart) -> None:
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

    def _map_model_response(self, message: ModelResponse) -> chat.ChatCompletionMessageParam | None:
        """Hook that determines how `ModelResponse` is mapped into `ChatCompletionMessageParam` objects before sending.

        Returns `None` to skip emitting any message for this `ModelResponse` (e.g. when it has no parts).

        Subclasses of `OpenAIChatModel` may override this method to provide their own mapping logic.
        """
        return self._MapModelResponseContext(self).map_assistant_message(message)

    def _map_carried_thinking(self, message: ModelResponse) -> chat.ChatCompletionUserMessageParam | None:
        """Build the user message that carries this `ModelResponse`'s reasoning ahead of its assistant message.

        That is reasoning that can't be sent natively, for a profile with
        [`mimics_assistant_message_formatting`][pydantic_ai.profiles.ModelProfile.mimics_assistant_message_formatting]:
        replayed in the assistant turn, it teaches the model to write thinking tags into its visible answers.
        `_MapModelResponseContext` decides which reasoning that is while mapping the assistant message, and
        leaves it out of that message. This maps the response a second time to collect it, so that
        `_map_model_response` keeps returning a single message for subclasses that override it.
        Returns `None` when there is nothing to carry.
        """
        if not any(isinstance(part, ThinkingPart) for part in message.parts):
            return None
        context = self._MapModelResponseContext(self)
        context.map_assistant_message(message)
        if not context.carried_thinkings:
            return None
        return chat.ChatCompletionUserMessageParam(role='user', content='\n\n'.join(context.carried_thinkings))

    def _map_finish_reason(self, key: str) -> FinishReason | None:
        """Hooks that maps a finish reason key to a [FinishReason][pydantic_ai.messages.FinishReason].

        This method may be overridden by subclasses of `OpenAIChatModel` to accommodate custom keys.
        """
        return CHAT_FINISH_REASON_MAP[key] if key in CHAT_FINISH_REASON_MAP else None

    async def _map_messages(  # noqa: C901
        self,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        *,
        model_settings: ModelSettings | None = None,
    ) -> list[chat.ChatCompletionMessageParam]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        openai_messages: list[chat.ChatCompletionMessageParam] = []
        profile = self.profile
        # DeepSeek-style APIs 400 on a tool-calling assistant turn that omits their thinking field while
        # the run is reasoning. A turn another model or provider produced, or one served with thinking off,
        # can lack it: a `FallbackModel` step, a resumed run, or a foreign native tool search that
        # `prepare_messages` splits into a synthesized `search_tools` turn. `'field'` mode sets the configured
        # field empty on every such turn from the first response that holds a `ThinkingPart` on, or, for a
        # profile that sets `openai_chat_thinking_requires_field_on_tool_calls`, on every such turn once the
        # request itself thinks, as it can before any step has thought; `'auto'` mode is handled after mapping,
        # below. A turn before the first `ThinkingPart` was already sent without the field, so a later response
        # that thinks doesn't add it there: the turn goes out the same way in every request, keeping the
        # cached prefix.
        send_back_thinking = profile.get('openai_chat_send_back_thinking_parts', 'auto')
        backfill_field = profile.get('openai_chat_thinking_field', None) if send_back_thinking == 'field' else None
        backfill = bool(
            backfill_field
            and profile.get('openai_chat_thinking_requires_field_on_tool_calls', False)
            and self._request_thinks(model_settings, model_request_parameters)
        )
        for message in messages:
            if isinstance(message, ModelRequest):
                async for item in self._map_user_message(message):
                    openai_messages.append(item)
            elif isinstance(message, ModelResponse):
                backfill = backfill or any(isinstance(part, ThinkingPart) for part in message.parts)
                if (carried_thinking := self._map_carried_thinking(message)) is not None:
                    openai_messages.append(carried_thinking)
                if (mapped := self._map_model_response(message)) is not None:
                    if (
                        backfill
                        and backfill_field
                        and mapped['role'] == 'assistant'
                        and mapped.get('tool_calls')
                        and backfill_field not in mapped
                    ):
                        mapped[backfill_field] = ''
                    openai_messages.append(mapped)
            else:
                assert_never(message)
        if send_back_thinking == 'auto':
            # `'auto'` learns the field only from the mapped messages, so it backfills only when they carry
            # exactly one of the names `_process_thinking` reads: a server that rejects unknown fields has
            # already accepted that one. DeepSeek's guide asks for the field on every turn of a request with
            # tools, but in practice it rejected a current-turn tool call without it and accepted an
            # earlier-turn one (`test_deepseek_auto_mode_thinking_backfill` records the accepted requests),
            # so earlier turns go out unchanged. A message backfilled here goes out without the field again
            # once a newer user message follows it.
            custom_field = profile.get('openai_chat_thinking_field', None)
            sent_fields = {
                name
                for message in openai_messages
                if message['role'] == 'assistant'
                for name in (custom_field, 'reasoning', 'reasoning_content')
                if name and name in message
            }
            if len(sent_fields) == 1:
                (sent_field,) = sent_fields
                last_user_index = max(
                    (i for i, message in enumerate(openai_messages) if message['role'] == 'user'), default=-1
                )
                for message in openai_messages[last_user_index + 1 :]:
                    if message['role'] == 'assistant' and message.get('tool_calls') and sent_field not in message:
                        message[sent_field] = ''
        system_prompt_role = profile.get('openai_system_prompt_role', None) or 'system'
        system_prompt_count = leading_system_message_count(openai_messages, system_prompt_role)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        if instruction_parts:
            if system_prompt_role == 'developer':
                instruction_messages: list[chat.ChatCompletionMessageParam] = [
                    chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                    for part in instruction_parts
                ]
            elif system_prompt_role == 'user':
                instruction_messages = [
                    chat.ChatCompletionUserMessageParam(role='user', content=part.content) for part in instruction_parts
                ]
            else:
                instruction_messages = [
                    chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
                    for part in instruction_parts
                ]
            openai_messages[system_prompt_count:system_prompt_count] = instruction_messages
        if (
            model_settings
            and model_settings.get('openai_cache_instructions')
            and profile.get('openai_supports_prompt_cache_breakpoints', False)
            # A `'user'` system prompt role can't be told apart from a real user turn, and merging
            # the leading messages collapses the boundary into one block, so neither can carry it.
            and system_prompt_role != 'user'
            and profile.get('openai_chat_supports_multiple_system_messages', True)
            # A dynamic system prompt changes between requests, so it can't sit in the cached prefix.
            and not has_dynamic_system_prompt(messages)
        ):
            static_count = cacheable_instruction_count(instruction_parts)
            breakpoint_index = system_prompt_count + static_count - 1
            if breakpoint_index >= 0:
                target = cast(
                    'chat.ChatCompletionSystemMessageParam | chat.ChatCompletionDeveloperMessageParam',
                    openai_messages[breakpoint_index],
                )
                content = target['content']
                content_parts = (
                    [ChatCompletionContentPartTextParam(type='text', text=content)]
                    if isinstance(content, str)
                    else list(content)
                )
                add_openai_prompt_cache_breakpoint(content_parts)
                target['content'] = content_parts
        if not self.profile.get('openai_chat_supports_multiple_system_messages', True):
            openai_messages = merge_leading_system_messages(openai_messages, system_prompt_role)
        # After instructions are inserted and system messages merged: the breakpoint may land on a system
        # message, whose string content it turns into a list, and the merge only joins strings.
        move_leading_cache_breakpoints(openai_messages, text_type='text')
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
        if self.profile.get('openai_supports_strict_tool_definition', True):  # pragma: no branch
            response_format_param['json_schema']['strict'] = o.strict
        return response_format_param

    def _map_tool_definition(self, f: ToolDefinition, model_settings: ModelSettings) -> chat.ChatCompletionToolParam:
        """Map a tool definition to an OpenAI tool parameter.

        This method may be overridden by subclasses to apply custom tool mappings. `model_settings` is
        typed as the base `ModelSettings` so subclass overrides can read provider-specific keys without
        narrowing the type.
        """
        tool_param: chat.ChatCompletionToolParam = {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description or '',
                'parameters': f.parameters_json_schema,
            },
        }
        if f.strict and self.profile.get('openai_supports_strict_tool_definition', True):
            tool_param['function']['strict'] = f.strict
        return tool_param

    async def _map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
        file_content: list[UserContent] = []
        for part in _sendable_request_parts(message.parts):
            if isinstance(part, SystemPromptPart):
                system_prompt_role = self.profile.get('openai_system_prompt_role', None)
                if system_prompt_role == 'developer':
                    yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
                elif system_prompt_role == 'user':
                    yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
                else:
                    yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
            elif isinstance(part, UserPromptPart):
                yield await self._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                tool_text, tool_file_content = part.model_response_str_and_user_content()
                file_content.extend(tool_file_content)
                yield chat.ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=_guard_tool_call_id(t=part),
                    content=tool_text,
                )
            elif isinstance(part, (InstructionBaselinePart, InstructionDeltaPart)):
                raise _unprojected_instruction_delta_error()
            elif isinstance(part, ModelSettingsDeltaPart):
                if (update := self._map_model_settings_delta(part)) is not None:
                    yield update
            elif isinstance(part, _UnpreparedPart):  # pragma: no cover
                raise _unprepared_part_error(part)
            elif isinstance(part, SpeechPart):
                # Unconverted realtime speech; `prepare_messages` turns these into `UserPromptPart`s in `Model.prepare_messages`.
                raise _unconverted_speech_part_error()
            else:
                assert_never(part)
        if file_content:
            yield await self._map_user_prompt(UserPromptPart(content=file_content))

    def _map_model_settings_delta(self, part: ModelSettingsDeltaPart) -> chat.ChatCompletionMessageParam | None:
        """Map a `ModelSettingsDeltaPart` to the message that changes the setting mid-conversation, or `None` for none.

        Chat Completions has no such message, so `prepare_messages` drops the parts and this raises. A subclass that
        sets `_renders_effort_deltas` for an API that extends Chat Completions with one overrides it.
        """
        raise _unprepared_part_error(part)

    async def _map_image_url_item(self, item: ImageUrl) -> ChatCompletionContentPartImageParam:
        """Map an ImageUrl to a chat completion image content part."""
        image_url: ImageURL = {'url': item.url}
        if metadata := item.vendor_metadata:
            image_url['detail'] = metadata.get('detail', 'auto')
        if item.force_download:
            image_content = await download_item(item, data_format='base64_uri', type_format='extension')
            image_url['url'] = image_content['data']
        return ChatCompletionContentPartImageParam(image_url=image_url, type='image_url')

    async def _map_binary_content_item(self, item: BinaryContent) -> ChatCompletionContentPartParam:
        """Map a BinaryContent item to a chat completion content part."""
        profile = self.profile
        if _is_text_like_media_type(item.media_type):
            # Inline text-like binary content as a text block
            return self._inline_text_file_part(
                item.data.decode('utf-8'),
                media_type=item.media_type,
                identifier=item.identifier,
            )
        elif item.is_image:
            image_url = ImageURL(url=item.data_uri)
            if metadata := item.vendor_metadata:
                image_url['detail'] = metadata.get('detail', 'auto')
            return ChatCompletionContentPartImageParam(image_url=image_url, type='image_url')
        elif item.is_audio:
            assert item.format in ('wav', 'mp3')
            if profile.get('openai_chat_audio_input_encoding', 'base64') == 'uri':
                audio = InputAudio(data=item.data_uri, format=item.format)
            else:
                audio = InputAudio(data=item.base64, format=item.format)
            return ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio')
        elif item.is_document:
            if not profile.get('openai_chat_supports_document_input', True):
                self._raise_document_input_not_supported_error()
            return File(
                file=FileFile(
                    file_data=item.data_uri,
                    filename=f'filename.{item.format}',
                ),
                type='file',
            )
        elif item.is_video:
            raise NotImplementedError('VideoUrl is not supported in OpenAI Chat Completions user prompts')
        else:  # pragma: no cover
            raise RuntimeError(f'Unsupported binary content type: {item.media_type}')

    async def _map_audio_url_item(self, item: AudioUrl) -> ChatCompletionContentPartInputAudioParam:
        """Map an AudioUrl to a chat completion audio content part."""
        profile = self.profile
        data_format = 'base64_uri' if profile.get('openai_chat_audio_input_encoding', 'base64') == 'uri' else 'base64'
        downloaded_item = await download_item(item, data_format=data_format, type_format='extension')
        assert downloaded_item['data_type'] in (
            'wav',
            'mp3',
        ), f'Unsupported audio format: {downloaded_item["data_type"]}'
        audio = InputAudio(data=downloaded_item['data'], format=downloaded_item['data_type'])
        return ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio')

    async def _map_document_url_item(self, item: DocumentUrl) -> ChatCompletionContentPartParam:
        """Map a DocumentUrl to a chat completion content part."""
        profile = self.profile
        # OpenAI Chat API's FileFile only supports base64-encoded data, not URLs.
        # Some providers (e.g., OpenRouter) support URLs via the profile flag.
        if not item.force_download and profile.get('openai_chat_supports_file_urls', False):
            return File(
                file=FileFile(
                    file_data=item.url,
                    filename=f'filename.{item.format}',
                ),
                type='file',
            )
        if _is_text_like_media_type(item.media_type):
            downloaded_text = await download_item(item, data_format='text')
            return self._inline_text_file_part(
                downloaded_text['data'],
                media_type=item.media_type,
                identifier=item.identifier,
            )
        else:
            if not profile.get('openai_chat_supports_document_input', True):
                self._raise_document_input_not_supported_error()
            downloaded_item = await download_item(item, data_format='base64_uri', type_format='extension')
            return File(
                file=FileFile(
                    file_data=downloaded_item['data'],
                    filename=f'filename.{downloaded_item["data_type"]}',
                ),
                type='file',
            )

    async def _map_video_url_item(self, item: VideoUrl) -> ChatCompletionContentPartParam:
        """Map a VideoUrl to a chat completion content part."""
        raise NotImplementedError('VideoUrl is not supported in OpenAI Chat Completions user prompts')

    async def _map_content_item(
        self,
        item: str
        | TextContent
        | ImageUrl
        | BinaryContent
        | AudioUrl
        | DocumentUrl
        | VideoUrl
        | UploadedFile
        | CachePoint,
    ) -> ChatCompletionContentPartParam | None:
        """Map a single content item to a chat completion content part, or None to filter it out."""
        if isinstance(item, str | TextContent):
            text = item if isinstance(item, str) else item.content
            return ChatCompletionContentPartTextParam(text=text, type='text')
        elif isinstance(item, ImageUrl):
            return await self._map_image_url_item(item)
        elif isinstance(item, BinaryContent):
            return await self._map_binary_content_item(item)
        elif isinstance(item, AudioUrl):
            return await self._map_audio_url_item(item)
        elif isinstance(item, DocumentUrl):
            return await self._map_document_url_item(item)
        elif isinstance(item, VideoUrl):
            return await self._map_video_url_item(item)
        elif isinstance(item, UploadedFile):
            self._validate_uploaded_file_provider(item)
            if item.media_type.startswith('image/'):
                # Chat Completions can only reference an uploaded file as a document `file` part;
                # `image_url` parts take a URL/data URI, not a `file_id`, so images can't be referenced by id.
                raise UserError(
                    'Referencing an uploaded image by `file_id` is not supported by OpenAIChatModel. '
                    'Use `ImageUrl` or `BinaryContent` for images, or use `OpenAIResponsesModel`.'
                )
            return File(
                file=FileFile(file_id=item.file_id),
                type='file',
            )
        elif isinstance(item, CachePoint):
            # Cache points are handled by `_map_user_prompt_content_item()` when supported.
            return None
        else:
            assert_never(item)

    async def _map_user_prompt_content_item(
        self, item: UserContent, content: list[ChatCompletionContentPartParam]
    ) -> None:
        """Map a single user-prompt content item onto the outgoing `content` list.

        Stable protected hook: subclasses override this to intercept user-prompt content items
        before the default mapping (e.g. `OpenRouterModel` translates `CachePoint` into a
        `cache_control` breakpoint on the preceding part).
        """
        if isinstance(item, CachePoint) and self.profile.get('openai_supports_prompt_cache_breakpoints', False):
            if not content:
                content.append(ChatCompletionContentPartTextParam(text='', type='text'))
            add_openai_prompt_cache_breakpoint(content)
        else:
            mapped_item = await self._map_content_item(item)
            if mapped_item is not None:
                content.append(mapped_item)

    async def _map_user_prompt(self, part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
        content: str | list[ChatCompletionContentPartParam]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                await self._map_user_prompt_content_item(item, content)
        return chat.ChatCompletionUserMessageParam(role='user', content=content)

    def _raise_document_input_not_supported_error(self) -> Never:
        if is_azure(self.client, self.system):
            raise UserError(
                "Azure's Chat Completions API does not support document input. "
                'Use `OpenAIResponsesModel` with `AzureProvider` instead.'
            )
        raise UserError(
            f'The {self._provider.name!r} provider does not support document input via the Chat Completions API.'
        )

    def _inline_text_file_part(
        self, text: str, *, media_type: str, identifier: str
    ) -> ChatCompletionContentPartTextParam:
        if not self.profile.get('supports_inline_system_prompts', False):
            # Inlined as user text, where the framework's own voice is `<system>`-tagged user text too.
            text = neutralize_voice_tag(text)
        return ChatCompletionContentPartTextParam(
            text=self._format_inlined_text_file(text, media_type=media_type, identifier=identifier),
            type='text',
        )
