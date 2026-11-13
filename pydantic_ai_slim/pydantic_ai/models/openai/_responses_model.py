"""`OpenAIResponsesModel`, the Responses API model."""

from __future__ import annotations as _annotations

import asyncio
import functools
import importlib
import warnings
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from functools import cached_property
from random import random
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Self, assert_never, cast, overload

import anyio
from httpx2 import URL as HTTPX2URL, Timeout as HTTPX2Timeout
from openai import (
    NOT_GIVEN,
    APIConnectionError,
    APIStatusError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AsyncStream,
    NotGiven,
    Omit,
    OpenAIError,
    omit as OMIT,
)
from openai.types import responses
from openai.types.chat.completion_create_params import Moderation
from openai.types.responses import ResponseCompactionItem, ToolSearchToolParam
from openai.types.responses.response_compaction_item_param_param import ResponseCompactionItemParamParam
from openai.types.responses.response_create_params import (
    ContextManagement,
    PromptCacheOptions as ResponsesPromptCacheOptions,
    ToolChoice as ResponsesToolChoice,
)
from openai.types.responses.response_input_file_content_param import ResponseInputFileContentParam
from openai.types.responses.response_input_image_content_param import ResponseInputImageContentParam
from openai.types.responses.response_input_item_param import ToolSearchCall as ToolSearchCallParam
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam
from openai.types.responses.response_reasoning_item_param import (
    Content as ReasoningContent,
    Summary as ReasoningSummary,
)
from openai.types.responses.tool_choice_allowed_param import ToolChoiceAllowedParam
from openai.types.responses.tool_choice_function_param import ToolChoiceFunctionParam
from openai.types.shared_params import Reasoning
from openai.types.websocket_connection_options import WebSocketConnectionOptions
from pydantic_core import to_json
from typing_extensions import TypedDict

from ... import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from ..._http import to_httpx2_timeout
from ..._instrumentation import current_model_request_span, get_instructions
from ..._output import DEFAULT_OUTPUT_TOOL_NAME
from ..._run_context import RunContext
from ..._thinking_part import render_replayed_thinking
from ..._utils import (
    guard_tool_call_id as _guard_tool_call_id,
    is_str_dict as _is_str_dict,
    now_utc as _now_utc,
    number_to_datetime,
)
from ...durable_exec._operation_backend import in_durable_unit
from ...exceptions import ModelConnectionError, SuspendedResponseExpired, UnsupportedSettingWarning, UserError
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
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelSettingsDeltaPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchReturnPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
    _compaction_part_is_wire_boundary,  # pyright: ignore[reportPrivateUsage]
    is_multi_modal_content,
)
from ...native_tools import (
    AbstractNativeTool,
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    WebSearchTool,
)
from ...native_tools._tool_search import ToolSearchTool
from ...output import OutputObjectDefinition
from ...profiles import ModelProfile, ModelProfileSpec, merge_profile
from ...profiles.openai import OpenAIModelProfile, OpenAISystemPromptRole
from ...providers import Provider, infer_provider
from ...providers._oauth_provider import OAuthCredentialsProvider
from ...providers.gateway import is_gateway_provider
from ...settings import CacheSetting, ModelSettings, merge_model_settings
from ...tools import ToolDefinition
from .. import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    OpenAIResponsesCompatibleProvider,
    StreamedResponse,
    ToolVisibility,
    _effort_window_baseline,  # pyright: ignore[reportPrivateUsage]
    _sendable_request_parts,  # pyright: ignore[reportPrivateUsage]
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unprepared_part_error,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
    _unprojected_instruction_delta_error,  # pyright: ignore[reportPrivateUsage]
    _with_default_user_agent,  # pyright: ignore[reportPrivateUsage]
    _without_model_settings_deltas,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from .._decode_errors import check_json_response, map_decode_errors
from .._history_digest import Hasher, continuation_digest, history_hasher
from .._sdk_retries import with_max_retries
from .._tool_choice import ResolvedToolChoice, resolve_tool_choice
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
from ._chat_model import preload_openai_sdk_resource_modules
from ._citations import (
    map_responses_citations,
    map_responses_citations_for_replay,
    responses_citations_provider_details,
    responses_output_text_annotations_ta,
)
from ._errors import (
    check_azure_content_filter,
    is_azure,
    map_api_errors,
    map_connection_error,
    map_status_error,
    response_error,
    response_error_type,
    status_error,
)
from ._history_digest import with_history_digest
from ._model_names import OpenAIModelName
from ._model_response_streaming import ModelResponseStreamedResponse
from ._native_tools import (
    map_code_interpreter_tool_call,
    map_code_interpreter_tool_param,
    map_file_search_tool_call,
    map_file_search_tool_param,
    map_image_generation_tool_call,
    map_mcp_call,
    map_mcp_list_tools,
    map_mcp_server_tool_param,
    map_openai_image_generation_tool,
    map_web_search_call_for_replay,
    map_web_search_tool_call,
    map_web_search_tool_param,
)
from ._reasoning import (
    drop_sampling_params_for_reasoning,
    map_reasoning_item,
    reasoning_active,
    resolve_openai_thinking_effort,
    set_reasoning_context,
)
from ._response_fields import (
    RESPONSES_FINISH_REASON_MAP,
    map_compaction_item,
    map_logprobs,
    map_usage,
    response_refusal,
    response_status_to_state,
    response_tool_call_id,
)
from ._responses_streaming import OpenAIResponsesStreamedResponse
from ._settings import (
    OpenAIChatModelSettings,
    OpenAIResponsesModelSettings,
    drop_unsupported_params,
    echoed_store,
    resolve_openai_service_tier,
    support_openai_tool_forcing,
    with_profile_unsupported_settings,
)
from ._tool_search import (
    build_tool_search_output_param,
    build_tool_search_return_part,
    find_search_tool_definition,
    has_tool_search,
    lacks_tool_search_output_identity,
    map_client_tool_search_call,
    map_tool_search_call,
    tool_search_call_ids_by_output,
    tool_search_namespace_for_synthesis,
    tool_search_replay_details,
)
from ._websocket_holder import ResponsesWebSocketHolder

if TYPE_CHECKING:
    from ..batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
    from ._responses_websocket import ResponsesWebSocketStream


_OPENAI_BACKGROUND_POLL_INTERVAL = 2.0


_RESPONSES_TRANSPORT_ATTRIBUTE = 'pydantic_ai.transport'


# Initial/max backoff and downward jitter follow the OpenAI SDK's retry algorithm.
_WEBSOCKET_RETRY_INITIAL_DELAY = 0.5


_WEBSOCKET_RETRY_MAX_DELAY = 8.0


_WEBSOCKET_RETRY_JITTER_RATIO = 0.25


_WEBSOCKET_RETRY_MAX_EXPONENT = 4  # The base delay has reached its cap after four doublings.


# Longer server-directed waits remain caller-owned rather than blocking a socket request.
_WEBSOCKET_RETRY_AFTER_LIMIT = 60.0


_DEFAULT_CLIENT_TOOL_SEARCH_DESCRIPTION = 'Search for relevant tools.'


class _OpenAIResponsesContinuationDetails(TypedDict, total=False):
    """Provider details for OpenAI Responses API continuation."""

    last_sequence_number: int


_ResponsesTimeout = float | HTTPX2Timeout | NotGiven


@dataclass
class _ResponsesRequestParams:
    """Typed request parameters shared by Responses API calls."""

    model: OpenAIModelName
    input: list[responses.ResponseInputItemParam]
    instructions: str | Omit
    parallel_tool_calls: bool | Omit
    tools: list[responses.ToolParam] | Omit
    tool_choice: ResponsesToolChoice | Omit
    previous_response_id: str | Omit
    conversation: str | Omit
    reasoning: Reasoning | Omit
    text: responses.ResponseTextConfigParam | Omit
    truncation: Literal['auto', 'disabled'] | Omit
    context_management: list[ContextManagement] | Omit


class _ResponsesCreateParams(TypedDict):
    """Body arguments of a response creation, shared by `client.responses.create` and the batch API."""

    model: OpenAIModelName
    input: list[responses.ResponseInputItemParam]
    instructions: str | Omit
    parallel_tool_calls: bool | Omit
    tools: list[responses.ToolParam] | Omit
    tool_choice: ResponsesToolChoice | Omit
    previous_response_id: str | Omit
    reasoning: Reasoning | Omit
    text: responses.ResponseTextConfigParam | Omit
    truncation: Literal['auto', 'disabled'] | Omit
    context_management: list[ContextManagement] | Omit
    max_output_tokens: int | Omit
    temperature: float | Omit
    top_p: float | Omit
    service_tier: responses.ServiceTier | Omit
    conversation: str | Omit
    top_logprobs: int | Omit
    store: bool | Omit | None
    user: str | Omit
    include: list[responses.ResponseIncludable] | Omit
    prompt_cache_key: str | Omit
    prompt_cache_retention: Literal['in_memory', '24h'] | Omit
    prompt_cache_options: ResponsesPromptCacheOptions | Omit
    moderation: Moderation | Omit


def batch_module():
    """The batch module, imported on first use, so an error importing it doesn't surface as a missing `openai` SDK."""
    from . import _batch

    return _batch


@dataclass(init=False)
class OpenAIResponsesModel(Model[AsyncOpenAI]):
    """A model that uses the OpenAI Responses API.

    The [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) is the
    new API for OpenAI models.

    If you are interested in the differences between the Responses API and the Chat Completions API,
    see the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).
    """

    unsupported_settings = frozenset(
        {'frequency_penalty', 'logit_bias', 'presence_penalty', 'seed', 'stop_sequences', 'top_k'}
    )

    supported_tool_deferral_modes = frozenset({'with_tool_search'})
    supported_tool_addition_modes = frozenset({'with_definitions'})
    _renders_effort_deltas = True
    # The API takes a lookalike note ahead of a request's tool results.
    _lookalike_notes_precede_tool_results = True
    # Responses only: `OpenAIChatModel` shares this model family's profile but has no compaction
    # surface, and the Responses API bills replayed pre-boundary items, so the trim is what makes
    # compaction actually compact here.
    #
    # The compaction item is an opaque blob the API decrypts; without it there is nothing to send in
    # place of the history the boundary would hide.
    compaction_requires_encrypted_content = True
    # `SystemPromptPart`s render as `system` input items *inside* the compacted window, and the item
    # keeps serving them: a latent directive that never fired before the boundary still governs
    # post-compaction replies without being re-sent (live-verified).
    compaction_retains_standing_prompt = True
    _honors_compaction_boundaries = True
    # Audio content goes nowhere the API reads it and video has no input type, so neither is sent.
    _unsupported_file_inputs = frozenset({'audio', 'video'})

    _model_name: OpenAIModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)
    _websocket: ResponsesWebSocketHolder = field(repr=False, compare=False)
    _history_digest_key: bytes | None = field(default=None, repr=False)
    _websocket_connection_options: WebSocketConnectionOptions | None = field(default=None, repr=False)
    _websocket_stream: ResponsesWebSocketStream | None = field(default=None, repr=False, compare=False)
    """The lane a copy of the model sends its one request on, over the Responses WebSocket."""

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        # A Responses WebSocket a request opens stays open until the model exits, so later runs reuse it.
        # Engines enter freshly resolved models inside individual units. That scope cannot amortize a socket.
        self._websocket.acquire(context=True, reusable=not in_durable_unit())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self._websocket.exit()
        return await super().__aexit__(exc_type, exc_val, exc_tb)

    def _run_scope(self) -> AbstractAsyncContextManager[None]:
        return self._websocket.lease()

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
        history_digest_key: bytes | None = None,
        websocket_connection_options: WebSocketConnectionOptions | None = None,
    ):
        """Initialize an OpenAI Responses model.

        Args:
            model_name: The name of the OpenAI model to use.
            provider: The provider to use. Defaults to `'openai'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
            history_digest_key: A secret to key each response's
                [`history_digest`][pydantic_ai.messages.ModelResponse.history_digest] with (an HMAC-SHA256), so
                stored history can't be used to test guesses at content redacted from it later. Every model
                continuing the same conversations needs the same key; a response recorded with another key, or
                none, is resent rather than continued from.
            websocket_connection_options: Options for the `websockets` connection of the Responses WebSocket that
                [`openai_transport='websocket'`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_transport]
                opens, typed as the OpenAI SDK's `WebSocketConnectionOptions`. `max_size` and `max_queue` default to
                `None`, which lifts the `websockets` limits on message size and on unread frames.
        """
        self._model_name = model_name
        self._history_digest_key = history_digest_key
        self._websocket_connection_options = websocket_connection_options
        # Shared by reference with the copies a request makes, so they all use one socket.
        self._websocket = ResponsesWebSocketHolder()

        if isinstance(provider, str):
            provider = infer_provider('gateway/openai' if provider == 'gateway' else provider)
        self._provider = provider

        super().__init__(settings=settings, profile=profile)

        preload_openai_sdk_resource_modules(self, self._provider.client)

    @property
    def client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def batch_max_requests(self) -> int | None:
        # A batch runs each request without streaming, which an endpoint that only streams can't serve.
        if not self.profile.get('openai_supports_batch_api', False) or self.profile.get(
            'openai_responses_requires_streaming', False
        ):
            return None
        return 50_000

    @property
    def batch_max_bytes(self) -> int | None:
        return None if self.batch_max_requests is None else batch_module().MAX_BATCH_BYTES

    def batch_request_size(self, request: EncodedBatchRequest) -> int:
        return batch_module().request_size(request)

    def _batch_credential(self) -> str:
        return batch_module().batch_credential(self)

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        return await batch_module().batch_encode(self, request)

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        with map_api_errors(self.model_name, self._provider.model_id_namespace):
            return await batch_module().batch_create(self, requests)

    async def batch_status(self, batch: Batch) -> Batch:
        with map_api_errors(self.model_name, self._provider.model_id_namespace):
            return await batch_module().batch_status(self, batch)

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        with map_api_errors(self.model_name, self._provider.model_id_namespace):
            return await batch_module().batch_results(self, batch)

    async def batch_cancel(self, batch: Batch) -> Batch:
        with map_api_errors(self.model_name, self._provider.model_id_namespace):
            return await batch_module().batch_cancel(self, batch)

    def batch_decode(self, result: BatchResult, request: BatchRequest) -> ModelResponse:
        return batch_module().batch_decode(self, result, request)

    @property
    def model_name(self) -> OpenAIModelName:
        """The model name."""
        return self._model_name

    def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        """Resolve the prompt cache retention requested by OpenAI settings or the unified `cache` setting."""
        return resolve_legacy_cache_retention(
            self.profile, self.settings, model_settings
        ) or super().resolve_cache_retention(model_settings)

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

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        model_settings, model_request_parameters = super().prepare_request(model_settings, model_request_parameters)
        model_settings = translate_openai_cache(self.profile, model_settings, model_request_parameters)
        if (
            model_settings is not None
            and model_settings.get('openai_transport') == 'websocket'
            and 'openai_previous_response_id' not in model_settings
            and 'openai_conversation_id' not in model_settings
        ):
            # Keep continuation consistent when a request in WebSocket mode falls back to HTTP.
            continued: OpenAIResponsesModelSettings = {**model_settings, 'openai_previous_response_id': 'auto'}
            return continued, model_request_parameters
        return model_settings, model_request_parameters

    def _unsupported_settings(self) -> frozenset[str]:
        return with_profile_unsupported_settings(self.unsupported_settings, self.profile)

    def _websocket_unsupported_reason(self) -> str | None:
        """Why this request cannot use a Responses WebSocket, or `None` when it can."""
        if not self.profile.get('openai_responses_supports_websocket', True):
            return (
                'its profile sets `openai_responses_supports_websocket=False`, as its provider, SDK client or model '
                "class can't open a Responses WebSocket. To try one on Azure OpenAI or the Pydantic AI Gateway "
                'with static credentials, pass '
                '`profile=OpenAIModelProfile(openai_responses_supports_websocket=True)` to the model.'
            )
        # Profiles are cached, but the HTTP client's authentication can change after the socket opened.
        # Azure clients and lightweight SDK wrappers do not initialize every optional attribute.
        client = self._provider.client
        if (
            getattr(client, '_api_key_provider', None) is not None
            or getattr(client, 'workload_identity', None) is not None
            or callable(getattr(client, '_azure_ad_token_provider', None))
            or getattr(getattr(client, '_client', None), 'auth', None) is not None
        ):
            return 'the SDK client resolves authentication per request; use HTTP to apply its current credentials.'
        try:
            importlib.import_module('pydantic_ai.models.openai._responses_websocket')
        except ImportError:
            return 'install `pydantic-ai-slim[openai-realtime]` to use Responses WebSockets.'
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return 'Responses WebSockets need the asyncio event loop.'
        return None

    def _warn_websocket_unsupported(self, reason: str) -> None:
        warnings.warn(
            f"The `openai_transport='websocket'` setting cannot open a socket for model {self.model_id!r}; using HTTP "
            f"with the mode's `openai_previous_response_id='auto'` default and history metadata: {reason}",
            UnsupportedSettingWarning,
            stacklevel=2,
        )

    @asynccontextmanager
    async def _websocket_transport(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncGenerator[Self | None]:
        """Yield a copy of this model bound to a lane of its Responses WebSocket, or `None` to send the request over HTTP.

        Only a request whose settings ask for `openai_transport='websocket'` uses the socket, and only while an agent
        run, `async with model:` or `async with agent:` holds it open on the request's event loop.
        """
        merged_settings = merge_model_settings(self.settings, model_settings)
        settings: OpenAIResponsesModelSettings = {**merged_settings} if merged_settings else {}
        if settings.get('openai_transport') != 'websocket':
            yield None
            return
        if (reason := self._websocket_unsupported_reason()) is not None:
            self._warn_websocket_unsupported(reason)
            yield None
            return
        loop = asyncio.get_running_loop()
        holder = self._websocket
        extra_body = settings.get('extra_body')
        with holder.lock:
            state = holder.states.get(loop)
            if state is not None and state.leases and state.reusable:
                state.in_flight += 1
            else:
                state = None
        if state is None:
            yield None
            return
        try:
            if (
                # The socket can't run a background response, or retrieve a suspended one.
                settings.get('openai_background')
                or (_is_str_dict(extra_body) and extra_body.get('background'))
                or self._get_continuation_info(messages, settings) is not None
            ):
                yield None
                return
            response = await self._reserve_websocket_lane(loop, messages, settings)
            if response is None:
                yield None
                return
            async with response:
                yield _utils.replace_no_init(self, _websocket_stream=response)
        finally:
            with holder.lock:
                state.in_flight -= 1
            await holder.close_if_unused(state)

    async def _reserve_websocket_lane(
        self, loop: asyncio.AbstractEventLoop, messages: Sequence[ModelMessage], settings: OpenAIResponsesModelSettings
    ) -> ResponsesWebSocketStream | None:
        """Reserve a lane for a request on this model's socket, opening one first if needed; `None` to use HTTP."""
        from ._responses_websocket import NAMED_LANE_LIMIT, ResponsesWebSocket

        headers = {key.lower(): value for key, value in settings.get('extra_headers', {}).items()}
        headers.setdefault('user-agent', get_user_agent())
        _, timeout = self._build_request_options(settings)
        stream_id_lanes = self.profile.get('openai_responses_supports_websocket_stream_id')
        if stream_id_lanes is None:
            # This is the one host-based exception `models/AGENTS.md` names: lanes on an endpoint that doesn't route
            # events by `stream_id` hand one conversation another's answer, while lanes off only send concurrent
            # requests over HTTP. OpenAI runs named lanes; Azure OpenAI and xAI document one response in flight per
            # connection, and xAI accepts a `stream_id` without echoing it on its events. The socket's host is
            # `websocket_base_url`'s when the SDK client sets one.
            # https://developers.openai.com/api/docs/guides/websocket-mode
            # https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/websockets
            # https://docs.x.ai/developers/advanced-api-usage/websocket-mode
            websocket_url = HTTPX2URL(self.client.websocket_base_url or self.client.base_url)
            stream_id_lanes = websocket_url.host == 'api.openai.com'
        connect = functools.partial(
            ResponsesWebSocket.connect,
            self.client,
            self.model_name,
            headers,
            self._resolve_websocket_timeout(timeout),
            self._websocket_connection_options or {},
            NAMED_LANE_LIMIT if stream_id_lanes else 0,
            functools.partial(status_error, self.model_name, self._provider.model_id_namespace),
        )
        # The lane that cached the conversation's latest response can continue from it.
        response_id = next(
            (
                m.provider_response_id
                for m in reversed(messages)
                if isinstance(m, ModelResponse) and m.provider_name == self.system
            ),
            None,
        )
        try:
            with map_api_errors(self.model_name, self._provider.model_id_namespace):
                return await self._websocket.states[loop].reserve(
                    connect, response_id, ResponsesWebSocket.effective_headers(self.client, headers)
                )
        except OpenAIError as exc:
            # The SDK raises this before connecting when it can't authenticate a socket for the client, like Bedrock
            # Mantle's SigV4 and native API key clients.
            self._warn_websocket_unsupported(f'the OpenAI SDK could not configure the WebSocket connection: {exc}')
            return None
        except ModelConnectionError as exc:
            # Reservation only opens a connection and selects a lane; it has not sent `response.create`.
            self._warn_websocket_unsupported(f'the WebSocket connection could not be established: {exc}')
            return None

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        """Cancel a suspended background response by cancelling its server-side job.

        `responses.cancel` only applies to background responses; calling it on an ordinary
        (foreground) response returns a 400. The `provider_details['background']` marker is
        stamped from the API's own `response.background` field (see `_process_response` and
        `OpenAIResponsesStreamedResponse._track_background`), so it explicitly and reliably
        distinguishes a cancellable background job from a normal streamed response that happens
        to be interrupted by `cancel()` — no need to infer background mode from the
        `continuation_delay` poll interval.
        """
        if (
            (response.provider_details or {}).get('background')
            and response.provider_response_id
            and response.provider_name == self.system
        ):
            with (
                map_api_errors(self.model_name, self._provider.model_id_namespace),
                map_decode_errors(self.model_name),
            ):
                await self.client.responses.cancel(response.provider_response_id)

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if response.state == 'suspended' and (response.provider_details or {}).get('background'):
            return _OPENAI_BACKGROUND_POLL_INTERVAL
        return None

    @cached_property
    def profile(self) -> OpenAIModelProfile:
        profile = super().profile
        # A model can be more capable on the Responses API than on Chat Completions, and the two
        # share one profile: DeepSeek honors `text.format` of type `json_schema` here while its
        # Chat Completions endpoint rejects it, so the generic flag can only be raised per model class.
        if profile.get('openai_responses_supports_json_schema_output', False):
            profile = merge_profile(profile, ModelProfile(supports_json_schema_output=True))
        if isinstance(self._provider, OAuthCredentialsProvider):
            # These providers authenticate HTTP requests without configuring the SDK socket client.
            profile = merge_profile(profile, OpenAIModelProfile(openai_responses_supports_websocket=False))
        return cast(OpenAIModelProfile, profile)

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        """Turn off Batch API and Responses WebSocket defaults for Azure clients and the gateway.

        Every model resolves this, so it reads only the client's type and the provider's identity, never client state
        such as URLs, which a mocked or partly built client may not have yet.
        """
        if is_azure(self._provider.client, self._provider.name) or is_gateway_provider(self._provider):
            # Azure batches need a batch deployment; the gateway doesn't serve the Batch API yet.
            profile = merge_profile(profile, OpenAIModelProfile(openai_supports_batch_api=False))
        # An Azure-named custom provider may advertise socket support with an ordinary OpenAI client.
        if isinstance(self._provider.client, AsyncAzureOpenAI) or is_gateway_provider(self._provider):
            # Responses WebSockets aren't verified on the Azure client, and the gateway's handshake fails with a 503
            # while HTTP succeeds (live, 2026-10-22).
            return merge_profile(profile, OpenAIModelProfile(openai_responses_supports_websocket=False))
        return profile

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the set of builtin tool types this model can handle."""
        return frozenset(
            {WebSearchTool, CodeExecutionTool, FileSearchTool, MCPServerTool, ImageGenerationTool, ToolSearchTool}
        )

    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        """Return `native_tool`'s restricting settings that the Responses `web_search` tool doesn't take: `max_uses`."""
        if isinstance(native_tool, WebSearchTool):
            return native_tool.restricting_settings - {'allowed_domains', 'blocked_domains', 'external_web_access'}
        return frozenset()

    async def compact_messages(
        self,
        request_context: ModelRequestContext,
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        """Compact messages using the OpenAI Responses compaction endpoint.

        This calls OpenAI's `responses.compact` API to produce an encrypted compaction
        that summarizes the conversation history. The returned `ModelResponse` contains
        a single `CompactionPart` that must be round-tripped in subsequent requests.

        Args:
            request_context: The model request context containing messages, settings, and parameters.
            instructions: Optional custom instructions for the compaction summarization.
                If provided, these override the agent-level instructions.

        Returns:
            A `ModelResponse` with a single `CompactionPart` containing the encrypted compaction data.
        """
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            request_context.model_settings,
            request_context.model_request_parameters,
        )
        response = await self._responses_compact(
            list(request_context.messages),
            cast(OpenAIResponsesModelSettings, model_settings or {}),
            model_request_parameters,
            instructions_override=instructions,
        )

        # Handle ModelResponse (e.g. from content filter)
        if isinstance(response, ModelResponse):  # pragma: no cover
            return response

        response = check_json_response(self.model_name, response)
        if not response.output:  # pragma: no cover
            raise UnexpectedModelBehavior('CompactedResponse returned with no output items')

        compaction = response.output[-1]
        if not isinstance(compaction, ResponseCompactionItem):  # pragma: no cover
            raise UnexpectedModelBehavior(f'Last item in response is not a compaction, got: {compaction.type}')

        # This compact call's input window explicitly planted the standing prompt (see
        # `_responses_compact`), so the minted part carries the provenance stamp that lets the
        # trim rely on the item's retention instead of re-sending the standing prompt.
        part = map_compaction_item(compaction, self.system, standing_prompt_planted=True)
        return ModelResponse(
            parts=[part],
            usage=map_usage(response, self._provider.name, self._provider.base_url, self.model_name),
            model_name=self._model_name,
            provider_response_id=response.id,
            # Marks this ModelResponse as coming from the stateless `/compact` endpoint.
            # `_get_previous_response_id_and_new_messages` uses this to break the auto-chain,
            # since compaction response ids cannot be used as `previous_response_id`.
            provider_details={'compaction': True},
            timestamp=_now_utc(),
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            canonical_system=self._canonical_system_stamp,
        )

    async def _responses_compact(
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        instructions_override: str | None = None,
    ) -> responses.CompactedResponse | ModelResponse:
        """Call the OpenAI Responses compaction endpoint."""
        previous_response_id, messages = self._resolve_previous_response_id(
            model_settings.get('openai_previous_response_id'),
            messages,
            store=self._resolve_store(model_settings),
            allow_no_new_messages=True,
        )
        # The endpoint doesn't take `configuration_update` items: the compacted window carries no effort change.
        # Removed after resolving the response to continue, which matches the history as stored.
        messages = _without_model_settings_deltas(messages)

        # Same ordering rule as `_build_responses_request_params`: the introduced-tools derivation
        # and the mapping must both see the trimmed history, and re-compacting only compacts the
        # current effective window rather than content an earlier compaction already replaced.
        # `standing_prompt_retained=False` (here and in the mapping below): blob-of-blob retention
        # decayed in probing, so the window sent for re-compaction plants the standing prompt
        # explicitly rather than relying on the previous compaction item to carry it forward.
        messages = self._trim_before_compaction(messages, standing_prompt_retained=False)
        instructions, openai_messages = await self._map_messages(
            messages,
            model_settings,
            model_request_parameters,
            previous_response_id=previous_response_id,
            standing_prompt_retained=False,
        )
        if instructions_override is not None:
            instructions = instructions_override

        try:
            with map_decode_errors(self.model_name):
                return await with_max_retries(self.client, model_settings).responses.compact(
                    input=openai_messages,
                    model=self.model_name,
                    instructions=instructions,
                    previous_response_id=previous_response_id or OMIT,
                )
        except APIStatusError as e:  # pragma: lax no cover
            if model_response := check_azure_content_filter(e, self.client, self.system, self.model_name):
                return model_response
            if e.status_code >= 400:
                raise map_status_error(e, self.model_name, self._provider.model_id_namespace) from e
            raise
        except APIConnectionError as e:  # pragma: lax no cover
            raise map_connection_error(e, self.model_name) from e

    async def request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        if self._websocket_stream is None:
            async with self._websocket_transport(messages, model_settings) as websocket_model:
                if websocket_model is not None:
                    return await websocket_model.request(messages, model_settings, model_request_parameters)
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = cast(OpenAIResponsesModelSettings, model_settings or {})

        if info := self._get_continuation_info(messages, settings):
            # Non-streaming retrieve: on `store=false` backends (Codex, which is also stream-only)
            # `_get_continuation_info` already rejected the continuation with `UserError`.
            suspended, response_id, _, _ = info
            response = await self._responses_retrieve(response_id, settings)
            # The retrieved response replaces the suspended one, the last message, and continues from what came before.
            return with_history_digest(
                self._process_response(response, settings, model_request_parameters),
                self._resumed_history_to_record(messages[:-1], suspended, settings, model_request_parameters),
            )
        elif self._websocket_stream is not None or self.profile.get('openai_responses_requires_streaming', False):
            # Stream-only backend (e.g. Codex subscription auth): drain a forced stream via the
            # streamed-response path, which handles `response.completed` arriving with an empty `output`.
            # Note: if a higher-level path to enforce streaming is ever added, this branch belongs there instead.
            stream = await self._responses_create(
                messages, stream=True, model_settings=settings, model_request_parameters=model_request_parameters
            )
            if isinstance(stream, ModelResponse):
                # A handled rejection (e.g. an Azure content filter) arrives as a finished
                # response, not a stream; same guard as `request_stream` below.
                return stream
            async with stream:
                streamed_response = await self._process_streamed_response(
                    stream,
                    settings,
                    model_request_parameters,
                    history=self._history_to_record(messages, settings, model_request_parameters),
                )
                async for _ in streamed_response:
                    pass
            return streamed_response.get()
        else:
            response = await self._responses_create(
                messages, stream=False, model_settings=settings, model_request_parameters=model_request_parameters
            )
            if isinstance(response, ModelResponse):
                return response
            return with_history_digest(
                self._process_response(response, settings, model_request_parameters),
                self._history_to_record(messages, settings, model_request_parameters),
            )

    def _history_to_record(
        self,
        messages: Sequence[ModelMessage],
        settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Hasher | None:
        """The history a new response is created from, to digest with it, when a later request may continue from it.

        In WebSocket mode that's every response, including one sent over HTTP instead, since a later socket request
        continues from it by default.
        A request's top-level instructions are omitted unless sent as input.
        """
        if settings.get('openai_previous_response_id') is None and settings.get('openai_transport') != 'websocket':
            return None
        return history_hasher(
            messages,
            self._history_digest_key,
            with_instructions=self._sends_instructions_as_input(model_request_parameters),
        )

    def _sends_instructions_as_input(self, model_request_parameters: ModelRequestParameters) -> bool:
        """Whether a request's instructions go out as a system input message rather than the `instructions` field.

        `json_object` output needs the word "JSON" in an input message, so prompted output moves them there.
        """
        return model_request_parameters.output_mode == 'prompted' and self.profile.get(
            'supports_json_object_output', False
        )

    def _resumed_history_to_record(
        self,
        history: Sequence[ModelMessage],
        suspended: ModelResponse,
        settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Hasher | None:
        """The history a retrieved background response continues: the messages before the suspended one it replaces.

        The server created it from the history the suspended response recorded, so if that, or the suspended response,
        has changed since, or it recorded none, none is returned: the merged response keeps the suspended one's digest,
        if any, which no longer matches, so it isn't continued from.
        """
        if not self._history_known_unchanged(suspended, history):
            return None
        return self._history_to_record(history, settings, model_request_parameters)

    def _history_known_unchanged(self, response: ModelResponse, history: Sequence[ModelMessage]) -> bool:
        """Whether `response`, and `history`, the messages before it, are as they were when the response was created.

        Without a recorded digest there's no telling, so a response that has none doesn't count.
        """
        if response.history_digest is None:
            return False
        return any(
            response.history_digest
            == continuation_digest(
                history_hasher(history, self._history_digest_key, with_instructions=with_instructions), response.parts
            )
            for with_instructions in (False, True)
        )

    async def count_tokens(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        if not self.profile.get('openai_supports_input_token_counting', True):
            raise UserError(
                f'Server-side token counting is not available for {self.system} models '
                f'({self.model_name!r}); the provider does not expose the input-tokens endpoint.'
            )
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = cast(OpenAIResponsesModelSettings, model_settings or {})

        # Validate that we have something meaningful to count tokens for.
        if (
            not messages
            and not settings.get('openai_previous_response_id')
            and not settings.get('openai_conversation_id')
        ):
            raise UserError('Cannot count tokens without any messages or a previous response ID.')

        profile = self.profile
        request_params = await self._build_responses_request_params(
            messages,
            settings,
            model_request_parameters,
            profile,
        )

        extra_headers, timeout = self._build_request_options(settings)
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            response = await with_max_retries(self.client, settings).responses.input_tokens.count(
                model=request_params.model,
                input=request_params.input,
                instructions=request_params.instructions,
                parallel_tool_calls=request_params.parallel_tool_calls,
                tools=request_params.tools,
                tool_choice=request_params.tool_choice,
                previous_response_id=request_params.previous_response_id,
                conversation=request_params.conversation,
                reasoning=request_params.reasoning,
                text=request_params.text,
                truncation=request_params.truncation,
                extra_headers=extra_headers,
                extra_body=settings.get('extra_body'),
                timeout=timeout,
            )

        return usage.RequestUsage(
            input_tokens=check_json_response(self.model_name, response).input_tokens,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        check_allow_model_requests()
        if self._websocket_stream is None:
            async with self._websocket_transport(messages, model_settings) as websocket_model:
                if websocket_model is not None:
                    async with websocket_model.request_stream(
                        messages, model_settings, model_request_parameters, run_context
                    ) as streamed_response:
                        yield streamed_response
                    return
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        settings = cast(OpenAIResponsesModelSettings, model_settings or {})

        if info := self._get_continuation_info(messages, settings):
            suspended, response_id, last_sequence_number, previous_model_name = info
            expected_response_id = response_id
            if last_sequence_number is None:
                # Some background responses were not previously streamed and have no resumable
                # sequence cursor. `retrieve(stream=True)` can block for a long time in this case,
                # so fall back to non-stream retrieve and return a static streamed wrapper.
                response = await self._responses_retrieve(response_id, settings)
                sr: StreamedResponse = ModelResponseStreamedResponse(
                    model_request_parameters=model_request_parameters,
                    _model_response=with_history_digest(
                        self._process_response(response, settings, model_request_parameters),
                        self._resumed_history_to_record(messages[:-1], suspended, settings, model_request_parameters),
                    ),
                )
                yield sr
                return
            response = await self._responses_retrieve(
                response_id, settings, stream=True, starting_after=last_sequence_number
            )
        else:
            previous_model_name = None
            expected_response_id = None
            response = await self._responses_create(
                messages, stream=True, model_settings=settings, model_request_parameters=model_request_parameters
            )
        if isinstance(response, ModelResponse):
            yield ModelResponseStreamedResponse(
                model_request_parameters=model_request_parameters,
                _model_response=response,
            )
            return
        async with response:
            sr = await self._process_streamed_response(
                response,
                settings,
                model_request_parameters,
                expected_model_name=previous_model_name,
                expected_response_id=expected_response_id,
                # A resumed stream has only the parts after the cursor, so it can't digest the whole response.
                history=None
                if expected_response_id
                else self._history_to_record(messages, settings, model_request_parameters),
            )
            yield sr

    def _process_provider_details(self, response: responses.Response) -> dict[str, Any] | None:
        """Map a complete Responses API response object to provider details.

        Subclasses may override this hook. It's called with the complete response, and when streaming, with the
        complete response carried by the terminal event that ends the stream. Built-in provider details take
        precedence on key collisions.
        """
        return None

    def _process_response(
        self,
        response: responses.Response,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        response = check_json_response(self.model_name, response)
        if error := response.error:
            raise response_error(
                self.model_name, error.code, error.message, error_type=response_error_type(error), in_stream=False
            )
        items = self._map_output_items(response, model_settings)
        refusal_text = response_refusal(response)

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] = dict(self._process_provider_details(response) or {})
        if (store := echoed_store(response, model_settings)) is not None:
            provider_details['store'] = store
        raw_finish_reason = details.reason if (details := response.incomplete_details) else response.status
        if raw_finish_reason:
            provider_details['finish_reason'] = raw_finish_reason
            finish_reason = RESPONSES_FINISH_REASON_MAP.get(raw_finish_reason)
        if response.created_at:  # pragma: no branch
            provider_details['timestamp'] = number_to_datetime(response.created_at)
        if response.conversation:
            provider_details['conversation_id'] = response.conversation.id
        if response.background:
            # Mark the response as a cancellable server-side background job (see
            # `cancel_suspended_response`), independent of the `continuation_delay` poll interval.
            provider_details['background'] = True

        if response.moderation:
            provider_details['moderation'] = response.moderation.model_dump()
        if response.service_tier:
            provider_details['service_tier'] = response.service_tier
        if response.prompt_cache_diagnostics is not None:
            provider_details['prompt_cache_diagnostics'] = response.prompt_cache_diagnostics.model_dump(mode='json')

        state = response_status_to_state(response.status, background=bool(response.background))
        if refusal_text is not None:
            items = []
            finish_reason = 'content_filter'
            provider_details.pop('finish_reason', None)
            provider_details['refusal'] = refusal_text

        return ModelResponse(
            parts=items,
            usage=map_usage(response, self._provider.name, self._provider.base_url, self.model_name),
            model_name=response.model,
            provider_response_id=response.id,
            timestamp=_now_utc(),
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            canonical_system=self._canonical_system_stamp,
            finish_reason=finish_reason,
            state=state,
            provider_details=provider_details or None,
        )

    def _map_output_items(  # noqa: C901
        self, response: responses.Response, model_settings: OpenAIResponsesModelSettings
    ) -> list[ModelResponsePart]:
        """Map the output items of a non-streamed response to parts."""
        tool_search_output_call_ids = tool_search_call_ids_by_output(response)
        tool_search_outputs = {
            tool_search_output_call_ids[item.id]: item
            for item in response.output
            if isinstance(item, responses.ResponseToolSearchOutputItem)
            and item.execution == 'server'
            and item.id in tool_search_output_call_ids
        }
        paired_tool_search_output_ids = {item.id for item in tool_search_outputs.values()}
        items: list[ModelResponsePart] = []
        for item in response.output:
            match item:
                case responses.ResponseReasoningItem():
                    items.extend(map_reasoning_item(item, self.system))
                case responses.ResponseOutputMessage():
                    items.extend(self._map_output_message(item, model_settings))
                case responses.ResponseFunctionToolCall():
                    items.append(self._map_function_tool_call(item, response.id))
                case responses.ResponseCodeInterpreterToolCall():
                    items.extend(self._map_code_interpreter_call(item))
                case responses.ResponseFunctionWebSearch():
                    items.extend(map_web_search_tool_call(item, self.system))
                case responses.ResponseToolSearchCall():
                    items.extend(self._map_tool_search_call(item, tool_search_outputs))
                case responses.ResponseToolSearchOutputItem():
                    if item.execution == 'server' and item.id not in paired_tool_search_output_ids:
                        items.append(build_tool_search_return_part(item.call_id or item.id, item, self.system))
                case responses.response_output_item.ImageGenerationCall():
                    items.extend(self._map_image_generation_call(item))
                case ResponseCompactionItem():
                    items.append(map_compaction_item(item, self.system))
                case responses.ResponseComputerToolCall():  # pragma: lax no cover
                    # Pydantic AI doesn't yet support the ComputerUse built-in tool
                    pass
                case responses.ResponseCustomToolCall():  # pragma: lax no cover
                    # Support is being implemented in https://github.com/pydantic/pydantic-ai/pull/2572
                    pass
                case responses.response_output_item.LocalShellCall():  # pragma: lax no cover
                    # Pydantic AI doesn't yet support the `codex-mini-latest` LocalShell built-in tool
                    pass
                case responses.ResponseFileSearchToolCall():
                    items.extend(map_file_search_tool_call(item, self.system))
                case responses.response_output_item.McpCall():
                    items.extend(map_mcp_call(item, self.system))
                case responses.response_output_item.McpListTools():
                    items.extend(map_mcp_list_tools(item, self.system))
                case responses.response_output_item.McpApprovalRequest():  # pragma: no cover
                    # Pydantic AI doesn't yet support McpApprovalRequest (explicit tool usage approval)
                    pass
                case _:  # pragma: lax no cover
                    pass
        return items

    def _map_output_message(
        self, item: responses.ResponseOutputMessage, model_settings: OpenAIResponsesModelSettings
    ) -> Iterator[TextPart]:
        for content in item.content:
            # Like a stream, which only builds text from its deltas, empty text (such as the message that
            # follows a generated image, or `text=null` from gateways like Bifrost) doesn't make a part.
            if not isinstance(content, responses.ResponseOutputText) or not content.text:
                continue
            citations = map_responses_citations(content.annotations, content.text)
            part_provider_details: dict[str, Any] = {}
            if citations:
                part_provider_details.update(responses_citations_provider_details(content.annotations))
            if content.logprobs:
                part_provider_details['logprobs'] = map_logprobs(content.logprobs)
            if model_settings.get('openai_include_raw_annotations') and content.annotations:
                part_provider_details['annotations'] = responses_output_text_annotations_ta.dump_python(
                    list(content.annotations), warnings=False
                )
            if item.phase is not None:
                part_provider_details['phase'] = item.phase
            yield TextPart(
                content.text,
                id=item.id,
                provider_name=self.system,
                provider_details=part_provider_details or None,
                citations=citations,
            )

    def _map_function_tool_call(self, item: responses.ResponseFunctionToolCall, response_id: str) -> ToolCallPart:
        # When the Responses API wraps a discovered deferred tool in a namespace
        # (tool-search flow), it requires the namespace to be round-tripped on
        # subsequent turns. Preserve it via `provider_details`.
        fn_provider_details: dict[str, Any] | None = None
        if item.namespace:
            fn_provider_details = {'namespace': item.namespace}
        return ToolCallPart(
            item.name,
            item.arguments,
            tool_call_id=response_tool_call_id(
                item.call_id,
                response_id if self.profile.get('openai_responses_tool_call_ids_are_response_scoped', False) else None,
            ),
            id=item.id,
            provider_name=self.system,
            provider_details=fn_provider_details,
        )

    def _map_code_interpreter_call(
        self, item: responses.ResponseCodeInterpreterToolCall
    ) -> Iterator[ModelResponsePart]:
        call_part, return_part, file_parts = map_code_interpreter_tool_call(item, self.system)
        yield call_part
        yield from file_parts
        yield return_part

    def _map_image_generation_call(
        self, item: responses.response_output_item.ImageGenerationCall
    ) -> Iterator[ModelResponsePart]:
        call_part, return_part, file_part = map_image_generation_tool_call(item, self.system)
        yield call_part
        if file_part:  # pragma: no branch
            yield file_part
        yield return_part

    def _map_tool_search_call(
        self,
        item: responses.ResponseToolSearchCall,
        tool_search_outputs: Mapping[str, responses.ResponseToolSearchOutputItem],
    ) -> Iterator[ModelResponsePart]:
        if item.execution == 'client':
            # Client-executed: emit a regular ToolCallPart so the standard agent-graph
            # tool-execution path runs the local `search_tools` function. The matching
            # `ToolReturnPart` is produced by the tool runner, not here.
            yield map_client_tool_search_call(item, self.system)
        else:
            call_part = map_tool_search_call(item, self.system)
            yield call_part
            if output_item := tool_search_outputs.get(call_part.tool_call_id):
                yield build_tool_search_return_part(call_part.tool_call_id, output_item, self.system)

    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        expected_model_name: OpenAIModelName | None = None,
        expected_response_id: str | None = None,
        history: Hasher | None = None,
    ) -> OpenAIResponsesStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response: _utils.PeekableAsyncStream[
            responses.ResponseStreamEvent, AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream
        ] = _utils.PeekableAsyncStream(response)
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            # Covered by the Codex forced-stream path, which drains empty streams through here.
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
        if isinstance(first_chunk, responses.ResponseErrorEvent):
            # Raise while the stream is being opened, so `FallbackModel` can still fall back on it.
            raise response_error(self.model_name, first_chunk.code, first_chunk.message, in_stream=True)

        if isinstance(first_chunk, responses.ResponseCreatedEvent):
            model_name = first_chunk.response.model
            provider_timestamp = (
                number_to_datetime(first_chunk.response.created_at) if first_chunk.response.created_at else None
            )
            background = bool(first_chunk.response.background)
            initial_state = response_status_to_state(first_chunk.response.status, background=background)
        else:
            # When `starting_after` is used, OpenAI may omit `response.created` and start
            # directly with delta events. Keep the previous model name (if known) so
            # continuation merging doesn't accidentally replace earlier parts.
            model_name = expected_model_name or self.model_name
            provider_timestamp = None
            # We only resume via `retrieve(stream=True, starting_after=...)` for a background job that
            # was still in progress, so start `'suspended'`: a clean EOF here without a terminal event
            # means it's still running and should be continued, not treated as done.
            background = True
            initial_state = 'suspended'

        tool_call_ids_are_response_scoped = self.profile.get(
            'openai_responses_tool_call_ids_are_response_scoped', False
        )
        is_websocket = False
        if self._websocket_stream is not None:
            from ._responses_websocket import ResponsesWebSocketStream

            # A retry can overflow to HTTP after this request copy reserved a socket lane.
            is_websocket = isinstance(response, ResponsesWebSocketStream)
        streamed_response = OpenAIResponsesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _model_settings=model_settings,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _model_id_namespace=self._provider.model_id_namespace,
            _provider_url=self._provider.base_url,
            _provider_timestamp=provider_timestamp,
            _history=history,
            _is_websocket=is_websocket,
            _tool_call_ids_are_response_scoped=tool_call_ids_are_response_scoped,
            _process_provider_details=self._process_provider_details,
        )
        streamed_response.canonical_system = self._canonical_system_stamp
        if tool_call_ids_are_response_scoped:
            # Response-scoped tool-call IDs are qualified with the response ID as chunks arrive, so the
            # ID must be known before iteration. This is only needed (and only set eagerly) for the
            # providers that opt in; other Responses streams keep resolving it during iteration.
            streamed_response.provider_response_id = (
                first_chunk.response.id
                if isinstance(first_chunk, responses.ResponseCreatedEvent)
                else expected_response_id
            )
        streamed_response.state = initial_state
        if background:
            # Stamp the background marker up front so the retry-delay gate (and `cancel_suspended_response`)
            # can tell a resumable background job from a foreground stream before any event is consumed;
            # `_track_background` keeps it stamped as status events arrive during iteration.
            streamed_response.provider_details = {**(streamed_response.provider_details or {}), 'background': True}
        return streamed_response

    def _effort_delta_request(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[bool, ModelRequestParameters]:
        """Decide whether this request sends effort changes as `configuration_update` items.

        It does when the history holds a mid-conversation change (a `ModelSettingsDeltaPart` that
        `prepare_messages` kept for this model) and nothing rules the item out: an explicit
        `openai_reasoning_effort`, which is sent as is, and server-side truncation or compaction, which OpenAI
        documents the item doesn't work with. The request then keeps the window's opening effort as its
        `reasoning.effort`, so its prefix matches the earlier requests'. Otherwise the caller removes the parts and
        the current effort goes out as `reasoning.effort`, as it would without them.

        Returns whether the parts stay, and the parameters whose `thinking` the request-level effort comes from.
        The caller removes the parts only from what it sends, after `_resolve_server_side_state`, which matches
        the history as stored against the response it continues.
        """
        baseline, has_changes = _effort_window_baseline(messages)
        # `prepare_messages` already dropped the parts on a model without the channel, but a direct `request`
        # call with stored history doesn't run it.
        if (
            has_changes
            and self.profile.get('supports_mid_conversation_effort', False)
            and not (
                self._overrides_unified_effort(model_settings)
                or model_settings.get('openai_truncation') == 'auto'
                or model_settings.get('openai_context_management')
                or any(
                    isinstance(part, CompactionPart)
                    for message in messages
                    if isinstance(message, ModelResponse)
                    for part in message.parts
                )
            )
        ):
            if baseline is not None and isinstance(model_request_parameters.thinking, str):
                model_request_parameters = replace(model_request_parameters, thinking=baseline)
            return True, model_request_parameters
        return False, model_request_parameters

    async def _build_responses_request_params(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        profile: OpenAIModelProfile,
    ) -> _ResponsesRequestParams:
        """Build typed request parameters shared by Responses API calls."""
        sends_effort_changes, effort_request_parameters = self._effort_delta_request(
            messages, model_settings, model_request_parameters
        )
        # Deliberately a separate variable: `messages` itself stays untrimmed for
        # `_resolve_server_side_state`, which recovers conversation/response IDs from responses the
        # trim would drop; `_map_messages` applies the same idempotent trim to whatever slice that
        # resolution hands it.
        trimmed_messages = self._trim_before_compaction(messages)
        resolved_tool_choice = resolve_tool_choice(model_settings, model_request_parameters)
        # A stored response keeps its input, so whatever a request moves into or out of it carries over to every
        # request that continues it. Decided by the setting, whatever it resolves to.
        uses_server_side_state = (
            model_settings.get('openai_previous_response_id') is not None
            or model_settings.get('openai_conversation_id') is not None
        )
        wire_request_parameters = self._wire_request_parameters(
            trimmed_messages,
            resolved_tool_choice,
            model_request_parameters,
            profile,
            uses_server_side_state=uses_server_side_state,
        )
        history_declared_tool_names = {
            tool.name
            for tool in wire_request_parameters.function_tools
            if wire_request_parameters.visibility_of(tool.name) == 'via_history'
        }
        function_tools, tool_choice = self._get_responses_tool_choice(
            resolved_tool_choice, model_settings, wire_request_parameters
        )
        extra_native_tools = model_settings.get('openai_native_tools', ())
        tools: list[responses.ToolParam] = (
            self._get_native_tools(wire_request_parameters) + list(extra_native_tools) + function_tools
        )
        if not tools and not history_declared_tool_names:
            tool_choice = None

        previous_response_id, conversation_id, messages = self._resolve_server_side_state(model_settings, messages)
        if not sends_effort_changes:
            messages = _without_model_settings_deltas(messages)

        instructions, openai_messages = await self._map_messages(
            messages,
            model_settings,
            wire_request_parameters,
            previous_response_id=previous_response_id,
        )
        reasoning = self._translate_thinking(model_settings, effort_request_parameters)

        system_prompt_role = profile.get('openai_system_prompt_role', None) or 'system'
        if self._should_relocate_cached_instructions(
            model_settings,
            profile,
            system_prompt_role,
            messages,
            openai_messages,
            uses_server_side_state=uses_server_side_state,
        ):
            instructions = self._relocate_cached_instructions(
                instructions,
                openai_messages,
                self._get_instruction_parts(messages, wire_request_parameters) or [],
                system_prompt_role,
            )

        text, instructions = self._responses_text_config(
            model_settings, model_request_parameters, profile, instructions, openai_messages, system_prompt_role
        )

        # When there are no input messages and we're not reusing server-side state,
        # the OpenAI API will reject a request without any input,
        # even if there are instructions.
        # To avoid this provide an explicit empty user message.
        if not openai_messages and not previous_response_id and not conversation_id:
            openai_messages.append(
                responses.EasyInputMessageParam(
                    role='user',
                    content='',
                )
            )

        # `parallel_tool_calls`, `truncation`, and `context_management` ride on the request params
        # rather than the create call, so they must honor the unsupported-settings seam here too —
        # `drop_unsupported_params` runs too late for them (and never runs for `count_tokens`).
        unsupported_settings = profile.get('openai_unsupported_model_settings', ())
        parallel_tool_calls = OMIT
        if tools and 'parallel_tool_calls' not in unsupported_settings:
            parallel_tool_calls = model_settings.get('parallel_tool_calls', OMIT)
        return _ResponsesRequestParams(
            model=self.model_name,
            input=openai_messages,
            instructions=instructions,
            parallel_tool_calls=parallel_tool_calls,
            tools=tools or OMIT,
            tool_choice=tool_choice or OMIT,
            previous_response_id=previous_response_id or OMIT,
            conversation=conversation_id or OMIT,
            reasoning=reasoning,
            text=text,
            truncation=OMIT
            if 'openai_truncation' in unsupported_settings
            else model_settings.get('openai_truncation', OMIT),
            context_management=OMIT
            if 'openai_context_management' in unsupported_settings
            else model_settings.get('openai_context_management', OMIT),
        )

    @staticmethod
    def _wire_request_parameters(
        trimmed_messages: list[ModelRequest | ModelResponse],
        resolved_tool_choice: ResolvedToolChoice,
        model_request_parameters: ModelRequestParameters,
        profile: OpenAIModelProfile,
        *,
        uses_server_side_state: bool,
    ) -> ModelRequestParameters:
        """The request parameters to send, with the reveals this request can't deliver in history redeclared in `tools`."""
        # Call-time import mirroring `models/__init__.py`'s `_tool_search` import: the toolsets
        # package imports `messages` while adapters load, so a module-level import would cycle.
        from ...toolsets._tool_search import parse_discovered_tools

        # A reveal whose `additional_tools` carrier sits before the compaction boundary is trimmed
        # with it, while the tool's `'via_history'` visibility still promises a history item — so it
        # would vanish from the request entirely. Until reveal state itself becomes boundary-aware
        # (#7225 derives it from the post-compaction window, which makes this branch inert), such a
        # tool is redeclared in `tools` with its schema: the compaction turn rebuilds the prefix
        # anyway, so the declaration costs nothing extra.
        post_trim_revealed = parse_discovered_tools(trimmed_messages)
        lost_reveals = {
            name
            for name, visibility in (model_request_parameters.tool_visibility or {}).items()
            if visibility == 'via_history' and name not in post_trim_revealed
        }
        # Before GPT-5.6, a `tool_choice` naming a tool that only an `additional_tools` item declares is
        # rejected ("Tool choice '<name>' not found in 'tools' parameter."), by name and in `allowed_tools`
        # alike, and so is declaring it in both places. A `'via_history'` tool the choice names is redeclared
        # in `tools` for this request instead, and the item leaves it out. Only such a request pays the cache miss.
        # With server-side state the item stays: a stored response without it would hide the tool from every
        # request that continues it, and redeclaring a forced name next to an item it holds gets a 500.
        chosen_reveals: set[str] = set()
        if (
            isinstance(resolved_tool_choice, tuple)
            and not profile.get('openai_responses_tool_choice_accepts_additional_tools', False)
            and not uses_server_side_state
        ):
            chosen_reveals = {
                name
                for name in resolved_tool_choice[1]
                if model_request_parameters.visibility_of(name) == 'via_history'
            }
        if redeclared := lost_reveals | chosen_reveals:
            tool_visibility: dict[str, ToolVisibility] = {
                **(model_request_parameters.tool_visibility or {}),
                **dict.fromkeys(sorted(redeclared), 'visible'),
            }
            return replace(model_request_parameters, tool_visibility=tool_visibility)
        return model_request_parameters

    @staticmethod
    def _should_relocate_cached_instructions(
        model_settings: OpenAIResponsesModelSettings,
        profile: OpenAIModelProfile,
        system_prompt_role: OpenAISystemPromptRole,
        messages: list[ModelRequest | ModelResponse],
        openai_messages: list[responses.ResponseInputItemParam],
        *,
        uses_server_side_state: bool,
    ) -> bool:
        """Whether `openai_cache_instructions` can move the instructions into the input to mark a cache breakpoint."""
        return bool(
            model_settings.get('openai_cache_instructions')
            and profile.get('openai_supports_prompt_cache_breakpoints', False)
            # A `'user'` system prompt role can't be told apart from a real user turn.
            and system_prompt_role != 'user'
            # Instructions relocated into a stored response's input would be replayed to any request
            # that continues it, next to that request's own instructions.
            and not uses_server_side_state
            # A dynamic system prompt changes between requests, so it can't sit in the cached prefix.
            and not has_dynamic_system_prompt(messages)
            # A compaction item retains the window's leading system items, so they aren't resent.
            and not any(isinstance(item, dict) and item.get('type') == 'compaction' for item in openai_messages)
        )

    def _responses_text_config(
        self,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        profile: OpenAIModelProfile,
        instructions: str | Omit,
        openai_messages: list[responses.ResponseInputItemParam],
        system_prompt_role: OpenAISystemPromptRole,
    ) -> tuple[responses.ResponseTextConfigParam | Omit, str | Omit]:
        """The `text` config for the output mode and verbosity, and the instructions left after JSON mode's move.

        Mutates `openai_messages` when JSON mode moves the instructions into them.
        """
        text: responses.ResponseTextConfigParam | Omit = OMIT
        if model_request_parameters.output_mode == 'native':
            output_object = model_request_parameters.output_object
            assert output_object is not None
            text = {'format': self._map_json_schema(output_object)}
        elif model_request_parameters.output_mode == 'prompted' and profile.get(
            'supports_json_object_output', False
        ):  # pragma: no branch
            text = {'format': {'type': 'json_object'}}

            # Without this trick, we'd hit this error:
            # > Response input messages must contain the word 'json' in some form to use 'text.format' of type 'json_object'.
            # Apparently they're only checking input messages for "JSON", not instructions.
            # `openai_cache_instructions` may already have moved them into the input messages.
            if isinstance(instructions, str):
                system_prompt_count = leading_system_message_count(openai_messages, system_prompt_role)
                openai_messages.insert(
                    system_prompt_count, responses.EasyInputMessageParam(role=system_prompt_role, content=instructions)
                )
                instructions = OMIT

        if verbosity := model_settings.get('openai_text_verbosity'):
            text_with_verbosity: responses.ResponseTextConfigParam = text if isinstance(text, dict) else {}
            text_with_verbosity['verbosity'] = verbosity
            text = text_with_verbosity
        return text, instructions

    @staticmethod
    def _relocate_cached_instructions(
        instructions: str | Omit,
        openai_messages: list[responses.ResponseInputItemParam],
        instruction_parts: list[InstructionPart],
        system_prompt_role: OpenAISystemPromptRole,
    ) -> str | Omit:
        """Move the instructions into leading input messages and mark a cache breakpoint after the last static one.

        The top-level `instructions` field cannot carry a breakpoint. Mutates `openai_messages` and
        returns what's left of the top-level `instructions`.
        """
        # An `additional_tools` item also has the `'developer'` role, but it's not a system prompt. Kept
        # apart from `leading_system_message_count`: checking `type` there would also move where
        # prompted-output instructions go for users who don't set `openai_cache_instructions`.
        system_prompt_count = next(
            (
                i
                for i, message in enumerate(openai_messages)
                if message.get('role') != system_prompt_role or message.get('type', 'message') != 'message'
            ),
            len(openai_messages),
        )
        breakpoint_index = system_prompt_count + cacheable_instruction_count(instruction_parts) - 1
        # With nothing static to cache, the instructions stay in the top-level field.
        if breakpoint_index >= 0:
            if instruction_parts:
                openai_messages[system_prompt_count:system_prompt_count] = [
                    responses.EasyInputMessageParam(role=system_prompt_role, content=part.content)
                    for part in instruction_parts
                ]
                instructions = OMIT
            target = cast(responses.EasyInputMessageParam, openai_messages[breakpoint_index])
            # A system prompt's content is already a list when a leading `CachePoint` attached its breakpoint there.
            content: str | responses.ResponseInputMessageContentListParam = target['content']
            if isinstance(content, str):
                content = [responses.ResponseInputTextParam(type='input_text', text=content)]
            add_openai_prompt_cache_breakpoint(content)
            target['content'] = content
        return instructions

    @staticmethod
    def _build_request_options(
        model_settings: OpenAIResponsesModelSettings,
    ) -> tuple[dict[str, str], _ResponsesTimeout]:
        extra_headers = _with_default_user_agent(model_settings.get('extra_headers', {}))
        timeout = to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN))
        return extra_headers, timeout

    def _resolve_websocket_timeout(self, timeout: _ResponsesTimeout) -> HTTPX2Timeout:
        """Resolve at each use: request preparation may change settings after lane reservation."""
        return HTTPX2Timeout(self.client.timeout if isinstance(timeout, NotGiven) else timeout)

    def _prepare_responses_settings(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
    ) -> OpenAIResponsesModelSettings:
        """Adjust the settings of a single request; subclasses override this to derive settings from the messages."""
        return model_settings

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
    ) -> AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream: ...

    async def _responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream | ModelResponse:
        given_settings = model_settings
        create_params, model_settings = await self._prepare_responses_create(
            messages, model_settings, model_request_parameters
        )
        extra_headers, timeout = self._build_request_options(model_settings)

        if (websocket_stream := self._websocket_stream) is not None:
            return await self._create_on_websocket(
                websocket_stream,
                create_params,
                timeout,
                messages,
                stream,
                model_settings,
                given_settings,
                model_request_parameters,
            )

        if model_settings.get('openai_transport') == 'websocket' and (span := current_model_request_span()) is not None:
            span.set_attribute(_RESPONSES_TRANSPORT_ATTRIBUTE, 'http')
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            try:
                return await with_max_retries(self.client, model_settings).responses.create(
                    **create_params,
                    stream=stream,
                    background=model_settings.get('openai_background', OMIT),
                    timeout=timeout,
                    extra_headers=extra_headers,
                    extra_body=model_settings.get('extra_body'),
                )
            except APIStatusError as e:
                if model_response := check_azure_content_filter(e, self.client, self.system, self.model_name):
                    return model_response
                raise

    async def _create_on_websocket(
        self,
        websocket_stream: ResponsesWebSocketStream,
        create_params: _ResponsesCreateParams,
        timeout: _ResponsesTimeout,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        given_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream | ModelResponse:
        if (span := current_model_request_span()) is not None:
            span.set_attribute(_RESPONSES_TRANSPORT_ATTRIBUTE, 'websocket')
        body = self._websocket_request_body(create_params, model_settings)
        try:
            if await websocket_stream.send(body, self._resolve_websocket_timeout(timeout)):
                # Only an explicit rejection before the response starts may be retried. Prime the stream so
                # later errors, including failures after partial output, stay with the caller's retry policy.
                websocket_stream.pending_event = await anext(websocket_stream)
                return websocket_stream
        except ModelHTTPError as exc:
            max_retries = model_settings.get('max_retries', self.client.max_retries)
            retry_after = exc.retry_after
            if (
                websocket_stream.rejection_retries >= max_retries
                or not (exc.status_code in (408, 409, 429) or exc.status_code >= 500)
                or (exc.headers or {}).get('x-should-retry') == 'false'
                or (retry_after is not None and retry_after > _WEBSOCKET_RETRY_AFTER_LIMIT)
            ):
                raise
            return await self._retry_rejected_websocket_request(
                websocket_stream, retry_after, max_retries, messages, stream, given_settings, model_request_parameters
            )
        return await self._replace_idle_closed_websocket(
            websocket_stream, messages, stream, model_settings, given_settings, model_request_parameters
        )

    @staticmethod
    def _websocket_request_body(
        create_params: _ResponsesCreateParams, model_settings: OpenAIResponsesModelSettings
    ) -> dict[str, object]:
        body: dict[str, object] = dict(create_params)
        if (extra_body := model_settings.get('extra_body')) is not None:
            if not _is_str_dict(extra_body):
                raise UserError('Responses WebSocket `extra_body` must have string keys.')
            body.update(extra_body)
        if any(key in body for key in ('stream', 'stream_id', 'type')):
            raise UserError('Responses WebSocket `extra_body` cannot set `stream`, `stream_id`, or `type`.')
        return {key: value for key, value in body.items() if not isinstance(value, Omit)}

    async def _retry_rejected_websocket_request(
        self,
        websocket_stream: ResponsesWebSocketStream,
        retry_after: float | None,
        max_retries: int,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        given_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream | ModelResponse:
        await websocket_stream.close()
        delay = retry_after
        if delay is None:
            delay = min(
                _WEBSOCKET_RETRY_INITIAL_DELAY
                * 2 ** min(websocket_stream.rejection_retries, _WEBSOCKET_RETRY_MAX_EXPONENT),
                _WEBSOCKET_RETRY_MAX_DELAY,
            )
            delay *= 1 - _WEBSOCKET_RETRY_JITTER_RATIO * random()
        await anyio.sleep(delay)
        retry = await self._reserve_websocket_lane(asyncio.get_running_loop(), messages, given_settings)
        if retry is None:
            http_settings: OpenAIResponsesModelSettings = {
                **given_settings,
                'max_retries': max_retries - websocket_stream.rejection_retries - 1,
            }
            return await _utils.replace_no_init(self, _websocket_stream=None)._responses_create(
                messages, stream, http_settings, model_request_parameters
            )
        retry.rejection_retries = websocket_stream.rejection_retries + 1
        return await self._create_on_websocket_lane(retry, messages, stream, given_settings, model_request_parameters)

    async def _replace_idle_closed_websocket(
        self,
        websocket_stream: ResponsesWebSocketStream,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        given_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream | ModelResponse:
        # The server closed the socket while it was idle, so nothing was sent: rebuild the request for a new socket
        # once, since what it can continue from depends on the lane.
        retry = (
            None
            if websocket_stream.retried
            else await self._reserve_websocket_lane(asyncio.get_running_loop(), messages, given_settings)
        )
        if retry is None:
            http_settings: OpenAIResponsesModelSettings = {
                **given_settings,
                'max_retries': max(
                    0,
                    model_settings.get('max_retries', self.client.max_retries) - websocket_stream.rejection_retries,
                ),
            }
            return await _utils.replace_no_init(self, _websocket_stream=None)._responses_create(
                messages, stream, http_settings, model_request_parameters
            )
        retry.retried = True
        retry.rejection_retries = websocket_stream.rejection_retries
        return await self._create_on_websocket_lane(retry, messages, stream, given_settings, model_request_parameters)

    async def _create_on_websocket_lane(
        self,
        retry: ResponsesWebSocketStream,
        messages: list[ModelRequest | ModelResponse],
        stream: bool,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent] | ResponsesWebSocketStream | ModelResponse:
        try:
            return await _utils.replace_no_init(self, _websocket_stream=retry)._responses_create(
                messages, stream, model_settings, model_request_parameters
            )
        except BaseException:
            await retry.close()
            raise

    async def _prepare_responses_create(
        self,
        messages: list[ModelRequest | ModelResponse],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[_ResponsesCreateParams, OpenAIResponsesModelSettings]:
        """Build the body arguments of a create call, and the settings that carry its request options."""
        profile = self.profile

        include = self._build_include(model_settings)

        request_params = await self._build_responses_request_params(
            messages,
            model_settings,
            model_request_parameters,
            profile,
        )
        # These helpers mutate the settings they receive.
        model_settings = self._prepare_responses_settings(messages, OpenAIResponsesModelSettings(**model_settings))
        drop_sampling_params_for_reasoning(profile, model_settings, model_request_parameters)
        keep_implicit_cache_without_breakpoints(model_settings, request_params.input)
        drop_unsupported_params(profile, model_settings)
        store = self._resolve_store(model_settings)

        # The SDK's Responses `PromptCacheOptions` has keys ours doesn't expose, so the TypedDicts aren't assignable.
        prompt_cache_options: ResponsesPromptCacheOptions | Omit = OMIT
        cache_options = model_settings.get('openai_prompt_cache_options')
        comparison_response_id = self._prompt_cache_comparison_response_id(messages, model_settings)
        if cache_options is not None or comparison_response_id is not None:
            prompt_cache_options = (
                # The SDK's `ttl` type only lists `'30m'`, but the Responses API also accepts `'12h'`.
                cast(ResponsesPromptCacheOptions, {**cache_options})
                if cache_options is not None
                else ResponsesPromptCacheOptions()
            )
            if comparison_response_id is not None:
                prompt_cache_options['comparison_response_id'] = comparison_response_id

        create_params = _ResponsesCreateParams(
            model=request_params.model,
            input=request_params.input,
            instructions=request_params.instructions,
            parallel_tool_calls=request_params.parallel_tool_calls,
            tools=request_params.tools,
            tool_choice=request_params.tool_choice,
            previous_response_id=request_params.previous_response_id,
            reasoning=request_params.reasoning,
            text=request_params.text,
            truncation=request_params.truncation,
            context_management=request_params.context_management,
            max_output_tokens=model_settings.get('max_tokens', OMIT),
            temperature=model_settings.get('temperature', OMIT),
            top_p=model_settings.get('top_p', OMIT),
            service_tier=resolve_openai_service_tier(model_settings),
            conversation=request_params.conversation,
            top_logprobs=model_settings.get('openai_top_logprobs', OMIT),
            store=store,
            user=model_settings.get('openai_user', OMIT),
            include=include or OMIT,
            prompt_cache_key=model_settings.get('openai_prompt_cache_key', OMIT),
            prompt_cache_retention=model_settings.get('openai_prompt_cache_retention', OMIT),
            prompt_cache_options=prompt_cache_options,
            moderation=model_settings.get('openai_moderation', OMIT),
        )
        return create_params, model_settings

    def _resolve_store(self, model_settings: OpenAIResponsesModelSettings) -> bool | Omit | None:
        if self.profile.get('openai_responses_requires_store_false', False):
            return False
        return model_settings.get('openai_store', OMIT)

    def _get_continuation_info(
        self, messages: list[ModelMessage], model_settings: OpenAIResponsesModelSettings
    ) -> tuple[ModelResponse, str, int | None, OpenAIModelName | None] | None:
        """If the last message is a suspended response from this provider, return it with continuation metadata."""
        if not messages:  # pragma: lax no cover
            return None
        last = messages[-1]
        if not isinstance(last, ModelResponse):
            return None
        if last.provider_name != self.system:  # pragma: lax no cover
            return None
        if not (last.state == 'suspended' and last.provider_response_id):  # pragma: lax no cover
            return None
        if self.profile.get('openai_responses_requires_store_false', False):
            # Continuation retrieves the suspended response server-side, but a `store=false`
            # backend (e.g. Codex subscription auth) never persisted it; without this guard the
            # retrieve fails at resume time as a misleading `SuspendedResponseExpired` on 404.
            raise UserError(
                f'Resuming a suspended run is not supported for {self.system} models '
                f'({self.model_name!r}): the backend requires `store=false`, so the suspended '
                'response was never persisted server-side.'
            )
        details: _OpenAIResponsesContinuationDetails = cast(
            _OpenAIResponsesContinuationDetails, last.provider_details or {}
        )
        last_sequence_number = details.get('last_sequence_number')
        return (
            last,
            last.provider_response_id,
            last_sequence_number,
            cast(OpenAIModelName | None, last.model_name),
        )

    def _build_include(
        self, model_settings: OpenAIResponsesModelSettings, *, is_retrieve: bool = False
    ) -> list[responses.ResponseIncludable]:
        """Build the include list for retrieve/create requests."""
        profile = self.profile
        include: list[responses.ResponseIncludable] = []
        if profile.get('openai_supports_encrypted_reasoning_content', False) and not is_retrieve:
            # OpenAI rejects `reasoning.encrypted_content` on any retrieve of a background
            # response ('Encrypted content cannot be requested for persisted responses'),
            # so only request it on create. Nothing is lost: a retrieved background response
            # never carries encrypted content, and continuation works via `previous_response_id`.
            include.append('reasoning.encrypted_content')
        if model_settings.get('openai_include_code_execution_outputs'):
            include.append('code_interpreter_call.outputs')
        if model_settings.get('openai_include_web_search_sources'):
            include.append('web_search_call.action.sources')
        if model_settings.get('openai_include_file_search_results'):
            include.append('file_search_call.results')
        if model_settings.get('openai_logprobs'):
            include.append('message.output_text.logprobs')
        return include

    @overload
    async def _responses_retrieve(
        self,
        response_id: str,
        model_settings: OpenAIResponsesModelSettings,
        *,
        stream: Literal[False] = False,
        starting_after: int | None = None,
    ) -> responses.Response: ...

    @overload
    async def _responses_retrieve(
        self,
        response_id: str,
        model_settings: OpenAIResponsesModelSettings,
        *,
        stream: Literal[True],
        starting_after: int | None = None,
    ) -> AsyncStream[responses.ResponseStreamEvent]: ...

    async def _responses_retrieve(
        self,
        response_id: str,
        model_settings: OpenAIResponsesModelSettings,
        *,
        stream: bool = False,
        starting_after: int | None = None,
    ) -> responses.Response | AsyncStream[responses.ResponseStreamEvent]:
        """Retrieve a background response by ID, optionally streaming."""
        include = self._build_include(model_settings, is_retrieve=True)
        extra_headers, timeout = self._build_request_options(model_settings)
        if model_settings.get('openai_transport') == 'websocket' and (span := current_model_request_span()) is not None:
            span.set_attribute(_RESPONSES_TRANSPORT_ATTRIBUTE, 'http')
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            try:
                return await with_max_retries(self.client, model_settings).responses.retrieve(
                    response_id=response_id,
                    include=include or OMIT,
                    starting_after=starting_after if starting_after is not None else OMIT,
                    stream=stream,
                    timeout=timeout,
                    extra_headers=extra_headers,
                )
            except APIStatusError as e:
                # A retrieve is only issued to resume a persisted suspended job, so a 404 means that job is
                # gone (past the provider's retention window). Surface a clear typed error instead of a
                # generic HTTP error; other statuses fall through to `map_api_errors`.
                if e.status_code == 404:
                    raise SuspendedResponseExpired(
                        f'The suspended response {response_id!r} could not be resumed because its server-side '
                        'job is no longer available (it may have expired).'
                    ) from e
                raise

    def _translate_thinking(
        self,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Reasoning | Omit:
        reasoning_effort = model_settings.get('openai_reasoning_effort', None)
        reasoning_mode = model_settings.get('openai_reasoning_mode', None)
        reasoning_context = model_settings.get('openai_reasoning_context', None)
        reasoning_summary = model_settings.get('openai_reasoning_summary', None)

        # Fall back to unified thinking when openai_reasoning_effort is not set
        if reasoning_effort is None and (thinking := model_request_parameters.thinking) is not None:
            reasoning_effort = resolve_openai_thinking_effort(
                thinking,
                self.profile,
                supports_max=self.profile.get('openai_responses_supports_max_reasoning_effort', False),
            )

        reasoning: Reasoning = {}
        if reasoning_effort:
            reasoning['effort'] = reasoning_effort
        if reasoning_mode and self.profile.get('openai_responses_supports_reasoning_mode', False):
            reasoning['mode'] = reasoning_mode
        set_reasoning_context(reasoning, reasoning_context, self.profile)
        if reasoning_summary:
            reasoning['summary'] = reasoning_summary
        return reasoning or OMIT

    def _request_thinks(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> bool:
        return reasoning_active(
            self.profile, cast(OpenAIResponsesModelSettings, model_settings or {}), model_request_parameters
        )

    def _settings_overriding_thinking(self, model_settings: Mapping[str, object]) -> tuple[str, ...]:
        # `_translate_thinking` sends an explicit effort instead of the one unified `thinking` maps to.
        return ('openai_reasoning_effort',) if model_settings.get('openai_reasoning_effort') is not None else ()

    def _overrides_unified_effort(self, model_settings: Mapping[str, object]) -> bool:
        return model_settings.get('openai_reasoning_effort') is not None

    def _get_responses_tool_choice(
        self,
        resolved_tool_choice: ResolvedToolChoice,
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[responses.FunctionToolParam], ResponsesToolChoice | None]:
        """Determine which tools to send and the API tool_choice value.

        Returns:
            A tuple of (filtered_function_tools, tool_choice).
            Note: builtin tools are handled separately and should be added to this list.
        """
        openai_profile = self.profile

        tool_choice: ResponsesToolChoice | None
        if resolved_tool_choice in ('auto', 'none'):
            tool_choice = resolved_tool_choice
        elif resolved_tool_choice == 'required':
            supports = support_openai_tool_forcing(
                self.model_name,
                openai_profile,
                model_settings,
                thinking=self._request_thinks(model_settings, model_request_parameters),
            )
            tool_choice = 'required' if supports else 'auto'
        elif isinstance(resolved_tool_choice, tuple):
            tool_choice_mode, tool_names = resolved_tool_choice
            tool_choice = self._map_tool_choice_restriction(
                tool_choice_mode, tool_names, model_settings, model_request_parameters
            )
        else:
            assert_never(resolved_tool_choice)

        # In client-executed tool search mode the `search_tools` function tool is
        # represented on the wire by `ToolSearchToolParam(execution='client')` — sending
        # it as a regular function tool too makes OpenAI emit `function_call` items
        # instead of `tool_search_call` items. The local dispatch path still sees the
        # function tool via `ModelRequestParameters.function_tools` (unaffected by this
        # filtering), so `_search_tools` continues to run normally.
        client_tool_search = has_tool_search(model_request_parameters)
        tools: list[responses.FunctionToolParam] = [
            self._map_tool_definition(t, visibility=model_request_parameters.visibility_of(t.name))
            for t in model_request_parameters.declared_tool_defs.values()
            if not (client_tool_search and t.tool_kind == 'tool-search')
        ]
        return tools, tool_choice

    def _map_tool_choice_restriction(
        self,
        tool_choice_mode: Literal['auto', 'required'],
        tool_names: set[str],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ResponsesToolChoice:
        supports = support_openai_tool_forcing(
            self.model_name,
            self.profile,
            model_settings,
            thinking=self._request_thinks(model_settings, model_request_parameters),
        )
        if tool_choice_mode == 'required' and len(tool_names) == 1 and supports:
            return ToolChoiceFunctionParam(type='function', name=next(iter(tool_names)))
        # `allowed_tools` filters via the API rather than the tools list,
        # so caching is preserved. Used for the multi-tool case and as the
        # cache-preserving fallback when forcing isn't supported.
        effective_mode = 'auto' if tool_choice_mode == 'auto' or not supports else tool_choice_mode
        return ToolChoiceAllowedParam(
            type='allowed_tools',
            mode=effective_mode,
            # Ignore names that are not currently available, keeping declaration order.
            # `tool_defs` rather than `declared_tool_defs`: a `'via_history'` tool travels in an
            # `additional_tools` item instead of `tools`, which GPT-5.6 and later accept in
            # `allowed_tools`; for earlier models `_build_responses_request_params` redeclares it
            # unless the request uses server-side state.
            tools=[{'type': 'function', 'name': n} for n in model_request_parameters.tool_defs if n in tool_names],
        )

    def _get_native_tools(self, model_request_parameters: ModelRequestParameters) -> list[responses.ToolParam]:
        tools: list[responses.ToolParam] = []
        has_image_generating_tool = False
        for tool in model_request_parameters.native_tools:
            match tool:
                case WebSearchTool():
                    tools.append(map_web_search_tool_param(tool))
                case FileSearchTool():
                    tools.append(map_file_search_tool_param(tool))
                case CodeExecutionTool():
                    has_image_generating_tool = True
                    tools.append(map_code_interpreter_tool_param(tool, self.system))
                case MCPServerTool():
                    tools.append(map_mcp_server_tool_param(tool))
                case ImageGenerationTool():
                    has_image_generating_tool = True
                    tools.append(map_openai_image_generation_tool(tool))
                case ToolSearchTool():
                    tools.append(self._map_tool_search_tool_param(tool, model_request_parameters))
                case _:  # pragma: no cover
                    raise UserError(
                        f'`{tool.__class__.__name__}` is not supported by `OpenAIResponsesModel`. If it should be, please file an issue.'
                    )

        if model_request_parameters.allow_image_output and not has_image_generating_tool:
            tools.append({'type': 'image_generation'})
        return tools

    @staticmethod
    def _map_tool_search_tool_param(
        tool: ToolSearchTool, model_request_parameters: ModelRequestParameters
    ) -> ToolSearchToolParam:
        # OpenAI's Responses API has no concept of named native strategies — the
        # `tool_search` builtin is one knob, decided server-side. Honor the user's
        # explicit choice or fail loudly; do not silently run a different algorithm.
        # `'custom'` is the internal marker for callable strategies and is handled
        # below — only named native strategies (`'bm25'`/`'regex'`) raise here.
        if tool.strategy is None:
            return ToolSearchToolParam(type='tool_search')
        if tool.strategy == 'custom':
            # With `strategy='custom'`, the search runs client-side via our local
            # `search_tools` function tool: the provider surfaces the search as a
            # `tool_search_call` with `execution='client'`, we dispatch it through the
            # normal function-call path, and reply with a `tool_search_output` that
            # carries the discovered tool definitions.
            search_tool_def = find_search_tool_definition(model_request_parameters)
            parameters = dict(search_tool_def.parameters_json_schema) if search_tool_def else {}
            # OpenAI's strict JSON schema mode is opt-in for client tool search — the
            # `additionalProperties: False` is what it prefers for closed-world schemas.
            parameters.setdefault('additionalProperties', False)
            # OpenAI's generated schema marks this optional, but the live Responses API
            # rejects client-executed `tool_search` without a non-null description.
            description = (
                search_tool_def.description
                if search_tool_def and search_tool_def.description
                else _DEFAULT_CLIENT_TOOL_SEARCH_DESCRIPTION
            )
            tool_search_param: ToolSearchToolParam = {
                'type': 'tool_search',
                'execution': 'client',
                'description': description,
                'parameters': parameters,
            }
            return tool_search_param
        raise UserError(
            f'`ToolSearch(strategy={tool.strategy!r})` is an Anthropic-native strategy '
            'and is not supported by OpenAI Responses. Use `strategy=None` (default, '
            "provider-managed), `strategy='keywords'` (local keyword matching), or "
            'a callable strategy.'
        )

    def _map_tool_definition(self, f: ToolDefinition, *, visibility: ToolVisibility) -> responses.FunctionToolParam:
        tool_param: responses.FunctionToolParam = {
            'name': f.name,
            'parameters': f.parameters_json_schema,
            'type': 'function',
            'description': f.description,
            'strict': bool(f.strict and self.profile.get('openai_supports_strict_tool_definition', True)),
        }
        # `'deferred'` visibility means "withhold this schema", and `prepare_request` only resolves
        # it where the API will accept it — which here means a
        # `tool_search` tool is along for the ride, without which this earns
        # `Invalid Value: 'tools.defer_loading'. Deferred tools require tools.tool_search.`
        if visibility == 'deferred':
            tool_param['defer_loading'] = True
        return tool_param

    def _resolve_previous_response_id(
        self,
        setting: str | None,
        messages: list[ModelMessage],
        *,
        store: bool | Omit | None = OMIT,
        allow_no_new_messages: bool = False,
        websocket_mode: bool = False,
    ) -> tuple[str | None, list[ModelMessage]]:
        # Resolve the effective `previous_response_id` and trim already-stored messages.
        #
        # A concrete ID in `setting` acts as a seed for the first request in a run (to
        # continue from a prior turn's stored response). On subsequent in-run requests
        # (retries, tool-call continuations), the most recent `provider_response_id`
        # from the message history takes precedence, so we chain to the latest stored
        # response instead of re-sending messages that are already server-side.
        #
        # A chain boundary in the tail (a compaction response, or a response whose prior
        # history has changed) holds even with a concrete seed: crossing it would re-inject
        # the context that compaction was meant to replace, or drop the change to the history.
        if setting is None:
            return None, messages
        auto_id, trimmed, at_boundary = self._get_previous_response_id_and_new_messages(
            messages, store=store, allow_no_new_messages=allow_no_new_messages, websocket_mode=websocket_mode
        )
        if auto_id is not None:
            return auto_id, trimmed
        if setting == 'auto' or at_boundary:
            return None, messages
        return setting, messages

    def _prompt_cache_comparison_response_id(
        self, messages: list[ModelRequest | ModelResponse], model_settings: OpenAIResponsesModelSettings
    ) -> str | None:
        """The response to compare this request against for prompt cache diagnostics, if they should be requested.

        That is the most recent response from this provider, unless a later compaction satisfies `_is_compaction_boundary`.
        OpenAI rejects an ID that doesn't start with `resp` with
        a 400, so a response from an endpoint that issues other IDs is skipped. So is one from the `/compact`
        endpoint: like `previous_response_id`, it's a boundary, and a request after it starts a new prefix anyway.
        """
        if not model_settings.get('openai_prompt_cache_diagnostics', True) or not self.profile.get(
            'openai_responses_supports_prompt_cache_diagnostics', False
        ):
            return None
        for message in reversed(messages):
            if isinstance(message, ModelResponse) and self._is_compaction_boundary(message):
                return None
            if isinstance(message, ModelResponse) and self.accepts_server_state(message):
                return self._comparable_response_id(message, model_settings)
        return None

    def _comparable_response_id(
        self, message: ModelResponse, model_settings: OpenAIResponsesModelSettings
    ) -> str | None:
        response_id = message.provider_response_id
        if not self._is_response_id_reachable(
            message,
            self._resolve_store(model_settings),
            websocket_mode=model_settings.get('openai_transport') == 'websocket',
        ):
            return None
        return response_id if response_id and response_id.startswith('resp_') else None

    def _is_compaction_boundary(self, response: ModelResponse) -> bool:
        """Keep accepted inline server state chainable; otherwise use the portable compaction boundary."""
        if self.accepts_server_state(response):
            return bool(response.provider_details and response.provider_details.get('compaction'))
        return any(
            isinstance(part, CompactionPart)
            and _compaction_part_is_wire_boundary(
                response,
                part,
                self.accepts_stateless_artifacts,
                requires_encrypted_content=self.compaction_requires_encrypted_content,
            )
            for part in response.parts
        )

    def _is_response_id_reachable(
        self, response: ModelResponse, store: bool | Omit | None, *, websocket_mode: bool = False
    ) -> bool:
        """Whether a request with the resolved `openai_store` value `store` can continue from `response`.

        A Responses WebSocket keeps the latest response on each lane in its cache, so a request on that lane can always
        continue from it. A response the socket created without storing it is kept nowhere else. Any other response
        is reachable wherever it's stored, as over HTTP, except from a socket request that doesn't store either: that
        conversation may never have been stored, as with Zero Data Retention, so only the lane's cache is relied on.
        """
        stream = self._websocket_stream
        if stream is not None:
            response_id = response.provider_response_id
            if response_id is not None and stream.lane.latest_response_id == response_id:
                return True
        details = response.provider_details or {}
        if (stream is not None or websocket_mode) and (store is False or details.get('store') is False):
            return False
        return details.get('transport') != 'websocket' or details.get('store') is True

    def _get_previous_response_id_and_new_messages(
        self,
        messages: list[ModelMessage],
        *,
        store: bool | Omit | None = OMIT,
        allow_no_new_messages: bool = False,
        websocket_mode: bool = False,
    ) -> tuple[str | None, list[ModelMessage], bool]:
        # Find the most recent `provider_response_id` from messages produced by this
        # provider, and return it along with the messages that came after it (which
        # still need to be sent as new input). When nothing suitable is found, returns
        # `(None, messages)` with the full list unchanged. The third element says whether
        # retained history has a chain boundary (an accepted compaction, or changed
        # history from this provider), which a concrete seed must not cross either.
        previous_response_id = None
        trimmed_messages: list[ModelMessage] = []
        for index in range(len(messages) - 1, -1, -1):
            m = messages[index]
            if isinstance(m, ModelResponse) and self._is_compaction_boundary(m):
                # Continue from the compaction item instead of restoring the older server-side context.
                return None, messages, True
            if isinstance(m, ModelResponse) and self.accepts_server_state(m):
                at_boundary = self._chain_break(m, messages[:index], store, websocket_mode=websocket_mode)
                if at_boundary is not None:
                    return None, messages, at_boundary
                previous_response_id = m.provider_response_id
                break
            trimmed_messages.append(m)

        if previous_response_id and (allow_no_new_messages or trimmed_messages):
            return previous_response_id, list(reversed(trimmed_messages)), False
        return None, messages, False

    def _chain_break(
        self,
        response: ModelResponse,
        prior_messages: list[ModelMessage],
        store: bool | Omit | None,
        *,
        websocket_mode: bool,
    ) -> bool | None:
        """`None` if a request can continue from `response`, else whether the break is a chain boundary."""
        # The server continues from this response and the history it was created with, so if
        # either has changed since (a history processor redacted or trimmed them, say),
        # continuing would silently drop the change: resend everything instead.
        if not self._history_known_unchanged(response, prior_messages):
            return True
        # A response cached only on another socket, or one that's gone, means replaying the history.
        if not self._is_response_id_reachable(response, store, websocket_mode=websocket_mode):
            return False
        return None

    def _resolve_server_side_state(
        self, model_settings: OpenAIResponsesModelSettings, messages: list[ModelMessage]
    ) -> tuple[str | None, str | None, list[ModelMessage]]:
        previous_response_id_setting = model_settings.get('openai_previous_response_id')
        conversation_id_setting = model_settings.get('openai_conversation_id')
        if previous_response_id_setting is not None and conversation_id_setting is not None:
            raise UserError(
                '`openai_previous_response_id` and `openai_conversation_id` cannot both be set because '
                'the OpenAI Responses API does not support `previous_response_id` with `conversation`.'
            )

        if conversation_id_setting is not None:
            conversation_id, messages = self._resolve_conversation_id(conversation_id_setting, messages)
            return None, conversation_id, messages

        previous_response_id, messages = self._resolve_previous_response_id(
            previous_response_id_setting,
            messages,
            store=self._resolve_store(model_settings),
            websocket_mode=model_settings.get('openai_transport') == 'websocket',
        )
        return previous_response_id, None, messages

    def _resolve_conversation_id(
        self, setting: Literal['auto'] | str, messages: list[ModelMessage]
    ) -> tuple[str | None, list[ModelMessage]]:
        if setting == 'auto':
            # Agent runs stamp the active Pydantic AI conversation ID on the final request.
            # Direct model calls may still pass an empty message list.
            pydantic_ai_conversation_id = next((m.conversation_id for m in messages[-1:]), None)
            return self._get_conversation_id_and_new_messages(
                messages, pydantic_ai_conversation_id=pydantic_ai_conversation_id
            )

        conversation_id, trimmed = self._get_conversation_id_and_new_messages(messages, openai_conversation_id=setting)
        if conversation_id is not None:
            return conversation_id, trimmed
        return setting, messages

    def _get_conversation_id_and_new_messages(
        self,
        messages: list[ModelMessage],
        *,
        openai_conversation_id: str | None = None,
        pydantic_ai_conversation_id: str | None = None,
    ) -> tuple[str | None, list[ModelMessage]]:
        trimmed_messages: list[ModelMessage] = []
        for m in reversed(messages):
            if isinstance(m, ModelResponse) and self.accepts_server_state(m):
                candidate = m.provider_details and m.provider_details.get('conversation_id')
                if (
                    pydantic_ai_conversation_id is not None
                    and m.conversation_id is not None
                    and m.conversation_id != pydantic_ai_conversation_id
                ):
                    trimmed_messages.append(m)
                    continue
                if isinstance(candidate, str) and (
                    openai_conversation_id is None or candidate == openai_conversation_id
                ):
                    return candidate, list(reversed(trimmed_messages))
            trimmed_messages.append(m)

        return None, messages

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        standing_prompt_retained: bool = True,
        previous_response_id: str | None = None,
    ) -> tuple[str | Omit, list[responses.ResponseInputItemParam]]:
        """Maps a `pydantic_ai.Message` to a `openai.types.responses.ResponseInputParam` i.e. the OpenAI Responses API input format.

        For `ThinkingParts`, this method:
        - Sends `signature` back as `encrypted_content` (for official OpenAI reasoning)
        - Sends `content` back as `summary` text
        - Sends `provider_details['raw_content']` back as `content` items (for gpt-oss raw CoT)

        Raw CoT is sent back to improve model performance in multi-turn conversations.

        """
        # `standing_prompt_retained=True` on ordinary requests: the compaction item retains the
        # window's leading `system` items (single hop, live-verified), so re-sending them would
        # duplicate the standing prompt. The re-compaction path passes `False`: retention decayed
        # across a second compaction in probing, so each freshly built window plants it explicitly.
        messages = self._trim_before_compaction(messages, standing_prompt_retained=standing_prompt_retained)
        profile = self.profile
        response_scoped_tool_call_ids = profile.get('openai_responses_tool_call_ids_are_response_scoped', False)
        context = _ResponsesMapMessagesContext(
            messages=messages,
            model_request_parameters=model_request_parameters,
            response_scoped_tool_call_ids=response_scoped_tool_call_ids,
            response_id=previous_response_id if response_scoped_tool_call_ids else None,
            send_item_ids=model_settings.get(
                'openai_send_reasoning_ids', profile.get('openai_supports_encrypted_reasoning_content', False)
            ),
            # Items from a response that wasn't stored can't be looked up by their ID later on.
            store_disabled=self._resolve_store(model_settings) is False,
            # With `execution='client'` tool search, `search_tools` calls/returns need to be
            # replayed as `tool_search_call` / `tool_search_output` items so the provider can
            # pair them with the builtin and unlock the discovered tools. The current
            # request's builtin configuration is the source of truth: a `ToolCallPart` for
            # `search_tools` belongs in the native replay flow whenever tool search is active.
            client_tool_search_active=has_tool_search(model_request_parameters),
        )
        for message_index, message in enumerate(messages):
            if isinstance(message, ModelRequest):
                await self._map_model_request(message, context)
            elif isinstance(message, ModelResponse):
                self._map_model_response(message_index, message, context)
            else:
                assert_never(message)
        move_leading_cache_breakpoints(context.openai_messages, text_type='input_text')
        instructions = get_instructions(messages, model_request_parameters) or OMIT
        return instructions, context.openai_messages

    async def _map_model_request(self, message: ModelRequest, context: _ResponsesMapMessagesContext) -> None:
        for part in _sendable_request_parts(message.parts):
            if isinstance(part, SystemPromptPart):
                context.openai_messages.append(
                    responses.EasyInputMessageParam(
                        role=self.profile.get('openai_system_prompt_role', None) or 'system', content=part.content
                    )
                )
            elif isinstance(part, UserPromptPart):
                context.openai_messages.append(await self._map_user_prompt(part))
            elif isinstance(part, (InstructionBaselinePart, InstructionDeltaPart)):
                raise _unprojected_instruction_delta_error()
            elif isinstance(part, ModelSettingsDeltaPart):
                self._map_model_settings_delta_part(part, context)
            elif isinstance(part, ToolAvailabilityDeltaPart):
                self._map_tool_availability_delta_part(part, context)
            elif isinstance(part, ToolReturnPart):
                await self._map_tool_return_part(part, context)
            elif isinstance(part, _UnpreparedPart):  # pragma: no cover
                raise _unprepared_part_error(part)
            elif isinstance(part, SpeechPart):  # pragma: no cover
                # Unconverted realtime speech; `prepare_messages` turns these into `UserPromptPart`s in `Model.prepare_messages`.
                raise _unconverted_speech_part_error()
            else:
                assert_never(part)

    def _map_model_settings_delta_part(
        self, part: ModelSettingsDeltaPart, context: _ResponsesMapMessagesContext
    ) -> None:
        # Only the mid-conversation changes render: a window's opening effort went out as the
        # request-level `reasoning.effort`. `_effort_delta_request` has already removed every
        # part when this request sends the current effort at request level instead.
        if not part.top_level and (effort := part.settings.get('thinking')) is not None:
            profile = self.profile
            context.openai_messages.append(
                responses.ResponseConfigurationUpdateItemParamParam(
                    type='configuration_update',
                    reasoning={
                        'effort': resolve_openai_thinking_effort(
                            effort,
                            profile,
                            supports_max=profile.get('openai_responses_supports_max_reasoning_effort', False),
                        )
                    },
                )
            )

    def _map_tool_availability_delta_part(
        self, part: ToolAvailabilityDeltaPart, context: _ResponsesMapMessagesContext
    ) -> None:
        if self.tool_addition_mode != 'with_definitions':
            # `prepare_messages` projects the delta onto the local tool-search exchange
            # for every model without native support, so arriving here means that
            # projection didn't run — reachable by calling `Model.request` directly, and
            # the same pipeline bug the other adapters raise on.
            #
            # Not because the API would reject the item: `gpt-5` and `gpt-4o`, both
            # outside the supported list, accept an `additional_tools` item and call the
            # tool it declares. So the raise is about the invariant, not the wire — but
            # silently rendering a shape whose support we haven't verified, for a tool
            # this same path has removed from `tools`, is how an availability change
            # goes missing with nothing to show for it.
            raise _unprepared_part_error(part)
        additional_tools = self._map_additional_tools(
            part.tools_added, context.model_request_parameters, rendered=context.rendered_additional_tools
        )
        if additional_tools['tools']:
            context.openai_messages.append(additional_tools)

    async def _map_tool_return_part(self, part: ToolReturnPart, context: _ResponsesMapMessagesContext) -> None:
        call_id = _guard_tool_call_id(t=part)
        call_id, _ = split_combined_tool_call_id(call_id)
        call_id = provider_response_tool_call_id(call_id, context.response_id)
        if call_id in context.client_replay_call_ids and isinstance(part, ToolSearchReturnPart):
            # Replay the local `search_tools` result as a
            # `tool_search_output` so the provider rehydrates the
            # discovered-tool state tied to the `tool_search_call`.
            context.openai_messages.append(
                build_tool_search_output_param(
                    part,
                    call_id,
                    'client',
                    'completed',
                    context.model_request_parameters,
                    self._map_tool_definition,
                )
            )
        else:
            output = await self._map_tool_return_output(part)
            item = FunctionCallOutput(
                type='function_call_output',
                call_id=call_id,
                output=output,
            )
            context.openai_messages.append(item)
            if (
                isinstance(part, ToolSearchReturnPart)
                and not context.client_tool_search_active
                and self.tool_addition_mode == 'with_definitions'
            ):
                additional_tools = self._map_additional_tools(
                    [match['name'] for match in part.discovered_tools],
                    context.model_request_parameters,
                    rendered=context.rendered_additional_tools,
                )
                if additional_tools['tools']:
                    context.openai_messages.append(additional_tools)

    def _map_model_response(
        self, message_index: int, message: ModelResponse, context: _ResponsesMapMessagesContext
    ) -> None:
        # Server-side state is only sent back to the endpoint that created it. A part from this provider in a
        # response from another (or none) has no endpoint of its own, so it's judged by name as before.
        same_endpoint = message.provider_name != self.system or self.accepts_server_state(message)
        response_context = _ResponsesMapModelResponseContext(
            message=message,
            same_endpoint=same_endpoint,
            response_from_same_provider=message.provider_name == self.system and same_endpoint,
            response_start=len(context.openai_messages),
        )
        response_parts: Sequence[ModelResponsePart] = message.parts
        if not self.profile.get('openai_responses_supports_interleaved_function_calls', True):
            response_parts = group_settled_portable_function_calls(
                context.messages,
                message_index,
                message,
                client_tool_search_active=context.client_tool_search_active,
            )
        for item in response_parts:
            self._map_response_part(item, response_context, context)
        if response_context.carried_thinkings:
            # See `ModelProfile.mimics_assistant_message_formatting`: reasoning replayed in an
            # assistant turn teaches the model to write thinking tags into its visible answers.
            context.openai_messages.insert(
                response_context.response_start,
                responses.EasyInputMessageParam(role='user', content='\n\n'.join(response_context.carried_thinkings)),
            )
        context.response_id = (
            message.provider_response_id
            if context.response_scoped_tool_call_ids and response_context.response_from_same_provider
            else None
        )

    def _map_response_part(
        self,
        item: ModelResponsePart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
    ) -> None:
        message = response_context.message
        from_same_provider = response_context.same_endpoint and (
            item.provider_name == self.system or (item.provider_name is None and message.provider_name == self.system)
        )
        response_id_from_same_provider = response_context.same_endpoint and (
            message.provider_name == self.system
            or (message.provider_name is None and item.provider_name == self.system)
        )
        response_context.response_from_same_provider |= response_id_from_same_provider
        # Two distinct gates. For native tool items whose state lives server-side
        # (web search, code interpreter, image generation, MCP), the item ID *is*
        # the replay payload, so `should_send_item_id` decides whether those items
        # are sent at all. Hosted tool-search items carry their state inline, so
        # they replay on `from_same_provider` alone and this flag only strips their
        # IDs (see the `openai_send_reasoning_ids` docstring).
        should_send_item_id = context.send_item_ids and from_same_provider

        match item:
            case TextPart():
                self._map_response_text_part(item, response_context, context, should_send_item_id=should_send_item_id)
            case ToolCallPart():
                self._map_response_tool_call_part(
                    item,
                    response_context,
                    context,
                    response_id_from_same_provider=response_id_from_same_provider,
                    should_send_item_id=should_send_item_id,
                )
            case NativeToolCallPart():
                self._map_response_native_tool_call_part(
                    item,
                    response_context,
                    context,
                    from_same_provider=from_same_provider,
                    should_send_item_id=should_send_item_id,
                )
            case NativeToolReturnPart():
                self._map_response_native_tool_return_part(
                    item,
                    response_context,
                    context,
                    from_same_provider=from_same_provider,
                    should_send_item_id=should_send_item_id,
                )
            case FilePart():
                # This was generated by the `ImageGenerationTool` or `CodeExecutionTool`,
                # and does not need to be sent back separately from the corresponding `NativeToolReturnPart`.
                # If `send_item_ids` is false, we won't send the `NativeToolReturnPart`, but OpenAI does not have a type for files from the assistant.
                pass
            case ThinkingPart():
                self._map_response_thinking_part(
                    item, response_context, context, should_send_item_id=should_send_item_id
                )
            case CompactionPart():
                self._map_response_compaction_part(item, response_context, context)
            case SpeechPart():  # pragma: no cover
                # Unconverted realtime speech; `prepare_messages` turns these into `TextPart`s in `Model.prepare_messages`.
                raise _unconverted_speech_part_error()
            case _:
                assert_never(item)

    def _map_response_text_part(
        self,
        item: TextPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
        *,
        should_send_item_id: bool,
    ) -> None:
        phase = (item.provider_details or {}).get('phase')
        send_phase = (
            self.profile.get('openai_supports_phase', False)
            and item.provider_name == self.system
            and phase in ('commentary', 'final_answer')
        )
        if item.id and should_send_item_id:
            message_item = response_context.message_item
            if message_item is None or message_item['id'] != item.id:  # pragma: no branch
                message_item = responses.ResponseOutputMessageParam(
                    role='assistant',
                    id=item.id,
                    content=[],
                    type='message',
                    status='completed',
                )
                response_context.message_item = message_item
                context.openai_messages.append(message_item)

            message_item['content'] = [
                *message_item['content'],
                responses.ResponseOutputTextParam(
                    text=item.content,
                    type='output_text',
                    # Citations are only replayed with the message ID of the output they came from.
                    annotations=map_responses_citations_for_replay(
                        (item.provider_details or {}).get('citations'), item.content
                    ),
                ),
            ]
            if send_phase:
                message_item['phase'] = phase
        else:
            easy_message_item = responses.EasyInputMessageParam(role='assistant', content=item.content)
            if send_phase:
                easy_message_item['phase'] = phase
            context.openai_messages.append(easy_message_item)

    def _map_response_tool_call_part(
        self,
        item: ToolCallPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
        *,
        response_id_from_same_provider: bool,
        should_send_item_id: bool,
    ) -> None:
        call_id = _guard_tool_call_id(t=item)
        call_id, id = split_combined_tool_call_id(call_id)
        call_id = provider_response_tool_call_id(
            call_id,
            response_context.message.provider_response_id
            if context.response_scoped_tool_call_ids and response_id_from_same_provider
            else None,
        )
        id = id or item.id

        if context.client_tool_search_active and isinstance(item, ToolSearchCallPart):
            # Replay the local `search_tools` call as a `tool_search_call`
            # with `execution='client'` so OpenAI re-attaches it to the
            # builtin and unlocks the discovered tools' schemas. Fires for
            # any `provider_name`, including `None` (cross-provider history
            # from a prior local-fallback turn) and other-provider names
            # (Google, etc.) — `execution='client'` is OpenAI's accepted
            # historical-replay shape regardless of the call's origin.
            context.client_replay_call_ids.add(call_id)
            tool_search_call: ToolSearchCallParam = {
                'type': 'tool_search_call',
                'execution': 'client',
                'arguments': item.args_as_dict() or {},
                'call_id': call_id,
                'status': 'completed',
            }
            if id and should_send_item_id:  # pragma: no branch
                tool_search_call['id'] = id
            context.openai_messages.append(tool_search_call)
        else:
            context.openai_messages.append(
                self._map_function_call(item, call_id, id, context, should_send_item_id=should_send_item_id)
            )

    def _map_function_call(
        self,
        item: ToolCallPart,
        call_id: str,
        id: str | None,
        context: _ResponsesMapMessagesContext,
        *,
        should_send_item_id: bool,
    ) -> responses.ResponseFunctionToolCallParam:
        param = responses.ResponseFunctionToolCallParam(
            name=item.tool_name,
            arguments=item.args_as_json_str(),
            call_id=call_id,
            type='function_call',
        )
        if self.profile.get('openai_responses_requires_function_call_status_none', False):
            # The SDK types `status` as a non-null literal, but vLLM before vllm-project/vllm#26706 requires `None`.
            param['status'] = None  # pyright: ignore[reportGeneralTypeIssues]
        if id and should_send_item_id:  # pragma: no branch
            param['id'] = id
        # The Responses API attaches a `namespace` to function calls
        # for discovered deferred tools (tool-search flow). Round-trip
        # it on replay or the API rejects the call as missing namespace.
        if (
            item.provider_name == self.system
            and item.provider_details
            and (ns := item.provider_details.get('namespace'))
        ):
            param['namespace'] = ns
        elif synthesized_ns := tool_search_namespace_for_synthesis(item.tool_name, context.model_request_parameters):
            # Cross-provider replay: prior turn ran on a non-OpenAI
            # provider, so no namespace was captured. See helper for the
            # evidence behind synthesizing `namespace = tool_name`.
            param['namespace'] = synthesized_ns
        return param

    def _map_response_native_tool_call_part(
        self,
        item: NativeToolCallPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
        *,
        from_same_provider: bool,
        should_send_item_id: bool,
    ) -> None:
        # Hosted tool-search items carry their state inline, so they replay to any provider
        # that accepts this one's stateless data; their IDs stay with the same provider.
        if (
            (from_same_provider or self.accepts_stateless_artifacts(response_context.message, item))
            and item.tool_name == ToolSearchTool.kind
            and item.tool_call_id
        ):
            call_id, status, _ = tool_search_replay_details(item)
            tool_search_call = responses.response_input_item_param.ToolSearchCall(
                call_id=call_id,
                arguments=item.args_as_dict() or {},
                type='tool_search_call',
                execution='server',
                status=status,
            )
            if should_send_item_id and item.id:
                tool_search_call['id'] = item.id
            context.openai_messages.append(tool_search_call)
        elif should_send_item_id:  # pragma: no branch
            self._map_stateful_native_tool_call(item, response_context, context)

    def _map_stateful_native_tool_call(
        self,
        item: NativeToolCallPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
    ) -> None:
        """Replay a native tool call whose state lives server-side: its item ID is the replay payload."""
        if (
            item.tool_name == CodeExecutionTool.kind
            and item.tool_call_id
            and (args := item.args_as_dict())
            and (container_id := args.get('container_id'))
        ):
            response_context.code_interpreter_item = responses.ResponseCodeInterpreterToolCallParam(
                id=item.tool_call_id,
                code=args.get('code'),
                container_id=container_id,
                outputs=None,  # These can be read server-side
                status='completed',
                type='code_interpreter_call',
            )
            context.openai_messages.append(response_context.code_interpreter_item)
        elif item.tool_name == WebSearchTool.kind and item.tool_call_id and (args := item.args_as_dict()):
            response_context.web_search_item = map_web_search_call_for_replay(item, args)
            context.openai_messages.append(response_context.web_search_item)
        elif item.tool_name == FileSearchTool.kind and item.tool_call_id and (args := item.args_as_dict()):
            response_context.file_search_item = cast(
                responses.ResponseFileSearchToolCallParam,
                {
                    'id': item.id or item.tool_call_id,
                    'queries': args.get('queries', []),
                    'status': 'completed',
                    'type': 'file_search_call',
                },
            )
            context.openai_messages.append(response_context.file_search_item)
        elif item.tool_name == ImageGenerationTool.kind and item.tool_call_id:
            # The API always resolves an `image_generation_call` by its ID, even when
            # `result` is sent inline, so one from an unstored response can't be sent back.
            if not context.store_disabled:
                # The cast is necessary because of https://github.com/openai/openai-python/issues/2648
                image_generation_item = cast(
                    responses.response_input_item_param.ImageGenerationCall,
                    {
                        'id': item.tool_call_id,
                        'type': 'image_generation_call',
                    },
                )
                context.openai_messages.append(image_generation_item)
        elif item.tool_name.startswith(MCPServerTool.kind):  # pragma: no branch
            self._map_mcp_server_tool_call(item, context)

    def _map_mcp_server_tool_call(self, item: NativeToolCallPart, context: _ResponsesMapMessagesContext) -> None:
        if (  # pragma: no branch
            item.tool_call_id
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
                context.openai_messages.append(mcp_list_tools_item)
            elif (  # pragma: no branch
                action == 'call_tool'
                and (tool_name := args.get('tool_name'))
                and (tool_args := args.get('tool_args')) is not None
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
                context.openai_messages.append(mcp_call_item)

    def _map_response_native_tool_return_part(
        self,
        item: NativeToolReturnPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
        *,
        from_same_provider: bool,
        should_send_item_id: bool,
    ) -> None:
        if (
            (from_same_provider or self.accepts_stateless_artifacts(response_context.message, item))
            and isinstance(item, NativeToolSearchReturnPart)
            and not lacks_tool_search_output_identity(item)
        ):
            call_id, status, provider_details = tool_search_replay_details(item)
            tool_search_output = build_tool_search_output_param(
                item,
                call_id,
                'server',
                status,
                context.model_request_parameters,
                self._map_tool_definition,
            )
            output_id = provider_details.get('id')
            if should_send_item_id and isinstance(output_id, str):
                tool_search_output['id'] = output_id
            context.openai_messages.append(tool_search_output)
        elif should_send_item_id:  # pragma: no branch
            status = item.content.get('status') if _is_str_dict(item.content) else None
            kind_to_item = {
                CodeExecutionTool.kind: response_context.code_interpreter_item,
                WebSearchTool.kind: response_context.web_search_item,
                FileSearchTool.kind: response_context.file_search_item,
            }
            if status and (builtin_item := kind_to_item.get(item.tool_name)) is not None:
                builtin_item['status'] = status
            elif item.tool_name == ImageGenerationTool.kind:
                # Image generation result does not need to be sent back, just the `id` off of `NativeToolCallPart`.
                pass
            elif item.tool_name.startswith(MCPServerTool.kind):
                # MCP call result does not need to be sent back, just the fields off of `NativeToolCallPart`.
                pass

    def _map_response_thinking_part(
        self,
        item: ThinkingPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
        *,
        should_send_item_id: bool,
    ) -> None:
        profile = self.profile
        # Get raw CoT content from provider_details if present and from this provider
        raw_content: list[str] | None = None
        if item.provider_name == self.system:
            raw_content = (item.provider_details or {}).get('raw_content')

        # Chat Completions stores its content/reasoning field name as a synthetic
        # part ID. Preserve opaque IDs from compatible Responses APIs unless they
        # collide with one of those sentinels without carrying native wire data.
        synthetic_chat_reasoning = item.id in ('content', 'reasoning', 'reasoning_content')
        native_reasoning = not synthetic_chat_reasoning or bool(item.signature or raw_content)
        # Encrypted reasoning carries its state inline, so it's also sent back to a provider
        # that accepts this one's encrypted data, with the item ID it was issued under.
        signature: str | None = None
        if (
            item.signature
            and profile.get('openai_supports_encrypted_reasoning_content', False)
            and self.accepts_stateless_artifacts(response_context.message, item)
        ):
            signature = item.signature
        if (
            item.id
            and native_reasoning
            and (should_send_item_id or raw_content or (context.send_item_ids and signature))
        ):
            self._map_thinking_to_reasoning_item(item, item.id, signature, raw_content, response_context, context)
        else:
            thinking_text, carry_in_user_turn = render_replayed_thinking(item.content, profile)
            if not carry_in_user_turn:
                context.openai_messages.append(responses.EasyInputMessageParam(role='assistant', content=thinking_text))
            elif item.content:
                # An empty wrapper in the user turn would tell the model the assistant thought nothing.
                response_context.carried_thinkings.append(thinking_text)

    def _map_thinking_to_reasoning_item(
        self,
        item: ThinkingPart,
        item_id: str,
        signature: str | None,
        raw_content: list[str] | None,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
    ) -> None:
        reasoning_item = response_context.reasoning_item
        if (reasoning_item is None or reasoning_item['id'] != item_id) and (
            signature or item.content or raw_content
        ):  # pragma: no branch
            reasoning_item = responses.ResponseReasoningItemParam(
                id=item_id,
                summary=[],
                encrypted_content=signature,
                type='reasoning',
            )
            response_context.reasoning_item = reasoning_item
            context.openai_messages.append(reasoning_item)

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
            reasoning_item['content'] = [ReasoningContent(text=text, type='reasoning_text') for text in raw_content]

    def _map_response_compaction_part(
        self,
        item: CompactionPart,
        response_context: _ResponsesMapModelResponseContext,
        context: _ResponsesMapMessagesContext,
    ) -> None:
        # The same predicate `_trim_before_compaction` drops history before a compaction part by.
        if (
            self.accepts_stateless_artifacts(response_context.message, item)
            and item.provider_details
            and 'encrypted_content' in item.provider_details
        ):
            context.openai_messages.append(
                ResponseCompactionItemParamParam(
                    id=item.id,
                    encrypted_content=item.provider_details['encrypted_content'],
                    type='compaction',
                )
            )

    def _map_additional_tools(
        self, tool_names: list[str], model_request_parameters: ModelRequestParameters, *, rendered: set[str]
    ) -> responses.response_input_item_param.AdditionalTools:
        """Build one `additional_tools` item for the given names, deduped across the request.

        A `'via_history'` name renders because this item is the only place its definition travels;
        a `'deferred'` name renders because its `tools`-array entry stays schema-withheld for
        byte-stable caching and the item is what delivers the schema on reveal
        (`test_tool_availability_delta_and_the_tools_cache_section` measures that property). A
        `'visible'` name is fully declared in `tools` and has nothing to add, and a `'withheld'`
        name — a delta can name one when a direct `Model.request` caller authors
        `revealed_tool_names` that don't cover the history's deltas — must not have the schema the
        request just withheld smuggled on via the item.

        `rendered` accumulates the names already declared by an earlier history part (a
        duplicated delta from a UI round-trip, a delta plus a replayed search return) — one
        declaration per request is enough, mirroring the Anthropic renderer's `tool_addition`
        dedupe.
        """
        tool_defs_by_name = {tool.name: tool for tool in model_request_parameters.function_tools}
        renderable: list[str] = []
        for name in tool_names:
            # Marking `rendered` per accepted name (like the Anthropic loop) also dedupes a name
            # repeated within this part's own list, not just across parts.
            if (
                name not in rendered
                and name in tool_defs_by_name
                and model_request_parameters.visibility_of(name) in ('deferred', 'via_history')
            ):
                renderable.append(name)
                rendered.add(name)
        return responses.response_input_item_param.AdditionalTools(
            type='additional_tools',
            role='developer',
            tools=[self._map_tool_definition(tool_defs_by_name[name], visibility='visible') for name in renderable],
        )

    def _map_json_schema(self, o: OutputObjectDefinition) -> responses.ResponseFormatTextJSONSchemaConfigParam:
        response_format_param: responses.ResponseFormatTextJSONSchemaConfigParam = {
            'type': 'json_schema',
            'name': o.name or DEFAULT_OUTPUT_TOOL_NAME,
            'schema': o.json_schema,
        }
        if o.description:
            response_format_param['description'] = o.description
        if self.profile.get('openai_supports_strict_tool_definition', True):  # pragma: no branch
            response_format_param['strict'] = o.strict
        return response_format_param

    async def _map_user_prompt(self, part: UserPromptPart) -> responses.EasyInputMessageParam:
        if isinstance(part.content, str):
            return responses.EasyInputMessageParam(role='user', content=part.content)
        content: list[responses.ResponseInputContentParam] = []
        for item in part.content:
            if isinstance(item, str | TextContent):
                text = item if isinstance(item, str) else item.content
                content.append(responses.ResponseInputTextParam(text=text, type='input_text'))
            elif isinstance(item, UploadedFile):
                content.append(self._map_uploaded_file_to_response_content(item))  # pyright: ignore[reportArgumentType]
            elif isinstance(item, CachePoint):
                self._map_user_prompt_cache_point(content)
            elif is_multi_modal_content(item):
                content.append(await OpenAIResponsesModel._map_file_to_response_content(item, 'user prompts'))  # pyright: ignore[reportArgumentType]
            else:
                raise RuntimeError(f'Unsupported content type: {type(item)}')  # pragma: no cover
        return responses.EasyInputMessageParam(role='user', content=content)

    def _map_user_prompt_cache_point(self, content: list[responses.ResponseInputContentParam]) -> None:
        if self.profile.get('openai_supports_prompt_cache_breakpoints', False):
            if not content:
                content.append(responses.ResponseInputTextParam(text='', type='input_text'))
            add_openai_prompt_cache_breakpoint(content)

    def _map_uploaded_file_to_response_content(
        self,
        item: UploadedFile,
    ) -> ResponseInputImageContentParam | ResponseInputFileContentParam:
        """Map an `UploadedFile` to its OpenAI Responses API content param.

        Raises `UserError` if the file was uploaded to a different provider (`provider_name != self.system`).

        Image uploads (an `image/*` media type) map to `input_image`, carrying `detail` from
        `vendor_metadata` (default `'auto'`); everything else maps to `input_file`. Opaque OpenAI
        Files-API ids (e.g. `file-...`) report `application/octet-stream`, so an image referenced by
        such an id without an explicit `image/*` media type also maps to `input_file`.
        """
        self._validate_uploaded_file_provider(item)
        return uploaded_file_to_response_content(item)

    @staticmethod
    async def _map_file_to_response_content(
        item: BinaryContent | ImageUrl | DocumentUrl | AudioUrl | VideoUrl,
        context: str,
    ) -> ResponseInputImageContentParam | ResponseInputFileContentParam:
        """Map a multimodal file item to its OpenAI Responses API content param."""
        if isinstance(item, BinaryContent):
            return binary_content_to_response_content(item, context)
        elif isinstance(item, ImageUrl):
            detail: Literal['auto', 'low', 'high'] = 'auto'
            image_url = item.url
            if metadata := item.vendor_metadata:
                detail = metadata.get('detail', 'auto')
            if item.force_download:
                downloaded = await download_item(item, data_format='base64_uri', type_format='extension')
                image_url = downloaded['data']
            return ResponseInputImageContentParam(
                image_url=image_url,
                type='input_image',
                detail=detail,
            )
        elif isinstance(item, (AudioUrl, DocumentUrl)):
            if item.force_download:
                downloaded = await download_item(item, data_format='base64_uri', type_format='extension')
                return ResponseInputFileContentParam(
                    type='input_file',
                    file_data=downloaded['data'],
                    filename=f'filename.{downloaded["data_type"]}',
                )
            return ResponseInputFileContentParam(
                type='input_file',
                file_url=item.url,
            )
        else:
            raise NotImplementedError(f'VideoUrl is not supported in OpenAI Responses {context}')

    async def _map_tool_return_output(
        self,
        part: ToolReturnPart,
    ) -> str | list[ResponseInputTextContentParam | ResponseInputImageContentParam | ResponseInputFileContentParam]:
        """Map a `ToolReturnPart` to OpenAI Responses API output format, supporting multimodal content.

        Iterates content directly to preserve order of mixed file/data content.
        """
        if not part.files:
            return part.model_response_str()

        output: list[
            ResponseInputTextContentParam | ResponseInputImageContentParam | ResponseInputFileContentParam
        ] = []

        for item in part.content_items(mode='str'):
            if isinstance(item, UploadedFile):
                output.append(self._map_uploaded_file_to_response_content(item))
            elif is_multi_modal_content(item):
                output.append(await OpenAIResponsesModel._map_file_to_response_content(item, 'tool returns'))  # pyright: ignore[reportArgumentType]
            elif isinstance(item, str):  # pragma: no branch
                output.append(ResponseInputTextContentParam(type='input_text', text=item))

        return output


@dataclass
class _ResponsesMapMessagesContext:
    """What `OpenAIResponsesModel._map_messages` carries from one message to the next."""

    messages: list[ModelMessage]
    model_request_parameters: ModelRequestParameters
    response_scoped_tool_call_ids: bool
    response_id: str | None
    send_item_ids: bool
    store_disabled: bool
    client_tool_search_active: bool
    client_replay_call_ids: set[str] = field(default_factory=set[str])
    # Mirrors the Anthropic renderer's per-request `tool_addition` dedupe: several history
    # parts may name the same revealed tool (duplicated deltas from a UI round-trip, a delta
    # plus a replayed search return), and one declaration per request is enough.
    rendered_additional_tools: set[str] = field(default_factory=set[str])
    openai_messages: list[responses.ResponseInputItemParam] = field(
        default_factory=list[responses.ResponseInputItemParam]
    )


@dataclass
class _ResponsesMapModelResponseContext:
    """What `OpenAIResponsesModel._map_model_response` carries from one part of a `ModelResponse` to the next."""

    message: ModelResponse
    same_endpoint: bool
    response_from_same_provider: bool
    response_start: int
    message_item: responses.ResponseOutputMessageParam | None = None
    reasoning_item: responses.ResponseReasoningItemParam | None = None
    web_search_item: responses.ResponseFunctionWebSearchParam | None = None
    file_search_item: responses.ResponseFileSearchToolCallParam | None = None
    code_interpreter_item: responses.ResponseCodeInterpreterToolCallParam | None = None
    carried_thinkings: list[str] = field(default_factory=list[str])


def group_settled_portable_function_calls(
    messages: list[ModelMessage],
    message_index: int,
    message: ModelResponse,
    *,
    client_tool_search_active: bool,
) -> Sequence[ModelResponsePart]:
    """Reorder one assistant turn's settled function calls to the end of their segment.

    Only for endpoints whose profile clears `openai_responses_supports_interleaved_function_calls`.
    Such an endpoint folds each call into the assistant message next to it, so an assistant item
    sitting between two calls splits them into separate messages, each carrying an unanswered call.
    Item IDs are deliberately *not* consulted: an endpoint that merges items this way derives an
    item's position from the surrounding sequence rather than its identity, so an ID pins nothing
    (verified against DeepSeek: the grouped order is accepted with reasoning, message and
    `function_call` IDs all present on the wire, while the interleaved order is rejected with them).

    Reordering is skipped when the turn carries a native or compaction item the provider owns, or
    when any of its calls is still unanswered — an unanswered call is rejected in either order, so
    grouping cannot rescue it.

    "Answered" is deliberately wire-local and stricter than `_agent_graph._dangling_tool_calls_by_response`:
    only the requests between this response and the next one count, and repeated call IDs are
    counted rather than shadowed. A model must not import the graph layer, and the two answer
    different questions, so they are allowed to disagree — but only in the direction where this one
    declines to reorder. Keep any change on that side.
    """
    parts = message.parts
    if any(isinstance(part, (NativeToolCallPart, NativeToolReturnPart, CompactionPart)) for part in parts):
        return parts

    unsettled_call_counts: dict[str, int] = {}
    # Active tool search calls are replayed as `tool_search_call` items, not function calls, so
    # neither they nor their answers count. A retried return carries no `tool_kind`; it is matched to
    # its search call by id and that call's name.
    search_call_names: dict[str, str] = {}
    for part in parts:
        if client_tool_search_active and isinstance(part, ToolSearchCallPart):
            search_call_names[part.tool_call_id] = part.tool_name
        elif isinstance(part, ToolCallPart):
            unsettled_call_counts[part.tool_call_id] = unsettled_call_counts.get(part.tool_call_id, 0) + 1
    for following_message_index in range(message_index + 1, len(messages)):
        following_message = messages[following_message_index]
        if isinstance(following_message, ModelResponse):
            break
        settle_answered_function_calls(
            following_message,
            unsettled_call_counts,
            search_call_names,
            client_tool_search_active=client_tool_search_active,
        )
    if unsettled_call_counts:
        return parts
    return group_function_calls_by_segment(parts, client_tool_search_active=client_tool_search_active)


def settle_answered_function_calls(
    request: ModelRequest,
    unsettled_call_counts: dict[str, int],
    search_call_names: Mapping[str, str],
    *,
    client_tool_search_active: bool,
) -> None:
    """Count down `unsettled_call_counts` for each function call `request` answers."""
    for part in request.parts:
        if (
            isinstance(part, ToolReturnPart)
            and not (client_tool_search_active and isinstance(part, ToolSearchReturnPart))
            and search_call_names.get(part.tool_call_id) != part.tool_name
            and (count := unsettled_call_counts.get(part.tool_call_id))
        ):
            if count == 1:
                del unsettled_call_counts[part.tool_call_id]
            else:
                unsettled_call_counts[part.tool_call_id] = count - 1


def group_function_calls_by_segment(
    parts: Sequence[ModelResponsePart], *, client_tool_search_active: bool
) -> list[ModelResponsePart]:
    """Move the function calls to the end of each segment, which a replayed tool search call ends."""
    grouped_parts: list[ModelResponsePart] = []
    segment: list[ModelResponsePart] = []
    for part in parts:
        if client_tool_search_active and isinstance(part, ToolSearchCallPart):
            flush_function_call_segment(grouped_parts, segment)
            grouped_parts.append(part)
        else:
            segment.append(part)
    flush_function_call_segment(grouped_parts, segment)
    return grouped_parts


def flush_function_call_segment(grouped_parts: list[ModelResponsePart], segment: list[ModelResponsePart]) -> None:
    """Append a segment's other parts, then its function calls, to `grouped_parts`, and empty the segment."""
    grouped_parts.extend(part for part in segment if not isinstance(part, ToolCallPart))
    grouped_parts.extend(part for part in segment if isinstance(part, ToolCallPart))
    segment.clear()


def split_combined_tool_call_id(combined_id: str) -> tuple[str, str | None]:
    # When reasoning, the Responses API requires the `ResponseFunctionToolCall` to be returned with both the `call_id` and `id` fields.
    # Before our `ToolCallPart` gained the `id` field alongside `tool_call_id` field, we combined the two fields into a single string stored on `tool_call_id`.
    if '|' in combined_id:
        call_id, id = combined_id.split('|', 1)
        return call_id, id
    else:
        return combined_id, None


def provider_response_tool_call_id(tool_call_id: str, response_id: str | None) -> str:
    """Restore a response-scoped provider tool-call ID from its normalized form."""
    if response_id is None:
        return tool_call_id
    return tool_call_id.removeprefix(f'{response_id}:')


def uploaded_file_to_response_content(
    item: UploadedFile,
) -> ResponseInputImageContentParam | ResponseInputFileContentParam:
    """Map an `UploadedFile` whose provider has already been checked to its OpenAI Responses API content param."""
    if item.media_type.startswith('image/'):
        detail: Literal['auto', 'low', 'high'] = 'auto'
        if metadata := item.vendor_metadata:
            detail = metadata.get('detail', 'auto')
        return ResponseInputImageContentParam(type='input_image', file_id=item.file_id, detail=detail)
    return ResponseInputFileContentParam(type='input_file', file_id=item.file_id)


def binary_content_to_response_content(
    item: BinaryContent, context: str
) -> ResponseInputImageContentParam | ResponseInputFileContentParam:
    """Map a `BinaryContent` to its OpenAI Responses API content param."""
    if item.is_image:
        detail: Literal['auto', 'low', 'high'] = 'auto'
        if metadata := item.vendor_metadata:
            detail = metadata.get('detail', 'auto')
        return ResponseInputImageContentParam(
            image_url=item.data_uri,
            type='input_image',
            detail=detail,
        )
    elif item.is_document:
        return ResponseInputFileContentParam(
            type='input_file',
            file_data=item.data_uri,
            filename=f'filename.{item.format}',
        )
    elif item.is_audio:
        raise NotImplementedError(f'BinaryContent with audio is not supported in OpenAI Responses {context}')
    elif item.is_video:
        raise NotImplementedError(f'BinaryContent with video is not supported in OpenAI Responses {context}')
    else:  # pragma: no cover
        raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
