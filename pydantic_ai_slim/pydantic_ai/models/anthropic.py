from __future__ import annotations as _annotations

import io
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Literal, TypeAlias, cast, overload

import pydantic_core
from pydantic import TypeAdapter
from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._http import to_httpx2_timeout
from .._run_context import RunContext
from .._tool_search import _NO_MATCHES_MESSAGE  # pyright: ignore[reportPrivateUsage]
from .._utils import guard_tool_call_id as _guard_tool_call_id, is_str_dict
from ..capabilities.abstract import AbstractCapability
from ..exceptions import ModelAPIError, UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    FinishReason,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
    is_multi_modal_content,
)
from ..native_tools import (
    SUPPORTED_NATIVE_TOOLS,
    AbstractNativeTool,
    AdvisorTool,
    CodeExecutionTool,
    MCPServerTool,
    MemoryTool,
    WebFetchTool,
    WebSearchTool,
)
from ..native_tools._tool_search import (
    ToolSearchArgs,
    ToolSearchMatch,
    ToolSearchTool,
)
from ..profiles import DEFAULT_THINKING_TAGS, ModelProfileSpec, merge_profile
from ..profiles.anthropic import (
    ANTHROPIC_THINKING_BUDGET_MAP,
    AnthropicCodeExecutionToolVersion,
    AnthropicEffort,
    AnthropicModelProfile,
    resolve_anthropic_effort,
)
from ..providers import Provider, infer_provider
from ..providers.anthropic import AsyncAnthropicClient
from ..settings import ModelSettings, ThinkingLevel, merge_model_settings
from ..tools import AgentDepsT, ToolDefinition
from ..toolsets._tool_search import discovered_tool_names_in_order
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    ToolVisibility,
    _standing_system_prompt_count,  # pyright: ignore[reportPrivateUsage]
    _suggest_known_model_id_from_provider_error,  # pyright: ignore[reportPrivateUsage]
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unsynthesized_tool_availability_delta_error,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from ._anthropic_containers import is_tool_result_only as _is_tool_result_only
from ._tool_choice import ResolvedToolChoice, resolve_tool_choice

_FINISH_REASON_MAP: dict[BetaStopReason, FinishReason | None] = {
    'compaction': 'stop',
    'end_turn': 'stop',
    'max_tokens': 'length',
    'model_context_window_exceeded': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
    'pause_turn': None,
    'refusal': 'content_filter',
}


def _revealed_deferred_tool_order(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
) -> list[str]:
    """Order revealed non-corpus deferred entries at the tail by their first reveal in history.

    Deferred entries are outside Anthropic's cache key, so this is a determinism rule, not a cache rule.
    """
    tool_defs = model_request_parameters.tool_defs
    return [
        name
        for name in discovered_tool_names_in_order(messages)
        if name in tool_defs
        and tool_defs[name].defer_loading
        and tool_defs[name].with_native is None
        and model_request_parameters.visibility_of(name) == 'deferred'
    ]


def _append_revealed_tool_params(tools: list[BetaToolUnionParam], revealed_tool_order: list[str]) -> None:
    """Move revealed deferred definitions behind the request's stable tool segments."""
    by_name = {tool.get('name'): tool for tool in tools}
    revealed_names = [name for name in revealed_tool_order if name in by_name]
    if revealed_names:
        revealed_name_set = set(revealed_names)
        tools[:] = [tool for tool in tools if tool.get('name') not in revealed_name_set] + [
            by_name[name] for name in revealed_names
        ]


try:
    from anthropic import (
        NOT_GIVEN,
        APIConnectionError,
        APIStatusError,
        AsyncAnthropicBedrock,  # pyright: ignore[reportPrivateImportUsage]
        AsyncAnthropicBedrockMantle,  # pyright: ignore[reportPrivateImportUsage]
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,  # pyright: ignore[reportPrivateImportUsage]
        AsyncStream,
        Omit,
        omit as OMIT,
    )
    from anthropic.types.anthropic_beta_param import AnthropicBetaParam
    from anthropic.types.beta import (
        BetaAdvisorTool20260301Param,
        BetaAdvisorToolResultBlock,
        BetaAdvisorToolResultBlockParam,
        BetaBase64PDFSourceParam,
        BetaBashCodeExecutionToolResultBlock,
        BetaBashCodeExecutionToolResultBlockParam,
        BetaCacheControlEphemeralParam,
        BetaCitationsConfigParam,
        BetaCitationsDelta,
        BetaCodeExecutionTool20250825Param,
        BetaCodeExecutionTool20260120Param,
        BetaCodeExecutionToolResultBlock,
        BetaCodeExecutionToolResultBlockContent,
        BetaCodeExecutionToolResultBlockParam,
        BetaCodeExecutionToolResultBlockParamContentParam,
        BetaCompactionBlock,
        BetaCompactionBlockParam,
        BetaCompactionContentBlockDelta,
        BetaContainerParams,
        BetaContainerUploadBlockParam,
        BetaContentBlock,
        BetaContentBlockParam,
        BetaContextManagementConfigParam,
        BetaDirectCaller,
        BetaFileDocumentSourceParam,
        BetaFileImageSourceParam,
        BetaImageBlockParam,
        BetaInputJSONDelta,
        BetaJSONOutputFormatParam,
        BetaMCPToolResultBlock,
        BetaMCPToolUseBlock,
        BetaMCPToolUseBlockParam,
        BetaMemoryTool20250818Param,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaMessageParam,
        BetaMessageTokensCount,
        BetaMetadataParam,
        BetaOutputConfigParam,
        BetaPlainTextSourceParam,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaRedactedThinkingBlock,
        BetaRedactedThinkingBlockParam,
        BetaRequestDocumentBlockParam,
        BetaRequestMCPServerToolConfigurationParam,
        BetaRequestMCPServerURLDefinitionParam,
        BetaServerToolCaller,
        BetaServerToolCaller20260120,
        BetaServerToolUseBlock,
        BetaServerToolUseBlockParam,
        BetaSignatureDelta,
        BetaStopReason,
        BetaTextBlock,
        BetaTextBlockParam,
        BetaTextDelta,
        BetaTextEditorCodeExecutionToolResultBlock,
        BetaTextEditorCodeExecutionToolResultBlockParam,
        BetaThinkingBlock,
        BetaThinkingBlockParam,
        BetaThinkingConfigParam,
        BetaThinkingDelta,
        BetaTokenTaskBudgetParam,
        BetaToolChoiceParam,
        BetaToolParam,
        BetaToolReferenceBlockParam,
        BetaToolSearchToolBm25_20251119Param,
        BetaToolSearchToolRegex20251119Param,
        BetaToolSearchToolResultBlock,
        BetaToolSearchToolResultBlockParam,
        BetaToolSearchToolResultErrorParam,
        BetaToolSearchToolSearchResultBlockParam,
        BetaToolUnionParam,
        BetaToolUseBlock,
        BetaToolUseBlockParam,
        BetaUsage,
        BetaWebFetchTool20250910Param,
        BetaWebFetchTool20260209Param,
        BetaWebFetchToolResultBlock,
        BetaWebFetchToolResultBlockParam,
        BetaWebSearchTool20250305Param,
        BetaWebSearchTool20260209Param,
        BetaWebSearchToolResultBlock,
        BetaWebSearchToolResultBlockContent,
        BetaWebSearchToolResultBlockParam,
        BetaWebSearchToolResultBlockParamContentParam,
        beta_tool_result_block_param,
    )
    from anthropic.types.beta.beta_advisor_tool_result_block import (
        Content as AdvisorToolResultBlockContent,
    )
    from anthropic.types.beta.beta_advisor_tool_result_block_param import (
        Content as AdvisorToolResultBlockParamContent,
    )
    from anthropic.types.beta.beta_bash_code_execution_tool_result_block import (
        Content as BashCodeExecutionToolResultBlockContent,
    )
    from anthropic.types.beta.beta_bash_code_execution_tool_result_block_param import (
        Content as BashCodeExecutionToolResultBlockParamContent,
    )
    from anthropic.types.beta.beta_text_editor_code_execution_tool_result_block import (
        Content as TextEditorCodeExecutionToolResultBlockContent,
    )
    from anthropic.types.beta.beta_text_editor_code_execution_tool_result_block_param import (
        Content as TextEditorCodeExecutionToolResultBlockParamContent,
    )
    from anthropic.types.beta.beta_user_location_param import BetaUserLocationParam
    from anthropic.types.beta.beta_web_fetch_tool_result_block_param import (
        Content as WebFetchToolResultBlockParamContent,
    )
    from anthropic.types.model_param import ModelParam

except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error

# `AsyncAnthropicBedrockMantle` uses the Messages API and supports automatic prompt caching (unlike the
# legacy `AsyncAnthropicBedrock` InvokeModel API), so it's not in `_NON_AUTOMATIC_CACHING_CLIENTS`. Fast
# mode is not available on any Bedrock transport, so it goes in `_FAST_MODE_UNSUPPORTED_CLIENTS`.
_NON_AUTOMATIC_CACHING_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicVertex)
_FAST_MODE_UNSUPPORTED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
# The legacy Bedrock InvokeModel API (`AsyncAnthropicBedrock`) doesn't support the `bm25` tool-search
# variant — it 400s with `BM25 tool search is not supported on Bedrock. Use tool_search_tool_regex instead.`
# — so we default to `regex` there. The other transports (Vertex, Foundry, and the Messages-API-based
# `AsyncAnthropicBedrockMantle`) expose the full Messages API, including both tool-search variants, so they
# keep the `bm25` default.
_BM25_TOOL_SEARCH_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock,)
# Anthropic web-tool availability is client/platform-specific:
# * `AsyncAnthropicBedrock` is the legacy Amazon Bedrock InvokeModel client, where web search/fetch
#   are unavailable.
# * Vertex AI supports basic web search only.
# * Direct Anthropic API, Claude Platform on AWS (`AsyncAnthropicBedrockMantle`), and Microsoft
#   Foundry support dynamic-filtering web tools on supported model profiles.
_WEB_SEARCH_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock,)
_WEB_FETCH_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicVertex)
_WEB_TOOLS_20260209_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicVertex)
# The advisor tool is available on the direct Anthropic API and Claude Platform on AWS
# (`AsyncAnthropicBedrockMantle`) only — not on the legacy Bedrock InvokeModel client, Vertex, or
# Foundry. `AsyncAnthropicBedrockMantle` isn't a subclass of `AsyncAnthropicBedrock`, so the plain
# isinstance tuple keeps it supported.
_ADVISOR_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicVertex, AsyncAnthropicFoundry)
# Mid-conversation `{'role': 'system'}` messages are published as available on the Anthropic API,
# Amazon Bedrock and Google Cloud, which leaves Microsoft Foundry as the only transport that has to
# fall back to the `<system>`-tagged user rendering. Verified on Bedrock rather than assumed: with
# `us.anthropic.claude-opus-4-8` the entry is served and acted on. An earlier version of this tuple
# excluded Bedrock and Vertex on the strength of a Bedrock test that used `claude-sonnet-5`, a model
# that ignores the entry on *every* transport — which measured the model, not the transport.
_INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS = (AsyncAnthropicFoundry,)

_ANTHROPIC_SAMPLING_PARAMS = ('temperature', 'top_p', 'top_k')
_ANTHROPIC_TASK_BUDGETS_BETA = 'task-budgets-2026-03-13'
_ANTHROPIC_FILES_API_BETA = 'files-api-2025-04-14'
_ANTHROPIC_COMPACT_EDIT_TYPE = 'compact_20260112'


@contextmanager
def _map_api_errors(model_name: str, model_id_namespace: str = 'anthropic') -> Generator[None]:
    try:
        yield
    except APIStatusError as e:
        if (status_code := e.status_code) >= 400:
            body: object | None = e.body
            suggested_model_id = None
            if _utils.is_str_dict(body) and _utils.is_str_dict(error := body.get('error')):
                if error.get('type') == 'not_found_error' and error.get('message') == f'model: {model_name}':
                    suggested_model_id = _suggest_known_model_id_from_provider_error(model_id_namespace, model_name)
            raise ModelHTTPError(
                status_code=status_code,
                model_name=model_name,
                body=body,
                headers=dict(e.response.headers),
                suggested_model_id=suggested_model_id,
            ) from e
        raise ModelAPIError(model_name=model_name, message=e.message) from e  # pragma: lax no cover
    except APIConnectionError as e:
        raise ModelAPIError(model_name=model_name, message=e.message) from e


LatestAnthropicModelNames = ModelParam
"""Anthropic model names from the installed SDK."""

AnthropicModelName = LatestAnthropicModelNames
"""Possible Anthropic model names.

The installed Anthropic SDK exposes the current literal set and still allows arbitrary string model names.
See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
"""

DEPRECATED_ANTHROPIC_MODELS: frozenset[str] = frozenset(
    {
        # https://platform.claude.com/docs/en/about-claude/model-deprecations
        # Retired 2026-04-20
        'claude-3-haiku-20240307',
        # Retired 2026-06-15
        'claude-opus-4-0',
        'claude-opus-4-20250514',
        'claude-sonnet-4-0',
        'claude-sonnet-4-20250514',
    }
)
"""Models that have been retired by Anthropic but are still present in the SDK's type definitions."""

_AnthropicCodeExecutionToolName: TypeAlias = Literal[
    'code_execution', 'bash_code_execution', 'text_editor_code_execution'
]
_AnthropicCodeExecutionProviderDetailToolName: TypeAlias = Literal['bash_code_execution', 'text_editor_code_execution']
_ANTHROPIC_CODE_EXECUTION_TOOL_NAMES: tuple[_AnthropicCodeExecutionToolName, ...] = (
    'code_execution',
    'bash_code_execution',
    'text_editor_code_execution',
)
_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL = 'anthropic_tool_name'
# See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#what-can-be-cached
# `tool_addition` accepted and honored verified live on `claude-opus-4-8`: a boundary on the block
# writes and reads back the full prefix. It can end a mid-conversation `system` entry, where a
# terminal `CachePoint`'s boundary lands on the entry's final block.
_ANTHROPIC_CACHEABLE_PARAM_TYPES = frozenset(
    {'text', 'tool_use', 'server_tool_use', 'image', 'tool_result', 'document', 'tool_addition'}
)
_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL = 'anthropic_caller'

AnthropicTaskBudget: TypeAlias = BetaTokenTaskBudgetParam
"""Anthropic task budget payload for `output_config.task_budget`."""


class AnthropicModelSettings(ModelSettings, total=False):
    """Settings used for an Anthropic model request."""

    # ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    anthropic_metadata: BetaMetadataParam
    """An object describing metadata about the request.

    Contains `user_id`, an external identifier for the user who is associated with the request.
    """

    anthropic_thinking: BetaThinkingConfigParam
    """Determine whether the model should generate a thinking block.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking) for more information.
    """

    anthropic_cache_tool_definitions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last tool definition.

    When enabled, the last tool in the `tools` array will have `cache_control` set,
    allowing Anthropic to cache tool definitions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """

    anthropic_service_tier: Literal['auto', 'standard_only']
    """The service tier to use for the model request.

    See https://docs.anthropic.com/en/docs/build-with-claude/latency-and-throughput for more information.
    """

    anthropic_cache_instructions: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last system prompt block.

    When enabled, the last system prompt will have `cache_control` set,
    allowing Anthropic to cache system instructions and reduce costs.
    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """

    anthropic_cache_messages: bool | Literal['5m', '1h']
    """Whether to add `cache_control` to the last message content block.

    This is an alternative to `anthropic_cache` for Anthropic-compatible gateways and
    proxies that accept the Anthropic message format but don't support the top-level
    automatic caching parameter.

    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.
    Cannot be combined with `anthropic_cache`.
    """

    anthropic_cache: bool | Literal['5m', '1h']
    """Enable prompt caching for multi-turn conversations.

    Passes a top-level `cache_control` parameter so the server automatically applies a
    cache breakpoint to the last cacheable block and moves it forward as conversations grow.

    On Bedrock and Vertex, automatic caching is not yet supported, so this falls back to
    per-block caching on the last user message. If the last content block already has
    `cache_control` from an explicit `CachePoint`, it is preserved.

    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.

    This can be combined with explicit cache breakpoints (`anthropic_cache_instructions`,
    `anthropic_cache_tool_definitions`, `CachePoint`). The automatic breakpoint counts as
    1 of Anthropic's 4 cache point slots; we automatically trim excess explicit breakpoints.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching
    for more information.
    """

    anthropic_effort: AnthropicEffort | None
    """The effort level for the model to use when generating a response.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/effort) for more information.
    """

    anthropic_task_budget: AnthropicTaskBudget
    """Task budget configuration for Anthropic beta requests.

    Maps to `output_config.task_budget`. Supported models are gated by the
    [`anthropic_supports_task_budgets`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_supports_task_budgets]
    profile flag, and Pydantic AI automatically enables Anthropic's required task-budget beta when
    this setting is present.

    Omit `remaining` unless you are intentionally carrying a budget across compaction
    or other rewritten context.
    """

    anthropic_container: BetaContainerParams | str | Literal[False]
    """Container configuration for multi-turn conversations.

    By default, if previous messages contain a container_id (from a prior response),
    it will be reused automatically. An id recovered that way is the only container
    Pydantic AI may replace on its own: if a request carrying both that id and
    `CodeExecutionTool` uploads comes back `500`, the id is dropped and the request is
    sent once more so the upload lands in a fresh container. Nothing else is replaced —
    not a container you set here, and not the id used to reconnect a paused turn — so a
    request pinning a container the API will not accept raises instead.

    Set to `False` to force a fresh container (ignore any `container_id` from history).
    Set to a container id string (e.g. `'container_xxx'`) to explicitly reuse a container,
    or to a `BetaContainerParams` dict (e.g. `{'skills': [...]}` or
    `{'id': 'container_xxx', 'skills': [...]}`) when passing Skills to the Anthropic
    Skills beta.
    """

    anthropic_code_execution_tool_version: AnthropicCodeExecutionToolVersion | Literal['auto']
    """Which Anthropic code execution tool version to send for `CodeExecutionTool`.

    Defaults to `'auto'`, which uses the default version from the model profile:
    `'20260120'` for Sonnet 4.5+ and Opus 4.5+, otherwise `'20250825'`.
    Set a concrete version to force that tool version; a `UserError` is raised if
    the selected model profile does not support that version.
    """

    anthropic_eager_input_streaming: bool
    """Whether to enable eager input streaming on tool definitions.

    When enabled, all tool definitions will have `eager_input_streaming` set to `True`,
    allowing Anthropic to stream tool call arguments incrementally instead of buffering
    the entire JSON before streaming. This reduces latency for tool calls with large inputs.
    See https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming for more information.
    """

    anthropic_betas: list[AnthropicBetaParam]
    """List of Anthropic beta features to enable for API requests.

    Each item can be a known beta name (e.g. 'interleaved-thinking-2025-05-14') or a custom string.
    Merged with auto-added betas (e.g. builtin tools) and any betas from
    extra_headers['anthropic-beta']. See the Anthropic docs for available beta features.
    """

    anthropic_speed: Literal['standard', 'fast']
    """The inference speed mode for this request.

    `'fast'` enables high output-tokens-per-second inference for supported models (currently Claude Opus 4.6, 4.7, 4.8, and 5).
    On unsupported models or clients, `anthropic_speed='fast'` is ignored with a `UserWarning`.
    Fast mode is a research preview and only available on the direct Anthropic API (not Bedrock, Vertex, or Foundry);
    see [the Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/fast-mode) for details.
    Note: switching between `'fast'` and `'standard'` invalidates the prompt cache.
    """

    anthropic_context_management: BetaContextManagementConfigParam
    """Context management configuration for automatic compaction.

    When configured, Anthropic will automatically compact older context when the
    input token count exceeds the configured threshold. The compaction produces
    a summary that replaces the compacted messages.

    See [the Anthropic docs](https://docs.anthropic.com/en/docs/build-with-claude/compaction) for more details.
    """


def _build_extra_body(model_settings: AnthropicModelSettings) -> object | None:
    """Merge the sampling settings into `extra_body`, which is how they reach the API now.

    `anthropic>=1` dropped `temperature`/`top_p`/`top_k` from the `messages.create()` signature, and
    passing one is a `TypeError`. The API still takes them, so they ride in `extra_body` — the route
    the SDK's own migration guide names — rather than being silently dropped. Models that reject them
    never get here: `_drop_unsupported_sampling_settings` has already removed them.

    An explicit `extra_body` entry wins over the setting of the same name, preserving the precedence
    the SDK gave it while the parameters were still named arguments.
    """
    sampling = {
        setting: value for setting in _ANTHROPIC_SAMPLING_PARAMS if (value := model_settings.get(setting)) is not None
    }
    extra_body = model_settings.get('extra_body')
    if not sampling:
        return extra_body
    if is_str_dict(extra_body):
        return {**sampling, **extra_body}
    return sampling


def _resolve_anthropic_service_tier(
    model_settings: AnthropicModelSettings,
) -> Literal['auto', 'standard_only'] | Omit:
    """Resolve the value to send as `service_tier` on the Anthropic request.

    Per-provider [`anthropic_service_tier`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_service_tier]
    wins; otherwise the top-level [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] is mapped
    (`'default'` → `'standard_only'`, `'auto'` → `'auto'`). `'flex'`/`'priority'` are dropped as Anthropic
    does not expose them via this field.
    """
    if anthropic_tier := model_settings.get('anthropic_service_tier'):
        return anthropic_tier
    unified = model_settings.get('service_tier')
    if unified == 'auto':
        return 'auto'
    if unified == 'default':
        return 'standard_only'
    return OMIT


@dataclass(init=False)
class AnthropicModel(Model[AsyncAnthropicClient]):
    """A model that uses the Anthropic API.

    Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    supported_tool_deferral_modes = frozenset({'standalone'})
    supported_tool_addition_modes = frozenset({'by_reference'})
    # The API ignores (and doesn't bill) blocks before a compaction block, so the trim only saves
    # request size here — but it also keeps a `tool_result` whose `tool_use` was trimmed away from
    # 400ing, since validation runs even on ignored content.
    #
    # Both declarations match the inherited defaults, restated because they're what makes this
    # adapter's trim correct rather than an accident: the summary is plaintext message content, so
    # there is no encrypted blob to require, and the standing prompt travels in the top-level
    # `system` parameter, rebuilt from the opening `SystemPromptPart`s on every request, so the trim
    # has to re-insert them.
    compaction_requires_encrypted_content = False
    compaction_retains_standing_prompt = False

    _model_name: AnthropicModelName = field(repr=False)
    _provider: Provider[AsyncAnthropicClient] = field(repr=False)

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['anthropic', 'gateway'] | Provider[AsyncAnthropicClient] = 'anthropic',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Anthropic model.

        Args:
            model_name: The name of the Anthropic model to use. List of model names available
                [here](https://docs.anthropic.com/en/docs/about-claude/models).
            provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
                instance of `Provider[AsyncAnthropicClient]`. Defaults to 'anthropic'.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
                The default 'anthropic' provider will use the default `..profiles.anthropic.anthropic_model_profile`.
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/anthropic' if provider == 'gateway' else provider)
        self._provider = provider

        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncAnthropicClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def model_name(self) -> AnthropicModelName:
        """The model name."""
        return self._model_name

    def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        """Resolve the longest retention requested by active Anthropic cache settings."""
        settings = merge_model_settings(self.settings, model_settings) or {}
        return self._max_prompt_cache_retention(
            settings.get('anthropic_cache'),
            settings.get('anthropic_cache_instructions'),
            settings.get('anthropic_cache_tool_definitions'),
            settings.get('anthropic_cache_messages'),
        )

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @cached_property
    def profile(self) -> AnthropicModelProfile:
        """The model profile.

        Anthropic web-tool availability depends on both model support and the client/platform, so the
        profile's `supported_native_tools` and `anthropic_supports_dynamic_filtering` are narrowed here
        for clients that don't support them (e.g. Bedrock, Vertex). `supports_inline_system_prompts` is
        narrowed the same way, and for the same reason: serving a `{'role': 'system'}` entry is a fact
        about the transport as much as about the model.
        """
        _profile = super().profile
        provider = self.provider
        if provider is None:
            return cast(AnthropicModelProfile, _profile)
        client = provider.client
        supported_native_tools = _profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
        if isinstance(client, _WEB_SEARCH_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {WebSearchTool}
        if isinstance(client, _WEB_FETCH_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {WebFetchTool}
        if isinstance(client, _ADVISOR_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {AdvisorTool}
        supports_dynamic_filtering = _profile.get('anthropic_supports_dynamic_filtering', False) and not isinstance(
            client, _WEB_TOOLS_20260209_UNSUPPORTED_CLIENTS
        )
        _profile = merge_profile(
            _profile,
            AnthropicModelProfile(
                supported_native_tools=supported_native_tools,
                anthropic_supports_dynamic_filtering=supports_dynamic_filtering,
                # Narrowed rather than handled in `_map_message` so `Model.prepare_messages` stays the
                # only place that knows the `<system>`-tagged fallback: where this is `False`, the
                # mid-conversation parts are rewritten before the adapter ever sees them.
                supports_inline_system_prompts=_profile.get('supports_inline_system_prompts', False)
                and not isinstance(client, _INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS),
            ),
        )
        return cast(AnthropicModelProfile, _profile)

    @property
    def tool_addition_mode(self) -> Literal['by_reference'] | None:
        """The effective addition mode, narrowed for transports without inline system messages."""
        mode = super().tool_addition_mode
        provider = self.provider
        if provider is not None and isinstance(provider.client, _INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS):
            return None
        return mode if mode == 'by_reference' else None

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """The set of builtin tool types this model can handle."""
        return frozenset(
            {WebSearchTool, CodeExecutionTool, WebFetchTool, MemoryTool, MCPServerTool, ToolSearchTool, AdvisorTool}
        )

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
        model_settings = cast(AnthropicModelSettings, model_settings or {})
        try:
            response = await self._messages_create(messages, False, model_settings, model_request_parameters)
            return self._process_response(response, model_request_parameters, model_settings)
        except ValueError as e:
            if 'Streaming is required' in str(e):
                # Anthropic SDK requires streaming for high max_tokens; fall back transparently
                # https://github.com/anthropics/anthropic-sdk-python/blob/49d639a671cb0ac30c767e8e1e68fdd5925205d5/src/anthropic/_base_client.py#L726
                stream = await self._messages_create(messages, True, model_settings, model_request_parameters)
                async with stream:
                    streamed_response = await self._process_streamed_response(
                        stream, model_request_parameters, model_settings
                    )
                    async for _ in streamed_response:
                        pass
                    return streamed_response.get()
            raise  # pragma: no cover

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )

        response = await self._messages_count_tokens(
            messages, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
        )

        return usage.RequestUsage(input_tokens=response.input_tokens)

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
        model_settings = cast(AnthropicModelSettings, model_settings or {})
        response = await self._messages_create(messages, True, model_settings, model_request_parameters)
        async with response:
            yield await self._process_streamed_response(response, model_request_parameters, model_settings)

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        profile = self.profile
        merged = merge_model_settings(self.settings, model_settings) or {}

        if (
            profile.get('anthropic_disallows_budget_thinking', False)
            and (anthropic_thinking := merged.get('anthropic_thinking'))
            and anthropic_thinking.get('type') == 'enabled'
        ):
            raise UserError(
                f'{self.model_name!r} does not support '
                "`anthropic_thinking={'type': 'enabled', 'budget_tokens': ...}`. "
                "Use `anthropic_thinking={'type': 'adaptive'}` and `anthropic_effort=...` instead."
            )

        supports_adaptive_thinking = profile.get('anthropic_supports_adaptive_thinking', False)
        supports_forced_tool_choice = profile.get('anthropic_supports_forced_tool_choice', True)
        thinking_type = _effective_thinking_type(
            merged.get('anthropic_thinking'),
            merged.get('thinking'),
            supports_adaptive_thinking=supports_adaptive_thinking,
        )
        # Tool Output resolves to a forced `tool_choice`, which Anthropic rejects alongside extended
        # thinking. Adaptive thinking is accepted — but only on models that accept forcing at all;
        # on the rest, Tool Output could only degrade to a soft `tool_choice='auto'` the model may
        # ignore, so they keep switching away from it whenever a thinking setting is configured.
        thinking_blocks_output_tools = thinking_type == 'enabled' or (
            thinking_type == 'adaptive' and not supports_forced_tool_choice
        )

        if model_request_parameters.output_tools and thinking_blocks_output_tools:
            supports_json_schema_output = profile.get('supports_json_schema_output', False)
            model_request_parameters = model_request_parameters.with_default_output_mode(
                'native' if supports_json_schema_output else 'prompted'
            )
            if (
                model_request_parameters.output_mode == 'tool' and not model_request_parameters.allow_text_output
            ):  # pragma: no branch
                # This would result in `tool_choice=required`, which isn't available here.
                suggested_output_type = 'NativeOutput' if supports_json_schema_output else 'PromptedOutput'
                remedy = f'Use `output_type={suggested_output_type}(...)` instead.'
                if thinking_type == 'adaptive':
                    raise UserError(
                        f'{self.model_name!r} does not support output tools when a thinking setting is '
                        f'configured, because it rejects the forced tool choice they require. {remedy}'
                    )
                if supports_adaptive_thinking:
                    remedy += " Alternatively, `anthropic_thinking={'type': 'adaptive'}` supports output tools."
                raise UserError(
                    f'Anthropic does not support extended thinking and output tools at the same time. {remedy}'
                )

        # Resolve 'auto' to the profile default here (a no-op if already resolved above) so the
        # strict-forcing check below also applies when native mode is reached via the profile default
        # rather than an explicit `NativeOutput(...)`; `super().prepare_request()` would otherwise only
        # resolve it after `customize_request_parameters()` has already transformed the schema.
        model_request_parameters = model_request_parameters.with_default_output_mode(
            self.profile.get('default_structured_output_mode', 'tool')
        )

        if model_request_parameters.output_mode == 'native':
            assert model_request_parameters.output_object is not None
            if model_request_parameters.output_object.strict is False:
                raise UserError(
                    'Setting `strict=False` on `output_type=NativeOutput(...)` is not allowed for Anthropic models.'
                )
            model_request_parameters = replace(
                model_request_parameters, output_object=replace(model_request_parameters.output_object, strict=True)
            )

        prepared_settings, model_request_parameters = super().prepare_request(model_settings, model_request_parameters)
        if profile.get('anthropic_disallows_sampling_settings', False) and prepared_settings:
            filtered: ModelSettings = {**prepared_settings}
            self._drop_unsupported_sampling_settings(filtered)
            prepared_settings = filtered or None
        return prepared_settings, model_request_parameters

    def _drop_unsupported_sampling_settings(self, model_settings: ModelSettings) -> None:
        dropped = {setting for setting in _ANTHROPIC_SAMPLING_PARAMS if setting in model_settings}
        extra_body = model_settings.get('extra_body')
        if is_str_dict(extra_body):
            dropped |= {setting for setting in _ANTHROPIC_SAMPLING_PARAMS if setting in extra_body}
            model_settings['extra_body'] = {
                key: value for key, value in extra_body.items() if key not in _ANTHROPIC_SAMPLING_PARAMS
            }

        for setting in _ANTHROPIC_SAMPLING_PARAMS:
            model_settings.pop(setting, None)

        if dropped:
            ordered = [setting for setting in _ANTHROPIC_SAMPLING_PARAMS if setting in dropped]
            warnings.warn(
                f'Sampling parameters {ordered} are not supported by {self.model_name!r}. These settings will be ignored.',
                UserWarning,
                stacklevel=2,
            )

    def _translate_thinking(
        self,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaThinkingConfigParam:
        """Get the thinking parameter, falling back to unified thinking."""
        if anthropic_thinking := model_settings.get('anthropic_thinking'):
            return anthropic_thinking
        thinking = model_request_parameters.thinking
        if thinking is None or thinking is False:
            return OMIT  # type: ignore[return-value]
        profile = self.profile
        if profile.get('anthropic_supports_adaptive_thinking', False):
            return {'type': 'adaptive'}
        return {'type': 'enabled', 'budget_tokens': ANTHROPIC_THINKING_BUDGET_MAP[thinking]}

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[BetaRawMessageStreamEvent]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
        """Calls the Anthropic API to create a message.

        This is the last step before sending the request to the API.
        Most preprocessing has happened in `prepare_request()`.
        """
        # Native search remains in the stable segment when a reveal lands. Revealed non-corpus deferred
        # entries are then appended in history order; Anthropic excludes them from its cache key.
        revealed_tool_order = _revealed_deferred_tool_order(messages, model_request_parameters)
        tools, tool_choice = self._prepare_tools_and_tool_choice(model_settings, model_request_parameters)
        tools, mcp_servers, native_tool_betas = self._add_native_tools(tools, model_request_parameters, model_settings)
        _append_revealed_tool_params(tools, revealed_tool_order)
        if tools and all(tool.get('defer_loading') is True for tool in tools):
            raise UserError(
                'Anthropic rejects a request whose wire tools all have `defer_loading=True`. '
                'Make at least one tool visible, or add a tool-search surface or capability catalog.'
            )

        auto_cache_control, resolved_cache_ttl = self._build_automatic_cache_control(model_settings)
        system_prompt, anthropic_messages = await self._map_message(messages, model_request_parameters, model_settings)
        self._apply_per_block_caching_fallback(resolved_cache_ttl, anthropic_messages)
        self._apply_explicit_message_caching(model_settings, anthropic_messages)
        self._limit_cache_points(
            system_prompt, anthropic_messages, tools, automatic_caching=auto_cache_control is not None
        )
        output_config = self._build_output_config(model_request_parameters, model_settings)
        anthropic_profile = self.profile
        betas, extra_headers = self._get_betas_and_extra_headers(
            model_settings, anthropic_profile, messages, model_request_parameters
        )
        betas.update(native_tool_betas)
        context_management = self._add_compaction_params(messages, betas, model_settings)
        self._validate_task_budget_vs_context_management(model_settings, context_management)
        container, container_from_history = self._get_container(messages, model_settings)

        async def create(
            container_param: BetaContainerParams | str | None,
        ) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
            return await self.client.beta.messages.create(
                max_tokens=model_settings.get('max_tokens', 4096),
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                output_config=output_config or OMIT,
                betas=sorted(betas) or OMIT,
                stream=stream,
                cache_control=auto_cache_control or OMIT,
                thinking=self._translate_thinking(model_settings, model_request_parameters),
                stop_sequences=model_settings.get('stop_sequences', OMIT),
                timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                metadata=model_settings.get('anthropic_metadata', OMIT),
                context_management=context_management or OMIT,
                container=container_param or OMIT,
                service_tier=_resolve_anthropic_service_tier(model_settings),
                speed=self._effective_speed(model_settings, anthropic_profile),
                extra_headers=extra_headers,
                extra_body=_build_extra_body(model_settings),
            )

        with _map_api_errors(self.model_name, self._provider.model_id_namespace):
            try:
                return await create(container)
            except APIStatusError as e:
                if (
                    e.status_code != 500
                    or not container_from_history
                    or not any(
                        is_str_dict(block) and block['type'] == 'container_upload'
                        for message in anthropic_messages
                        for block in message['content']
                    )
                ):
                    raise
                return await create(None)

    @staticmethod
    def _add_compaction_params(
        messages: list[ModelMessage],
        betas: set[str],
        model_settings: AnthropicModelSettings,
    ) -> BetaContextManagementConfigParam | None:
        """Add compaction beta and default context_management when messages contain CompactionParts.

        This ensures CompactionParts can be round-tripped even without AnthropicCompaction active.
        """
        has_compaction_parts = any(
            isinstance(part, CompactionPart) for msg in messages if isinstance(msg, ModelResponse) for part in msg.parts
        )
        if has_compaction_parts:
            betas.add('compact-2026-01-12')
        context_management = model_settings.get('anthropic_context_management')
        if has_compaction_parts and context_management is None:
            context_management = cast(
                BetaContextManagementConfigParam, {'edits': [{'type': _ANTHROPIC_COMPACT_EDIT_TYPE}]}
            )
        return context_management

    def _get_betas_and_extra_headers(
        self,
        model_settings: AnthropicModelSettings,
        anthropic_profile: AnthropicModelProfile,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[set[str], dict[str, str]]:
        """Prepare beta features list and extra headers for API request.

        Handles merging custom `anthropic-beta` header from `extra_headers` into betas set,
        auto-attaching the Files API beta when messages contain an Anthropic `UploadedFile`,
        and ensuring `User-Agent` is set.
        """
        extra_headers = dict(model_settings.get('extra_headers', {}))
        extra_headers.setdefault('User-Agent', get_user_agent())

        betas: set[str] = set()

        if model_settings.get('anthropic_context_management'):
            betas.add('compact-2026-01-12')
        if self._get_task_budget(model_settings) is not None:
            betas.add(_ANTHROPIC_TASK_BUDGETS_BETA)

        if model_settings.get('anthropic_speed') == 'fast' and self._client_supports_fast_speed(anthropic_profile):
            betas.add('fast-mode-2026-02-01')

        if betas_from_setting := model_settings.get('anthropic_betas'):
            betas.update(str(b) for b in betas_from_setting)

        if beta_header := extra_headers.pop('anthropic-beta', None):
            betas.update({stripped_beta for beta in beta_header.split(',') if (stripped_beta := beta.strip())})

        if self._messages_use_anthropic_uploaded_file(messages):
            betas.add(_ANTHROPIC_FILES_API_BETA)
        # `prepare_messages` converts local search returns into availability deltas before this
        # scan. Native search returns replay under the tool-search beta, so delta parts are the
        # complete trigger set for the mid-conversation-tool-changes beta.
        # Function tools only, to stay co-extensive with the `tool_addition` renderer: output
        # tools are never revealed, so a (forged or replayed) delta naming one renders nothing
        # and must not add the beta either.
        function_tool_defs = {tool_def.name: tool_def for tool_def in model_request_parameters.function_tools}
        if self.tool_addition_mode == 'by_reference' and any(
            name in model_request_parameters.declared_tool_defs
            and name in function_tool_defs
            and model_request_parameters.visibility_of(name) != 'visible'
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolAvailabilityDeltaPart)
            for name in part.tools_added
        ):
            betas.add('mid-conversation-tool-changes-2026-07-01')

        return betas, extra_headers

    def _messages_use_anthropic_uploaded_file(self, messages: list[ModelMessage]) -> bool:
        """Whether any normalized message contains an Anthropic-hosted `UploadedFile`.

        Used to gate auto-attachment of the `files-api-2025-04-14` beta header.
        Mirrors the per-item `provider_name == self.system` check the request
        mappers (`_map_user_prompt`, `_map_message`'s `ToolReturnPart` branch)
        already perform — so the beta is added exactly when the wire shape
        requires it. `UploadedFile`s for other providers are intentionally
        ignored here; they will raise the existing `UserError` later in the
        request-mapping path.
        """
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    for item in part.content:
                        if isinstance(item, UploadedFile) and item.provider_name == self.system:
                            return True
                elif isinstance(part, ToolReturnPart):
                    for item in part.content_items(mode='raw'):
                        if isinstance(item, UploadedFile) and item.provider_name == self.system:
                            return True
        return False

    def _effective_speed(
        self, model_settings: AnthropicModelSettings, anthropic_profile: AnthropicModelProfile
    ) -> Literal['standard', 'fast'] | Omit:
        """Speed to send to the API, or OMIT when the model or client does not support the `speed` parameter."""
        s = model_settings.get('anthropic_speed')
        if s in ('standard', 'fast') and self._client_supports_fast_speed(anthropic_profile):
            return s
        if s == 'fast':
            warnings.warn(
                f"anthropic_speed='fast' is not supported by {self.model_name} on this client; the setting will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return OMIT

    def _client_supports_fast_speed(self, anthropic_profile: AnthropicModelProfile) -> bool:
        """Fast mode is only available on the direct Anthropic API (not Bedrock, Vertex, or Foundry)."""
        return anthropic_profile.get('anthropic_supports_fast_speed', False) and not isinstance(
            self.client, _FAST_MODE_UNSUPPORTED_CLIENTS
        )

    def _get_container(
        self, messages: list[ModelMessage], model_settings: AnthropicModelSettings
    ) -> tuple[BetaContainerParams | str | None, bool]:
        """Resolve the `container` request parameter, and whether we resolved it from history.

        The second element is `True` only for an id recovered by the history scan below. That is the
        one container the caller never chose, so it is the only one we may drop and re-resolve when
        the API rejects it. A user-set `anthropic_container` and a `pause_turn` reconnect id are
        both deliberate and stay on the wire even when they fail.

        The Anthropic SDK types `container` as `BetaContainerParams | str`, and the
        live API accepts both forms *except* for one specific shape: a dict carrying
        only `id` and nothing else, which is rejected with
        `container: Input should be a valid string`. `{"skills": [...]}`,
        `{"id": x, "skills": [...]}`, and the raw `"x"` string all work — only
        `{"id": x}` alone is broken server-side.

        So when the user passes that only-broken shape, we transparently unwrap it to
        the string the server wants. Every other shape is passed through untouched so
        the Skills path (`{"skills": ...}` / `{"id": ..., "skills": ...}`) keeps
        working. History-based reuse is always sent as the raw id string since we
        never have skills to attach there.
        """
        if (container := model_settings.get('anthropic_container')) is not None:
            if container is False:
                return None, False
            if isinstance(container, dict) and set(container) == {'id'} and (cid := container.get('id')):
                return cid, False
            return container, False
        # On pause_turn continuation, pass just the container ID string to reconnect.
        # Re-passing BetaContainerParams triggers a prefill rejection on some models
        # (e.g. Sonnet 4-6) even though plain string ID works fine.
        if messages and isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended':
            if messages[-1].provider_details:
                return messages[-1].provider_details.get('container_id'), False
            return None, False

        for m in reversed(messages):
            if isinstance(m, ModelResponse) and m.provider_name == self.system and m.provider_details:
                if cid := m.provider_details.get('container_id'):
                    return cid, True
        return None, False

    async def _messages_count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessageTokensCount:
        # Anthropic docs: https://platform.claude.com/docs/en/api/messages-count-tokens
        # The `count_tokens` endpoint rejects server-side tools (`web_search`, `code_execution`,
        # `web_fetch`, `tool_search`) and the `mcp_servers` param with a 400, so restrict the wire
        # `tools` list to client-side tools like `MemoryTool`, which are accepted and contribute
        # real tokens. This undercounts the prompt by the server tools' definitions, but it's the
        # only way to get a count until Anthropic supports them.
        # TODO: Remove this workaround if Anthropic starts accepting server tools on `count_tokens`.
        # Recheck: POST /v1/messages/count_tokens with a web_search tool. Docs now list those tools: https://platform.claude.com/docs/en/api/messages/count_tokens
        count_tokens_parameters = replace(
            model_request_parameters,
            native_tools=[tool for tool in model_request_parameters.native_tools if isinstance(tool, MemoryTool)],
        )
        # Params that keep `ToolSearchTool` but drop `AdvisorTool`. Keeping tool search means the
        # count describes the request we'd really send, deferred tools and all: `function_tools`
        # aren't stripped here, so they keep their `defer_loading` and the tool-search replay still
        # renders the same `tool_reference` wire shape as `/v1/messages`. The endpoint honors the flag
        # rather than ignoring it — one deferred 30-field tool counts 440 tokens with it and 1761
        # without on `claude-opus-4-8` — so this is the difference between counting the prompt and
        # counting the hidden schemas too.
        # Dropping advisor makes `advisor_active` False so its call/result history blocks are stripped
        # during replay — the advisor tool is a server tool that `count_tokens` rejects (and that
        # `_add_native_tools` keeps off the wire below), and replaying advisor blocks without the tool
        # definition would 400.
        map_parameters = replace(
            model_request_parameters,
            native_tools=[tool for tool in model_request_parameters.native_tools if not isinstance(tool, AdvisorTool)],
        )

        # standalone function to make it easier to override
        revealed_tool_order = _revealed_deferred_tool_order(messages, map_parameters)
        tools, tool_choice = self._prepare_tools_and_tool_choice(model_settings, map_parameters)
        # `count_tokens_parameters` here, not `map_parameters`: the server-side tool definitions are
        # what the endpoint rejects, so they're the one thing that has to differ from the real request.
        tools, mcp_servers, native_tool_betas = self._add_native_tools(tools, count_tokens_parameters, model_settings)
        _append_revealed_tool_params(tools, revealed_tool_order)

        auto_cache_control, resolved_cache_ttl = self._build_automatic_cache_control(model_settings)
        system_prompt, anthropic_messages = await self._map_message(messages, map_parameters, model_settings)
        self._apply_per_block_caching_fallback(resolved_cache_ttl, anthropic_messages)
        self._apply_explicit_message_caching(model_settings, anthropic_messages)
        self._limit_cache_points(
            system_prompt, anthropic_messages, tools, automatic_caching=auto_cache_control is not None
        )
        output_config = self._build_output_config(model_request_parameters, model_settings)
        anthropic_profile = self.profile
        betas, extra_headers = self._get_betas_and_extra_headers(
            model_settings, anthropic_profile, messages, model_request_parameters
        )
        betas.update(native_tool_betas)
        context_management = self._add_compaction_params(messages, betas, model_settings)
        self._validate_task_budget_vs_context_management(model_settings, context_management)
        if isinstance(self.client, AsyncAnthropicBedrock):
            from ._anthropic_bedrock_count_tokens import count_tokens_via_bedrock

            with _map_api_errors(self.model_name, self._provider.model_id_namespace):
                return await count_tokens_via_bedrock(
                    self.client,
                    self._model_name,
                    system=system_prompt or OMIT,
                    messages=anthropic_messages,
                    max_tokens=model_settings.get('max_tokens', 4096),
                    tools=tools or OMIT,
                    tool_choice=tool_choice or OMIT,
                    mcp_servers=mcp_servers or OMIT,
                    betas=sorted(betas) or OMIT,
                    output_config=output_config or OMIT,
                    cache_control=auto_cache_control or OMIT,
                    thinking=self._translate_thinking(model_settings, model_request_parameters),
                    context_management=context_management or OMIT,
                    timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                    speed=self._effective_speed(model_settings, anthropic_profile),
                    extra_headers=extra_headers,
                    extra_body=model_settings.get('extra_body'),
                )

        with _map_api_errors(self.model_name, self._provider.model_id_namespace):
            return await self.client.beta.messages.count_tokens(
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                betas=sorted(betas) or OMIT,
                output_config=output_config or OMIT,
                cache_control=auto_cache_control or OMIT,
                thinking=self._translate_thinking(model_settings, model_request_parameters),
                context_management=context_management or OMIT,
                timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                speed=self._effective_speed(model_settings, anthropic_profile),
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
            )

    def _process_response(  # noqa: C901
        self,
        response: BetaMessage,
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        items: list[ModelResponsePart] = []
        builtin_tool_calls: dict[str, NativeToolCallPart] = {}
        enabled_server_tool_names = self._get_enabled_server_tool_names(model_request_parameters, model_settings)
        server_tool_result_ids = {
            item.tool_use_id
            for item in response.content
            if isinstance(
                item,
                BetaWebSearchToolResultBlock
                | BetaWebFetchToolResultBlock
                | BetaCodeExecutionToolResultBlock
                | BetaBashCodeExecutionToolResultBlock
                | BetaTextEditorCodeExecutionToolResultBlock
                | BetaToolSearchToolResultBlock
                | BetaAdvisorToolResultBlock,
            )
        }
        for item in response.content:
            if isinstance(item, BetaTextBlock):
                items.append(TextPart(content=item.text))
            elif isinstance(item, BetaServerToolUseBlock):
                if item.name not in enabled_server_tool_names and item.id not in server_tool_result_ids:
                    continue
                call_part = _map_server_tool_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaWebSearchToolResultBlock):
                items.append(_map_web_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaToolSearchToolResultBlock):
                items.append(_map_tool_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaCodeExecutionToolResultBlock):
                items.append(_map_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaBashCodeExecutionToolResultBlock):
                items.append(_map_bash_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaTextEditorCodeExecutionToolResultBlock):
                items.append(_map_text_editor_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaWebFetchToolResultBlock):
                items.append(_map_web_fetch_tool_result_block(item, self.system))
            elif isinstance(item, BetaAdvisorToolResultBlock):
                items.append(_map_advisor_tool_result_block(item, self.system))
            elif isinstance(item, BetaRedactedThinkingBlock):
                items.append(
                    ThinkingPart(id='redacted_thinking', content='', signature=item.data, provider_name=self.system)
                )
            elif isinstance(item, BetaThinkingBlock):
                items.append(ThinkingPart(content=item.thinking, signature=item.signature, provider_name=self.system))
            elif isinstance(item, BetaMCPToolUseBlock):
                call_part = _map_mcp_server_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaMCPToolResultBlock):
                call_part = builtin_tool_calls.get(item.tool_use_id)
                items.append(_map_mcp_server_result_block(item, call_part, self.system))
            elif isinstance(item, BetaCompactionBlock):
                items.append(
                    CompactionPart(
                        content=item.content,
                        provider_name=self.system,
                        provider_details=_compaction_provider_details(item.encrypted_content),
                    )
                )
            else:
                assert isinstance(item, BetaToolUseBlock), f'unexpected item type {type(item)}'
                items.append(
                    ToolCallPart(
                        tool_name=item.name,
                        args=cast(dict[str, Any], item.input),
                        tool_call_id=item.id,
                    )
                )

        finish_reason: FinishReason | None = None
        provider_details: dict[str, Any] | None = None
        if raw_finish_reason := response.stop_reason:  # pragma: no branch
            provider_details = {'finish_reason': raw_finish_reason}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
        if response.stop_details is not None:
            provider_details = provider_details or {}
            if response.stop_details.explanation is not None:
                provider_details['refusal'] = response.stop_details.explanation
            if response.stop_details.category is not None:
                provider_details['refusal_category'] = response.stop_details.category
        if response.container:
            provider_details = provider_details or {}
            provider_details['container_id'] = response.container.id

        return ModelResponse(
            parts=items,
            usage=_map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            finish_reason=finish_reason,
            state='suspended' if response.stop_reason == 'pause_turn' else 'complete',
            provider_details=provider_details,
        )

    def _get_enabled_server_tool_names(
        self, model_request_parameters: ModelRequestParameters, model_settings: AnthropicModelSettings
    ) -> frozenset[str]:
        """Wire names that may legitimately appear in a `server_tool_use` block for this request.

        Derived from the same `_add_native_tools` call that builds the request payload, so the
        filter can't drift from what is actually sent to the API.
        """
        native_tools, _, _ = self._add_native_tools([], model_request_parameters, model_settings)
        # `BetaMCPToolsetParam` is the only union member without a `name`, and `_add_native_tools`
        # never emits it into `tools` (MCP servers are returned separately).
        enabled_server_tool_names = {tool['name'] for tool in native_tools if 'name' in tool}  # pragma: no branch
        # The native memory tool is client-executed and surfaces as a regular `tool_use` block.
        enabled_server_tool_names.discard('memory')

        implicit_code_execution_names = {'code_execution', 'bash_code_execution', 'text_editor_code_execution'}
        if 'code_execution' in enabled_server_tool_names:
            enabled_server_tool_names.update(implicit_code_execution_names)
        # The 20260209 web tools provision code execution for dynamic filtering server-side.
        if self.profile.get('anthropic_supports_dynamic_filtering', False) and any(
            isinstance(tool, WebSearchTool | WebFetchTool) for tool in model_request_parameters.native_tools
        ):
            enabled_server_tool_names.update(implicit_code_execution_names)
        return frozenset(enabled_server_tool_names)

    async def _process_streamed_response(
        self,
        response: AsyncStream[BetaRawMessageStreamEvent],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> StreamedResponse:
        peekable_response: _utils.PeekableAsyncStream[
            BetaRawMessageStreamEvent, AsyncStream[BetaRawMessageStreamEvent]
        ] = _utils.PeekableAsyncStream(response)
        with _map_api_errors(self.model_name, self._provider.model_id_namespace):
            first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        assert isinstance(first_chunk, BetaRawMessageStartEvent)

        # On Bedrock the SDK drops SSE event types, so a leading Bedrock-only chunk
        # (e.g. `amazon-bedrock-invocationMetrics`) is non-validating `construct_type`d
        # into `BetaRawMessageStartEvent(message=None)`. Fall back to the configured model
        # name rather than dereference `first_chunk.message.model` (https://github.com/pydantic/pydantic-ai/issues/5774). The
        # iterator below skips these `message=None` events.
        model_name = first_chunk.message.model if first_chunk.message is not None else self.model_name  # pyright: ignore[reportUnnecessaryComparison]

        return AnthropicStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _response=peekable_response,
            _provider_name=self._provider.name,
            _model_id_namespace=self._provider.model_id_namespace,
            _provider_url=self._provider.base_url,
            _enabled_server_tool_names=self._get_enabled_server_tool_names(model_request_parameters, model_settings),
        )

    def _get_code_execution_tool_version(
        self, model_settings: AnthropicModelSettings
    ) -> AnthropicCodeExecutionToolVersion:
        version = model_settings.get('anthropic_code_execution_tool_version', 'auto')
        profile = self.profile
        if version == 'auto':
            return profile.get('anthropic_default_code_execution_tool_version', '20250825')
        if version not in profile.get('anthropic_supported_code_execution_tool_versions', ('20250825',)):
            supported_versions = ', '.join(
                f'{supported_version!r}'
                for supported_version in profile.get('anthropic_supported_code_execution_tool_versions', ('20250825',))
            )
            raise UserError(
                f'`anthropic_code_execution_tool_version={version!r}` is not supported by model '
                f'{self.model_name!r}. Supported versions are: {supported_versions}.'
            )
        return version

    def _resolve_tool_search_strategy(self, strategy: Literal['bm25', 'regex'] | None) -> Literal['bm25', 'regex']:
        """Resolve which native tool-search variant to send.

        `bm25` is the default, except on clients that don't support it (the legacy Bedrock
        InvokeModel API), where `regex` is the default and an explicit `bm25` is rejected.
        """
        if isinstance(self.client, _BM25_TOOL_SEARCH_UNSUPPORTED_CLIENTS):
            if strategy == 'bm25':
                raise UserError(
                    "ToolSearch(strategy='bm25') is not supported by the `AsyncAnthropicBedrock` client; "
                    "use ToolSearch(strategy='regex') instead, or leave the strategy unset to use the default."
                )
            return 'regex'
        return strategy or 'bm25'

    @staticmethod
    def _map_web_search_tool(
        tool: WebSearchTool, supports_dynamic_filtering: bool
    ) -> BetaWebSearchTool20260209Param | BetaWebSearchTool20250305Param:
        user_location = BetaUserLocationParam(type='approximate', **tool.user_location) if tool.user_location else None
        if supports_dynamic_filtering:
            return BetaWebSearchTool20260209Param(
                name='web_search',
                type='web_search_20260209',
                max_uses=tool.max_uses,
                allowed_domains=tool.allowed_domains,
                blocked_domains=tool.blocked_domains,
                user_location=user_location,
            )
        return BetaWebSearchTool20250305Param(
            name='web_search',
            type='web_search_20250305',
            max_uses=tool.max_uses,
            allowed_domains=tool.allowed_domains,
            blocked_domains=tool.blocked_domains,
            user_location=user_location,
        )

    @staticmethod
    def _map_web_fetch_tool(
        tool: WebFetchTool, supports_dynamic_filtering: bool
    ) -> tuple[BetaWebFetchTool20260209Param | BetaWebFetchTool20250910Param, str | None]:
        citations = BetaCitationsConfigParam(enabled=tool.enable_citations) if tool.enable_citations else None
        if supports_dynamic_filtering:
            return (
                BetaWebFetchTool20260209Param(
                    name='web_fetch',
                    type='web_fetch_20260209',
                    max_uses=tool.max_uses,
                    allowed_domains=tool.allowed_domains,
                    blocked_domains=tool.blocked_domains,
                    citations=citations,
                    max_content_tokens=tool.max_content_tokens,
                ),
                None,
            )
        return (
            BetaWebFetchTool20250910Param(
                name='web_fetch',
                type='web_fetch_20250910',
                max_uses=tool.max_uses,
                allowed_domains=tool.allowed_domains,
                blocked_domains=tool.blocked_domains,
                citations=citations,
                max_content_tokens=tool.max_content_tokens,
            ),
            'web-fetch-2025-09-10',
        )

    def _add_native_tools(  # noqa: C901
        self,
        tools: list[BetaToolUnionParam],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[list[BetaToolUnionParam], list[BetaRequestMCPServerURLDefinitionParam], set[str]]:
        beta_features: set[str] = set()
        mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] = []
        supports_dynamic_filtering = self.profile.get('anthropic_supports_dynamic_filtering', False)

        for tool in model_request_parameters.native_tools:
            if isinstance(tool, WebSearchTool):
                tools.append(self._map_web_search_tool(tool, supports_dynamic_filtering))
            elif isinstance(tool, CodeExecutionTool):  # pragma: no branch
                tool_version = self._get_code_execution_tool_version(model_settings)
                tools.append(_map_code_execution_tool(tool_version))
                # Cross-provider files are dropped silently here, not raised via
                # `_validate_uploaded_file_provider`; intentional per https://github.com/pydantic/pydantic-ai/issues/4338 (ignore over raise).
                if tool.files and any(file.provider_name == self.system for file in tool.files):
                    beta_features.add('files-api-2025-04-14')
            elif isinstance(tool, WebFetchTool):  # pragma: no branch
                web_fetch_tool, beta_feature = self._map_web_fetch_tool(tool, supports_dynamic_filtering)
                tools.append(web_fetch_tool)
                if beta_feature is not None:
                    beta_features.add(beta_feature)
            elif isinstance(tool, MemoryTool):  # pragma: no branch
                if 'memory' not in {tool.name for tool in model_request_parameters.declared_function_tools}:
                    raise UserError("Native `MemoryTool` requires a 'memory' tool to be defined.")
                # Replace the memory tool definition with the native memory tool
                tools = [tool for tool in tools if tool.get('name') != 'memory']
                tools.append(BetaMemoryTool20250818Param(name='memory', type='memory_20250818'))
                beta_features.add('context-management-2025-06-27')
            elif isinstance(tool, ToolSearchTool):  # pragma: no branch
                # Custom-callable strategies go through the regular `search_tools` function tool
                # (which is already in `function_tools`), so no server-side builtin is emitted.
                if tool.strategy != 'custom':
                    if self._resolve_tool_search_strategy(tool.strategy) == 'regex':
                        tools.append(
                            BetaToolSearchToolRegex20251119Param(
                                type='tool_search_tool_regex_20251119',
                                name='tool_search_tool_regex',
                            )
                        )
                    else:
                        tools.append(
                            BetaToolSearchToolBm25_20251119Param(
                                type='tool_search_tool_bm25_20251119',
                                name='tool_search_tool_bm25',
                            )
                        )
                # No `beta_features.add(...)`: tool search is GA on Sonnet/Opus/Haiku 4.5+ and the
                # provisional `tool-search-tool-2025-11-19` beta header is rejected by the API.
            elif isinstance(tool, AdvisorTool):  # pragma: no branch
                tools.append(_map_advisor_tool(tool))
                beta_features.add('advisor-tool-2026-03-01')
            elif isinstance(tool, MCPServerTool) and tool.url:
                mcp_server_url_definition_param = BetaRequestMCPServerURLDefinitionParam(
                    type='url',
                    name=tool.id,
                    url=tool.url,
                )
                if tool.allowed_tools is not None:  # pragma: no branch
                    mcp_server_url_definition_param['tool_configuration'] = BetaRequestMCPServerToolConfigurationParam(
                        enabled=bool(tool.allowed_tools),
                        allowed_tools=tool.allowed_tools,
                    )
                if tool.authorization_token:  # pragma: no cover
                    mcp_server_url_definition_param['authorization_token'] = tool.authorization_token
                mcp_servers.append(mcp_server_url_definition_param)
                beta_features.add('mcp-client-2025-04-04')
            else:
                raise UserError(  # pragma: no cover
                    f'`{tool.__class__.__name__}` is not supported by `AnthropicModel`. If it should be, please file an issue.'
                )
        return tools, mcp_servers, beta_features

    def _prepare_tools_and_tool_choice(
        self,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[BetaToolUnionParam], BetaToolChoiceParam | None]:
        """Determine which tools to send and the API tool_choice value.

        Returns:
            A tuple of (filtered_tools, tool_choice).
        """
        tool_defs = model_request_parameters.declared_tool_defs
        resolved_tool_choice = resolve_tool_choice(model_settings, model_request_parameters)
        supports_forced_tool_choice = self.profile.get('anthropic_supports_forced_tool_choice', True)
        supports_adaptive_thinking = self.profile.get('anthropic_supports_adaptive_thinking', False)

        tool_choice: BetaToolChoiceParam

        if resolved_tool_choice in ('auto', 'none'):
            # tool_choice = {'type': resolved_tool_choice}`: pyright can't narrow this properly
            tool_choice = {'type': 'auto'} if resolved_tool_choice == 'auto' else {'type': 'none'}
        elif resolved_tool_choice == 'required':
            supports = _support_tool_forcing(
                model_settings,
                model_request_parameters,
                resolved_tool_choice,
                "tool_choice='required'",
                supports_forced_tool_choice=supports_forced_tool_choice,
                supports_adaptive_thinking=supports_adaptive_thinking,
            )
            tool_choice = {'type': 'any'} if supports else {'type': 'auto'}
        elif isinstance(resolved_tool_choice, tuple):
            tool_choice_mode, tool_names = resolved_tool_choice
            supports = _support_tool_forcing(
                model_settings,
                model_request_parameters,
                resolved_tool_choice,
                supports_forced_tool_choice=supports_forced_tool_choice,
                supports_adaptive_thinking=supports_adaptive_thinking,
            )
            if tool_choice_mode == 'required' and len(tool_names) == 1:
                if supports:
                    tool_choice = {'type': 'tool', 'name': next(iter(tool_names))}
                else:
                    # Forcing not supported (extended thinking, or a model that rejects it outright):
                    # filter so the model can only see the requested tool, since `auto` alone
                    # wouldn't restrict the choice.
                    # Breaks caching, but Anthropic doesn't support limiting tools via API arg.
                    tool_defs = {k: v for k, v in tool_defs.items() if k in tool_names}
                    tool_choice = {'type': 'auto'}
            else:
                # Breaks caching, but Anthropic doesn't support limiting tools via API arg
                tool_defs = {k: v for k, v in tool_defs.items() if k in tool_names}
                tool_choice = {'type': 'auto'} if tool_choice_mode == 'auto' or not supports else {'type': 'any'}
        else:
            assert_never(resolved_tool_choice)

        if not tool_defs:
            return [], None

        # `'deferred'` visibility means "withhold this tool's schema", which
        # `prepare_request` only leaves set on models that can unhide it again — so the flag goes on
        # the wire as it stands, with no second opinion from here. Anthropic unhides through a
        # `tool_reference` block, from a `tool_addition` or a tool-search result, and the tool keeps
        # the flag afterwards so an advertised entry never graduates into the cache-keyed segment.
        tools: list[BetaToolUnionParam] = [
            self._map_tool_definition(t, model_settings, visibility=model_request_parameters.visibility_of(t.name))
            for t in tool_defs.values()
        ]

        # Add cache_control to the last non-deferred tool if enabled. Anthropic rejects
        # `cache_control` on tools with `defer_loading=True` (`Tools with defer_loading
        # cannot use prompt caching`); they're hidden from the model until tool search
        # discovers them, so they aren't part of the cacheable prompt prefix anyway.
        if cache_tool_defs := model_settings.get('anthropic_cache_tool_definitions'):
            ttl: Literal['5m', '1h'] = '5m' if cache_tool_defs is True else cache_tool_defs
            for tool in reversed(tools):
                if tool.get('defer_loading') is not True:
                    tool['cache_control'] = self._build_cache_control(ttl)
                    break

        if 'parallel_tool_calls' in model_settings and tool_choice['type'] != 'none':
            tool_choice['disable_parallel_tool_use'] = not model_settings['parallel_tool_calls']

        return tools, tool_choice

    async def _map_message(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        messages = self._trim_before_compaction(messages)
        system_prompt_parts: list[str] = []
        anthropic_messages: list[BetaMessageParam] = []
        # Cross-provider files are dropped silently here, not raised via
        # `_validate_uploaded_file_provider`; intentional per https://github.com/pydantic/pydantic-ai/issues/4338 (ignore over raise).
        pending_container_uploads = [
            file.file_id
            for tool in model_request_parameters.native_tools
            if isinstance(tool, CodeExecutionTool) and tool.files
            for file in tool.files
            if file.provider_name == self.system
        ]
        # Whenever this request withholds a tool's schema, render any local-shape `search_tools`
        # history exchanges as Anthropic's "client-side" tool-search wire — `tool_use` paired with a
        # `tool_result` whose `content` is a `tool_reference` array. That block is the reveal: it
        # unlocks the withheld schemas server-side without forcing the model to re-search, and works
        # regardless of the current turn's strategy (default native, named native, or custom
        # callable), covering the no-history single-turn custom callable case too (where the local
        # `search_tools` exchange is created on this turn).
        #
        # Reading the deferred tools rather than the presence of a `ToolSearchTool` is deliberate:
        # tool search is one thing that can trigger a reveal, not what makes a reveal legal. A run
        # whose deferred tools are all capability-gated sends no search tool at all and still needs
        # this, and Anthropic agrees — a `tool_reference` result with no tool-search tool in the
        # request returns 200 and the model calls the revealed tool (verified live on
        # `claude-sonnet-5` and `claude-opus-4-8`).
        # A `None` visibility means unresolved parameters from a direct `Model` call (every
        # production path resolves via `prepare_request` first); the authored flag is the stand-in.
        deferred_tools_active = any(
            model_request_parameters.visibility_of(t.name) == 'deferred'
            for t in model_request_parameters.function_tools
        )
        # The API 400s if advisor blocks appear in history without the advisor tool in the current
        # request, so when it's absent we strip advisor call/result blocks during replay (per
        # Anthropic's docs). When present, blocks — including a dangling pause_turn call — round-trip
        # verbatim (no pairing logic needed).
        advisor_active = any(isinstance(t, AdvisorTool) for t in model_request_parameters.native_tools)
        # Filter `tool_reference` entries during replay against the tools the current turn
        # will actually send: Anthropic rejects references to tools not in the wire `tools`
        # list (e.g. an MCP that failed to register this turn). The previously-discovered
        # name still lives in history; it just isn't worth replaying as a tool reference.
        #
        # `tool_defs` rather than `function_tools` so this matches what `_prepare_tools_and_tool_choice`
        # actually sends, which includes output tools. Nothing generates a reveal naming an output tool
        # today — the framework's only generator reads `function_tools`, and the UI adapters round-trip
        # names from it — so this changes no current behavior. It keeps the filter honest against the
        # wire regardless, rather than silently dropping a block whenever the two sets diverge.
        declared_tool_names = set(model_request_parameters.declared_tool_defs)
        # Only the opening `SystemPromptPart`s in the first request are the run's own system prompt and
        # hoist to the top-level `system` parameter. Later ones are mid-conversation operator instructions:
        # where we support them they reach us verbatim (rather than `<system>`-tagged by `prepare_messages`)
        # and it's on us to render them as a `{'role': 'system'}` entry, so adding an instruction leaves the
        # cached prefix the top-level `system` parameter sits in untouched. Where we don't,
        # `prepare_messages` has already rewritten them and none reach this branch — bar the adapter's
        # direct entry points, `count_tokens` and `request`, where hoisting them is the safe reading.
        inline_system_prompts = self.profile.get('supports_inline_system_prompts', False)
        # Already narrowed for transports that can't serve the `system` role, so this covers both halves
        # of the gate — and it's the same flag the beta header is added under.
        supports_tool_availability_delta = self.tool_addition_mode == 'by_reference'
        leading_request = next((m for m in messages if isinstance(m, ModelRequest)), None)
        rendered_tool_additions: set[str] = set()
        tool_defs_by_name = {tool.name: tool for tool in model_request_parameters.function_tools}
        for m in messages:
            if isinstance(m, ModelRequest):
                standing_prompt_count = _standing_system_prompt_count(m) if m is leading_request else 0
                user_content_params: list[BetaContentBlockParam] = []
                mid_conversation_system_prompts: list[str] = []
                tool_availability_blocks: list[dict[str, Any]] = []
                # `CachePoint`s authored after a mid-conversation instruction or a tool availability
                # change, as the number of user blocks that preceded each one. They can't be placed
                # while mapping: both render after this request's user blocks in the `system` entry
                # despite being authored before the marker, so whether they fall inside the boundary
                # depends on whether anything else follows the marker.
                deferred_cache_points: list[tuple[int, Literal['5m', '1h']]] = []
                for part_index, request_part in enumerate(m.parts):
                    if isinstance(request_part, SystemPromptPart):
                        if not inline_system_prompts or part_index < standing_prompt_count:
                            system_prompt_parts.append(request_part.content)
                        else:
                            # System-voice parts within one request all precede the same assistant
                            # response. Rendering an interleaved entry after sibling user content is
                            # deterministic and preserves the only ordering boundary that matters.
                            mid_conversation_system_prompts.append(request_part.content)
                    elif isinstance(request_part, UserPromptPart):
                        async for content in self._map_user_prompt(request_part):
                            if isinstance(content, CachePoint):
                                if mid_conversation_system_prompts or tool_availability_blocks:
                                    deferred_cache_points.append((len(user_content_params), content.ttl))
                                else:
                                    # A `CachePoint` asks to cache everything up to that point, and it
                                    # normally attaches to the block before it in this same user message.
                                    # A mid-conversation instruction used to leave one there — it was
                                    # folded in as `<system>`-tagged text — and now goes to its own
                                    # `system` entry instead, so `[SystemPromptPart,
                                    # UserPromptPart([CachePoint(), ...])]` arrives here with nothing to
                                    # attach to and used to raise. The boundary is still well defined
                                    # whenever the conversation has a previous message: it's the end of
                                    # that message, which is everything the entry and this turn build on.
                                    # Only a `CachePoint` with no prior content anywhere still raises,
                                    # which is the case the error is actually about.
                                    self._add_cache_control_to_last_param(
                                        user_content_params or _last_message_content(anthropic_messages),
                                        ttl=content.ttl,
                                    )
                            else:
                                user_content_params.append(content)
                    elif isinstance(request_part, ToolAvailabilityDeltaPart):
                        if not supports_tool_availability_delta:
                            # `prepare_messages` projects the delta onto the local tool-search exchange
                            # for every model and transport that can't render it natively, so arriving
                            # here means that projection didn't run — the same pipeline bug the other
                            # adapters raise on, reachable by calling `Model.request` directly. Raising
                            # matches them; rendering the blocks anyway would send them without the
                            # `mid-conversation-tool-changes` beta header, which is added under this
                            # same condition, and earn a 400 in place of an explanation.
                            raise _unsynthesized_tool_availability_delta_error()
                        # Both block types carry a `tool_reference`, and the API rejects one naming a
                        # tool this request doesn't declare: `tool_addition/tool_removal references
                        # unknown tool '...'`. Replayed history routinely names tools that have since
                        # gone — the turn announcing a removal is the last one that still declares it,
                        # and a tool added then removed is absent from every turn after that. So a
                        # block that can no longer be referenced is dropped, not asserted away: the
                        # tool's absence from `tools` already tells the model what the block would.
                        # `declared_tool_names` is the one bound above and shared with the tool-search
                        # replay filters — same question, so it should be the same answer.
                        for name in request_part.tools_added:
                            tool_def = tool_defs_by_name.get(name)
                            if (
                                name in declared_tool_names
                                and name not in rendered_tool_additions
                                and tool_def is not None
                                and model_request_parameters.visibility_of(name) != 'visible'
                            ):
                                tool_availability_blocks.append(
                                    {'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': name}}
                                )
                                rendered_tool_additions.add(name)
                    elif isinstance(request_part, ToolReturnPart):
                        tool_result_content: list[beta_tool_result_block_param.Content] = []

                        custom_tool_refs, custom_empty_message = _build_custom_tool_search_replay_blocks(
                            request_part, deferred_tools_active, declared_tool_names
                        )
                        if custom_tool_refs:
                            tool_result_block_param = beta_tool_result_block_param.BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=custom_tool_refs,
                                is_error=False,
                            )
                            user_content_params.append(tool_result_block_param)
                            continue
                        if custom_tool_refs is not None:
                            # Empty-results path on the custom-callable strategy. Anthropic
                            # rejects an empty `tool_result.content` list, so we send the
                            # `message` text from the typed return (set by the toolset's
                            # `_empty_return`) as a single text block instead.
                            empty_message = custom_empty_message or _NO_MATCHES_MESSAGE
                            tool_result_block_param = beta_tool_result_block_param.BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=[BetaTextBlockParam(text=empty_message, type='text')],
                                is_error=False,
                            )
                            user_content_params.append(tool_result_block_param)
                            continue

                        for item in request_part.content_items(mode='str', wrap_if_error=False):
                            if isinstance(item, UploadedFile):
                                self._validate_uploaded_file_provider(item)
                                if item.media_type.startswith('image/'):
                                    tool_result_content.append(
                                        BetaImageBlockParam(
                                            source=BetaFileImageSourceParam(file_id=item.file_id, type='file'),
                                            type='image',
                                        )
                                    )
                                elif item.media_type.startswith(('text/', 'application/')):
                                    tool_result_content.append(
                                        BetaRequestDocumentBlockParam(
                                            source=BetaFileDocumentSourceParam(file_id=item.file_id, type='file'),
                                            type='document',
                                        )
                                    )
                                else:
                                    raise UserError(
                                        f'Unsupported media type {item.media_type!r} for Anthropic file upload. '
                                        'Only image and document (text/application) types are supported.'
                                    )
                            elif is_multi_modal_content(item):
                                tool_result_content.append(await self._map_file_to_content_block(item, 'tool returns'))  # pyright: ignore[reportArgumentType]
                            elif isinstance(item, str):  # pragma: no branch
                                tool_result_content.append(BetaTextBlockParam(text=item, type='text'))

                        tool_result_block_param = beta_tool_result_block_param.BetaToolResultBlockParam(
                            tool_use_id=_guard_tool_call_id(t=request_part),
                            type='tool_result',
                            content=tool_result_content or '',
                            is_error=request_part.outcome == 'failed',
                        )
                        user_content_params.append(tool_result_block_param)
                    elif isinstance(request_part, SpeechPart):  # pragma: no cover
                        # Unconverted realtime speech; `prepare_messages` turns these into `UserPromptPart`s.
                        raise _unconverted_speech_part_error()
                    elif isinstance(request_part, RetryPromptPart):  # pragma: no branch
                        if request_part.tool_name is None:
                            text = request_part.model_response()
                            retry_param = BetaTextBlockParam(type='text', text=text)
                        else:
                            retry_param = beta_tool_result_block_param.BetaToolResultBlockParam(
                                tool_use_id=_guard_tool_call_id(t=request_part),
                                type='tool_result',
                                content=request_part.model_response(),
                                is_error=True,
                            )
                        user_content_params.append(retry_param)
                # A marker that ends the request has the instruction authored before it and nothing
                # authored after it, so the `system` entry's final block is exactly its boundary. A
                # marker with content after it can't have both, and lands where it was authored: the
                # instruction then sits outside the boundary, which caches a prefix of what was asked
                # for rather than more than was asked for.
                system_entry_cache_ttl: Literal['5m', '1h'] | None = None
                for marked_at, ttl in deferred_cache_points:
                    if marked_at == len(user_content_params):
                        system_entry_cache_ttl = ttl
                    else:
                        self._add_cache_control_to_last_param(
                            user_content_params[:marked_at] or _last_message_content(anthropic_messages), ttl=ttl
                        )
                if len(user_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='user', content=user_content_params))
                if mid_conversation_system_prompts or tool_availability_blocks:
                    system_content_params: list[BetaContentBlockParam] = [
                        BetaTextBlockParam(text=content, type='text') for content in mid_conversation_system_prompts
                    ]
                    system_content_params.extend(cast('list[BetaContentBlockParam]', tool_availability_blocks))
                    # The boundary covers the whole system entry, availability blocks included — a
                    # terminal `CachePoint` lands on the entry's final block, never inside the entry.
                    if system_entry_cache_ttl is not None:
                        self._add_cache_control_to_last_param(system_content_params, ttl=system_entry_cache_ttl)
                    anthropic_messages.append(BetaMessageParam(role='system', content=system_content_params))
            elif isinstance(m, ModelResponse):
                assistant_content_params: list[
                    BetaTextBlockParam
                    | BetaToolUseBlockParam
                    | BetaServerToolUseBlockParam
                    | BetaWebSearchToolResultBlockParam
                    | BetaCodeExecutionToolResultBlockParam
                    | BetaBashCodeExecutionToolResultBlockParam
                    | BetaTextEditorCodeExecutionToolResultBlockParam
                    | BetaWebFetchToolResultBlockParam
                    | BetaToolSearchToolResultBlockParam
                    | BetaAdvisorToolResultBlockParam
                    | BetaThinkingBlockParam
                    | BetaRedactedThinkingBlockParam
                    | BetaMCPToolUseBlockParam
                    | BetaMCPToolResultBlock
                    | BetaCompactionBlockParam
                ] = []
                for response_part in m.parts:
                    if isinstance(response_part, TextPart):
                        if response_part.content:
                            assistant_content_params.append(BetaTextBlockParam(text=response_part.content, type='text'))
                    elif isinstance(response_part, ToolCallPart):
                        tool_use_block_param = BetaToolUseBlockParam(
                            id=_guard_tool_call_id(t=response_part),
                            type='tool_use',
                            name=response_part.tool_name,
                            input=response_part.args_as_dict(),
                        )
                        assistant_content_params.append(tool_use_block_param)
                    elif isinstance(response_part, ThinkingPart):
                        # An empty signature (e.g. from an interrupted stream) is never valid,
                        # so fall back to tagged text rather than triggering a 400 from the API.
                        if response_part.provider_name == self.system and response_part.signature:
                            if response_part.id == 'redacted_thinking':
                                assistant_content_params.append(
                                    BetaRedactedThinkingBlockParam(
                                        data=response_part.signature,
                                        type='redacted_thinking',
                                    )
                                )
                            else:
                                assistant_content_params.append(
                                    BetaThinkingBlockParam(
                                        thinking=response_part.content,
                                        signature=response_part.signature,
                                        type='thinking',
                                    )
                                )
                        elif response_part.content:  # pragma: no branch
                            start_tag, end_tag = self.profile.get('thinking_tags', DEFAULT_THINKING_TAGS)
                            assistant_content_params.append(
                                BetaTextBlockParam(
                                    text='\n'.join([start_tag, response_part.content, end_tag]), type='text'
                                )
                            )
                    elif isinstance(response_part, NativeToolCallPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name == WebSearchTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='web_search',
                                    input=response_part.args_as_dict(),
                                )
                                _add_anthropic_caller_param(server_tool_use_block_param, response_part)
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name in (
                                CodeExecutionTool.kind,
                                'bash_code_execution',
                                'text_editor_code_execution',
                            ):
                                anthropic_tool_name = _get_anthropic_code_execution_tool_name(response_part)
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name=anthropic_tool_name,
                                    input=response_part.args_as_dict(),
                                )
                                _add_anthropic_caller_param(server_tool_use_block_param, response_part)
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name == WebFetchTool.kind:
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='web_fetch',
                                    input=response_part.args_as_dict(),
                                )
                                _add_anthropic_caller_param(server_tool_use_block_param, response_part)
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name == AdvisorTool.kind:
                                if not advisor_active:
                                    continue
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name='advisor',
                                    input=response_part.args_as_dict(),
                                )
                                _add_anthropic_caller_param(server_tool_use_block_param, response_part)
                                assistant_content_params.append(server_tool_use_block_param)
                            elif response_part.tool_name == ToolSearchTool.kind:
                                # Round-trip the native variant (bm25/regex) so we don't
                                # silently rewrite the algorithm. `_map_server_tool_use_block`
                                # stashes it in `provider_details['strategy']`. Clients that don't
                                # support `bm25` (legacy Bedrock InvokeModel) always replay as `regex`.
                                details = response_part.provider_details or {}
                                strategy: Literal['bm25', 'regex'] = (
                                    'regex'
                                    if details.get('strategy') == 'regex'
                                    or isinstance(self.client, _BM25_TOOL_SEARCH_UNSUPPORTED_CLIENTS)
                                    else 'bm25'
                                )
                                native_name = (
                                    'tool_search_tool_regex' if strategy == 'regex' else 'tool_search_tool_bm25'
                                )
                                # Rebuild the variant-specific wire shape from the
                                # cross-provider `queries` slot. bm25 expects `{"query": "..."}`,
                                # regex expects `{"pattern": "..."}`. Joining with a space
                                # matches Anthropic's single-string input on either variant.
                                args_dict = response_part.args_as_dict()
                                if 'queries' in args_dict:
                                    raw_queries = args_dict.get('queries')
                                    queries: list[str] = (
                                        [q for q in cast('list[Any]', raw_queries) if isinstance(q, str)]
                                        if isinstance(raw_queries, list)
                                        else []
                                    )
                                    wire_key = 'pattern' if strategy == 'regex' else 'query'
                                    wire_input: dict[str, Any] = {wire_key: ' '.join(queries)}
                                else:
                                    wire_input = args_dict
                                server_tool_use_block_param = BetaServerToolUseBlockParam(
                                    id=tool_use_id,
                                    type='server_tool_use',
                                    name=native_name,
                                    input=wire_input,
                                )
                                _add_anthropic_caller_param(server_tool_use_block_param, response_part)
                                assistant_content_params.append(server_tool_use_block_param)
                            elif (
                                response_part.tool_name.startswith(MCPServerTool.kind)
                                and (server_id := response_part.tool_name.split(':', 1)[1])
                                and (args := response_part.args_as_dict())
                                and (tool_name := args.get('tool_name'))
                                and (tool_args := args.get('tool_args')) is not None
                            ):  # pragma: no branch
                                mcp_tool_use_block_param = BetaMCPToolUseBlockParam(
                                    id=tool_use_id,
                                    type='mcp_tool_use',
                                    server_name=server_id,
                                    name=tool_name,
                                    input=tool_args,
                                )
                                assistant_content_params.append(mcp_tool_use_block_param)
                    elif isinstance(response_part, NativeToolReturnPart):
                        if response_part.provider_name == self.system:
                            tool_use_id = _guard_tool_call_id(t=response_part)
                            if response_part.tool_name in (
                                WebSearchTool.kind,
                                'web_search_tool_result',  # Backward compatibility
                            ) and isinstance(response_part.content, dict | list):
                                block = BetaWebSearchToolResultBlockParam(
                                    tool_use_id=tool_use_id,
                                    type='web_search_tool_result',
                                    content=cast(
                                        BetaWebSearchToolResultBlockParamContentParam,
                                        response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                    ),
                                )
                                _add_anthropic_caller_param(block, response_part)
                                assistant_content_params.append(block)
                            elif response_part.tool_name in (
                                CodeExecutionTool.kind,
                                'code_execution_tool_result',  # Backward compatibility
                                'bash_code_execution',
                                'text_editor_code_execution',
                            ) and isinstance(response_part.content, dict | list):
                                match _get_anthropic_code_execution_tool_name(response_part):
                                    case 'code_execution':
                                        assistant_content_params.append(
                                            BetaCodeExecutionToolResultBlockParam(
                                                tool_use_id=tool_use_id,
                                                type='code_execution_tool_result',
                                                content=cast(
                                                    BetaCodeExecutionToolResultBlockParamContentParam,
                                                    response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                                ),
                                            )
                                        )
                                    case 'bash_code_execution':
                                        assistant_content_params.append(
                                            BetaBashCodeExecutionToolResultBlockParam(
                                                tool_use_id=tool_use_id,
                                                type='bash_code_execution_tool_result',
                                                content=cast(
                                                    BashCodeExecutionToolResultBlockParamContent,
                                                    response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                                ),
                                            )
                                        )
                                    case 'text_editor_code_execution':  # pragma: no branch
                                        assistant_content_params.append(
                                            BetaTextEditorCodeExecutionToolResultBlockParam(
                                                tool_use_id=tool_use_id,
                                                type='text_editor_code_execution_tool_result',
                                                content=cast(
                                                    TextEditorCodeExecutionToolResultBlockParamContent,
                                                    response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                                ),
                                            )
                                        )
                            elif response_part.tool_name == WebFetchTool.kind and isinstance(
                                response_part.content, dict
                            ):
                                block = BetaWebFetchToolResultBlockParam(
                                    tool_use_id=tool_use_id,
                                    type='web_fetch_tool_result',
                                    content=cast(
                                        WebFetchToolResultBlockParamContent,
                                        response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                    ),
                                )
                                _add_anthropic_caller_param(block, response_part)
                                assistant_content_params.append(block)
                            elif response_part.tool_name == AdvisorTool.kind:
                                # Drop advisor result blocks when continuing without the advisor tool: the
                                # API 400s if they appear in history without the tool in the request, and
                                # Anthropic's docs prescribe stripping them (mirrors the orphan tool-search
                                # drop above). When kept, the discriminated content dict round-trips verbatim
                                # so the server can decrypt a redacted advisor result on the next turn.
                                if advisor_active and isinstance(response_part.content, dict):
                                    assistant_content_params.append(
                                        BetaAdvisorToolResultBlockParam(
                                            tool_use_id=tool_use_id,
                                            type='advisor_tool_result',
                                            content=cast(
                                                AdvisorToolResultBlockParamContent,
                                                response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                            ),
                                        )
                                    )
                            elif isinstance(response_part, NativeToolSearchReturnPart):
                                assistant_content_params.append(
                                    _build_tool_search_replay_block(response_part, tool_use_id, declared_tool_names)
                                )
                            elif response_part.tool_name.startswith(MCPServerTool.kind) and isinstance(
                                response_part.content, dict
                            ):  # pragma: no branch
                                mcp_content = cast(
                                    'dict[str, Any]',
                                    response_part.content,  # pyright: ignore[reportUnknownMemberType]
                                )
                                assistant_content_params.append(
                                    BetaMCPToolResultBlock(
                                        tool_use_id=tool_use_id,
                                        type='mcp_tool_result',
                                        **mcp_content,
                                    )
                                )
                    elif isinstance(response_part, CompactionPart):
                        if response_part.provider_name == self.system:  # pragma: no branch
                            compaction_block = BetaCompactionBlockParam(
                                content=response_part.content, type='compaction'
                            )
                            if (
                                response_part.provider_details
                                and response_part.provider_details.get('encrypted_content') is not None
                            ):
                                compaction_block['encrypted_content'] = response_part.provider_details[
                                    'encrypted_content'
                                ]
                            assistant_content_params.append(compaction_block)
                    elif isinstance(response_part, FilePart):  # pragma: no cover
                        # Files generated by models are not sent back to models that don't themselves generate files.
                        pass
                    elif isinstance(response_part, SpeechPart):  # pragma: no cover
                        # Unconverted realtime speech; `prepare_messages` turns these into `TextPart`s in `Model.prepare_messages`.
                        raise _unconverted_speech_part_error()
                    else:
                        assert_never(response_part)
                if len(assistant_content_params) > 0:
                    anthropic_messages.append(BetaMessageParam(role='assistant', content=assistant_content_params))
            else:
                assert_never(m)

        _drop_unpaired_native_tool_calls(anthropic_messages)
        _place_system_messages_before_generation(anthropic_messages)
        _anchor_system_messages(anthropic_messages)

        if pending_container_uploads:
            for message in anthropic_messages:
                if message['role'] == 'user':
                    existing = message['content']
                    assert isinstance(existing, list)
                    if _is_tool_result_only(existing):
                        continue
                    extra: list[BetaContainerUploadBlockParam] = [
                        {'type': 'container_upload', 'file_id': file_id} for file_id in pending_container_uploads
                    ]
                    message['content'] = [*existing, *extra]

        instruction_parts = self._get_instruction_parts(messages, model_request_parameters)
        system_prompt = '\n\n'.join(system_prompt_parts)

        # Build system prompt blocks: each instruction part becomes a separate text block.
        # When anthropic_cache_instructions is enabled, the cache point goes after the last
        # static instruction (or at the end if all instructions are static).
        cache_instructions = model_settings.get('anthropic_cache_instructions')

        if instruction_parts or cache_instructions:
            system_prompt_blocks: list[BetaTextBlockParam] = []

            if system_prompt:
                system_prompt_blocks.append(BetaTextBlockParam(type='text', text=system_prompt))

            if instruction_parts:
                for part in instruction_parts:
                    system_prompt_blocks.append(BetaTextBlockParam(type='text', text=part.content))

            if system_prompt_blocks and cache_instructions:
                ttl: Literal['5m', '1h'] = '5m' if cache_instructions is True else cache_instructions
                # Find the last block that corresponds to a static instruction.
                # system_prompt_blocks layout: [system_prompt_block?, ...instruction_blocks]
                # instruction_parts are sorted static-first, so find the boundary.
                if instruction_parts:
                    has_dynamic = any(p.dynamic for p in instruction_parts)
                    if has_dynamic:
                        # Cache after the last static instruction block
                        num_prefix_blocks = 1 if system_prompt else 0
                        num_static = sum(1 for p in instruction_parts if not p.dynamic)
                        if num_static > 0:
                            cache_block_idx = num_prefix_blocks + num_static - 1
                        else:
                            # All dynamic: cache the system prompt block if it exists
                            cache_block_idx = 0 if system_prompt else None
                    else:
                        # All static: cache the last block
                        cache_block_idx = len(system_prompt_blocks) - 1
                else:
                    # No instruction parts, just system prompt: cache it
                    cache_block_idx = 0

                if cache_block_idx is not None:
                    system_prompt_blocks[cache_block_idx]['cache_control'] = self._build_cache_control(ttl)

            if system_prompt_blocks:
                return system_prompt_blocks, anthropic_messages

        return system_prompt, anthropic_messages

    @staticmethod
    def _limit_cache_points(
        system_prompt: str | list[BetaTextBlockParam],
        anthropic_messages: list[BetaMessageParam],
        tools: list[BetaToolUnionParam],
        *,
        automatic_caching: bool = False,
    ) -> None:
        """Limit the number of cache points in the request to Anthropic's maximum.

        Anthropic enforces a maximum of 4 cache points per request. This method ensures
        compliance by counting existing cache points and removing excess ones from messages.

        When automatic_caching is enabled, the server-applied breakpoint uses 1 of the 4
        available slots, so the budget for explicit breakpoints is reduced to 3.

        Strategy:
        1. Count cache points in system_prompt (can be multiple if list of blocks)
        2. Count cache points in tools (can be in any position, not just last)
        3. Raise UserError if system + tools already exceed the budget
        4. Calculate remaining budget for message cache points
        5. Traverse messages from newest to oldest, keeping the most recent cache points
           within the remaining budget
        6. Remove excess cache points from older messages to stay within limit

        Cache point priority (always preserved):
        - System prompt cache points
        - Tool definition cache points
        - Message cache points (newest first, oldest removed if needed)

        Raises:
            UserError: If system_prompt and tools combined already exceed the budget.
                      This indicates a configuration error that cannot be auto-fixed.
        """
        MAX_CACHE_POINTS = 3 if automatic_caching else 4

        # Count existing cache points in system prompt
        used_cache_points = (
            sum(1 for block in system_prompt if 'cache_control' in cast(dict[str, Any], block))
            if isinstance(system_prompt, list)
            else 0
        )

        # Count existing cache points in tools (any tool may have cache_control)
        # Note: cache_control can be in the middle of tools list if builtin tools are added after
        for tool in tools:
            if 'cache_control' in tool:
                used_cache_points += 1

        # Calculate remaining cache points budget for messages
        remaining_budget = MAX_CACHE_POINTS - used_cache_points
        if remaining_budget < 0:  # pragma: no cover
            raise UserError(
                f'Too many cache points for Anthropic request. '
                f'System prompt and tool definitions already use {used_cache_points} cache points, '
                f'which exceeds the maximum of {MAX_CACHE_POINTS}.'
            )
        # Remove excess cache points from messages (newest to oldest)
        for message in reversed(anthropic_messages):
            content = message['content']
            if isinstance(content, str):  # pragma: no cover
                continue

            # Process content blocks in reverse order (newest first)
            for block in reversed(cast(list[BetaContentBlockParam], content)):
                block_dict = cast(dict[str, Any], block)

                if 'cache_control' in block_dict:
                    if remaining_budget > 0:
                        remaining_budget -= 1
                    else:
                        # Exceeded limit, remove this cache point
                        del block_dict['cache_control']

    def _build_cache_control(self, ttl: Literal['5m', '1h'] = '5m') -> BetaCacheControlEphemeralParam:
        """Build a cache control dict with the given TTL.

        Args:
            ttl: The cache time-to-live ('5m' or '1h').

        Returns:
            A cache control dict with the specified TTL.
        """
        return BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

    def _build_automatic_cache_control(
        self, model_settings: AnthropicModelSettings
    ) -> tuple[BetaCacheControlEphemeralParam | None, Literal['5m', '1h'] | None]:
        """Resolve cache settings and build the top-level cache_control parameter.

        Returns:
            A tuple of (top_level_param, resolved_ttl).
            top_level_param is the cache_control for the API (None on unsupported clients).
            resolved_ttl is the effective TTL (None if caching is not enabled), used by
            _apply_per_block_caching_fallback on clients that don't support automatic caching.
        """
        auto_cache = model_settings.get('anthropic_cache')
        cache_messages = model_settings.get('anthropic_cache_messages')

        if auto_cache and cache_messages:
            raise UserError('`anthropic_cache` and `anthropic_cache_messages` cannot both be enabled.')

        if not auto_cache:
            return None, None

        ttl: Literal['5m', '1h'] = '5m' if auto_cache is True else auto_cache
        # Bedrock and Vertex do not support the top-level cache_control parameter
        # (automatic caching). Per-block fallback is handled by _apply_per_block_caching_fallback.
        # Bedrock: https://github.com/anthropics/anthropic-sdk-python/issues/939
        # Vertex: https://github.com/anthropics/anthropic-sdk-python/issues/653
        # Foundry supports automatic caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching
        if isinstance(self.client, _NON_AUTOMATIC_CACHING_CLIENTS):
            return None, ttl
        return self._build_cache_control(ttl), ttl

    def _apply_per_block_caching_fallback(
        self,
        resolved_ttl: Literal['5m', '1h'] | None,
        anthropic_messages: list[BetaMessageParam],
    ) -> None:
        """Apply per-block message caching as a fallback for automatic caching on unsupported platforms.

        Bedrock and Vertex do not support the top-level `cache_control` parameter used by
        `anthropic_cache` for automatic caching. As a fallback, this applies per-block
        `cache_control` to the last content block of the last user message.

        Args:
            resolved_ttl: The resolved TTL from `_build_automatic_cache_control`, or None
                if caching is not enabled.
            anthropic_messages: The list of Anthropic message params to apply fallback to.
        """
        if resolved_ttl and isinstance(self.client, _NON_AUTOMATIC_CACHING_CLIENTS):
            self._apply_message_cache_control(anthropic_messages, resolved_ttl)

    def _apply_explicit_message_caching(
        self,
        model_settings: AnthropicModelSettings,
        anthropic_messages: list[BetaMessageParam],
    ) -> None:
        """Apply per-block message caching when `anthropic_cache_messages` is enabled.

        Mutually exclusive with `anthropic_cache` (enforced by `_build_automatic_cache_control`).
        """
        if cache_messages := model_settings.get('anthropic_cache_messages'):
            ttl: Literal['5m', '1h'] = '5m' if cache_messages is True else cache_messages
            self._apply_message_cache_control(anthropic_messages, ttl)

    def _apply_message_cache_control(
        self,
        anthropic_messages: list[BetaMessageParam],
        ttl: Literal['5m', '1h'],
    ) -> None:
        """Apply per-block `cache_control` to the last content block of the last message.

        If the last block already has `cache_control` (e.g. from an explicit `CachePoint`),
        it is left unchanged to preserve the user's chosen TTL.

        Assumes `anthropic_messages` is non-empty.
        """
        last_message = anthropic_messages[-1]
        content = last_message['content']
        if isinstance(content, str):  # pragma: no cover
            last_message['content'] = [
                BetaTextBlockParam(
                    type='text',
                    text=content,
                    cache_control=self._build_cache_control(ttl),
                )
            ]
        else:
            content_blocks = cast(list[BetaContentBlockParam], content)
            self._add_cache_control_to_last_cacheable_param(content_blocks, ttl)

    def _add_cache_control_to_last_cacheable_param(
        self, params: list[BetaContentBlockParam], ttl: Literal['5m', '1h'] = '5m'
    ) -> None:
        for param in reversed(params):
            if not is_str_dict(param):  # pragma: no cover
                continue
            if 'cache_control' in param:
                return
            if param['type'] in _ANTHROPIC_CACHEABLE_PARAM_TYPES:
                param['cache_control'] = self._build_cache_control(ttl)
                return

    def _add_cache_control_to_last_param(
        self, params: list[BetaContentBlockParam], ttl: Literal['5m', '1h'] = '5m'
    ) -> None:
        """Add cache control to the last content block param.

        See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.

        Args:
            params: List of content block params to modify.
            ttl: The cache time-to-live ('5m' or '1h').
        """
        _add_cache_control_param(params, self._build_cache_control(ttl))

    @staticmethod
    def _map_binary_data(data: bytes, media_type: str) -> BetaImageBlockParam | BetaRequestDocumentBlockParam:
        if media_type.startswith('image/'):
            return BetaImageBlockParam(
                source={'data': io.BytesIO(data), 'media_type': media_type, 'type': 'base64'},  # pyright: ignore[reportArgumentType]
                type='image',
            )
        elif media_type == 'application/pdf':
            return BetaRequestDocumentBlockParam(
                source=BetaBase64PDFSourceParam(
                    data=io.BytesIO(data),
                    media_type='application/pdf',
                    type='base64',
                ),
                type='document',
            )
        elif media_type == 'text/plain':
            return BetaRequestDocumentBlockParam(
                source=BetaPlainTextSourceParam(data=data.decode('utf-8'), media_type=media_type, type='text'),
                type='document',
            )
        else:  # pragma: no cover
            raise RuntimeError(f'Unsupported binary content media type for Anthropic: {media_type}')

    @staticmethod
    async def _map_image_url(item: ImageUrl) -> BetaImageBlockParam:
        if item.force_download:
            downloaded = await download_item(item, data_format='bytes')
            return AnthropicModel._map_binary_data(downloaded['data'], item.media_type)  # pyright: ignore[reportReturnType]
        return BetaImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')

    @staticmethod
    async def _map_document_url(item: DocumentUrl) -> BetaRequestDocumentBlockParam:
        if item.media_type == 'application/pdf':
            if item.force_download:
                downloaded = await download_item(item, data_format='bytes')
                return AnthropicModel._map_binary_data(downloaded['data'], item.media_type)  # pyright: ignore[reportReturnType]
            return BetaRequestDocumentBlockParam(source={'url': item.url, 'type': 'url'}, type='document')
        elif item.media_type == 'text/plain':
            downloaded_item = await download_item(item, data_format='text')
            return BetaRequestDocumentBlockParam(
                source=BetaPlainTextSourceParam(data=downloaded_item['data'], media_type=item.media_type, type='text'),
                type='document',
            )
        else:  # pragma: no cover
            raise RuntimeError(f'Unsupported document media type: {item.media_type}')

    @staticmethod
    async def _map_file_to_content_block(
        item: BinaryContent | ImageUrl | DocumentUrl | AudioUrl | VideoUrl,
        context: str,
    ) -> BetaImageBlockParam | BetaRequestDocumentBlockParam:
        """Map a multimodal file item to its Anthropic API content block."""
        if isinstance(item, BinaryContent):
            if item.is_image or item.is_document:
                return AnthropicModel._map_binary_data(item.data, item.media_type)
            raise NotImplementedError(f'Unsupported binary content type in Anthropic {context}: {item.media_type}')
        elif isinstance(item, ImageUrl):
            return await AnthropicModel._map_image_url(item)
        elif isinstance(item, DocumentUrl):
            return await AnthropicModel._map_document_url(item)
        elif isinstance(item, AudioUrl):
            raise NotImplementedError(f'AudioUrl is not supported in Anthropic {context}')
        else:
            raise NotImplementedError(f'VideoUrl is not supported in Anthropic {context}')

    async def _map_user_prompt(
        self,
        part: UserPromptPart,
    ) -> AsyncGenerator[BetaContentBlockParam | CachePoint]:
        if isinstance(part.content, str):
            if part.content:  # Only yield non-empty text
                yield BetaTextBlockParam(text=part.content, type='text')
        else:
            for item in part.content:
                if isinstance(item, str | TextContent):
                    text = item if isinstance(item, str) else item.content
                    if text:  # Only yield non-empty text
                        yield BetaTextBlockParam(text=text, type='text')
                elif isinstance(item, CachePoint):
                    yield item
                elif isinstance(item, UploadedFile):
                    self._validate_uploaded_file_provider(item)
                    if item.media_type.startswith('image/'):
                        yield BetaImageBlockParam(
                            source=BetaFileImageSourceParam(file_id=item.file_id, type='file'),
                            type='image',
                        )
                    elif item.media_type.startswith(('text/', 'application/')):
                        yield BetaRequestDocumentBlockParam(
                            source=BetaFileDocumentSourceParam(file_id=item.file_id, type='file'),
                            type='document',
                        )
                    else:
                        raise UserError(
                            f'Unsupported media type {item.media_type!r} for Anthropic file upload. '
                            'Only image and document (text/application) types are supported.'
                        )
                elif is_multi_modal_content(item):
                    yield await AnthropicModel._map_file_to_content_block(item, 'user prompts')  # pyright: ignore[reportArgumentType]
                else:
                    raise RuntimeError(f'Unsupported content type: {type(item)}')  # pragma: no cover

    def _map_tool_definition(
        self,
        f: ToolDefinition,
        model_settings: AnthropicModelSettings,
        *,
        visibility: ToolVisibility,
    ) -> BetaToolParam:
        """Maps a `ToolDefinition` dataclass to an Anthropic `BetaToolParam` dictionary."""
        tool_param: BetaToolParam = {
            'name': f.name,
            'description': f.description or '',
            'input_schema': f.parameters_json_schema,
        }
        if f.strict and self.profile.get('supports_json_schema_output', False):
            tool_param['strict'] = f.strict
        if model_settings.get('anthropic_eager_input_streaming'):
            tool_param['eager_input_streaming'] = True
        if visibility == 'deferred':
            tool_param['defer_loading'] = True
        return tool_param

    def _build_output_config(
        self, model_request_parameters: ModelRequestParameters, model_settings: AnthropicModelSettings
    ) -> BetaOutputConfigParam | None:
        output_format: BetaJSONOutputFormatParam | None = None
        if model_request_parameters.output_mode == 'native':
            assert model_request_parameters.output_object is not None
            output_format = {'type': 'json_schema', 'schema': model_request_parameters.output_object.json_schema}

        effort: AnthropicEffort | None = model_settings.get('anthropic_effort')
        # Fall back to unified thinking effort level when anthropic_effort is not set
        # Only map effort level strings; bare True just enables thinking without a specific effort
        profile = self.profile
        if (
            effort is None
            and profile.get('anthropic_supports_effort', False)
            and isinstance(model_request_parameters.thinking, str)
        ):
            effort = resolve_anthropic_effort(
                model_request_parameters.thinking,
                supports_xhigh=profile.get('anthropic_supports_xhigh_effort', False),
            )

        if effort is not None:
            self._validate_effort_vs_disabled_thinking(effort, model_settings)

        task_budget = self._get_task_budget(model_settings)

        if output_format is None and effort is None and task_budget is None:
            return None

        config: BetaOutputConfigParam = {}
        if output_format is not None:
            config['format'] = output_format
        if effort is not None:
            config['effort'] = effort
        if task_budget is not None:
            config['task_budget'] = task_budget
        return config

    def _validate_effort_vs_disabled_thinking(
        self, effort: AnthropicEffort, model_settings: AnthropicModelSettings
    ) -> None:
        """Reject `xhigh`/`max` effort combined with explicitly disabled thinking.

        Claude Opus 5 caps effort at `high` once thinking is disabled, while Claude Opus 4.8 accepts
        every effort level in that combination. Fail fast with a helpful message rather than letting
        the API return an opaque 400.
        """
        if effort not in ('xhigh', 'max'):
            return
        if not self.profile.get('anthropic_disallows_top_effort_when_thinking_disabled', False):
            return
        thinking = model_settings.get('anthropic_thinking')
        if thinking is None or thinking.get('type') != 'disabled':
            return
        raise UserError(
            f'Model {self.model_name!r} does not support `anthropic_effort={effort!r}` while '
            "`anthropic_thinking={'type': 'disabled'}`. Use an effort of 'high' or below, or enable thinking."
        )

    def _get_task_budget(self, model_settings: AnthropicModelSettings) -> AnthropicTaskBudget | None:
        task_budget = model_settings.get('anthropic_task_budget')
        if task_budget is None:
            return None

        profile = self.profile
        if not profile.get('anthropic_supports_task_budgets', False):
            raise UserError(
                f'Model {self.model_name!r} does not support `anthropic_task_budget`. '
                'See https://platform.claude.com/docs/en/build-with-claude/task-budgets for the supported models.'
            )

        return task_budget

    @staticmethod
    def _validate_task_budget_vs_context_management(
        model_settings: AnthropicModelSettings,
        context_management: BetaContextManagementConfigParam | None,
    ) -> None:
        # Anthropic rejects requests that combine `task_budget.remaining` with a
        # server-side compaction edit (the API tracks the budget itself). Fail fast with a
        # helpful message rather than letting the API return an opaque 400.
        task_budget = model_settings.get('anthropic_task_budget')
        if task_budget is None or 'remaining' not in task_budget:
            return
        if not isinstance(context_management, dict):
            return
        edits = context_management.get('edits') or ()
        if any(isinstance(e, dict) and e.get('type') == _ANTHROPIC_COMPACT_EDIT_TYPE for e in edits):
            raise UserError(
                '`anthropic_task_budget.remaining` cannot be combined with `AnthropicCompaction`: '
                'Anthropic rejects this combination because server-side compaction tracks the budget itself. '
                'Use `remaining` for client-side budget tracking, or `AnthropicCompaction` '
                'for server-side compaction — not both.'
            )


@dataclass(init=False)
class AnthropicCompaction(AbstractCapability[AgentDepsT]):
    """Compaction capability for Anthropic models.

    Configures automatic context management via Anthropic's `context_management`
    API parameter. Compaction triggers server-side when input tokens exceed
    the configured threshold.

    Example usage:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicCompaction

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[AnthropicCompaction(token_threshold=100_000)],
    )
    ```
    """

    def __init__(
        self,
        *,
        token_threshold: int = 150_000,
        instructions: str | None = None,
        pause_after_compaction: bool = False,
    ) -> None:
        """Initialize the Anthropic compaction capability.

        Args:
            token_threshold: Compact when input tokens exceed this threshold. Minimum 50,000.
            instructions: Custom instructions for the compaction summarization.
            pause_after_compaction: If `True`, the response will stop after the compaction block
                with `stop_reason='compaction'`, allowing explicit handling.
        """
        self.token_threshold = token_threshold
        self.instructions = instructions
        self.pause_after_compaction = pause_after_compaction

    def get_model_settings(self) -> Callable[[RunContext[AgentDepsT]], ModelSettings]:
        edit: dict[str, Any] = {
            'type': _ANTHROPIC_COMPACT_EDIT_TYPE,
            'trigger': {'type': 'input_tokens', 'value': self.token_threshold},
        }
        if self.pause_after_compaction:
            edit['pause_after_compaction'] = True
        if self.instructions is not None:
            edit['instructions'] = self.instructions

        def resolve(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            # Append our edit to any existing context_management the user may have configured,
            # preserving other fields (not just edits).
            existing_cm: dict[str, Any] = {}
            if ctx.model_settings:
                raw_cm = cast(dict[str, Any], ctx.model_settings).get('anthropic_context_management')
                if isinstance(raw_cm, dict):  # pragma: no branch
                    existing_cm = {**cast(dict[str, Any], raw_cm)}
            existing_edits = cast(list[dict[str, Any]], existing_cm.get('edits', []))
            existing_cm['edits'] = [*existing_edits, edit]
            return cast(ModelSettings, {'anthropic_context_management': existing_cm})

        return resolve

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'AnthropicCompaction'


_COMPACTION_TOKEN_KEYS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def _extract_usage_details(response_usage: BetaUsage | BetaMessageDeltaUsage) -> dict[str, int]:
    """Extract Anthropic usage into a flat dict, preserving compaction and advisor iteration totals.

    Anthropic's top-level `input_tokens`/`output_tokens` exclude both compaction and advisor iteration
    usage (see <https://docs.anthropic.com/en/docs/build-with-claude/compaction#understanding-usage>),
    so they're kept as-is here and the iteration totals are recorded under `compaction_*` / `advisor_*`
    keys. `_map_usage` sums the compaction totals back into the request totals at extraction time (which
    also keeps streaming correct: the fixed totals set by the start event survive the merge with delta
    events that only carry the top-level values). Advisor totals are deliberately NOT summed back, since
    advisor tokens are billed at the advisor model's rates, not the executor's, and folding them into the
    request totals would misprice the request via genai-prices.
    """
    details: dict[str, int] = {}
    for key in _COMPACTION_TOKEN_KEYS:
        if isinstance((value := getattr(response_usage, key, None)), int):
            details[key] = value

    # Anthropic bills thinking tokens inside `output_tokens`, so this is a readable subset of the
    # output total rather than an additive one, matching `reasoning_tokens` on OpenAI and
    # `thoughts_tokens` on Google.
    output_tokens_details = response_usage.output_tokens_details
    if output_tokens_details is not None and (thinking_tokens := output_tokens_details.thinking_tokens):
        details['thinking_tokens'] = thinking_tokens

    iterations = response_usage.iterations
    if not iterations:
        return details

    compaction_iterations = [it for it in iterations if it.type == 'compaction']
    advisor_iterations = [it for it in iterations if it.type == 'advisor_message']
    if not compaction_iterations and not advisor_iterations:
        return details

    # Both compaction and advisor iterations are separate from executor turns, so exclude them from
    # the `message_iterations` count.
    details['message_iterations'] = len(iterations) - len(compaction_iterations) - len(advisor_iterations)

    if compaction_iterations:
        details['compaction_iterations'] = len(compaction_iterations)
        for key in _COMPACTION_TOKEN_KEYS:
            if compaction_total := sum(getattr(it, key) for it in compaction_iterations):
                details[f'compaction_{key}'] = compaction_total

    if advisor_iterations:
        details['advisor_iterations'] = len(advisor_iterations)
        # The advisor iteration token fields share the names in `_COMPACTION_TOKEN_KEYS`. These are
        # recorded for observability only; `_map_usage` must not fold them into the request totals.
        for key in _COMPACTION_TOKEN_KEYS:
            if advisor_total := sum(getattr(it, key) for it in advisor_iterations):
                details[f'advisor_{key}'] = advisor_total

    return details


def _map_usage(
    message: BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent,
    provider: str,
    provider_url: str,
    model: str,
    existing_usage: usage.RequestUsage | None = None,
) -> usage.RequestUsage:
    if isinstance(message, BetaMessage):
        response_usage = message.usage
    elif isinstance(message, BetaRawMessageStartEvent):
        if message.message is None:  # pyright: ignore[reportUnnecessaryComparison]
            # On Bedrock the Anthropic SDK drops SSE event types, so Bedrock-only chunks
            # (e.g. `amazon-bedrock-invocationMetrics`) are non-validating `construct_type`d
            # into `BetaRawMessageStartEvent(message=None)`, violating the type annotation.
            # The metrics chunk's token counts duplicate the canonical `message_start` /
            # `message_delta` usage, so dropping them here avoids double-counting.
            return existing_usage or usage.RequestUsage()
        response_usage = message.message.usage
    elif isinstance(message, BetaRawMessageDeltaEvent):
        response_usage = message.usage
    else:
        assert_never(message)

    # In streaming, usage appears in different events.
    # The values are cumulative, meaning new values should replace existing ones entirely.
    details = (existing_usage.details if existing_usage else {}) | _extract_usage_details(response_usage)

    # Anthropic reports top-level tokens excluding compaction iteration usage; add the
    # compaction totals back in so the extracted `RequestUsage` reflects the real request cost.
    usage_for_extraction = dict(details)
    for key in _COMPACTION_TOKEN_KEYS:
        if compaction_value := details.get(f'compaction_{key}'):
            usage_for_extraction[key] = usage_for_extraction.get(key, 0) + compaction_value

    # Note: genai-prices already extracts cache_creation_input_tokens and cache_read_input_tokens
    # from the Anthropic response and maps them to cache_write_tokens and cache_read_tokens
    return usage.RequestUsage.extract(
        dict(model=model, usage=usage_for_extraction),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='anthropic',
        details=details,
    )


def _compaction_provider_details(encrypted_content: str | None) -> dict[str, Any] | None:
    if encrypted_content is not None:
        return {'encrypted_content': encrypted_content}
    return None


@dataclass
class AnthropicStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Anthropic models."""

    _model_name: AnthropicModelName
    _response: _utils.PeekableAsyncStream[BetaRawMessageStreamEvent, AsyncStream[BetaRawMessageStreamEvent]]
    _provider_name: str
    _model_id_namespace: str
    _provider_url: str
    _enabled_server_tool_names: frozenset[str]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        with _map_api_errors(self._model_name, self._model_id_namespace):
            current_block: BetaContentBlock | None = None
            ignored_server_tool_use_indices: set[int] = set()

            builtin_tool_calls: dict[str, NativeToolCallPart] = {}
            async for event in self._response:
                if isinstance(event, BetaRawMessageStartEvent):
                    if event.message is None:  # pyright: ignore[reportUnnecessaryComparison]
                        # See `_map_usage`: Bedrock emits type-less chunks the SDK constructs
                        # as `BetaRawMessageStartEvent(message=None)`. Skip them entirely so we
                        # don't dereference `event.message.id` / `.container` below.
                        continue
                    self._usage = _map_usage(event, self._provider_name, self._provider_url, self._model_name)
                    self.provider_response_id = event.message.id
                    if event.message.container:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['container_id'] = event.message.container.id

                elif isinstance(event, BetaRawContentBlockStartEvent):
                    current_block = event.content_block
                    if isinstance(current_block, BetaTextBlock) and current_block.text:
                        for event_ in self._parts_manager.handle_text_delta(
                            vendor_part_id=event.index, content=current_block.text
                        ):
                            yield event_
                    elif isinstance(current_block, BetaThinkingBlock):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            content=current_block.thinking,
                            signature=current_block.signature,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(current_block, BetaRedactedThinkingBlock):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            id='redacted_thinking',
                            signature=current_block.data,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(current_block, BetaToolUseBlock):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            tool_name=current_block.name,
                            args=cast(dict[str, Any], current_block.input) or None,
                            tool_call_id=current_block.id,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaServerToolUseBlock):
                        if current_block.name not in self._enabled_server_tool_names:
                            # Unlike non-streaming, this cannot pre-scan later result blocks, so a gap in the
                            # enabled-name set would leave their return part orphaned. Result presence is only
                            # advisory: newer web tools can omit paired blocks with `response_inclusion: "excluded"`,
                            # which pydantic-ai does not currently send.
                            ignored_server_tool_use_indices.add(event.index)
                            continue
                        call_part = _map_server_tool_use_block(current_block, self.provider_name)
                        builtin_tool_calls[call_part.tool_call_id] = call_part
                        # In streaming, the block's `input` is empty at start and arrives via
                        # subsequent `BetaInputJSONDelta` events. Emit with `args=None` so the
                        # accumulating JSON deltas can attach as a string; the
                        # `BetaRawContentBlockStopEvent` handler below normalizes the final
                        # value back to the canonical part shape (matching non-streaming).
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=replace(call_part, args=None),
                        )
                    elif isinstance(current_block, BetaWebSearchToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_web_search_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaToolSearchToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_tool_search_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaCodeExecutionToolResultBlock):  # pragma: no cover
                        # Legacy code execution responses used this bare `code_execution_tool_result` shape.
                        # Current code execution tool versions emit the named bash/text-editor blocks below.
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaBashCodeExecutionToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_bash_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaTextEditorCodeExecutionToolResultBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_text_editor_code_execution_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaWebFetchToolResultBlock):  # pragma: lax no cover
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_web_fetch_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaAdvisorToolResultBlock):
                        # The advisor result block arrives fully formed in a single `content_block_start`
                        # (no deltas), same as web search results.
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_advisor_tool_result_block(current_block, self.provider_name),
                        )
                    elif isinstance(current_block, BetaMCPToolUseBlock):
                        call_part = _map_mcp_server_use_block(current_block, self.provider_name)
                        builtin_tool_calls[call_part.tool_call_id] = call_part

                        args_json = call_part.args_as_json_str()
                        # Drop the final `{}}` so that we can add tool args deltas
                        args_json_delta = args_json[:-3]
                        assert args_json_delta.endswith('"tool_args":'), (
                            f'Expected {args_json_delta!r} to end in `"tool_args":`'
                        )

                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index, part=replace(call_part, args=None)
                        )
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args=args_json_delta,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaMCPToolResultBlock):
                        call_part = builtin_tool_calls.get(current_block.tool_use_id)
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=_map_mcp_server_result_block(current_block, call_part, self.provider_name),
                        )
                    elif isinstance(current_block, BetaCompactionBlock):
                        yield self._parts_manager.handle_part(
                            vendor_part_id=event.index,
                            part=CompactionPart(
                                content=current_block.content,
                                provider_name=self.provider_name,
                                provider_details=_compaction_provider_details(current_block.encrypted_content),
                            ),
                        )

                elif isinstance(event, BetaRawContentBlockDeltaEvent):
                    if event.index in ignored_server_tool_use_indices:
                        continue
                    if isinstance(event.delta, BetaTextDelta):
                        for event_ in self._parts_manager.handle_text_delta(
                            vendor_part_id=event.index, content=event.delta.text
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaThinkingDelta):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            content=event.delta.thinking,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaSignatureDelta):
                        for event_ in self._parts_manager.handle_thinking_delta(
                            vendor_part_id=event.index,
                            signature=event.delta.signature,
                            provider_name=self.provider_name,
                        ):
                            yield event_
                    elif isinstance(event.delta, BetaInputJSONDelta):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args=event.delta.partial_json,
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(event.delta, BetaCompactionContentBlockDelta):
                        if event.delta.content:  # pragma: no branch
                            # Re-emit part with updated content; replaces the initial block start part
                            yield self._parts_manager.handle_part(
                                vendor_part_id=event.index,
                                part=CompactionPart(
                                    content=event.delta.content,
                                    provider_name=self.provider_name,
                                    provider_details=_compaction_provider_details(event.delta.encrypted_content),
                                ),
                            )
                    # TODO(Marcelo): We need to handle citations.
                    elif isinstance(event.delta, BetaCitationsDelta):
                        pass

                elif isinstance(event, BetaRawMessageDeltaEvent):
                    self._usage = _map_usage(
                        event, self._provider_name, self._provider_url, self._model_name, self._usage
                    )
                    if raw_finish_reason := event.delta.stop_reason:  # pragma: no branch
                        self.provider_details = self.provider_details or {}
                        self.provider_details['finish_reason'] = raw_finish_reason
                        self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)
                        self.state = 'suspended' if raw_finish_reason == 'pause_turn' else 'complete'
                    if event.delta.stop_details is not None:
                        self.provider_details = self.provider_details or {}
                        if event.delta.stop_details.explanation is not None:
                            self.provider_details['refusal'] = event.delta.stop_details.explanation
                        if event.delta.stop_details.category is not None:
                            self.provider_details['refusal_category'] = event.delta.stop_details.category
                    if event.delta.container:
                        self.provider_details = self.provider_details or {}
                        self.provider_details['container_id'] = event.delta.container.id

                elif isinstance(event, BetaRawContentBlockStopEvent):  # pragma: no branch
                    if event.index in ignored_server_tool_use_indices:
                        ignored_server_tool_use_indices.remove(event.index)
                    elif isinstance(current_block, BetaMCPToolUseBlock):
                        maybe_event = self._parts_manager.handle_tool_call_delta(
                            vendor_part_id=event.index,
                            args='}',
                        )
                        if maybe_event is not None:  # pragma: no branch
                            yield maybe_event
                    elif isinstance(current_block, BetaServerToolUseBlock) and current_block.name in (
                        'tool_search_tool_regex',
                        'tool_search_tool_bm25',
                    ):
                        # The streaming start emitted the part with `args=None`; JSON deltas
                        # have since accumulated as a string. Re-emit with the normalized
                        # cross-provider `ToolSearchArgs` shape so downstream code (history
                        # replay, typed-part dispatch) sees the same structure as the
                        # non-streaming `_process_response` path produces.
                        existing = self._parts_manager.get_part_by_vendor_id(event.index)
                        if isinstance(existing, NativeToolSearchCallPart):  # pragma: no branch
                            yield self._parts_manager.handle_part(
                                vendor_part_id=event.index,
                                part=_finalize_streamed_tool_search_call_part(existing),
                            )
                    current_block = None
                elif isinstance(event, BetaRawMessageStopEvent):  # pragma: no branch
                    current_block = None

    async def close_stream(self) -> None:
        await self._response.source.close()

    @property
    def model_name(self) -> AnthropicModelName:
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


def _build_custom_tool_search_replay_blocks(
    request_part: ToolReturnPart, deferred_tools_active: bool, declared_tool_names: set[str]
) -> tuple[list[BetaToolReferenceBlockParam] | None, str | None]:
    """Tool-search replay payload for the Anthropic `tool_result` block.

    Reads the typed [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent]
    off `part.content` (the local `search_tools` return shape) and unpacks it into:

    * `tool_references`: matched tools, ready to be wrapped in `BetaToolReferenceBlockParam`s.
    * `empty_message`: fallback text to send when no matches were found (Anthropic
      rejects an empty `tool_result` content list).

    Returns `(None, None)` when the current request withholds no tool schemas — nothing for a
    reference to unhide — or when this isn't a typed `search_tools` return; the caller then falls
    through to the default text-formatting path. Fires for any active tool-search strategy (default
    native, named native, or custom callable), so cross-provider history (e.g. a prior local turn on
    Google) gets re-shaped into Anthropic's "client-side" tool_search wire when the current turn runs
    on Anthropic. Both flavors live in one helper because the wire shape is the same: `tool_use` +
    `tool_result` with `tool_reference` content blocks.

    `declared_tool_names` filters the references against the tools currently in
    `function_tools` on the wire — Anthropic rejects `tool_reference` entries for
    tools not in the request's `tools` list (e.g. an MCP server that failed to
    register this turn).
    """
    if not deferred_tools_active:
        return None, None
    if not isinstance(request_part, ToolSearchReturnPart):
        return None, None
    refs = [
        BetaToolReferenceBlockParam(tool_name=match['name'], type='tool_reference')
        for match in request_part.discovered_tools
        if match['name'] in declared_tool_names
    ]
    return refs, request_part.message


def _build_tool_search_replay_block(
    response_part: NativeToolSearchReturnPart, tool_use_id: str, declared_tool_names: set[str]
) -> BetaToolSearchToolResultBlockParam:
    """Reconstruct an Anthropic tool-search result block for history replay.

    Reads the cross-provider
    [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent] off
    `content` and any error fields the parse-time mapper stashed on `provider_details`.

    `declared_tool_names` filters references against the tools currently in
    `function_tools` on the wire — Anthropic rejects `tool_reference` entries for
    tools not in the request's `tools` list (e.g. an MCP server that failed to
    register this turn).
    """
    err = response_part.provider_details or {}
    inner: BetaToolSearchToolResultErrorParam | BetaToolSearchToolSearchResultBlockParam
    if err.get('error_code') is not None:
        # `BetaToolSearchToolResultErrorParam` only carries `error_code` (no
        # `error_message`); the parse-time mapper stashes the message for
        # observability but it doesn't make it back onto the wire.
        inner = BetaToolSearchToolResultErrorParam(
            type='tool_search_tool_result_error',
            error_code=err['error_code'],
        )
    else:
        tool_refs = [
            BetaToolReferenceBlockParam(tool_name=match['name'], type='tool_reference')
            for match in response_part.discovered_tools
            if match['name'] in declared_tool_names
        ]
        inner = BetaToolSearchToolSearchResultBlockParam(
            type='tool_search_tool_search_result',
            tool_references=tool_refs,
        )
    return BetaToolSearchToolResultBlockParam(
        tool_use_id=tool_use_id,
        type='tool_search_tool_result',
        content=inner,
    )


_BUILTIN_TOOL_KIND_BY_SERVER_TOOL_USE_NAME: dict[str, str] = {
    'web_search': WebSearchTool.kind,
    'code_execution': CodeExecutionTool.kind,
    'web_fetch': WebFetchTool.kind,
}


def _anthropic_code_execution_tool_provider_details(
    tool_name: _AnthropicCodeExecutionProviderDetailToolName,
) -> dict[str, _AnthropicCodeExecutionProviderDetailToolName]:
    return {_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL: tool_name}


def _anthropic_caller_provider_details(
    caller: BetaDirectCaller | BetaServerToolCaller | BetaServerToolCaller20260120 | None,
) -> dict[str, Any]:
    # A `direct` caller is the implicit default (the server tool was invoked directly, not from within a
    # code execution tool), so it carries no state worth round-tripping; only non-direct callers (e.g.
    # `code_execution_*`) need to be preserved and re-emitted on replay.
    if caller is None or caller.type == 'direct':
        return {}
    return {_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL: caller.model_dump(mode='json')}


def _add_anthropic_caller_param(
    block: BetaServerToolUseBlockParam | BetaWebSearchToolResultBlockParam | BetaWebFetchToolResultBlockParam,
    part: NativeToolCallPart | NativeToolReturnPart,
) -> None:
    if part.provider_details is None:
        return

    caller = part.provider_details.get(_ANTHROPIC_SERVER_TOOL_CALLER_DETAIL)
    if not _utils.is_str_dict(caller):
        return

    # Re-emit whatever caller `_anthropic_caller_provider_details` stored. That producer only stores
    # genuine SDK callers, so replaying the stored value as-is keeps the read/write paths symmetric and
    # forward-compatible with new caller types, rather than dropping any not in a hardcoded allow-list.
    block['caller'] = caller  # pyright: ignore[reportGeneralTypeIssues]


def _get_anthropic_code_execution_tool_name(
    part: NativeToolCallPart | NativeToolReturnPart,
) -> _AnthropicCodeExecutionToolName:
    if part.provider_details:
        tool_name = part.provider_details.get(_ANTHROPIC_CODE_EXECUTION_TOOL_NAME_DETAIL)
        if isinstance(tool_name, str) and tool_name in _ANTHROPIC_CODE_EXECUTION_TOOL_NAMES:
            return tool_name

    if part.tool_name in ('bash_code_execution', 'text_editor_code_execution'):
        return cast(_AnthropicCodeExecutionToolName, part.tool_name)

    if isinstance(part, NativeToolReturnPart) and _utils.is_str_dict(part.content):
        content_type = part.content.get('type')
        if isinstance(content_type, str):
            if content_type.startswith('bash_code_execution'):
                return 'bash_code_execution'
            elif content_type.startswith('text_editor_code_execution'):
                return 'text_editor_code_execution'

    return 'code_execution'


def _map_code_execution_tool(version: AnthropicCodeExecutionToolVersion) -> BetaToolUnionParam:
    match version:
        case '20250825':
            return BetaCodeExecutionTool20250825Param(name='code_execution', type='code_execution_20250825')
        case '20260120':
            return BetaCodeExecutionTool20260120Param(name='code_execution', type='code_execution_20260120')
        case _:
            assert_never(version)


def _map_advisor_tool(tool: AdvisorTool) -> BetaAdvisorTool20260301Param:
    param = BetaAdvisorTool20260301Param(type='advisor_20260301', name='advisor', model=tool.model)
    if tool.max_uses is not None:
        param['max_uses'] = tool.max_uses
    if tool.max_tokens is not None:
        param['max_tokens'] = tool.max_tokens
    if tool.caching is not None:
        param['caching'] = BetaCacheControlEphemeralParam(type='ephemeral', ttl=tool.caching)
    return param


def _map_server_tool_use_block(item: BetaServerToolUseBlock, provider_name: str) -> NativeToolCallPart:
    tool_args = cast(dict[str, Any], item.input) or None
    if item.name in ('web_search', 'code_execution', 'web_fetch'):
        kind = _BUILTIN_TOOL_KIND_BY_SERVER_TOOL_USE_NAME[item.name]
        part = NativeToolCallPart(
            provider_name=provider_name,
            tool_name=kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details=_anthropic_caller_provider_details(item.caller) or None,
        )
        if item.name == 'code_execution':
            part.otel_metadata = {'code_arg_name': 'code', 'code_arg_language': 'python'}
        return part
    if item.name in ('tool_search_tool_regex', 'tool_search_tool_bm25'):
        # Normalize the wire payload into the cross-provider `{"queries": [...]}` shape
        # carried on the typed call part. bm25 emits `{"query": "..."}`, regex emits
        # `{"pattern": "..."}`. The variant goes on `provider_details` so same-provider
        # replay can pick the original tool name back out.
        normalized_args = _normalize_tool_search_args(tool_args, item.name)
        provider_details: dict[str, Any] = {
            'strategy': 'regex' if item.name == 'tool_search_tool_regex' else 'bm25',
            **_anthropic_caller_provider_details(item.caller),
        }
        return NativeToolSearchCallPart(
            provider_name=provider_name,
            args=normalized_args,
            tool_call_id=item.id,
            provider_details=provider_details,
        )
    if item.name == 'bash_code_execution':
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=CodeExecutionTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details={
                **_anthropic_code_execution_tool_provider_details('bash_code_execution'),
                **_anthropic_caller_provider_details(item.caller),
            },
        )
    if item.name == 'text_editor_code_execution':
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=CodeExecutionTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details={
                **_anthropic_code_execution_tool_provider_details('text_editor_code_execution'),
                **_anthropic_caller_provider_details(item.caller),
            },
        )
    if item.name == 'advisor':
        # The advisor `server_tool_use` block always carries an empty `input` ({}), so `tool_args`
        # is None and `args` stays None on the round-tripped call part.
        return NativeToolCallPart(
            provider_name=provider_name,
            tool_name=AdvisorTool.kind,
            args=tool_args,
            tool_call_id=item.id,
            provider_details=_anthropic_caller_provider_details(item.caller) or None,
        )
    assert_never(item.name)


_USER_ANCHOR_TEXT = '.'
"""The minimal user turn a `system` entry is given when nothing else precedes it.

The API takes a `system` entry only after a user turn (or an assistant turn ending in a server tool
result), so an instruction enqueued after the model's last response has nothing legal to sit behind.
A single period is the cheapest thing that satisfies the rule while asserting nothing on the user's
behalf; the alternative is degrading the instruction to `<system>`-tagged text, which the recordings
show the model reads as a preference it may overrule.
"""


def _add_cache_control_param(
    params: list[BetaContentBlockParam], cache_control: BetaCacheControlEphemeralParam
) -> None:
    """Attach an already-built `cache_control` to the last content block param.

    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for more information.
    """
    if not params:
        raise UserError(
            'CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to. '
            'To cache system instructions or tool definitions, use the `anthropic_cache_instructions` or `anthropic_cache_tool_definitions` settings instead.'
        )

    # Cast needed because BetaContentBlockParam is a union including response Block types (Pydantic models)
    # that don't support dict operations, even though at runtime we only have request Param types (TypedDicts).
    last_param = cast(dict[str, Any], params[-1])
    if last_param['type'] not in _ANTHROPIC_CACHEABLE_PARAM_TYPES:
        raise UserError(f'Cache control not supported for param type: {last_param["type"]}')

    last_param['cache_control'] = cache_control


def _last_message_content(anthropic_messages: list[BetaMessageParam]) -> list[BetaContentBlockParam]:
    """The content blocks of the last rendered message, or an empty list if there's nothing to attach to.

    Only used to give a leading `CachePoint` somewhere to land. A `str` content body can't carry
    `cache_control`, and neither can a conversation that hasn't rendered a message yet, so both return
    empty and let the caller raise the error that explains the situation.
    """
    if not anthropic_messages:
        return []
    content = anthropic_messages[-1]['content']
    # Returned as-is, not copied: the caller attaches `cache_control` by mutating the block in place, so
    # it has to be the list the message actually holds.
    return content if isinstance(content, list) else []


def _drop_unpaired_native_tool_calls(anthropic_messages: list[BetaMessageParam]) -> None:  # noqa: C901
    """Drop native tool calls that Anthropic will reject without a rendered result.

    A result can arrive in a later response, or exist as a `NativeToolReturnPart` but fail to render.
    Anthropic accepts an unpaired native call only while every later message is a user turn containing
    only concurrent client-tool results. That shape is `_is_tool_result_only`, shared with the
    `container_upload` injection so the two cannot drift.
    """
    returned_native_tool_call_ids: set[str] = set()
    for message in anthropic_messages:
        content = message['content']
        assert isinstance(content, list)
        for block in content:
            if isinstance(block, BetaMCPToolResultBlock):
                returned_native_tool_call_ids.add(block.tool_use_id)
            elif is_str_dict(block) and (tool_use_id := block.get('tool_use_id')) is not None:
                returned_native_tool_call_ids.add(tool_use_id)

    suffix_is_tool_result_only = True
    for index in range(len(anthropic_messages) - 1, -1, -1):
        message = anthropic_messages[index]
        content = message['content']
        assert isinstance(content, list)
        kept: list[BetaContentBlockParam] = []
        for block in content:
            # Preserve the union type before `is_str_dict` narrows `block` to `dict[str, Any]`.
            block_param = block
            if isinstance(block, BetaMCPToolResultBlock) or not is_str_dict(block):
                kept.append(block_param)
                continue

            unpaired = (
                block['type'] in ('server_tool_use', 'mcp_tool_use')
                and block['id'] not in returned_native_tool_call_ids
            )
            tool_search = block.get('name') in ('tool_search_tool_bm25', 'tool_search_tool_regex')
            if unpaired and (not suffix_is_tool_result_only or tool_search):
                # Tool search is retried instead of preserved in flight because Bedrock rejects
                # that shape even though the direct Anthropic API accepts other native tools there.
                # Keep a cache boundary on the nearest preceding cacheable block. Moving it
                # forward would cache content the user placed outside the boundary. If no such
                # block exists, the boundary disappears with the block that carried it.
                if (cache_control := block.get('cache_control')) is not None:
                    carriers: list[BetaContentBlockParam] = []
                    for preceding_content in [
                        preceding_message['content'] for preceding_message in anthropic_messages[:index]
                    ] + [kept]:
                        assert isinstance(preceding_content, list)
                        for param in preceding_content:
                            # Preserve the union type before `is_str_dict` narrows `param`.
                            param_value = param
                            if is_str_dict(param) and param['type'] in _ANTHROPIC_CACHEABLE_PARAM_TYPES:
                                carriers.append(param_value)
                    if carriers:
                        _add_cache_control_param(carriers, cache_control)
                continue
            kept.append(block_param)

        if len(kept) != len(content):
            if not kept:
                # Anthropic rejects empty messages; an assistant turn containing only the
                # dropped call has nothing left to send.
                del anthropic_messages[index]
                continue
            message['content'] = kept

        suffix_is_tool_result_only = (
            suffix_is_tool_result_only and message['role'] == 'user' and _is_tool_result_only(kept)
        )


def _anchor_system_messages(anthropic_messages: list[BetaMessageParam]) -> None:
    """Give each `system` section a user turn to follow, if it doesn't already have one.

    This runs *after* `_place_system_messages_before_generation`, on final positions, because an
    entry's predecessor isn't known until the slide has finished with it. Anchoring while mapping
    instead put the anchor between an assistant `tool_use` and the `tool_result` that answers it,
    whenever the two were separated by a system-only request: the slide then moved the entry past the
    result and left the anchor stranded, and the API rejects that with `tool_use ids were found
    without tool_result blocks immediately after`. Deciding here, the same history needs no anchor at
    all — the entry lands behind the tool result, which is a user turn.

    Walking in reverse keeps the untouched indexes valid. A `system` entry behind another only has to
    satisfy the rule as a section, so it defers to the entry ahead of it. An assistant turn ending in
    a server tool result is a legal predecessor too, but it gets an anchor anyway rather than a live
    verification we can't cheaply record: the cost is one wasted turn in a shape that needs a response
    truncated right after a server tool call, and the alternative is an unproven exception.
    """
    for index in range(len(anthropic_messages) - 1, -1, -1):
        if anthropic_messages[index]['role'] != 'system':
            continue
        if index > 0 and anthropic_messages[index - 1]['role'] in ('user', 'system'):
            continue
        anthropic_messages.insert(
            index, BetaMessageParam(role='user', content=[BetaTextBlockParam(text=_USER_ANCHOR_TEXT, type='text')])
        )


def _place_system_messages_before_generation(anthropic_messages: list[BetaMessageParam]) -> None:
    """Move each `system` entry to just before the assistant turn it governs.

    A `{'role': 'system'}` entry is accepted only immediately before an assistant turn or at the end
    of the array. An entry that a *user* turn ended up behind is therefore illegal where it stands —
    which can't be decided while mapping, because a `ModelResponse` whose parts all drop out (an
    empty `TextPart`, an orphan tool-search call) renders to nothing, so reading ahead in the message
    list calls placements legal that the wire rejects.

    Sliding it forward past those user turns rather than degrading it keeps it an operator
    instruction. It costs nothing semantically: an instruction only ever governs the generation that
    follows it, and that's the same assistant turn either way — the entry just stops matching the
    position it was authored at. This is the same trade the mapping loop makes inside a single
    request, where `[user, system, user]` parts render as `user, user, system`.

    Only *user* turns are hopped over, so an earlier instruction stops behind a later one instead of
    overtaking it and inverting the order they were given in. A `system` entry directly after another
    is a placement the API accepts — the group as a whole still precedes the generation.

    Walking in reverse keeps the indexes still to be visited valid, since moving an entry later never
    disturbs anything before it. The mapping loop is the only thing that emits the `system` role, so
    scanning for it finds exactly the entries it wrote, with no bookkeeping to keep in step.
    """
    for index in range(len(anthropic_messages) - 1, -1, -1):
        if anthropic_messages[index]['role'] != 'system':
            continue
        target = index + 1
        while target < len(anthropic_messages) and anthropic_messages[target]['role'] == 'user':
            target += 1
        if target == index + 1:
            continue
        _leave_cache_boundary_behind(anthropic_messages, index)
        anthropic_messages.insert(target - 1, anthropic_messages.pop(index))


def _leave_cache_boundary_behind(anthropic_messages: list[BetaMessageParam], index: int) -> None:
    """Keep a sliding `system` entry's cache boundary at the position it was authored at.

    A `CachePoint` that ends a request lands on the entry's final block, because the entry renders
    after the request's user blocks even though the instruction was authored before them. Carrying it
    along on the slide would drag the boundary over the turns the entry hops and cache content nobody
    marked, so the boundary stays and the entry travels without it. What that costs is the instruction
    itself, which is the one thing that can't be both inside the boundary and after the turns it now
    follows; the cached prefix stays a prefix of what was authored before the marker.

    The block it moves to is the end of the message the entry currently sits behind — which the entry
    is about to stop sitting behind — so an unmoved entry never reaches here and never loses its
    boundary. `_add_cache_control_param` raises if that block can't carry one, or if there's nothing
    behind the entry at all, which is the same `CachePoint` with nothing to attach to that the mapping
    loop raises on.
    """
    content = anthropic_messages[index]['content']
    if not isinstance(content, list):  # pragma: no cover
        return
    last_block = cast(dict[str, Any], content[-1])
    if (cache_control := last_block.pop('cache_control', None)) is None:
        return
    _add_cache_control_param(_last_message_content(anthropic_messages[:index]), cache_control)


def _normalize_tool_search_args(tool_args: dict[str, Any] | None, tool_name: str) -> ToolSearchArgs:
    """Normalize an Anthropic `tool_search_tool_*.input` payload into `ToolSearchArgs`.

    Wire keys differ by variant: bm25 emits `{"query": "..."}`, regex emits
    `{"pattern": "..."}`. Both map to the cross-provider canonical
    `{"queries": [...]}` shape. Used by both the non-streaming `_process_response` path
    (which has the full input at once) and the streaming finalizer (which has only
    accumulated JSON-string deltas to reparse at content_block_stop time).
    """
    wire_key = 'pattern' if tool_name == 'tool_search_tool_regex' else 'query'
    raw = (tool_args or {}).get(wire_key, '')
    queries = [raw] if isinstance(raw, str) else []
    return {'queries': queries}


def _finalize_streamed_tool_search_call_part(part: NativeToolSearchCallPart) -> NativeToolSearchCallPart:
    """Finalize a streamed tool-search call's args.

    Converts a `NativeToolSearchCallPart` whose `args` accumulated as a JSON string
    (via `BetaInputJSONDelta`) into the canonical dict shape produced by the
    non-streaming path. Already-canonical dict args (typed `ToolSearchArgs`) pass
    through unchanged; `None` finalizes to an empty `queries` list.
    """
    if isinstance(part.args, dict):
        return part
    if isinstance(part.args, str):
        try:
            parsed: dict[str, Any] | None = cast(dict[str, Any], pydantic_core.from_json(part.args))
        except ValueError:  # pragma: no cover
            # Malformed partial args.
            parsed = None
    else:
        parsed = None
    # `_map_server_tool_use_block` stashes the variant on `provider_details['strategy']`
    # at content_block_start; map it back to the wire-shape tool name so the variant's
    # input key (`pattern` vs `query`) is honored.
    strategy = (part.provider_details or {}).get('strategy')
    tool_name = 'tool_search_tool_regex' if strategy == 'regex' else 'tool_search_tool_bm25'
    return replace(part, args=_normalize_tool_search_args(parsed, tool_name))


def _map_tool_search_tool_result_block(
    item: BetaToolSearchToolResultBlock, provider_name: str
) -> NativeToolSearchReturnPart:
    """Map a tool-search result block into a typed [`NativeToolSearchReturnPart`][pydantic_ai.messages.NativeToolSearchReturnPart].

    Writes a cross-provider [`ToolSearchReturnContent`][pydantic_ai.messages.ToolSearchReturnContent] to `content` (no
    provider-shape smuggling) and stashes the Anthropic-specific error fields on
    `provider_details` so we can faithfully reconstruct the original block on replay.
    """
    block = item.content
    provider_details: dict[str, Any] | None = None
    matches: list[ToolSearchMatch] = []
    if block.type == 'tool_search_tool_search_result':
        matches = [{'name': ref.tool_name} for ref in block.tool_references]
    else:  # tool_search_tool_result_error
        provider_details = {'error_code': block.error_code, 'error_message': block.error_message}
    return NativeToolSearchReturnPart(
        provider_name=provider_name,
        content={'discovered_tools': matches},
        tool_call_id=item.tool_use_id,
        provider_details=provider_details,
    )


web_search_tool_result_content_ta: TypeAdapter[BetaWebSearchToolResultBlockContent] = TypeAdapter(
    BetaWebSearchToolResultBlockContent
)


def _map_web_search_tool_result_block(item: BetaWebSearchToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=WebSearchTool.kind,
        content=web_search_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=_anthropic_caller_provider_details(item.caller) or None,
    )


code_execution_tool_result_content_ta: TypeAdapter[BetaCodeExecutionToolResultBlockContent] = TypeAdapter(
    BetaCodeExecutionToolResultBlockContent
)


def _map_code_execution_tool_result_block(
    item: BetaCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


bash_code_execution_tool_result_content_ta: TypeAdapter[BashCodeExecutionToolResultBlockContent] = TypeAdapter(
    BashCodeExecutionToolResultBlockContent
)


def _map_bash_code_execution_tool_result_block(
    item: BetaBashCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=bash_code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=_anthropic_code_execution_tool_provider_details('bash_code_execution'),
    )


text_editor_code_execution_tool_result_content_ta: TypeAdapter[TextEditorCodeExecutionToolResultBlockContent] = (
    TypeAdapter(TextEditorCodeExecutionToolResultBlockContent)
)


def _map_text_editor_code_execution_tool_result_block(
    item: BetaTextEditorCodeExecutionToolResultBlock, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=text_editor_code_execution_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=_anthropic_code_execution_tool_provider_details('text_editor_code_execution'),
    )


def _map_web_fetch_tool_result_block(item: BetaWebFetchToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=WebFetchTool.kind,
        # Store just the content field (BetaWebFetchBlock) which has {content, type, url, retrieved_at}
        content=item.content.model_dump(mode='json'),
        tool_call_id=item.tool_use_id,
        provider_details=_anthropic_caller_provider_details(item.caller) or None,
    )


advisor_tool_result_content_ta: TypeAdapter[AdvisorToolResultBlockContent] = TypeAdapter(AdvisorToolResultBlockContent)


def _map_advisor_tool_result_block(item: BetaAdvisorToolResultBlock, provider_name: str) -> NativeToolReturnPart:
    # Store the discriminated content dict verbatim (plaintext, redacted, or error). Round-tripping it
    # unchanged is what lets the server decrypt a redacted advisor result on the next turn — the client
    # can't read it, and no variant needs special-casing.
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=AdvisorTool.kind,
        content=advisor_tool_result_content_ta.dump_python(item.content, mode='json'),
        tool_call_id=item.tool_use_id,
    )


def _map_mcp_server_use_block(item: BetaMCPToolUseBlock, provider_name: str) -> NativeToolCallPart:
    return NativeToolCallPart(
        provider_name=provider_name,
        tool_name=':'.join([MCPServerTool.kind, item.server_name]),
        args={
            'action': 'call_tool',
            'tool_name': item.name,
            'tool_args': cast(dict[str, Any], item.input),
        },
        tool_call_id=item.id,
    )


def _map_mcp_server_result_block(
    item: BetaMCPToolResultBlock, call_part: NativeToolCallPart | None, provider_name: str
) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        provider_name=provider_name,
        tool_name=call_part.tool_name if call_part else MCPServerTool.kind,
        content=item.model_dump(mode='json', include={'content', 'is_error'}),
        tool_call_id=item.tool_use_id,
    )


def _effective_thinking_type(
    anthropic_thinking: BetaThinkingConfigParam | None,
    unified_thinking: ThinkingLevel | None,
    *,
    supports_adaptive_thinking: bool,
) -> Literal['enabled', 'adaptive'] | None:
    """Resolve the effective Anthropic thinking type for the output-tool and tool-forcing guards.

    Extended thinking (`{'type': 'enabled'}`) is incompatible with forced tool use and Tool Output;
    adaptive thinking is compatible with both. Unified thinking maps to `adaptive` when the profile
    advertises it and to `enabled` otherwise — the same mapping `_translate_thinking` uses to build
    the wire payload. Returns `'enabled'`, `'adaptive'`, or `None` when thinking is off.
    """
    if anthropic_thinking:
        thinking_type = anthropic_thinking.get('type')
        return thinking_type if thinking_type in ('enabled', 'adaptive') else None
    if unified_thinking:
        return 'adaptive' if supports_adaptive_thinking else 'enabled'
    return None


def _support_tool_forcing(
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
    resolved_tool_choice: ResolvedToolChoice,
    context: str = 'forcing specific tools',
    *,
    supports_forced_tool_choice: bool = True,
    supports_adaptive_thinking: bool = False,
) -> bool:
    """A forced `tool_choice` ('required'/specific tool) isn't always compatible with Anthropic.

    Extended thinking rejects forcing (adaptive thinking does not), and some models
    (e.g. Claude Fable 5, Claude Mythos Preview) reject it unconditionally.
    We only raise an error if the user explicitly set a forcing value; a forcing value that came
    from the `tool_choice` resolution logic falls back softly to 'auto'.
    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#forcing-tool-use
    """
    # `params.thinking` is checked too since Model.prepare_request strips unified `thinking` from
    # model_settings into params.thinking before the tool-choice helpers run.
    thinking_type = _effective_thinking_type(
        model_settings.get('anthropic_thinking'),
        model_request_parameters.thinking or model_settings.get('thinking'),
        supports_adaptive_thinking=supports_adaptive_thinking,
    )

    if supports_forced_tool_choice and thinking_type != 'enabled':
        return True

    explicit_choice = model_settings.get('tool_choice')
    if explicit_choice == 'required' or isinstance(explicit_choice, list):
        if not supports_forced_tool_choice:
            raise UserError(f"Anthropic does not support {context} for this model. Use `tool_choice='auto'`.")
        adaptive_hint = (
            " Alternatively, `anthropic_thinking={'type': 'adaptive'}` supports forcing."
            if supports_adaptive_thinking
            else ''
        )
        raise UserError(
            f'Anthropic does not support {context} with extended thinking. '
            f"Disable thinking or use `tool_choice='auto'`.{adaptive_hint}"
        )

    if resolved_tool_choice == 'required' or isinstance(resolved_tool_choice, tuple):
        return False

    return True
