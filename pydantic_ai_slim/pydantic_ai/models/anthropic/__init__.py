from __future__ import annotations as _annotations

import base64
import warnings
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, assert_never, cast, overload

from typing_extensions import TypedDict

from ... import UnexpectedModelBehavior, _utils, usage
from ..._context_size import mark_context_input_tokens
from ..._http import to_httpx2_timeout
from ..._run_context import RunContext
from ..._thinking_part import REDACTED_THINKING_PART_IDS, render_replayed_thinking
from ..._tool_search import _NO_MATCHES_MESSAGE  # pyright: ignore[reportPrivateUsage]
from ..._utils import guard_tool_call_id as _guard_tool_call_id, is_str_dict
from ...capabilities.abstract import AbstractCapability
from ...exceptions import UserError
from ...messages import (
    ERROR_OUTCOMES,
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
    NativeToolSearchReturnPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
    _is_request_scoped,  # pyright: ignore[reportPrivateUsage]
    is_multi_modal_content,
)
from ...native_tools import (
    SUPPORTED_NATIVE_TOOLS,
    AbstractNativeTool,
    AdvisorTool,
    CodeExecutionTool,
    MCPServerTool,
    MemoryTool,
    WebFetchTool,
    WebSearchTool,
)
from ...native_tools._tool_search import ToolSearchTool
from ...output import StructuredOutputMode
from ...profiles import ModelProfile, ModelProfileSpec, merge_profile
from ...profiles.anthropic import (
    ANTHROPIC_SAMPLING_PARAMS,
    ANTHROPIC_THINKING_BUDGET_MAP,
    AnthropicCodeExecutionToolVersion,
    AnthropicEffort,
    AnthropicModelProfile,
    resolve_anthropic_effort,
)
from ...providers import Provider, infer_provider
from ...providers._bedrock_model_names import (
    bedrock_claude_cache_retentions,
    bedrock_claude_supports_automatic_caching,
    bedrock_claude_supports_mid_conversation_effort,
    bedrock_lacks_claude_structured_output,
)
from ...providers.anthropic import AsyncAnthropicClient
from ...providers.gateway import is_gateway_provider
from ...settings import CacheConfig, CacheRetention, CacheSetting, ModelSettings, ThinkingLevel, merge_model_settings
from ...tools import AgentDepsT, ToolDefinition
from ...toolsets._tool_search import discovered_tool_names_in_order
from .. import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
    ToolVisibility,
    _declare_lone_newcomers,  # pyright: ignore[reportPrivateUsage]
    _effort_window_baseline,  # pyright: ignore[reportPrivateUsage]
    _request_scoped_tail_texts,  # pyright: ignore[reportPrivateUsage]
    _sendable_request_parts,  # pyright: ignore[reportPrivateUsage]
    _standing_system_prompt_count,  # pyright: ignore[reportPrivateUsage]
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unprepared_part_error,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
    _unprojected_instruction_delta_error,  # pyright: ignore[reportPrivateUsage]
    _without_model_settings_deltas,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from .._anthropic_schema_limits import default_output_mode_within_limits, limit_exceeded_error, model_exceeded_limit
from .._decode_errors import check_json_response, map_decode_errors
from .._prompt_cache import (
    excess_cache_points,
    instruction_cache_boundaries,
    previous_tail_needing_breakpoint,
    split_cache_setting,
    split_instruction_cache_points,
)
from .._sdk_retries import with_max_retries
from .._tool_choice import resolve_tool_choice, support_tool_forcing, tool_forcing_unavailable_reason


def _native_tool_is_a_tools_entry(tool: AbstractNativeTool) -> bool:
    """Whether `_add_native_tools` sends this native tool as an entry in the request's `tools`.

    An MCP server goes in `mcp_servers` instead, and a custom tool search strategy is served by the
    `search_tools` function tool, which `function_tools` already carries.
    """
    if isinstance(tool, MCPServerTool):
        return False
    return not (isinstance(tool, ToolSearchTool) and tool.strategy == 'custom')


def _revealed_deferred_tool_order(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
) -> list[str]:
    """Order revealed non-corpus deferred entries at the tail by their first reveal in history.

    Deferred entries are outside Anthropic's cache key, so this is a determinism rule, not a cache rule.
    A mid-run newcomer (`introduced_tool_names`) is declared deferred for the same reason and joins them.

    Without a tool-search surface, `resolve_request_tools` declares a deferred tool from the first request,
    so its reveal leaves it where earlier requests sent it: moving it would reorder entries already sent.
    """
    tool_defs = model_request_parameters.tool_defs
    pre_declared = not any(isinstance(tool, ToolSearchTool) for tool in model_request_parameters.native_tools)
    return [
        name
        for name in discovered_tool_names_in_order(messages)
        if name in tool_defs
        and (
            name in model_request_parameters.introduced_tool_names
            or (tool_defs[name].defer_loading and not pre_declared)
        )
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
        DEFAULT_TIMEOUT,
        NOT_GIVEN,
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
        BetaAdvisorToolResultBlock,
        BetaAdvisorToolResultBlockParam,
        BetaBase64PDFSourceParam,
        BetaBashCodeExecutionToolResultBlock,
        BetaBashCodeExecutionToolResultBlockParam,
        BetaCacheControlEphemeralParam,
        BetaCitationsConfigParamParam,
        BetaCodeExecutionToolResultBlock,
        BetaCodeExecutionToolResultBlockParam,
        BetaCompactionBlock,
        BetaCompactionBlockParam,
        BetaCompactionConfigParam,
        BetaContainerParams,
        BetaContainerUploadBlockParam,
        BetaContentBlockParam,
        BetaContextManagementConfigParam,
        BetaDiagnosticsParam,
        BetaFileDocumentSourceParam,
        BetaFileImageSourceParam,
        BetaImageBlockParam,
        BetaJSONOutputFormatParam,
        BetaMCPToolResultBlock,
        BetaMCPToolUseBlock,
        BetaMCPToolUseBlockParam,
        BetaMemoryTool20250818Param,
        BetaMessage,
        BetaMessageParam,
        BetaMessageTokensCount,
        BetaMetadataParam,
        BetaOutputConfigParam,
        BetaPlainTextSourceParam,
        BetaRawMessageStartEvent,
        BetaRawMessageStreamEvent,
        BetaRedactedThinkingBlock,
        BetaRedactedThinkingBlockParam,
        BetaRequestDocumentBlockParam,
        BetaRequestMCPServerToolConfigurationParam,
        BetaRequestMCPServerURLDefinitionParam,
        BetaRequestMCPToolResultBlockParam,
        BetaServerToolUseBlock,
        BetaServerToolUseBlockParam,
        BetaTextBlock,
        BetaTextBlockParam,
        BetaTextEditorCodeExecutionToolResultBlock,
        BetaTextEditorCodeExecutionToolResultBlockParam,
        BetaThinkingBlock,
        BetaThinkingBlockParam,
        BetaThinkingConfigParam,
        BetaTokenTaskBudgetParam,
        BetaToolChoiceParam,
        BetaToolParam,
        BetaToolSearchToolBm25_20251119Param,
        BetaToolSearchToolRegex20251119Param,
        BetaToolSearchToolResultBlock,
        BetaToolSearchToolResultBlockParam,
        BetaToolUnionParam,
        BetaToolUseBlock,
        BetaToolUseBlockParam,
        BetaWebFetchTool20250910Param,
        BetaWebFetchTool20260209Param,
        BetaWebFetchToolResultBlock,
        BetaWebFetchToolResultBlockParam,
        BetaWebSearchTool20250305Param,
        BetaWebSearchTool20260209Param,
        BetaWebSearchToolResultBlock,
        BetaWebSearchToolResultBlockParam,
        BetaWebSearchToolResultBlockParamContentParam,
        beta_tool_result_block_param,
    )
    from anthropic.types.beta.beta_advisor_tool_result_block_param import Content as AdvisorToolResultBlockParamContent
    from anthropic.types.beta.beta_user_location_param import BetaUserLocationParam
    from anthropic.types.beta.beta_web_fetch_tool_result_block_param import (
        Content as WebFetchToolResultBlockParamContent,
    )
    from anthropic.types.model_param import ModelParam

    from . import _batch
    from ._bedrock_count_tokens import BEDROCK_ANTHROPIC_VERSION, count_tokens_via_bedrock
    from ._cache_control import (
        ANTHROPIC_CACHEABLE_PARAM_TYPES,
        add_cache_control_param,
        blocks_before_request_scoped_text,
        is_effort_entry,
        last_cacheable_block_has_cache_control,
        last_message_content,
        raise_earlier_breakpoint_ttls,
    )
    from ._citations import (
        check_citations_tool_choice,
        citations_fit_text,
        citations_included,
        citations_provider_details,
        fetched_page,
        fetched_pages_from_history,
        map_citation_for_replay,
        map_citations,
        with_document_citations,
    )
    from ._compaction import (
        ON_DEMAND_COMPACTION_BETA,
        compaction_instructions,
        is_signed_compaction,
        settle_output_config,
        settle_tool_choice,
        without_unanswered_tool_calls,
    )
    from ._containers import is_tool_result_only
    from ._errors import is_expired_container_error, map_api_errors
    from ._message_passes import (
        anchor_system_messages,
        drop_unpaired_native_tool_calls,
        end_tool_results_before_system_messages,
        place_system_messages_before_generation,
    )
    from ._native_tools import (
        add_anthropic_caller_param,
        advisor_tool_result_content_ta,
        bash_code_execution_tool_result_content_ta,
        build_code_execution_tool_result_block,
        build_custom_tool_search_replay_blocks,
        build_tool_search_replay_block,
        code_execution_tool_result_content_ta,
        get_anthropic_code_execution_tool_name,
        map_advisor_tool,
        map_advisor_tool_result_block,
        map_bash_code_execution_tool_result_block,
        map_code_execution_tool,
        map_code_execution_tool_result_block,
        map_mcp_server_result_block,
        map_mcp_server_use_block,
        map_server_tool_use_block,
        map_text_editor_code_execution_tool_result_block,
        map_tool_search_tool_result_block,
        map_web_fetch_tool_result_block,
        map_web_search_tool_result_block,
        text_editor_code_execution_tool_result_content_ta,
        web_search_tool_result_content_ta,
    )
    from ._response import (
        FINISH_REASON_MAP,
        compaction_provider_details,
        last_pass_input_tokens,
        map_usage,
        mark_continuation_context_input_tokens,
        report_input_transformations,
    )
    from ._streaming import AnthropicEventStream, AnthropicStreamedResponse
    from ._thinking_binding import (
        ANTHROPIC_THINKING_BINDING_BETA,
        AnthropicStaleThinkingBlockWarning,
        drop_stale_thinking_blocks,
        is_stale_thinking_block_error,
        mark_history_to_drop_stale_thinking_blocks_for_count_tokens,
        thinking_with_stale_block_history,
        warn_stale_thinking_block_recovery,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `anthropic` to use the Anthropic model, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error

if TYPE_CHECKING:
    from ..batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest

__all__ = (
    'AnthropicCompaction',
    'AnthropicModel',
    'AnthropicModelName',
    'AnthropicModelSettings',
    'AnthropicStaleThinkingBlockWarning',
    'AnthropicStreamedResponse',
    'AnthropicTaskBudget',
    'DEPRECATED_ANTHROPIC_MODELS',
    'LatestAnthropicModelNames',
    'advisor_tool_result_content_ta',
    'bash_code_execution_tool_result_content_ta',
    'code_execution_tool_result_content_ta',
    'text_editor_code_execution_tool_result_content_ta',
    'web_search_tool_result_content_ta',
)

# Fast mode is not available on any Bedrock transport, so it goes in `_FAST_MODE_UNSUPPORTED_CLIENTS`.
_CACHE_SETTINGS_KEYS = (
    'anthropic_cache',
    'anthropic_cache_instructions',
    'anthropic_cache_tool_definitions',
    'anthropic_cache_messages',
)
_FAST_MODE_UNSUPPORTED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
# The Message Batches API is the Claude API's own. Bedrock and Vertex batch through their platforms' batch inference
# jobs instead, which take files in cloud storage rather than a request list.
_BATCH_UNSUPPORTED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
# `_THINKING_BINDING_UNSUPPORTED_CLIENTS` is a client-class boundary, not a base-URL one: a plain
# `AsyncAnthropic` carrying a proxy or Pydantic AI Gateway base URL keeps the flag. Anthropic documents
# the binding beta for Amazon Bedrock and Google Cloud, and the SDK carries it through the legacy
# Bedrock `anthropic_beta` body field and Vertex beta routes. Verified live on the Bedrock Messages API
# (`AsyncAnthropicBedrockMantle`) with `anthropic.claude-opus-5-5`: it takes `block_binding` under this beta in
# the `anthropic-beta` header, and rejects a stale block or drops it as the Claude API does
# (https://github.com/pydantic/pydantic-ai/issues/11481). Foundry has no published support for this beta, so it
# stays excluded until the provider contract changes or a live request verifies it.
_THINKING_BINDING_UNSUPPORTED_CLIENTS = (AsyncAnthropicFoundry,)
# The legacy Bedrock InvokeModel API (`AsyncAnthropicBedrock`) accepts the `bm25` tool-search variant wherever it
# accepts `regex`, but an unset strategy resolves to `regex` there: it was the only variant Bedrock accepted,
# and resolving to `bm25` would change the search algorithm for every request that doesn't set one. For the same
# reason, recorded `bm25` calls replay as `regex` there unless the request asks for `bm25`. The other transports
# (Vertex, Foundry, and the Bedrock Messages API's `AsyncAnthropicBedrockMantle`) default to `bm25`.
_REGEX_DEFAULT_TOOL_SEARCH_CLIENTS = (AsyncAnthropicBedrock,)
# Anthropic web-tool availability is client/platform-specific:
# * `AsyncAnthropicBedrock` is the legacy Amazon Bedrock InvokeModel client, where web search/fetch
#   are unavailable.
# * `AsyncAnthropicBedrockMantle` is the Bedrock Messages API. Verified live, it rejects `web_search_20250305`,
#   `web_search_20260209`, `web_fetch_20250910` and `web_fetch_20260209` with a 400 ("tool type ... is not
#   supported for this model") on every Claude model it serves. https://github.com/pydantic/pydantic-ai/issues/11481
# * Vertex AI supports basic web search only.
# * Direct Anthropic API, Claude Platform on AWS (`AsyncAnthropicAWS`, an `AsyncAnthropic` subclass), and
#   Microsoft Foundry support dynamic-filtering web tools on supported model profiles.
_WEB_SEARCH_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle)
_WEB_FETCH_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle, AsyncAnthropicVertex)
_WEB_TOOLS_20260209_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle, AsyncAnthropicVertex)
# Anthropic Files API file IDs and code execution container IDs aren't sent through the Bedrock, Vertex AI and Foundry
# clients. See `AnthropicModelProfile.anthropic_accepts_file_and_container_ids`.
_FILE_AND_CONTAINER_IDS_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicVertex, AsyncAnthropicFoundry)
# Server-side tools whose results carry their state inline, so they replay to any client that accepts the
# response's stateless data; see `AnthropicModel._replays_native_tool_part`. `web_search_tool_result` is the
# kind older histories recorded web search results under.
_STATELESS_NATIVE_TOOL_KINDS = frozenset(
    {WebSearchTool.kind, 'web_search_tool_result', WebFetchTool.kind, ToolSearchTool.kind, AdvisorTool.kind}
)
# The advisor tool is documented for the direct Anthropic API and Claude Platform on AWS only. Claude Platform on AWS
# is `AsyncAnthropicAWS`, an `AsyncAnthropic` subclass, so it keeps the tool. The Bedrock Messages API
# (`AsyncAnthropicBedrockMantle`, which isn't an `AsyncAnthropicBedrock` subclass) is listed explicitly: verified
# live, it rejects `advisor_20260301` with a 400 ("tool type ... is not supported for this model") on Claude Opus
# 5.5, Sonnet 5 and Haiku 4.5. https://github.com/pydantic/pydantic-ai/issues/11481
_ADVISOR_UNSUPPORTED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicVertex,
    AsyncAnthropicFoundry,
)
# Neither Bedrock API serves Anthropic's code execution tool. Verified live on every Claude model tried, with both
# `code_execution_20250825` and `code_execution_20260120`: the legacy InvokeModel API (`AsyncAnthropicBedrock`) rejects
# them with a 400 ("tool type ... is not supported for this model", or on older models "Input tag ... does not match
# any of the expected tags"), and the Bedrock Messages API (`AsyncAnthropicBedrockMantle`) with a 400 ("tool type ...
# is not supported for this model"). Claude Platform on AWS (`AsyncAnthropicAWS`, an `AsyncAnthropic` subclass) is a
# separate transport and keeps the tool. https://github.com/pydantic/pydantic-ai/issues/11513
_CODE_EXECUTION_UNSUPPORTED_CLIENTS = (AsyncAnthropicBedrock, AsyncAnthropicBedrockMantle)
# Mid-conversation `{'role': 'system'}` messages are published as available on the Anthropic API,
# Amazon Bedrock and Google Cloud, which leaves Microsoft Foundry as the only transport that has to
# fall back to the `<system>`-tagged user rendering. Verified on Bedrock rather than assumed: with
# `us.anthropic.claude-opus-4-8` the entry is served and acted on. An earlier version of this tuple
# excluded Bedrock and Vertex on the strength of a Bedrock test that used `claude-sonnet-5`, a model
# that ignores the entry on *every* transport — which measured the model, not the transport.
_INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS = (AsyncAnthropicFoundry,)
# Per-message effort (`mid-conversation-output-config-2026-07-01`) is documented for the Claude API, Google Cloud
# and Amazon Bedrock. Foundry isn't listed, so it sends effort changes as the request-level setting. Both Bedrock
# APIs serve a narrower model list, which `AnthropicModel.profile` applies. The Bedrock Messages API
# (`AsyncAnthropicBedrockMantle`) accepts the beta on `anthropic.claude-opus-5-5` but rejects a per-message
# `output_config` on `anthropic.claude-opus-5` ("Extra inputs are not permitted"), as the legacy InvokeModel
# API does. https://github.com/pydantic/pydantic-ai/issues/11481
# https://platform.claude.com/docs/en/build-with-claude/effort#per-message-effort-beta
_MID_CONVERSATION_EFFORT_UNSUPPORTED_CLIENTS = (AsyncAnthropicFoundry,)
_MID_CONVERSATION_EFFORT_BETA = 'mid-conversation-output-config-2026-07-01'
# Cache diagnostics are documented as Claude API only. Verified live: the Bedrock InvokeModel and Messages APIs
# both reject the `diagnostics` field with a 400 ("Extra inputs are not permitted").
_CACHE_DIAGNOSTICS_UNSUPPORTED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
# The direct API's `count_tokens` accepts the web search, web fetch, code execution, tool search and advisor tools,
# verified live 2026-10-27 on the Claude 5.5 models. These clients' count endpoints aren't verified to accept them, so
# `anthropic_count_tokens_accepts_server_tools` stays off for them. The legacy Bedrock one rejects `defer_loading`
# with a 400 whether or not the search tool is sent. Claude Platform on AWS (`AsyncAnthropicAWS`, an `AsyncAnthropic`
# subclass) is documented to serve the Claude API's own endpoints, so it keeps the flag on, unverified.
_COUNT_TOKENS_SERVER_TOOLS_UNVERIFIED_CLIENTS = (
    AsyncAnthropicBedrock,
    AsyncAnthropicBedrockMantle,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
# Anthropic's explicit strict-schema limits were verified live on the Messages API and on both Bedrock clients. Vertex
# and Foundry weren't reachable with our credentials, so they keep sending such requests for the API to judge.
_STRICT_SCHEMA_LIMITS_UNVERIFIED_CLIENTS = (AsyncAnthropicFoundry, AsyncAnthropicVertex)

_ANTHROPIC_TASK_BUDGETS_BETA = 'task-budgets-2026-03-13'
_ANTHROPIC_TURN_SCOPED_SYSTEM_BETA = 'mid-conversation-system-clear-at-2026-08-21'
_ANTHROPIC_FILES_API_BETA = 'files-api-2025-04-14'
_ANTHROPIC_COMPACT_EDIT_TYPE = 'compact_20260112'


def _with_max_retries(client: AsyncAnthropicClient, model_settings: ModelSettings) -> AsyncAnthropicClient:
    """The client to send one request with, carrying `ModelSettings['max_retries']` when it's set."""
    if model_settings.get('max_retries') is None or not isinstance(client, AsyncAnthropicBedrock):
        return with_max_retries(client, model_settings)
    # `AsyncAnthropicBedrock.with_options()` doesn't copy `aws_profile`, so its copy would sign requests with
    # the default AWS credential chain instead of the profile the client was built with.
    return with_max_retries(client, model_settings, carry_over={'aws_profile': client.aws_profile})


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

_AssistantContentBlockParam: TypeAlias = (
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
    | BetaRequestMCPToolResultBlockParam
    | BetaCompactionBlockParam
)

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

    Where the model profile's `supports_auto_cache` is `False`, such as on the legacy `AsyncAnthropicBedrock`
    client with models not documented to support it, this falls back to per-block caching on the last user message.
    If the last content block already has `cache_control` from an explicit `CachePoint`, it is preserved.

    If `True`, uses TTL='5m'. You can also specify '5m' or '1h' directly.

    This can be combined with explicit cache breakpoints (`anthropic_cache_instructions`,
    `anthropic_cache_tool_definitions`, `CachePoint`). The automatic breakpoint counts as
    1 of Anthropic's 4 cache point slots; we automatically trim excess explicit breakpoints.
    See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching
    for more information.
    """

    anthropic_cache_diagnostics: bool
    """Request [cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics) on every request.

    When enabled, each request carries a `diagnostics` object naming the most recent Anthropic response in the
    message history as `previous_message_id` (or `null` when there is none, which opts the request in so the next
    one can be compared against it). Anthropic compares the two requests and reports where the prompt prefix
    first diverged. The result is available in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] under the
    `'cache_diagnostics'` key, as Anthropic's `diagnostics` object: `{'cache_miss_reason': {'type': ...,
    'cache_missed_input_tokens': ...}}`, or `{'cache_miss_reason': None}` when the comparison was still running.
    The key is absent when Anthropic found no divergence.

    Diagnostics are free and don't affect caching, but Anthropic retains a short-lived fingerprint (hashes and
    token counts, never prompt content) for each request that carries them, and HIPAA-enabled organizations have
    requests using them rejected. They are only available on the Claude API, so the setting is ignored on Amazon
    Bedrock, Google Cloud Vertex AI and Microsoft Foundry. Defaults to `False`.
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


def _build_extra_body(
    model_settings: AnthropicModelSettings,
    thinking_override: dict[str, object] | None = None,
    *,
    bedrock: bool = False,
) -> object | None:
    """Merge the sampling settings, and a `thinking` object the SDK cannot type, into `extra_body`.

    `anthropic>=1` dropped `temperature`/`top_p`/`top_k` from the `messages.create()` signature, and
    passing one is a `TypeError`. The API still takes them, so they ride in `extra_body` — the route
    the SDK's own migration guide names — rather than being silently dropped. Models that reject them
    never get here: `_drop_unsupported_sampling_settings` has already removed them.

    On the legacy Bedrock client, `anthropic_version` is pinned to `bedrock-2023-05-31`. `anthropic>=2`
    defaults it to the Messages API version `2026-10-21` instead, which Bedrock rejects with `Invalid API
    version` for some Claude models (Sonnet 4.5 and Opus 4.5, though not Haiku 4.5, when verified live).
    The SDK fills the field in only when the body has none. Reported upstream as
    https://github.com/anthropics/anthropic-sdk-python/issues/2031; drop the pin once that is fixed.

    An explicit `extra_body` entry wins over the setting of the same name, preserving the precedence
    the SDK gave it while the parameters were still named arguments.
    """
    fields: dict[str, Any] = {
        setting: value for setting in ANTHROPIC_SAMPLING_PARAMS if (value := model_settings.get(setting)) is not None
    }
    if thinking_override is not None:
        fields['thinking'] = thinking_override
    if bedrock:
        fields['anthropic_version'] = BEDROCK_ANTHROPIC_VERSION
    extra_body = model_settings.get('extra_body')
    if not fields:
        return extra_body
    if not _is_str_mapping(extra_body):
        return fields
    merged = {**fields, **extra_body}
    if thinking_override is not None:
        # The override was built from the effective wire object, including any caller-provided
        # `extra_body['thinking']`, so it must replace that original object to add `block_binding`.
        merged['thinking'] = thinking_override
    return merged


def _extra_body_thinking(model_settings: AnthropicModelSettings) -> dict[str, object] | None:
    """The `thinking` object a caller hand-rolled in `extra_body`, if any.

    `extra_body` is how a caller reaches a field the installed SDK cannot type yet, so it can carry
    `thinking` — including a `block_binding` of their own, which must be honored rather than
    retried over.
    """
    extra_body = model_settings.get('extra_body')
    if not _is_str_mapping(extra_body):
        return None
    thinking = extra_body.get('thinking')
    if not _is_str_mapping(thinking):
        return None
    return dict(thinking)


def _is_str_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether a value is a mapping whose keys are all strings."""
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(key, str) for key in value)  # pyright: ignore[reportUnknownVariableType]


def _effective_thinking(
    model_settings: AnthropicModelSettings, thinking: BetaThinkingConfigParam | Omit
) -> dict[str, object] | Omit:
    """Resolve the thinking object that reaches the wire after `extra_body` precedence."""
    if (caller_thinking := _extra_body_thinking(model_settings)) is not None:
        return caller_thinking
    return OMIT if isinstance(thinking, Omit) else dict(thinking)


_DEFAULT_MAX_TOKENS = 16384
"""The `max_tokens` sent when the request doesn't set one and the model's maximum output is unknown.

It stays under the SDK's non-streaming limit (`_MAX_NON_STREAMING_TOKENS`).
"""

_MAX_NON_STREAMING_TOKENS = 21_333
"""The largest `max_tokens` the Anthropic SDK sends without streaming with its default timeout.

The SDK expects a response to take up to an hour per 128,000 tokens, and requires streaming past 10 minutes.
"""

_LEGACY_DEFAULT_MAX_TOKENS = 4096
"""The default `max_tokens` for models that reject input plus `max_tokens` beyond the context window."""

_MIN_TOKENS_AFTER_THINKING_BUDGET = 4096
"""The room the default `max_tokens` leaves beyond an extended thinking `budget_tokens`."""


class _MessageCreateParams(TypedDict):
    max_tokens: int
    system: str | list[BetaTextBlockParam] | Omit
    messages: list[BetaMessageParam]
    model: str
    tools: list[BetaToolUnionParam] | Omit
    tool_choice: BetaToolChoiceParam | Omit
    mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] | Omit
    output_config: BetaOutputConfigParam | Omit
    cache_control: BetaCacheControlEphemeralParam | Omit
    thinking: BetaThinkingConfigParam | Omit
    stop_sequences: list[str] | Omit
    metadata: BetaMetadataParam | Omit
    context_management: BetaContextManagementConfigParam | Omit
    compaction: BetaCompactionConfigParam | Omit
    container: BetaContainerParams | str | Omit
    service_tier: Literal['auto', 'standard_only'] | Omit
    diagnostics: BetaDiagnosticsParam | Omit
    extra_body: object | None


@dataclass(kw_only=True)
class _PreparedMessagesCreate:
    """The `messages.create` parameters of one request, before any recovery retry changes them.

    `_messages_create` sends them, retrying with a different container or thinking object when the API
    rejects one; a batch request encodes the initial values once, as it gets no retry.
    """

    system_prompt: str | list[BetaTextBlockParam]
    anthropic_messages: list[BetaMessageParam]
    tools: list[BetaToolUnionParam]
    tool_choice: BetaToolChoiceParam | None
    mcp_servers: list[BetaRequestMCPServerURLDefinitionParam]
    output_config: BetaOutputConfigParam | None
    auto_cache_control: BetaCacheControlEphemeralParam | None
    max_tokens: int
    stop_sequences: list[str] | Omit
    effective_thinking: dict[str, object] | Omit
    betas: set[str]
    extra_headers: dict[str, str]
    context_management: BetaContextManagementConfigParam | None
    compaction: BetaCompactionConfigParam | None
    container: BetaContainerParams | str | None
    container_from_history: bool
    initial_thinking: BetaThinkingConfigParam | Omit
    initial_betas: set[str]
    initial_thinking_override: dict[str, object] | None
    initial_effective_thinking: dict[str, object] | Omit


def _default_max_tokens(thinking: dict[str, object] | Omit, profile: AnthropicModelProfile) -> int:
    """The `max_tokens` to send when the request doesn't set one.

    That's the model's maximum output, so responses are only cut off at the model's limit. Above about 21,000 tokens
    the request is streamed behind the scenes (see `_messages_create`). Models that reject input
    plus `max_tokens` beyond the context window keep a lower default, so conversations close to the window still fit.
    Extended thinking's `budget_tokens` counts toward `max_tokens`, and Anthropic rejects a request whose `max_tokens`
    isn't greater than the budget, so a large budget raises a lower default to leave room for the answer.
    """
    if profile.get('anthropic_rejects_max_tokens_beyond_context_window', False):
        default = _LEGACY_DEFAULT_MAX_TOKENS
    else:
        default = profile.get('anthropic_max_output_tokens') or _DEFAULT_MAX_TOKENS
    wire_thinking: dict[str, object] = {} if isinstance(thinking, Omit) else thinking
    budget = wire_thinking.get('budget_tokens') if wire_thinking.get('type') == 'enabled' else None
    return max(default, (budget if isinstance(budget, int) else 0) + _MIN_TOKENS_AFTER_THINKING_BUDGET)


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

    unsupported_settings = frozenset({'frequency_penalty', 'logit_bias', 'presence_penalty', 'seed'})

    supported_tool_deferral_modes = frozenset({'standalone'})
    supported_tool_addition_modes = frozenset({'by_reference'})
    _renders_effort_deltas = True
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
    _honors_compaction_boundaries = True
    # The Messages API has no audio or video block, and its document block takes a PDF or plain text.
    _unsupported_file_inputs = frozenset({'audio', 'video'})
    _supported_document_media_types = frozenset({'application/pdf', 'text/plain'})

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
    def batch_max_requests(self) -> int | None:
        return 100_000 if self.profile.get('anthropic_supports_message_batches', False) else None

    @property
    def batch_max_bytes(self) -> int | None:
        return None if self.batch_max_requests is None else _batch.MAX_BATCH_BYTES

    def _no_batch_api_message(self) -> str:
        if is_gateway_provider(self._provider):
            return f'`AnthropicModel` {self.model_name!r} has no batch API through the Pydantic AI Gateway yet.'
        if isinstance(self.client, _BATCH_UNSUPPORTED_CLIENTS):
            return (
                f'`AnthropicModel` {self.model_name!r} has no batch API on this platform: Message Batches are served '
                'by the Claude API, with the `anthropic:` provider.'
            )
        return super()._no_batch_api_message()

    def batch_request_size(self, request: EncodedBatchRequest) -> int:
        return _batch.request_size(request)

    def _batch_credential(self) -> str:
        return _batch.batch_credential(self)

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        return await _batch.batch_encode(self, request)

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        with map_api_errors(self.model_name):
            return await _batch.batch_create(self, requests)

    async def batch_status(self, batch: Batch) -> Batch:
        with map_api_errors(self.model_name):
            return await _batch.batch_status(self, batch)

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        with map_api_errors(self.model_name):
            return await _batch.batch_results(self, batch)

    async def batch_cancel(self, batch: Batch) -> Batch:
        with map_api_errors(self.model_name):
            return await _batch.batch_cancel(self, batch)

    def batch_decode(self, result: BatchResult, request: BatchRequest) -> ModelResponse:
        return _batch.batch_decode(self, result, request)

    @property
    def model_name(self) -> AnthropicModelName:
        """The model name."""
        return self._model_name

    def _has_provider_cache_settings(self, merged_settings: ModelSettings) -> bool:
        return any(key in merged_settings for key in _CACHE_SETTINGS_KEYS)

    def _effective_cache_settings(self, merged_settings: ModelSettings) -> tuple[CacheSetting | None, ...]:
        # Mirrors `prepare_request` precedence: when any explicit `anthropic_cache*` setting is present, the
        # unified value contributes nothing, since it also adds nothing to the request.
        if self._has_provider_cache_settings(merged_settings):
            settings = cast(AnthropicModelSettings, merged_settings)
            return (
                settings.get('anthropic_cache'),
                settings.get('anthropic_cache_instructions'),
                settings.get('anthropic_cache_tool_definitions'),
                settings.get('anthropic_cache_messages'),
            )
        return super()._effective_cache_settings(merged_settings)

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
        about the transport as much as about the model. Thinking-block binding is likewise narrowed, but
        by client class rather than by base URL: its beta and drop-block retry are verified against
        Anthropic's Messages API, which a proxied or gateway `AsyncAnthropic` still reaches.
        """
        _profile = super().profile
        client = self._provider.client
        supported_native_tools = _profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
        if isinstance(client, _WEB_SEARCH_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {WebSearchTool}
        if isinstance(client, _WEB_FETCH_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {WebFetchTool}
        if isinstance(client, _ADVISOR_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {AdvisorTool}
        if isinstance(client, _CODE_EXECUTION_UNSUPPORTED_CLIENTS):
            supported_native_tools = supported_native_tools - {CodeExecutionTool}
        supports_dynamic_filtering = _profile.get('anthropic_supports_dynamic_filtering', False) and not isinstance(
            client, _WEB_TOOLS_20260209_UNSUPPORTED_CLIENTS
        )
        # Narrowed rather than handled in `_map_system_prompt_part` so `Model.prepare_messages` stays the only place
        # that knows the `<system>`-tagged fallback: where this is `False`, the mid-conversation parts are
        # rewritten before the adapter ever sees them.
        supports_inline_system_prompts = _profile.get('supports_inline_system_prompts', False) and not isinstance(
            client, _INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS
        )
        supports_mid_conversation_effort = (
            _profile.get('supports_mid_conversation_effort', False)
            and not isinstance(client, _MID_CONVERSATION_EFFORT_UNSUPPORTED_CLIENTS)
            and (
                not isinstance(client, AsyncAnthropicBedrock | AsyncAnthropicBedrockMantle)
                or bedrock_claude_supports_mid_conversation_effort(self._model_name)
            )
        )
        _profile = merge_profile(
            _profile,
            AnthropicModelProfile(
                supported_native_tools=supported_native_tools,
                anthropic_supports_dynamic_filtering=supports_dynamic_filtering,
                supports_inline_system_prompts=supports_inline_system_prompts,
                supports_mid_conversation_effort=supports_mid_conversation_effort,
                # A request-scoped entry is a `{'role': 'system'}` entry, so it's only native where the role is.
                supports_request_scoped_system_prompts=_profile.get('supports_request_scoped_system_prompts', False)
                and supports_inline_system_prompts,
                anthropic_binds_thinking_blocks=_profile.get('anthropic_binds_thinking_blocks', False)
                and not isinstance(client, _THINKING_BINDING_UNSUPPORTED_CLIENTS),
                anthropic_accepts_file_and_container_ids=_profile.get('anthropic_accepts_file_and_container_ids', True)
                and not isinstance(client, _FILE_AND_CONTAINER_IDS_UNSUPPORTED_CLIENTS),
            ),
        )
        if isinstance(client, AsyncAnthropicBedrock):
            # AWS grants the 1-hour cache TTL to only a subset of Claude models, and withholds structured output
            # from some that have it on the direct API, matching `BedrockConverseModel`.
            _profile = merge_profile(
                _profile,
                AnthropicModelProfile(
                    supported_cache_retentions=bedrock_claude_cache_retentions(self._model_name),
                    supports_json_schema_output=_profile.get('supports_json_schema_output', False)
                    and not bedrock_lacks_claude_structured_output(self._model_name),
                ),
            )
        return cast(AnthropicModelProfile, _profile)

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        """Set the defaults that depend on the client, before the user's `profile=` applies.

        `anthropic_supports_message_batches` follows the client class: only the Claude API's own client serves
        Message Batches, and not through the Pydantic AI Gateway, which doesn't serve them yet.

        `anthropic_count_tokens_accepts_server_tools` is on for `AsyncAnthropic` clients and off for the platform
        clients whose count endpoints aren't verified to accept server tools, unless the provider's profile already
        sets it.

        The legacy InvokeModel API rejects top-level automatic caching with a 400 on the models it doesn't serve
        from the Messages API infrastructure, so `supports_auto_cache` is narrowed by model on the legacy Bedrock
        client, and `anthropic_cache` and the unified `cache` setting place a breakpoint on the last block there
        instead. Narrowing the default rather than the resolved profile lets a `supports_auto_cache` the user sets
        explicitly win, for models this list doesn't know yet.

        Anthropic's strict-schema limits are only checked on the clients they were verified on, and a user whose
        Vertex or Foundry deployment is known to enforce them can opt in with `anthropic_enforces_strict_schema_limits`.
        """
        client = self._provider.client
        profile = merge_profile(
            profile,
            AnthropicModelProfile(
                anthropic_supports_message_batches=not isinstance(client, _BATCH_UNSUPPORTED_CLIENTS)
                and not is_gateway_provider(self._provider)
            ),
        )
        if 'anthropic_count_tokens_accepts_server_tools' not in profile:
            profile = merge_profile(
                profile,
                AnthropicModelProfile(
                    anthropic_count_tokens_accepts_server_tools=not isinstance(
                        client, _COUNT_TOKENS_SERVER_TOOLS_UNVERIFIED_CLIENTS
                    )
                ),
            )
        if isinstance(client, AsyncAnthropicBedrock) and not bedrock_claude_supports_automatic_caching(
            self._model_name
        ):
            profile = merge_profile(profile, AnthropicModelProfile(supports_auto_cache=False))
        if isinstance(client, _STRICT_SCHEMA_LIMITS_UNVERIFIED_CLIENTS):
            profile = merge_profile(profile, AnthropicModelProfile(anthropic_enforces_strict_schema_limits=False))
        return profile

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

    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        """Return `native_tool`'s restricting settings that Anthropic's web tools don't take: `external_web_access=False`."""
        if isinstance(native_tool, WebSearchTool | WebFetchTool):
            return native_tool.restricting_settings - {'allowed_domains', 'blocked_domains', 'max_uses'}
        return frozenset()

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
        response = await self._messages_create(messages, False, model_settings, model_request_parameters)
        model_response = await self._finish_response(response, messages, model_settings, model_request_parameters)
        mark_continuation_context_input_tokens(messages, model_response)
        return model_response

    async def _finish_response(
        self,
        response: BetaMessage | AnthropicEventStream,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Turn the response to a non-streaming `_messages_create` call into a `ModelResponse`."""
        response = check_json_response(self.model_name, response)
        if isinstance(response, BetaMessage):
            return self._process_response(
                response, model_request_parameters, model_settings, fetched_pages_from_history(messages)
            )
        # The request was streamed behind the scenes, see `_messages_create`. `map_api_errors` maps a transport
        # error that breaks the stream off, both while it opens and while it's read.
        async with response.source:
            streamed_response = await self._process_streamed_response(
                response, model_request_parameters, model_settings, fetched_pages_from_history(messages)
            )
            async for _ in streamed_response:
                pass
        return streamed_response.get()

    async def compact_messages(
        self,
        request_context: ModelRequestContext,
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        """Compact messages with Anthropic's on-demand compaction.

        This sends the conversation to the Messages API with the `compaction` parameter, which summarizes it instead
        of replying. It reads the conversation's prompt cache, like any other request with the same prefix. The
        returned `ModelResponse` holds a single `CompactionPart` with the readable summary, and its signature in
        `provider_details`. Keep it in the message history: later requests send it back in place of the messages
        before it, with the `compact-2026-09-04` beta that it needs. Only Claude models that support on-demand
        compaction accept the block.

        The parameters Anthropic rejects on a compaction request are left out of it: `stop_sequences`, structured
        output's `output_config.format`, the `remaining` task budget and `anthropic_context_management`. A forcing
        `tool_choice` is relaxed to `auto`, and the calls of a trailing response that has no tool results yet are
        dropped. Images and documents in the summarized messages don't survive compaction.

        Args:
            request_context: The model request context containing messages, settings, and parameters.
            instructions: Optional focus for the summary, such as what to keep. Anthropic replaces its whole
                summarization prompt with the instructions it's given, so the focus is sent inside a full
                summarization prompt rather than on its own. On OpenAI, `instructions` replaces the agent's
                instructions for the compaction request instead.

        Returns:
            A `ModelResponse` with a single `CompactionPart` containing the summary.

        Raises:
            UserError: If the model doesn't support on-demand compaction, according to its
                [`anthropic_supports_on_demand_compaction`][pydantic_ai.profiles.anthropic.AnthropicModelProfile.anthropic_supports_on_demand_compaction]
                profile flag.
            UnexpectedModelBehavior: If Anthropic returns no summary, for example when `max_tokens` cuts it off.
        """
        check_allow_model_requests()
        if not self.profile.get('anthropic_supports_on_demand_compaction', False):
            raise UserError(
                f'{self.model_name!r} does not support on-demand compaction. '
                'See https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand for the supported models.'
            )
        model_settings, model_request_parameters = self.prepare_request(
            request_context.model_settings,
            request_context.model_request_parameters,
        )
        model_settings = cast(AnthropicModelSettings, model_settings or {})
        compaction = BetaCompactionConfigParam(type='summarize')
        if instructions and (focus := instructions.strip()):
            compaction['instructions'] = compaction_instructions(focus)
        messages = without_unanswered_tool_calls(list(request_context.messages))
        response = await self._messages_create(
            messages, False, model_settings, model_request_parameters, compaction=compaction
        )
        model_response = await self._finish_response(response, messages, model_settings, model_request_parameters)

        # A summary that's missing or empty is no compaction boundary, so the history it would replace can't be dropped.
        match model_response.parts:
            case [CompactionPart(content=str(content))] if content:
                # The usage is the conversation that was summarized, but the context the response leaves behind
                # is the summary alone, its output: a size trigger mustn't read the old conversation back.
                mark_context_input_tokens(model_response, 0)
                return model_response
            case _:
                stop_reason = (model_response.provider_details or {}).get('finish_reason')
                raise UnexpectedModelBehavior(
                    f'Anthropic returned no compaction summary (stop reason {stop_reason!r}).'
                    + (' Raise `max_tokens` to leave room for the summary.' if stop_reason == 'max_tokens' else '')
                )

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

        return usage.RequestUsage(input_tokens=check_json_response(self.model_name, response).input_tokens)

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
        async with response.source:
            streamed_response = await self._process_streamed_response(
                response, model_request_parameters, model_settings, fetched_pages_from_history(messages)
            )
            yield streamed_response
            mark_continuation_context_input_tokens(messages, streamed_response)

    def _request_thinks(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> bool:
        return _request_thinking_type(self.profile, model_settings, model_request_parameters) is not None

    def _overrides_unified_effort(self, model_settings: Mapping[str, object]) -> bool:
        return model_settings.get('anthropic_effort') is not None

    def _settings_overriding_thinking(self, model_settings: Mapping[str, object]) -> tuple[str, ...]:
        overriding: list[str] = []
        # `_translate_thinking` sends an explicit thinking config instead of the unified one.
        if model_settings.get('anthropic_thinking'):
            overriding.append('anthropic_thinking')
        # `_build_output_config` sends an explicit effort instead of the one an effort-level `thinking` maps to.
        if (
            model_settings.get('anthropic_effort') is not None
            and isinstance(model_settings.get('thinking'), str)
            and self.profile.get('anthropic_supports_effort', False)
        ):
            overriding.append('anthropic_effort')
        return tuple(overriding)

    def _default_structured_output_mode(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> StructuredOutputMode:
        # Extended thinking rejects the forced tool choice Tool Output relies on.
        if (
            model_request_parameters.output_tools
            and _request_thinking_type(self.profile, model_settings, model_request_parameters) == 'enabled'
        ):
            mode: StructuredOutputMode = (
                'native' if self.profile.get('supports_json_schema_output', False) else 'prompted'
            )
        else:
            mode = super()._default_structured_output_mode(model_settings, model_request_parameters)
        return default_output_mode_within_limits(
            self, mode, model_settings, model_request_parameters, self._strict_schema_limit_exceeded
        )

    def _strict_schema_limit_exceeded(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> str | None:
        """The limit the request's strict schemas exceed, where the API enforces Anthropic's limits."""
        return model_exceeded_limit(
            self,
            model_settings,
            model_request_parameters,
            supports_strict_tools=lambda: self.profile.get('supports_json_schema_output', False),
            sent_tools=lambda settings, params: self._tool_choice_declarations(
                cast(AnthropicModelSettings, settings or {}), params
            )[0].values(),
        )

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

        # Resolve 'auto' here so the strict-forcing check below also applies when native mode is reached
        # via the default rather than an explicit `NativeOutput(...)`; `super().prepare_request()` would
        # otherwise only resolve it after `customize_request_parameters()` has already transformed the schema.
        model_request_parameters = model_request_parameters.with_default_output_mode(
            self._default_structured_output_mode(merged, model_request_parameters)
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
        if exceeded := self._strict_schema_limit_exceeded(prepared_settings, model_request_parameters):
            raise limit_exceeded_error(exceeded, self.model_name)
        if profile.get('anthropic_disallows_sampling_settings', False) and prepared_settings:
            filtered: ModelSettings = {**prepared_settings}
            self._drop_unsupported_sampling_settings(filtered)
            prepared_settings = filtered or None
        # Explicit `anthropic_cache*` settings take precedence over the unified `cache` setting.
        if (cache := model_request_parameters.cache) and not any(
            key in (prepared_settings or {}) for key in _CACHE_SETTINGS_KEYS
        ):
            prepared_settings = self._translate_cache(cast(AnthropicModelSettings, prepared_settings or {}), cache)
        return prepared_settings, model_request_parameters

    def _drop_unsupported_sampling_settings(self, model_settings: ModelSettings) -> None:
        dropped = {setting for setting in ANTHROPIC_SAMPLING_PARAMS if setting in model_settings}
        extra_body = model_settings.get('extra_body')
        if is_str_dict(extra_body):
            dropped |= {setting for setting in ANTHROPIC_SAMPLING_PARAMS if setting in extra_body}
            model_settings['extra_body'] = {
                key: value for key, value in extra_body.items() if key not in ANTHROPIC_SAMPLING_PARAMS
            }

        for setting in ANTHROPIC_SAMPLING_PARAMS:
            model_settings.pop(setting, None)

        if dropped:
            ordered = [setting for setting in ANTHROPIC_SAMPLING_PARAMS if setting in dropped]
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
        if thinking is False:
            if self.profile.get('thinking_always_enabled', False):
                # A model that can't turn thinking off thinks as little as it can: adaptively at the lowest effort,
                # which `_build_output_config` adds. A model without an effort setting is left at its default.
                if self.profile.get('anthropic_supports_effort', False):
                    return {'type': 'adaptive'}
                return OMIT  # type: ignore[return-value]
            if self.profile.get('anthropic_disables_thinking_with_between_tools', False):
                return {'type': 'between_tools'}
            if self.profile.get('thinking_enabled_by_default', False):
                # Omitting `thinking` leaves it on for these models.
                return {'type': 'disabled'}
            return OMIT  # type: ignore[return-value]
        if thinking is None:
            return OMIT  # type: ignore[return-value]
        if self.profile.get('anthropic_supports_adaptive_thinking', False):
            return {'type': 'adaptive'}
        return {'type': 'enabled', 'budget_tokens': ANTHROPIC_THINKING_BUDGET_MAP[thinking]}

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        compaction: BetaCompactionConfigParam | None = None,
    ) -> AnthropicEventStream:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        compaction: BetaCompactionConfigParam | None = None,
    ) -> BetaMessage | AnthropicEventStream:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        compaction: BetaCompactionConfigParam | None = None,
    ) -> BetaMessage | AnthropicEventStream:
        """Calls the Anthropic API to create a message.

        This is the last step before sending the request to the API.
        Most preprocessing has happened in `prepare_request()`.

        A non-streaming request is streamed anyway when the SDK requires it for its `max_tokens`, so its response
        can be a stream too.

        With `compaction`, this is a compaction request, which samples no reply: the parameters Anthropic rejects
        on one are settled here, on the request as it would otherwise be sent, so the rest of the request (and the
        prompt cache it reads) stays the same.
        """
        prepared = await self._prepare_messages_create(
            messages, model_settings, model_request_parameters, compaction=compaction
        )
        anthropic_profile = self.profile
        client = _with_max_retries(self.client, model_settings)

        async def create(
            container_param: BetaContainerParams | str | None,
            thinking: BetaThinkingConfigParam | Omit,
            betas: set[str],
            thinking_override: dict[str, object] | None,
        ) -> BetaMessage | AnthropicEventStream:
            max_tokens = prepared.max_tokens

            async def send(stream: bool) -> BetaMessage | AsyncStream[BetaRawMessageStreamEvent]:
                return await client.beta.messages.create(
                    **self._messages_create_params(
                        messages, model_settings, prepared, container_param, thinking, thinking_override
                    ),
                    betas=sorted(betas) or OMIT,
                    stream=stream,
                    timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                    speed=self._effective_speed(model_settings, anthropic_profile),
                    extra_headers=prepared.extra_headers,
                )

            async def open_stream() -> AnthropicEventStream:
                raw_stream = cast(AsyncStream[BetaRawMessageStreamEvent], await send(True))
                event_stream: AnthropicEventStream = _utils.PeekableAsyncStream(raw_stream)
                try:
                    # An error that stops the response, like an expired container, arrives as the first event,
                    # so peek it here to reach the retries below.
                    await event_stream.peek()
                except BaseException:
                    await raw_stream.close()
                    raise
                return event_stream

            if stream:
                return await open_stream()
            # The SDK refuses a non-streaming request it expects to take over 10 minutes, but only with its default
            # timeout. A default `max_tokens` above that limit is the model's maximum output, so stream it when a
            # custom timeout is set too, rather than hold one connection open for the whole response.
            if (
                'max_tokens' not in model_settings
                and max_tokens > _MAX_NON_STREAMING_TOKENS
                and ('timeout' in model_settings or self.client.timeout != DEFAULT_TIMEOUT)
            ):
                return await open_stream()
            try:
                return cast(BetaMessage, await send(False))
            except ValueError as e:
                if 'Streaming is required' not in str(e):
                    raise
                return await open_stream()

        initial_thinking = prepared.initial_thinking
        initial_betas = prepared.initial_betas
        initial_thinking_override = prepared.initial_thinking_override
        retry_container = prepared.container
        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            try:
                return await create(prepared.container, initial_thinking, initial_betas, initial_thinking_override)
            except APIStatusError as error:
                if (
                    is_expired_container_error(error)
                    and prepared.container_from_history
                    and any(
                        is_str_dict(block) and block['type'] == 'container_upload'
                        for message in prepared.anthropic_messages
                        for block in message['content']
                    )
                ):
                    try:
                        return await create(None, initial_thinking, initial_betas, initial_thinking_override)
                    except APIStatusError as fallback_error:
                        # Classify what the fallback hit, not the 500 that caused it, so a stale
                        # thinking block rejected only on this attempt still reaches the retry
                        # below. The history-resolved container is gone, so it stays dropped there.
                        error = fallback_error
                        retry_container = None
                if not is_stale_thinking_block_error(anthropic_profile, prepared.initial_effective_thinking, error):
                    raise error

            result = await create(
                retry_container,
                OMIT,
                prepared.betas | {ANTHROPIC_THINKING_BINDING_BETA},
                drop_stale_thinking_blocks(prepared.effective_thinking),
            )
            warn_stale_thinking_block_recovery(self.model_name)
            return result

    def _messages_create_params(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        prepared: _PreparedMessagesCreate,
        container: BetaContainerParams | str | None,
        thinking: BetaThinkingConfigParam | Omit,
        thinking_override: dict[str, object] | None,
    ) -> _MessageCreateParams:
        return {
            'max_tokens': prepared.max_tokens,
            'system': prepared.system_prompt or OMIT,
            'messages': prepared.anthropic_messages,
            'model': self._model_name,
            'tools': prepared.tools or OMIT,
            'tool_choice': prepared.tool_choice or OMIT,
            'mcp_servers': prepared.mcp_servers or OMIT,
            'output_config': prepared.output_config or OMIT,
            'cache_control': prepared.auto_cache_control or OMIT,
            'thinking': thinking,
            'stop_sequences': prepared.stop_sequences,
            'metadata': model_settings.get('anthropic_metadata', OMIT),
            'context_management': prepared.context_management or OMIT,
            'compaction': prepared.compaction or OMIT,
            'container': container or OMIT,
            'service_tier': _resolve_anthropic_service_tier(model_settings),
            'diagnostics': self._cache_diagnostics(messages, model_settings),
            'extra_body': _build_extra_body(
                model_settings, thinking_override, bedrock=isinstance(self.client, AsyncAnthropicBedrock)
            ),
        }

    async def _prepare_messages_create(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        compaction: BetaCompactionConfigParam | None = None,
    ) -> _PreparedMessagesCreate:
        model_request_parameters = self._declare_newcomers_left_alone(model_settings, model_request_parameters)
        # Native search remains in the stable segment when a reveal lands. Revealed non-corpus deferred
        # entries are then appended in history order; Anthropic excludes them from its cache key.
        messages, effort_request_parameters = self._effort_delta_request(
            messages, model_settings, model_request_parameters
        )
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
        request_scoped_texts = _request_scoped_tail_texts(messages)
        self._apply_per_block_caching_fallback(resolved_cache_ttl, anthropic_messages, request_scoped_texts)
        self._apply_explicit_message_caching(model_settings, anthropic_messages, request_scoped_texts)
        # The API rejects automatic caching when the last block's explicit breakpoint has a different TTL, and
        # ignores it when the TTL is the same, since both mark the same breakpoint. The explicit one wins.
        auto_cache_control = None if last_cacheable_block_has_cache_control(anthropic_messages) else auto_cache_control
        self._limit_cache_points(
            system_prompt, anthropic_messages, tools, automatic_caching=auto_cache_control is not None
        )
        raise_earlier_breakpoint_ttls(system_prompt, anthropic_messages, tools, auto_cache_control)
        output_config = self._build_output_config(effort_request_parameters, model_settings)
        anthropic_profile = self.profile
        thinking = self._translate_thinking(model_settings, model_request_parameters)
        effective_thinking = _effective_thinking(model_settings, thinking)
        betas, extra_headers = self._get_betas_and_extra_headers(
            model_settings, anthropic_profile, messages, model_request_parameters, effective_thinking
        )
        betas.update(native_tool_betas)
        context_management = self._add_compaction_params(messages, betas, model_settings)
        stop_sequences = model_settings.get('stop_sequences', OMIT)
        if compaction is not None:
            betas.add(ON_DEMAND_COMPACTION_BETA)
            context_management = None
            stop_sequences = OMIT
            tool_choice = settle_tool_choice(tool_choice)
            output_config = settle_output_config(output_config)
        self._validate_task_budget_vs_context_management(model_settings, context_management)
        container, container_from_history = self._get_container(messages, model_settings)
        recovery_messages, compaction_boundary = self._stale_thinking_block_recovery_history(messages)
        initial_thinking, initial_betas, initial_thinking_override, initial_effective_thinking = (
            thinking_with_stale_block_history(
                anthropic_profile,
                recovery_messages,
                thinking,
                effective_thinking,
                betas,
                compaction_boundary=compaction_boundary,
            )
        )
        return _PreparedMessagesCreate(
            system_prompt=system_prompt,
            anthropic_messages=anthropic_messages,
            tools=tools,
            tool_choice=tool_choice,
            mcp_servers=mcp_servers,
            output_config=output_config,
            auto_cache_control=auto_cache_control,
            max_tokens=model_settings.get('max_tokens', _default_max_tokens(effective_thinking, anthropic_profile)),
            stop_sequences=stop_sequences,
            effective_thinking=effective_thinking,
            betas=betas,
            extra_headers=extra_headers,
            context_management=context_management,
            compaction=compaction,
            container=container,
            container_from_history=container_from_history,
            initial_thinking=initial_thinking,
            initial_betas=initial_betas,
            initial_thinking_override=initial_thinking_override,
            initial_effective_thinking=initial_effective_thinking,
        )

    def _stale_thinking_block_recovery_history(
        self, messages: list[ModelMessage]
    ) -> tuple[list[ModelMessage], ModelResponse | None]:
        """The wire-visible history and response whose metadata describes the compacted-away request."""
        trimmed_messages = self._trim_before_compaction(messages)
        if trimmed_messages is messages:
            return messages, None

        # The shared trim may re-insert a standing-prompt request before the boundary response, but it
        # removes every earlier response. `input_transformations` on the boundary response describes
        # the request that produced the compaction block, not the next post-compaction request.
        boundary = next(message for message in trimmed_messages if isinstance(message, ModelResponse))
        return trimmed_messages, boundary

    @staticmethod
    def _add_compaction_params(
        messages: list[ModelMessage],
        betas: set[str],
        model_settings: AnthropicModelSettings,
    ) -> BetaContextManagementConfigParam | None:
        """Add the compaction betas, and default context_management, that the messages' CompactionParts need.

        This ensures CompactionParts can be round-tripped even without AnthropicCompaction active. A part
        without `provider_name` is a summary the application wrote, sent as text, so it turns nothing on.
        A signed block from [`compact_messages`][pydantic_ai.models.anthropic.AnthropicModel.compact_messages]
        needs its own beta, and Anthropic rejects it alongside the threshold compaction edit, so it gets no default.
        """
        compaction_parts = [
            part
            for msg in messages
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, CompactionPart) and part.provider_name is not None
        ]
        has_signed_compaction = any(is_signed_compaction(part) for part in compaction_parts)
        if has_signed_compaction:
            betas.add(ON_DEMAND_COMPACTION_BETA)
        has_compaction_parts = any(not is_signed_compaction(part) for part in compaction_parts)
        if has_compaction_parts:
            betas.add('compact-2026-01-12')
        context_management = model_settings.get('anthropic_context_management')
        if has_compaction_parts and not has_signed_compaction and context_management is None:
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
        thinking: dict[str, object] | Omit,
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

        # `thinking.block_binding` is a 400 (`Extra inputs are not permitted`) without its beta,
        # so the beta follows the field rather than the profile flag — a user who sets
        # `block_binding` through `anthropic_thinking` gets it too.
        # Membership, not truthiness: `block_binding: {}` is a valid request meaning "every binding
        # default", and it needs the beta exactly as much as a populated one does.
        if not isinstance(thinking, Omit) and 'block_binding' in thinking:
            betas.add(ANTHROPIC_THINKING_BINDING_BETA)

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
        declared_tool_defs, _ = self._tool_choice_declarations(model_settings, model_request_parameters)
        if self.tool_addition_mode == 'by_reference' and any(
            name in declared_tool_defs
            and name in function_tool_defs
            and model_request_parameters.visibility_of(name) != 'visible'
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolAvailabilityDeltaPart)
            for name in part.tools_added
        ):
            betas.add('mid-conversation-tool-changes-2026-07-01')
        # Co-extensive with the renderer in `_map_system_prompt_part`: `clear_at` is a 400 without the beta. A user
        # who can't use the beta opts out by setting the profile flag to `False`, which falls back to
        # leaving finished request-scoped prompts out of the request.
        if anthropic_profile.get('supports_request_scoped_system_prompts', False) and any(
            _is_request_scoped(part)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            betas.add(_ANTHROPIC_TURN_SCOPED_SYSTEM_BETA)
        # Co-extensive with the renderer in `_map_message`: the messages here are the ones
        # `_effort_delta_request` left, which hold a mid-conversation change only when it renders.
        if any(
            isinstance(part, ModelSettingsDeltaPart)
            and not part.top_level
            and part.settings.get('thinking') is not None
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            betas.add(_MID_CONVERSATION_EFFORT_BETA)

        return betas, extra_headers

    def _messages_use_anthropic_uploaded_file(self, messages: list[ModelMessage]) -> bool:
        """Whether any normalized message contains an Anthropic-hosted `UploadedFile`.

        Used to gate auto-attachment of the `files-api-2025-04-14` beta header.
        Mirrors the per-item `provider_name == self.system` check the request
        mappers (`_map_user_prompt`, `_map_tool_return_part`)
        already perform — so the beta is added exactly when the wire shape
        requires it. `UploadedFile`s for other providers are intentionally
        ignored here; they will raise the existing `UserError` later in the
        request-mapping path, as will Anthropic ones on a client without the Files API.
        """
        if not self.profile.get('anthropic_accepts_file_and_container_ids', True):
            return False
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

    def _cache_diagnostics(
        self, messages: list[ModelMessage], model_settings: AnthropicModelSettings
    ) -> BetaDiagnosticsParam | Omit:
        """Build the `diagnostics` request field when `anthropic_cache_diagnostics` is enabled.

        The comparison baseline is the most recent response from this provider. Anthropic rejects an id that
        doesn't start with `msg_` with a 400, so a response from an Anthropic-compatible endpoint that issues other
        ids (OpenRouter's `gen-...`, for example) opts in without a baseline instead.
        """
        if not model_settings.get('anthropic_cache_diagnostics') or isinstance(
            self.client, _CACHE_DIAGNOSTICS_UNSUPPORTED_CLIENTS
        ):
            return OMIT
        previous_message_id: str | None = None
        for message in reversed(messages):
            if isinstance(message, ModelResponse) and self.accepts_server_state(message):
                if (response_id := message.provider_response_id) and response_id.startswith('msg_'):
                    previous_message_id = response_id
                break
        return BetaDiagnosticsParam(previous_message_id=previous_message_id)

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
        if not self.profile.get('anthropic_accepts_file_and_container_ids', True):
            return None, False
        # On pause_turn continuation, pass just the container ID string to reconnect.
        # Re-passing BetaContainerParams triggers a prefill rejection on some models
        # (e.g. Sonnet 4-6) even though plain string ID works fine.
        if messages and isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended':
            if messages[-1].provider_details:
                return messages[-1].provider_details.get('container_id'), False
            return None, False

        for m in reversed(messages):
            if isinstance(m, ModelResponse) and self.accepts_server_state(m) and m.provider_details:
                if cid := m.provider_details.get('container_id'):
                    return cid, True
        return None, False

    async def _messages_count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> BetaMessageTokensCount:
        # Anthropic docs: https://platform.claude.com/docs/en/api/messages/count_tokens
        # The wire `tools` list keeps client-side tools like `MemoryTool`, which contribute real tokens, and the
        # server tools where `anthropic_count_tokens_accepts_server_tools` says the endpoint takes them. Verified
        # 2026-10-27 on the Claude 5.5 models: with tool search, the advisor tool and its replayed history, or
        # any web search, web fetch or code execution version the SDK lists, the count equals the real request's
        # `input_tokens`, though https://platform.claude.com/docs/en/build-with-claude/token-counting says the
        # endpoint rejects every server tool but the advisor. Without a server tool, a request whose function tools
        # are all deferred can't be counted: the endpoint rejects a `tools` list in which every tool is deferred.
        # Where the flag is off, server tools are dropped, as the endpoint once rejected them
        # (https://github.com/pydantic/pydantic-ai/issues/5702), which undercounts the prompt by their definitions.
        # TODO: Send `mcp_servers` once the endpoint accepts it. Rechecked 2026-10-27, it rejects the param with a
        # 400: "The mcp_servers parameter is not supported on the count_tokens endpoint".
        counted_native_tools: tuple[type[AbstractNativeTool], ...] = (
            (MemoryTool, ToolSearchTool, WebSearchTool, WebFetchTool, CodeExecutionTool, AdvisorTool)
            if self.profile.get('anthropic_count_tokens_accepts_server_tools', False)
            else (MemoryTool,)
        )
        count_tokens_parameters = replace(
            model_request_parameters,
            native_tools=[
                tool for tool in model_request_parameters.native_tools if isinstance(tool, counted_native_tools)
            ],
        )
        # Params that keep `ToolSearchTool`, and `AdvisorTool` where the count keeps it. Keeping tool search means the
        # count describes the request we'd really send, deferred tools and all: `function_tools`
        # aren't stripped here, so they keep their `defer_loading` and the tool-search replay still
        # renders the same `tool_reference` wire shape as `/v1/messages`. The endpoint honors the flag
        # rather than ignoring it — one deferred 30-field tool counts 440 tokens with it and 1761
        # without on `claude-opus-4-8` — so this is the difference between counting the prompt and
        # counting the hidden schemas too.
        # Where the count drops advisor, dropping it here too makes `advisor_active` False so its
        # call/result history blocks are stripped during replay: replaying them without the tool
        # definition would 400.
        map_parameters = replace(
            self._declare_newcomers_left_alone(model_settings, model_request_parameters),
            native_tools=[
                tool
                for tool in model_request_parameters.native_tools
                if not isinstance(tool, AdvisorTool) or isinstance(tool, counted_native_tools)
            ],
        )

        # standalone function to make it easier to override
        messages, effort_request_parameters = self._effort_delta_request(
            messages, model_settings, model_request_parameters
        )

        revealed_tool_order = _revealed_deferred_tool_order(messages, map_parameters)
        tools, tool_choice = self._prepare_tools_and_tool_choice(model_settings, map_parameters)
        # `count_tokens_parameters` here, not `map_parameters`: the server-side tool definitions left off
        # the count are the one thing that has to differ from the real request.
        tools, mcp_servers, native_tool_betas = self._add_native_tools(tools, count_tokens_parameters, model_settings)
        _append_revealed_tool_params(tools, revealed_tool_order)

        auto_cache_control, resolved_cache_ttl = self._build_automatic_cache_control(model_settings)
        system_prompt, anthropic_messages = await self._map_message(messages, map_parameters, model_settings)
        request_scoped_texts = _request_scoped_tail_texts(messages)
        self._apply_per_block_caching_fallback(resolved_cache_ttl, anthropic_messages, request_scoped_texts)
        self._apply_explicit_message_caching(model_settings, anthropic_messages, request_scoped_texts)
        # The API rejects automatic caching when the last block's explicit breakpoint has a different TTL, and
        # ignores it when the TTL is the same, since both mark the same breakpoint. The explicit one wins.
        auto_cache_control = None if last_cacheable_block_has_cache_control(anthropic_messages) else auto_cache_control
        self._limit_cache_points(
            system_prompt, anthropic_messages, tools, automatic_caching=auto_cache_control is not None
        )
        raise_earlier_breakpoint_ttls(system_prompt, anthropic_messages, tools, auto_cache_control)
        output_config = self._build_output_config(effort_request_parameters, model_settings)
        anthropic_profile = self.profile
        thinking = self._translate_thinking(model_settings, model_request_parameters)
        effective_thinking = _effective_thinking(model_settings, thinking)
        betas, extra_headers = self._get_betas_and_extra_headers(
            model_settings, anthropic_profile, messages, model_request_parameters, effective_thinking
        )
        betas.update(native_tool_betas)
        context_management = self._add_compaction_params(messages, betas, model_settings)
        self._validate_task_budget_vs_context_management(model_settings, context_management)
        recovery_messages, compaction_boundary = self._stale_thinking_block_recovery_history(messages)
        initial_thinking, initial_betas, initial_thinking_override, initial_effective_thinking = (
            thinking_with_stale_block_history(
                anthropic_profile,
                recovery_messages,
                thinking,
                effective_thinking,
                betas,
                compaction_boundary=compaction_boundary,
                include_count_tokens_recovery=True,
            )
        )
        client = _with_max_retries(self.client, model_settings)

        async def count(
            thinking: BetaThinkingConfigParam | Omit,
            betas: set[str],
            thinking_override: dict[str, object] | None,
        ) -> BetaMessageTokensCount:
            extra_body = (
                _build_extra_body(model_settings, thinking_override)
                if thinking_override is not None
                else model_settings.get('extra_body')
            )
            if isinstance(client, AsyncAnthropicBedrock):
                return await count_tokens_via_bedrock(
                    client,
                    self._model_name,
                    system=system_prompt or OMIT,
                    messages=anthropic_messages,
                    max_tokens=model_settings.get(
                        'max_tokens', _default_max_tokens(effective_thinking, anthropic_profile)
                    ),
                    tools=tools or OMIT,
                    tool_choice=tool_choice or OMIT,
                    mcp_servers=mcp_servers or OMIT,
                    betas=sorted(betas) or OMIT,
                    output_config=output_config or OMIT,
                    cache_control=auto_cache_control or OMIT,
                    thinking=thinking,
                    context_management=context_management or OMIT,
                    timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                    speed=self._effective_speed(model_settings, anthropic_profile),
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )

            return await client.beta.messages.count_tokens(
                system=system_prompt or OMIT,
                messages=anthropic_messages,
                model=self._model_name,
                tools=tools or OMIT,
                tool_choice=tool_choice or OMIT,
                mcp_servers=mcp_servers or OMIT,
                betas=sorted(betas) or OMIT,
                output_config=output_config or OMIT,
                cache_control=auto_cache_control or OMIT,
                thinking=thinking,
                context_management=context_management or OMIT,
                timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                speed=self._effective_speed(model_settings, anthropic_profile),
                extra_headers=extra_headers,
                extra_body=extra_body,
            )

        with map_api_errors(self.model_name, self._provider.model_id_namespace), map_decode_errors(self.model_name):
            try:
                return await count(initial_thinking, initial_betas, initial_thinking_override)
            except APIStatusError as error:
                if not is_stale_thinking_block_error(anthropic_profile, initial_effective_thinking, error):
                    raise

            result = await count(
                OMIT,
                betas | {ANTHROPIC_THINKING_BINDING_BETA},
                drop_stale_thinking_blocks(effective_thinking),
            )
            mark_history_to_drop_stale_thinking_blocks_for_count_tokens(messages)
            warn_stale_thinking_block_recovery(self.model_name)
            return result

    def _process_response(  # noqa: C901
        self,
        response: BetaMessage,
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
        history_fetched_pages: Sequence[tuple[str, str]] = (),
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
        fetched_pages = [
            *history_fetched_pages,
            *(
                page
                for item in response.content
                if isinstance(item, BetaWebFetchToolResultBlock) and (page := fetched_page(item))
            ),
        ]
        for item in response.content:
            if isinstance(item, BetaTextBlock):
                if citations := map_citations(item.citations or [], item.text, fetched_pages):
                    items.append(
                        TextPart(
                            content=item.text,
                            provider_name=self.system,
                            provider_details=citations_provider_details(item.citations or []),
                            citations=citations,
                        )
                    )
                else:
                    items.append(TextPart(content=item.text))
            elif isinstance(item, BetaServerToolUseBlock):
                if item.name not in enabled_server_tool_names and item.id not in server_tool_result_ids:
                    continue
                call_part = map_server_tool_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaWebSearchToolResultBlock):
                items.append(map_web_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaToolSearchToolResultBlock):
                items.append(map_tool_search_tool_result_block(item, self.system))
            elif isinstance(item, BetaCodeExecutionToolResultBlock):
                items.append(map_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaBashCodeExecutionToolResultBlock):
                items.append(map_bash_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaTextEditorCodeExecutionToolResultBlock):
                items.append(map_text_editor_code_execution_tool_result_block(item, self.system))
            elif isinstance(item, BetaWebFetchToolResultBlock):
                items.append(map_web_fetch_tool_result_block(item, self.system))
            elif isinstance(item, BetaAdvisorToolResultBlock):
                items.append(map_advisor_tool_result_block(item, self.system))
            elif isinstance(item, BetaRedactedThinkingBlock):
                items.append(
                    ThinkingPart(id='redacted_thinking', content='', signature=item.data, provider_name=self.system)
                )
            elif isinstance(item, BetaThinkingBlock):
                items.append(ThinkingPart(content=item.thinking, signature=item.signature, provider_name=self.system))
            elif isinstance(item, BetaMCPToolUseBlock):
                call_part = map_mcp_server_use_block(item, self.system)
                builtin_tool_calls[call_part.tool_call_id] = call_part
                items.append(call_part)
            elif isinstance(item, BetaMCPToolResultBlock):
                call_part = builtin_tool_calls.get(item.tool_use_id)
                items.append(map_mcp_server_result_block(item, call_part, self.system))
            elif isinstance(item, BetaCompactionBlock):
                items.append(
                    CompactionPart(
                        content=item.content,
                        provider_name=self.system,
                        provider_details=compaction_provider_details(item.encrypted_content, item.signature),
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
            finish_reason = FINISH_REASON_MAP.get(raw_finish_reason)
        if response.stop_details is not None:
            provider_details = provider_details or {}
            if response.stop_details.explanation is not None:
                provider_details['refusal'] = response.stop_details.explanation
            if response.stop_details.category is not None:
                provider_details['refusal_category'] = response.stop_details.category
        if response.container:
            provider_details = provider_details or {}
            provider_details['container_id'] = response.container.id
        if response.input_transformations:
            provider_details = provider_details or {}
            provider_details['input_transformations'] = report_input_transformations(response.input_transformations)
        if response.diagnostics is not None:
            provider_details = provider_details or {}
            provider_details['cache_diagnostics'] = response.diagnostics.model_dump(mode='json')

        model_response = ModelResponse(
            parts=items,
            usage=map_usage(response, self._provider.name, self._provider.base_url, self._model_name),
            model_name=response.model,
            provider_response_id=response.id,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            canonical_system=self._canonical_system_stamp,
            finish_reason=finish_reason,
            state='suspended' if response.stop_reason == 'pause_turn' else 'complete',
            provider_details=provider_details,
        )
        if (context_input_tokens := last_pass_input_tokens(response.usage)) is not None:
            mark_context_input_tokens(model_response, context_input_tokens)
        return model_response

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
        response: AnthropicEventStream,
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
        history_fetched_pages: Sequence[tuple[str, str]] = (),
    ) -> AnthropicStreamedResponse:
        first_chunk = await response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert isinstance(first_chunk, BetaRawMessageStartEvent)

        # On Bedrock the SDK drops SSE event types, so a leading Bedrock-only chunk
        # (e.g. `amazon-bedrock-invocationMetrics`) is non-validating `construct_type`d
        # into `BetaRawMessageStartEvent(message=None)`. Fall back to the configured model
        # name rather than dereference `first_chunk.message.model` (https://github.com/pydantic/pydantic-ai/issues/5774). The
        # iterator below skips these `message=None` events.
        model_name = first_chunk.message.model if first_chunk.message is not None else self.model_name  # pyright: ignore[reportUnnecessaryComparison]

        streamed_response = AnthropicStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=model_name,
            _response=response,
            _provider_name=self._provider.name,
            _model_id_namespace=self._provider.model_id_namespace,
            _provider_url=self._provider.base_url,
            _enabled_server_tool_names=self._get_enabled_server_tool_names(model_request_parameters, model_settings),
            _history_fetched_pages=history_fetched_pages,
        )
        streamed_response.canonical_system = self._canonical_system_stamp
        return streamed_response

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

        An explicit strategy is sent as given. An unset one resolves to `bm25`, except on the legacy
        Bedrock InvokeModel API, where it resolves to `regex`.
        """
        if strategy is None and isinstance(self.client, _REGEX_DEFAULT_TOOL_SEARCH_CLIENTS):
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
        tool: WebFetchTool, supports_dynamic_filtering: bool, include_citations: bool
    ) -> tuple[BetaWebFetchTool20260209Param | BetaWebFetchTool20250910Param, str | None]:
        enable_citations = tool.enable_citations or include_citations
        citations = BetaCitationsConfigParamParam(enabled=True) if enable_citations else None
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
                tools.append(map_code_execution_tool(tool_version))
                # Cross-provider files are dropped silently here, not raised via
                # `_validate_uploaded_file_provider`; intentional per https://github.com/pydantic/pydantic-ai/issues/4338 (ignore over raise).
                if self._container_upload_file_ids(tool):
                    beta_features.add('files-api-2025-04-14')
            elif isinstance(tool, WebFetchTool):  # pragma: no branch
                web_fetch_tool, beta_feature = self._map_web_fetch_tool(
                    tool, supports_dynamic_filtering, citations_included(model_settings, model_request_parameters)
                )
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
                tools.append(map_advisor_tool(tool))
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

    def _tool_choice_declarations(
        self,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[dict[str, ToolDefinition], BetaToolChoiceParam]:
        """The tool definitions to declare, after any `tool_choice` restriction, and the API `tool_choice` value."""
        tool_defs = model_request_parameters.declared_tool_defs
        resolved_tool_choice = resolve_tool_choice(model_settings, model_request_parameters)

        tool_choice: BetaToolChoiceParam

        if resolved_tool_choice in ('auto', 'none'):
            # tool_choice = {'type': resolved_tool_choice}`: pyright can't narrow this properly
            tool_choice = {'type': 'auto'} if resolved_tool_choice == 'auto' else {'type': 'none'}
        elif resolved_tool_choice == 'required':
            supports = _support_tool_forcing(self.model_name, self.profile, model_settings, model_request_parameters)
            tool_choice = {'type': 'any'} if supports else {'type': 'auto'}
        elif isinstance(resolved_tool_choice, tuple):
            tool_choice_mode, tool_names = resolved_tool_choice
            supports = _support_tool_forcing(self.model_name, self.profile, model_settings, model_request_parameters)
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
        return tool_defs, tool_choice

    def _declare_newcomers_left_alone(
        self, model_settings: AnthropicModelSettings, model_request_parameters: ModelRequestParameters
    ) -> ModelRequestParameters:
        """Declare mid-run newcomers plainly when `tool_choice` leaves no other plainly declared tool.

        Resolution keeps a newcomer deferred while another tool is plainly declared, but a `tool_choice`
        restriction can filter that tool out of the request, leaving only deferred declarations, which
        Anthropic rejects. Applied before the tools and messages are mapped, so the newcomer's reveal
        blocks are dropped along with its `defer_loading` flag.
        """
        tool_defs, _ = self._tool_choice_declarations(model_settings, model_request_parameters)
        return _declare_lone_newcomers(
            model_request_parameters,
            tool_defs.keys(),
            wire_native_tools=[
                tool for tool in model_request_parameters.native_tools if _native_tool_is_a_tools_entry(tool)
            ],
        )

    def _prepare_tools_and_tool_choice(
        self,
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[BetaToolUnionParam], BetaToolChoiceParam | None]:
        """Determine which tools to send and the API tool_choice value.

        Returns:
            A tuple of (filtered_tools, tool_choice).
        """
        tool_defs, tool_choice = self._tool_choice_declarations(model_settings, model_request_parameters)
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

    async def _map_message(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> tuple[str | list[BetaTextBlockParam], list[BetaMessageParam]]:
        """Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
        messages = self._trim_before_compaction(messages)
        map_context = _MapMessagesContext(
            model_request_parameters=model_request_parameters,
            include_citations=citations_included(model_settings, model_request_parameters),
            # Whenever this request withholds a tool's schema, render any local-shape `search_tools`
            # history exchanges as Anthropic's "client-side" tool-search wire — `tool_use` paired with a
            # `tool_result` whose `content` is a `tool_reference` array. That block is the reveal: it
            # unlocks the withheld schemas server-side without forcing the model to re-search, and works
            # regardless of the current request's strategy (default native, named native, or custom
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
            deferred_tools_active=any(
                model_request_parameters.visibility_of(t.name) == 'deferred'
                for t in model_request_parameters.function_tools
            ),
            # The API 400s if advisor blocks appear in history without the advisor tool in the current
            # request, so when it's absent we strip advisor call/result blocks during replay (per
            # Anthropic's docs). When present, blocks — including a dangling pause_turn call — round-trip
            # verbatim (no pairing logic needed).
            advisor_active=any(isinstance(t, AdvisorTool) for t in model_request_parameters.native_tools),
            bm25_requested=any(
                isinstance(t, ToolSearchTool) and t.strategy == 'bm25' for t in model_request_parameters.native_tools
            ),
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
            # Read after the `tool_choice` restriction, which leaves tools out of the request: a reveal naming
            # one of those would be just as unknown as one naming a tool that's gone.
            declared_tool_names=set(self._tool_choice_declarations(model_settings, model_request_parameters)[0]),
            tool_defs_by_name={tool.name: tool for tool in model_request_parameters.function_tools},
        )
        leading_request = next((m for m in messages if isinstance(m, ModelRequest)), None)
        for m in messages:
            if isinstance(m, ModelRequest):
                standing_prompt_count = _standing_system_prompt_count(m) if m is leading_request else 0
                await self._map_model_request(m, standing_prompt_count, map_context)
            elif isinstance(m, ModelResponse):
                self._map_model_response(m, map_context)
            else:
                assert_never(m)
        anthropic_messages = map_context.anthropic_messages

        drop_unpaired_native_tool_calls(anthropic_messages)
        place_system_messages_before_generation(anthropic_messages)
        anchor_system_messages(anthropic_messages)
        self._add_container_uploads(anthropic_messages, model_request_parameters)
        # After the container uploads, which skip turns of only tool results, so the block doesn't move them.
        if self.profile.get('anthropic_distrusts_system_after_tool_result', False):
            end_tool_results_before_system_messages(anthropic_messages)
        check_citations_tool_choice(
            model_settings, model_request_parameters, anthropic_messages, map_context.include_citations
        )
        system_prompt = self._build_system_prompt(
            map_context.system_prompt_parts, messages, model_request_parameters, model_settings
        )
        return system_prompt, anthropic_messages

    def _container_upload_file_ids(self, tool: CodeExecutionTool) -> list[str]:
        """The IDs of the Anthropic-hosted files to upload to the code execution container, where the API has one."""
        if not tool.files or not self.profile.get('anthropic_accepts_file_and_container_ids', True):
            return []
        return [file.file_id for file in tool.files if file.provider_name == self.system]

    def _validate_uploaded_file_provider(self, item: UploadedFile) -> None:
        super()._validate_uploaded_file_provider(item)
        if not self.profile.get('anthropic_accepts_file_and_container_ids', True):
            raise UserError(
                f'`UploadedFile` {item.file_id!r} is an Anthropic Files API file, which '
                f'`{type(self.client).__name__}` does not support. Send the file content instead.'
            )

    def _add_container_uploads(
        self, anthropic_messages: list[BetaMessageParam], model_request_parameters: ModelRequestParameters
    ) -> None:
        """Add each Anthropic-hosted `CodeExecutionTool` file to every user message that can carry a `container_upload`."""
        # Cross-provider files are dropped silently here, not raised via
        # `_validate_uploaded_file_provider`; intentional per https://github.com/pydantic/pydantic-ai/issues/4338 (ignore over raise).
        pending_container_uploads = [
            file_id
            for tool in model_request_parameters.native_tools
            if isinstance(tool, CodeExecutionTool)
            for file_id in self._container_upload_file_ids(tool)
        ]
        if not pending_container_uploads:
            return
        for message in anthropic_messages:
            if message['role'] == 'user':
                existing = message['content']
                assert isinstance(existing, list)
                if is_tool_result_only(existing):
                    continue
                extra: list[BetaContainerUploadBlockParam] = [
                    {'type': 'container_upload', 'file_id': file_id} for file_id in pending_container_uploads
                ]
                message['content'] = [*existing, *extra]

    def _build_system_prompt(
        self,
        system_prompt_parts: list[str],
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        model_settings: AnthropicModelSettings,
    ) -> str | list[BetaTextBlockParam]:
        """The top-level `system` parameter: the hoisted system prompt, then one block per instruction part."""
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        system_prompt = '\n\n'.join(system_prompt_parts)
        cache_instructions = model_settings.get('anthropic_cache_instructions')
        if not instruction_parts and not (system_prompt and cache_instructions):
            return system_prompt

        # Each instruction part becomes a separate text block after the system prompt's.
        leading_texts = [system_prompt] if system_prompt else []
        system_prompt_blocks = [
            BetaTextBlockParam(type='text', text=text)
            for text in [*leading_texts, *(part.content for part in instruction_parts)]
        ]
        if cache_instructions:
            ttl: Literal['5m', '1h'] = '5m' if cache_instructions is True else cache_instructions
            # After the literal instructions (or the system prompt when there are none), and after the append
            # blocks; see `instruction_cache_boundaries`. `_limit_cache_points` drops the second first.
            static_parts, stable_parts = instruction_cache_boundaries(instruction_parts)
            static_count = len(leading_texts) + static_parts
            if static_count:
                system_prompt_blocks[static_count - 1]['cache_control'] = self._build_cache_control(ttl)
            stable_count = len(leading_texts) + stable_parts
            if stable_count > static_count:
                system_prompt_blocks[stable_count - 1]['cache_control'] = self._build_cache_control(ttl)
        return system_prompt_blocks

    async def _map_model_request(
        self, message: ModelRequest, standing_prompt_count: int, map_context: _MapMessagesContext
    ) -> None:
        """Append a `ModelRequest`'s user turn, then its `system` entries, to `map_context.anthropic_messages`."""
        request_context = _MapModelRequestContext()
        for part_index, request_part in enumerate(_sendable_request_parts(message.parts)):
            if isinstance(request_part, SystemPromptPart):
                self._map_system_prompt_part(
                    request_part, part_index < standing_prompt_count, request_context, map_context
                )
            elif isinstance(request_part, UserPromptPart):
                await self._map_user_prompt_part(request_part, request_context, map_context)
            elif isinstance(request_part, (InstructionBaselinePart, InstructionDeltaPart)):
                raise _unprojected_instruction_delta_error()
            elif isinstance(request_part, ModelSettingsDeltaPart):
                # Only the mid-conversation changes render: a window's opening effort went out as the request-level
                # `output_config.effort`. `_effort_delta_request` has already removed every part when this request
                # sends the current effort at request level instead.
                if not request_part.top_level and (effort := request_part.settings.get('thinking')) is not None:
                    request_context.effort_change = resolve_anthropic_effort(
                        effort,
                        supports_xhigh=self.profile.get('anthropic_supports_xhigh_effort', False),
                        supports_max=self.profile.get('anthropic_supports_max_effort', True),
                    )
            elif isinstance(request_part, ToolAvailabilityDeltaPart):
                self._map_tool_availability_delta_part(
                    request_part, request_context.tool_availability_blocks, map_context
                )
            elif isinstance(request_part, ToolReturnPart):
                request_context.user_content_params.append(await self._map_tool_return_part(request_part, map_context))
            elif isinstance(request_part, SpeechPart):
                # Unconverted realtime speech; `prepare_messages` turns these into `UserPromptPart`s.
                raise _unconverted_speech_part_error()
            elif isinstance(request_part, _UnpreparedPart):  # pragma: no cover
                raise _unprepared_part_error(request_part)
            else:
                assert_never(request_part)
        self._append_model_request_turns(request_context, map_context.anthropic_messages)

    def _map_system_prompt_part(
        self,
        part: SystemPromptPart,
        standing: bool,
        request_context: _MapModelRequestContext,
        map_context: _MapMessagesContext,
    ) -> None:
        # Narrowed to transports that serve the role, like `supports_inline_system_prompts` below. Where
        # this is `False`, `prepare_messages` has already left out every request-scoped part whose
        # request has been answered.
        if part.scope == 'request' and self.profile.get('supports_request_scoped_system_prompts', False):
            # Its own entry, so `clear_at` covers it alone. It needs no cache-point
            # bookkeeping: it's never inside a cached prefix, and `cache_control` on it
            # is a 400.
            request_context.request_scoped_prompts.append(part.content)
        # Only the opening `SystemPromptPart`s in the first request are the run's own system prompt and
        # hoist to the top-level `system` parameter. Later ones are mid-conversation operator instructions:
        # where we support them they reach us verbatim (rather than `<system>`-tagged by `prepare_messages`)
        # and it's on us to render them as a `{'role': 'system'}` entry, so adding an instruction leaves the
        # cached prefix the top-level `system` parameter sits in untouched. Where we don't,
        # `prepare_messages` has already rewritten them and none reach this branch — bar the adapter's
        # direct entry points, `count_tokens` and `request`, where hoisting them is the safe reading.
        elif standing or not self.profile.get('supports_inline_system_prompts', False):
            map_context.system_prompt_parts.append(part.content)
        else:
            # System-voice parts within one request all precede the same assistant
            # response. Rendering an interleaved entry after sibling user content is
            # deterministic and preserves the only ordering boundary that matters.
            request_context.mid_conversation_system_prompts.append(part.content)

    async def _map_user_prompt_part(
        self, part: UserPromptPart, request_context: _MapModelRequestContext, map_context: _MapMessagesContext
    ) -> None:
        user_content_params = request_context.user_content_params
        async for content in self._map_user_prompt(part, include_citations=map_context.include_citations):
            if not isinstance(content, CachePoint):
                user_content_params.append(content)
            elif request_context.mid_conversation_system_prompts or request_context.tool_availability_blocks:
                request_context.deferred_cache_points.append((len(user_content_params), content.ttl))
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
                    user_content_params or last_message_content(map_context.anthropic_messages),
                    ttl=content.ttl,
                )

    def _append_model_request_turns(
        self, request_context: _MapModelRequestContext, anthropic_messages: list[BetaMessageParam]
    ) -> None:
        user_content_params = request_context.user_content_params
        # A marker that ends the request has the instruction authored before it and nothing
        # authored after it, so the `system` entry's final block is exactly its boundary. A
        # marker with content after it can't have both, and lands where it was authored: the
        # instruction then sits outside the boundary, which caches a prefix of what was asked
        # for rather than more than was asked for.
        system_entry_cache_ttl: Literal['5m', '1h'] | None = None
        for marked_at, ttl in request_context.deferred_cache_points:
            if marked_at == len(user_content_params):
                system_entry_cache_ttl = ttl
            else:
                self._add_cache_control_to_last_param(
                    user_content_params[:marked_at] or last_message_content(anthropic_messages), ttl=ttl
                )
        if len(user_content_params) > 0:
            anthropic_messages.append({'role': 'user', 'content': user_content_params})
        if request_context.mid_conversation_system_prompts or request_context.tool_availability_blocks:
            system_content_params: list[BetaContentBlockParam] = [
                BetaTextBlockParam(text=content, type='text')
                for content in request_context.mid_conversation_system_prompts
            ]
            system_content_params.extend(cast('list[BetaContentBlockParam]', request_context.tool_availability_blocks))
            # The boundary covers the whole system entry, availability blocks included — a
            # terminal `CachePoint` lands on the entry's final block, never inside the entry.
            if system_entry_cache_ttl is not None:
                self._add_cache_control_to_last_param(system_content_params, ttl=system_entry_cache_ttl)
            anthropic_messages.append({'role': 'system', 'content': system_content_params})
        if request_context.effort_change is not None:
            # An effort-only entry carries no content, so the placement rules for mid-conversation system entries
            # don't apply to it, and it goes after the request's text entry, which has to follow the user turn
            # directly. After a turn with new input it sets the effort for the reply to that turn; after a turn of
            # only tool results, it takes effect from the next user turn with new input. It holds until a later
            # entry replaces it.
            anthropic_messages.append(
                {'role': 'system', 'content': [], 'output_config': {'effort': request_context.effort_change}}
            )
        if request_context.request_scoped_prompts:
            # Sent with every request, current or not: once a later user turn exists the API stops
            # rendering it, and leaving it in place is what keeps the cached prefix and the thinking
            # blocks bound to it valid. It goes after the request's other system entry so that one
            # can still carry the request's cache boundary.
            anthropic_messages.append(
                {
                    'role': 'system',
                    'content': [
                        BetaTextBlockParam(text=content, type='text')
                        for content in request_context.request_scoped_prompts
                    ],
                    'clear_at': 'next_user_message',
                }
            )

    def _map_tool_availability_delta_part(
        self,
        part: ToolAvailabilityDeltaPart,
        tool_availability_blocks: list[dict[str, Any]],
        map_context: _MapMessagesContext,
    ) -> None:
        # `tool_addition_mode` is already narrowed for transports that can't serve the `system` role, so
        # this covers both halves of the gate — and it's the same flag the beta header is added under.
        if self.tool_addition_mode != 'by_reference':
            # `prepare_messages` projects the delta onto the local tool-search exchange
            # for every model and transport that can't render it natively, so arriving
            # here means that projection didn't run — the same pipeline bug the other
            # adapters raise on, reachable by calling `Model.request` directly. Raising
            # matches them; rendering the blocks anyway would send them without the
            # `mid-conversation-tool-changes` beta header, which is added under this
            # same condition, and earn a 400 in place of an explanation.
            raise _unprepared_part_error(part)
        # Both block types carry a `tool_reference`, and the API rejects one naming a
        # tool this request doesn't declare: `tool_addition/tool_removal references
        # unknown tool '...'`. Replayed history routinely names tools that have since
        # gone — the turn announcing a removal is the last one that still declares it,
        # and a tool added then removed is absent from every turn after that. So a
        # block that can no longer be referenced is dropped, not asserted away: the
        # tool's absence from `tools` already tells the model what the block would.
        # `declared_tool_names` is the one shared with the tool-search replay filters —
        # same question, so it should be the same answer.
        for name in part.tools_added:
            tool_def = map_context.tool_defs_by_name.get(name)
            if (
                name in map_context.declared_tool_names
                and name not in map_context.rendered_tool_additions
                and tool_def is not None
                and map_context.model_request_parameters.visibility_of(name) != 'visible'
            ):
                tool_availability_blocks.append(
                    {'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': name}}
                )
                map_context.rendered_tool_additions.add(name)

    async def _map_tool_return_part(
        self, part: ToolReturnPart, map_context: _MapMessagesContext
    ) -> beta_tool_result_block_param.BetaToolResultBlockParam:
        custom_tool_refs, custom_empty_message = build_custom_tool_search_replay_blocks(
            part, map_context.deferred_tools_active, map_context.declared_tool_names
        )
        if custom_tool_refs:
            return beta_tool_result_block_param.BetaToolResultBlockParam(
                tool_use_id=_guard_tool_call_id(t=part),
                type='tool_result',
                content=custom_tool_refs,
                is_error=False,
            )
        if custom_tool_refs is not None:
            # No reference to send on the custom-callable strategy. Anthropic rejects an
            # empty `tool_result.content` list, so we send the helper's text as a single
            # text block instead: the typed return's own no-matches message, or a line
            # naming matches this request no longer declares.
            return beta_tool_result_block_param.BetaToolResultBlockParam(
                tool_use_id=_guard_tool_call_id(t=part),
                type='tool_result',
                content=[BetaTextBlockParam(text=custom_empty_message or _NO_MATCHES_MESSAGE, type='text')],
                is_error=False,
            )

        tool_result_content: list[beta_tool_result_block_param.Content] = []
        for item in part.content_items(mode='str', wrap_if_error=False):
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
                        with_document_citations(
                            BetaRequestDocumentBlockParam(
                                source=BetaFileDocumentSourceParam(file_id=item.file_id, type='file'),
                                type='document',
                            ),
                            map_context.include_citations,
                        )
                    )
                else:
                    raise UserError(
                        f'Unsupported media type {item.media_type!r} for Anthropic file upload. '
                        'Only image and document (text/application) types are supported.'
                    )
            elif is_multi_modal_content(item):
                tool_result_content.append(
                    await self._map_file_to_content_block(
                        item,  # pyright: ignore[reportArgumentType]
                        'tool returns',
                        include_citations=map_context.include_citations,
                    )
                )
            elif isinstance(item, str):  # pragma: no branch
                tool_result_content.append(BetaTextBlockParam(text=item, type='text'))

        return beta_tool_result_block_param.BetaToolResultBlockParam(
            tool_use_id=_guard_tool_call_id(t=part),
            type='tool_result',
            content=tool_result_content or '',
            is_error=part.outcome in ERROR_OUTCOMES,
        )

    def _map_model_response(self, message: ModelResponse, map_context: _MapMessagesContext) -> None:
        """Append a `ModelResponse`'s assistant turn to `map_context.anthropic_messages`."""
        assistant_content_params: list[_AssistantContentBlockParam] = []
        carried_thinking_params: list[BetaTextBlockParam] = []
        for response_part in message.parts:
            if isinstance(response_part, TextPart):
                if response_part.content:
                    text_block = BetaTextBlockParam(text=response_part.content, type='text')
                    raw_citations = (response_part.provider_details or {}).get('citations')
                    if (
                        # Web search citations point into the search results by an encrypted index, so they go
                        # wherever those results do; see `_replays_native_tool_part`.
                        self.accepts_stateless_artifacts(message, response_part)
                        and isinstance(raw_citations, list)
                        and citations_fit_text(response_part)
                        # Only web search citations are sent back, and clients without web search, such as Bedrock,
                        # reject them.
                        and WebSearchTool in self.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
                        and (
                            citations := [
                                param
                                for citation in cast(list[object], raw_citations)
                                if _utils.is_str_dict(citation) and (param := map_citation_for_replay(citation))
                            ]
                        )
                    ):
                        text_block['citations'] = citations
                    assistant_content_params.append(text_block)
            elif isinstance(response_part, ToolCallPart):
                assistant_content_params.append(
                    BetaToolUseBlockParam(
                        id=_guard_tool_call_id(t=response_part),
                        type='tool_use',
                        name=response_part.tool_name,
                        input=response_part.args_as_dict(),
                    )
                )
            elif isinstance(response_part, ThinkingPart):
                self._map_thinking_part(message, response_part, assistant_content_params, carried_thinking_params)
            elif isinstance(response_part, NativeToolCallPart):
                if self._replays_native_tool_part(message, response_part):
                    self._map_native_tool_call_part(response_part, assistant_content_params, map_context)
            elif isinstance(response_part, NativeToolReturnPart):
                if self._replays_native_tool_part(message, response_part):
                    self._map_native_tool_return_part(response_part, assistant_content_params, map_context)
            elif isinstance(response_part, CompactionPart):
                self._map_compaction_part(message, response_part, assistant_content_params)
            elif isinstance(response_part, FilePart):  # pragma: no cover
                # Files generated by models are not sent back to models that don't themselves generate files.
                pass
            elif isinstance(response_part, SpeechPart):  # pragma: no cover
                # Unconverted realtime speech; `prepare_messages` turns these into `TextPart`s in `Model.prepare_messages`.
                raise _unconverted_speech_part_error()
            else:
                assert_never(response_part)
        self._append_model_response_turns(
            assistant_content_params, carried_thinking_params, map_context.anthropic_messages
        )

    @staticmethod
    def _append_model_response_turns(
        assistant_content_params: list[_AssistantContentBlockParam],
        carried_thinking_params: list[BetaTextBlockParam],
        anthropic_messages: list[BetaMessageParam],
    ) -> None:
        if carried_thinking_params:
            # Reasoning replayed in an assistant turn teaches a model that imitates assistant
            # formatting to write `<thinking>` tags into its visible answers
            # (https://github.com/pydantic/pydantic-ai/issues/5869), so it gets a user turn of its
            # own, built here so its position is fixed while mapping. It goes *ahead* of any
            # `system` entry this request emitted, so that entry has no user turn to hop and keeps
            # the cache boundary it was authored with (see `place_system_messages_before_generation`).
            insert_at = len(anthropic_messages)
            while insert_at > 0 and anthropic_messages[insert_at - 1]['role'] == 'system':
                insert_at -= 1
            anthropic_messages.insert(insert_at, {'role': 'user', 'content': carried_thinking_params})
        if len(assistant_content_params) > 0:
            anthropic_messages.append({'role': 'assistant', 'content': assistant_content_params})

    def _map_compaction_part(
        self,
        message: ModelResponse,
        part: CompactionPart,
        assistant_content_params: list[_AssistantContentBlockParam],
    ) -> None:
        # The same predicate `_trim_before_compaction` drops history before a compaction part by.
        if self.accepts_stateless_artifacts(message, part):  # pragma: no branch
            compaction_block = BetaCompactionBlockParam(content=part.content, type='compaction')
            details = part.provider_details or {}
            if (encrypted_content := details.get('encrypted_content')) is not None:
                compaction_block['encrypted_content'] = encrypted_content
            if (signature := details.get('signature')) is not None:
                compaction_block['signature'] = signature
            assistant_content_params.append(compaction_block)

    def _map_thinking_part(
        self,
        message: ModelResponse,
        part: ThinkingPart,
        assistant_content_params: list[_AssistantContentBlockParam],
        carried_thinking_params: list[BetaTextBlockParam],
    ) -> None:
        # An empty signature (e.g. from an interrupted stream) is never valid,
        # so fall back to tagged text rather than triggering a 400 from the API.
        if part.signature and self.accepts_stateless_artifacts(message, part):
            # `BedrockConverseModel` records redacted thinking as `redacted_content`, with the same data.
            if part.id in REDACTED_THINKING_PART_IDS:
                assistant_content_params.append(
                    BetaRedactedThinkingBlockParam(data=part.signature, type='redacted_thinking')
                )
            else:
                assistant_content_params.append(
                    BetaThinkingBlockParam(thinking=part.content, signature=part.signature, type='thinking')
                )
        elif part.content:  # pragma: no branch
            thinking_text, carry_in_user_turn = render_replayed_thinking(part.content, self.profile)
            thinking_param = BetaTextBlockParam(text=thinking_text, type='text')
            if carry_in_user_turn:
                carried_thinking_params.append(thinking_param)
            else:
                assistant_content_params.append(thinking_param)

    def _replays_native_tool_part(
        self, message: ModelResponse, part: NativeToolCallPart | NativeToolReturnPart
    ) -> bool:
        """Whether a server-side tool call or result in history is sent back.

        Search, fetch and advisor results carry their state inline, encrypted where it's sensitive, so they're
        sent to any client that accepts the response's stateless data. Code execution and MCP results refer to
        state kept by the account that ran them, such as a container, so they only go back to the same provider.
        They're judged by `provider_name` alone, not by `accepts_server_state`, until a live check against two
        endpoints shows whether an endpoint rejects blocks from another: the one exception to judging server-side
        state by endpoint. The container itself is judged by endpoint in `_get_container`.
        """
        if part.tool_name in _STATELESS_NATIVE_TOOL_KINDS:
            return self.accepts_stateless_artifacts(message, part)
        return part.provider_name == self.system

    def _map_native_tool_call_part(
        self,
        part: NativeToolCallPart,
        assistant_content_params: list[_AssistantContentBlockParam],
        map_context: _MapMessagesContext,
    ) -> None:
        tool_use_id = _guard_tool_call_id(t=part)
        if part.tool_name == WebSearchTool.kind:
            name, tool_input = 'web_search', part.args_as_dict()
        elif part.tool_name in (
            CodeExecutionTool.kind,
            'bash_code_execution',
            'text_editor_code_execution',
        ):
            name, tool_input = get_anthropic_code_execution_tool_name(part), part.args_as_dict()
        elif part.tool_name == WebFetchTool.kind:
            name, tool_input = 'web_fetch', part.args_as_dict()
        elif part.tool_name == AdvisorTool.kind:
            if not map_context.advisor_active:
                return
            name, tool_input = 'advisor', part.args_as_dict()
        elif part.tool_name == ToolSearchTool.kind:
            name, tool_input = self._map_tool_search_call_input(part, map_context.bm25_requested)
        else:
            if (
                part.tool_name.startswith(MCPServerTool.kind)
                and (server_id := part.tool_name.split(':', 1)[1])
                and (args := part.args_as_dict())
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
            return
        server_tool_use_block_param = BetaServerToolUseBlockParam(
            id=tool_use_id,
            type='server_tool_use',
            name=name,
            input=tool_input,
        )
        add_anthropic_caller_param(server_tool_use_block_param, part)
        assistant_content_params.append(server_tool_use_block_param)

    def _map_tool_search_call_input(
        self, part: NativeToolCallPart, bm25_requested: bool
    ) -> tuple[Literal['tool_search_tool_regex', 'tool_search_tool_bm25'], dict[str, Any]]:
        # Round-trip the native variant (bm25/regex) so we don't
        # silently rewrite the algorithm. `map_server_tool_use_block`
        # stashes it in `provider_details['strategy']`. On the legacy Bedrock
        # InvokeModel API, `bm25` calls replay as `regex` unless this request
        # asks for `bm25` (see `_REGEX_DEFAULT_TOOL_SEARCH_CLIENTS`).
        details = part.provider_details or {}
        strategy: Literal['bm25', 'regex'] = (
            'regex'
            if details.get('strategy') == 'regex'
            or (isinstance(self.client, _REGEX_DEFAULT_TOOL_SEARCH_CLIENTS) and not bm25_requested)
            else 'bm25'
        )
        native_name = 'tool_search_tool_regex' if strategy == 'regex' else 'tool_search_tool_bm25'
        # Rebuild the variant-specific wire shape from the
        # cross-provider `queries` slot. bm25 expects `{"query": "..."}`,
        # regex expects `{"pattern": "..."}`. Joining with a space
        # matches Anthropic's single-string input on either variant.
        args_dict = part.args_as_dict()
        if 'queries' not in args_dict:
            return native_name, args_dict
        raw_queries = args_dict.get('queries')
        queries: list[str] = (
            [q for q in cast('list[Any]', raw_queries) if isinstance(q, str)] if isinstance(raw_queries, list) else []
        )
        wire_key = 'pattern' if strategy == 'regex' else 'query'
        return native_name, {wire_key: ' '.join(queries)}

    def _map_native_tool_return_part(
        self,
        part: NativeToolReturnPart,
        assistant_content_params: list[_AssistantContentBlockParam],
        map_context: _MapMessagesContext,
    ) -> None:
        tool_use_id = _guard_tool_call_id(t=part)
        if part.tool_name in (
            WebSearchTool.kind,
            'web_search_tool_result',  # Backward compatibility
        ) and isinstance(part.content, dict | list):
            block = BetaWebSearchToolResultBlockParam(
                tool_use_id=tool_use_id,
                type='web_search_tool_result',
                content=cast(
                    BetaWebSearchToolResultBlockParamContentParam,
                    part.content,  # pyright: ignore[reportUnknownMemberType]
                ),
            )
            add_anthropic_caller_param(block, part)
            assistant_content_params.append(block)
        elif part.tool_name in (
            CodeExecutionTool.kind,
            'code_execution_tool_result',  # Backward compatibility
            'bash_code_execution',
            'text_editor_code_execution',
        ) and isinstance(part.content, dict | list):
            assistant_content_params.append(build_code_execution_tool_result_block(part, tool_use_id))
        elif part.tool_name == WebFetchTool.kind and isinstance(part.content, dict):
            block = BetaWebFetchToolResultBlockParam(
                tool_use_id=tool_use_id,
                type='web_fetch_tool_result',
                content=cast(
                    WebFetchToolResultBlockParamContent,
                    part.content,  # pyright: ignore[reportUnknownMemberType]
                ),
            )
            add_anthropic_caller_param(block, part)
            assistant_content_params.append(block)
        elif part.tool_name == AdvisorTool.kind:
            # Drop advisor result blocks when continuing without the advisor tool: the
            # API 400s if they appear in history without the tool in the request, and
            # Anthropic's docs prescribe stripping them (mirrors the orphan tool-search
            # drop above). When kept, the discriminated content dict round-trips verbatim
            # so the server can decrypt a redacted advisor result on the next turn.
            if map_context.advisor_active and isinstance(part.content, dict):
                assistant_content_params.append(
                    BetaAdvisorToolResultBlockParam(
                        tool_use_id=tool_use_id,
                        type='advisor_tool_result',
                        content=cast(
                            AdvisorToolResultBlockParamContent,
                            part.content,  # pyright: ignore[reportUnknownMemberType]
                        ),
                    )
                )
        elif isinstance(part, NativeToolSearchReturnPart):
            assistant_content_params.append(
                build_tool_search_replay_block(part, tool_use_id, map_context.declared_tool_names)
            )
        elif part.tool_name.startswith(MCPServerTool.kind) and isinstance(part.content, dict):  # pragma: no branch
            mcp_content = cast(
                'dict[str, Any]',
                part.content,  # pyright: ignore[reportUnknownMemberType]
            )
            assistant_content_params.append(
                BetaRequestMCPToolResultBlockParam(
                    content=mcp_content['content'],
                    is_error=mcp_content['is_error'],
                    tool_use_id=tool_use_id,
                    type='mcp_tool_result',
                )
            )

    @staticmethod
    def _limit_cache_points(
        system_prompt: str | list[BetaTextBlockParam],
        anthropic_messages: list[BetaMessageParam],
        tools: list[BetaToolUnionParam],
        *,
        automatic_caching: bool = False,
    ) -> None:
        """Limit the number of cache points in the request to Anthropic's maximum.

        Tool definition cache points and the first instruction cache point always take priority; excess
        message cache points are removed oldest-first, after the instruction cache point behind the
        `on_change='append'` blocks (see `split_instruction_cache_points`). When automatic caching
        is enabled, the server-applied breakpoint uses 1 of the available slots, reducing the budget for
        explicit breakpoints by one.

        Raises:
            UserError: If system_prompt and tools combined already exceed the budget.
                      This indicates a configuration error that cannot be auto-fixed.
        """
        # Anthropic enforces a maximum of 4 cache points per request.
        max_points = 4 - (1 if automatic_caching else 0)
        system_points = (
            [cast('dict[str, Any]', block) for block in system_prompt if 'cache_control' in block]
            if isinstance(system_prompt, list)
            else []
        )
        reserved, optional_points = split_instruction_cache_points(system_points)
        # cache_control can be in the middle of the tools list if builtin tools are added after.
        reserved += sum(1 for tool in tools if 'cache_control' in tool)

        message_blocks = (
            cast('dict[str, Any]', block)
            for message in reversed(anthropic_messages)
            if not isinstance(message['content'], str)
            for block in reversed(cast('list[BetaContentBlockParam]', message['content']))
        )
        for block_dict in excess_cache_points(
            chain(message_blocks, optional_points),
            max_points=max_points,
            reserved=reserved,
            is_cache_point=lambda block: 'cache_control' in block,
            description='Anthropic request',
        ):
            del block_dict['cache_control']

    def _build_cache_control(self, ttl: Literal['5m', '1h'] = '5m') -> BetaCacheControlEphemeralParam:
        """Build a cache control dict with the given TTL.

        Args:
            ttl: The cache time-to-live ('5m' or '1h').

        Returns:
            A cache control dict with the specified TTL.
        """
        return BetaCacheControlEphemeralParam(type='ephemeral', ttl=ttl)

    def _translate_cache(
        self, model_settings: AnthropicModelSettings, cache: Literal[True] | CacheRetention | CacheConfig
    ) -> AnthropicModelSettings:
        """Map the unified `cache` setting onto Anthropic cache settings.

        Only called when no explicit `anthropic_cache*` setting is present (those take
        precedence in `prepare_request`). The static instructions and tool definitions always get
        breakpoints, since Anthropic only reads cache entries written at a breakpoint and a new
        conversation can only share that prefix. The conversation is cached with automatic caching
        where the client supports it, and with a breakpoint on its last block on the legacy Bedrock client.
        """
        retention, messages = split_cache_setting(cache)
        ttl: Literal['5m', '1h'] = retention if retention in ('5m', '1h') else '5m'
        translated = model_settings.copy()
        translated['anthropic_cache_instructions'] = ttl
        translated['anthropic_cache_tool_definitions'] = ttl
        if messages:
            if self.profile.get('supports_auto_cache', False):
                translated['anthropic_cache'] = ttl
            else:
                translated['anthropic_cache_messages'] = ttl
        return translated

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
        # Where the profile disclaims automatic caching (such as the legacy Bedrock client on older models),
        # the per-block fallback is handled by _apply_per_block_caching_fallback.
        if not self.profile.get('supports_auto_cache', False):
            return None, ttl
        return self._build_cache_control(ttl), ttl

    def _apply_per_block_caching_fallback(
        self,
        resolved_ttl: Literal['5m', '1h'] | None,
        anthropic_messages: list[BetaMessageParam],
        request_scoped_texts: tuple[str, ...] = (),
    ) -> None:
        """Apply per-block message caching where `anthropic_cache`'s automatic breakpoint can't be relied on.

        Where the model profile's `supports_auto_cache` is `False` (such as on the legacy Bedrock client with models
        that don't take the top-level `cache_control` parameter), this applies per-block
        `cache_control` to the last content block of the last message.

        It's also applied, on every client, when the request ends with request-scoped text the next request
        won't send (see `_request_scoped_tail_texts`). The automatic breakpoint lands on that text, writing a
        cache entry no later request can read, so an explicit one goes on the last block before it. A
        native request-scoped entry needs none of this: automatic caching already skips it.

        Args:
            resolved_ttl: The resolved TTL from `_build_automatic_cache_control`, or None
                if caching is not enabled.
            anthropic_messages: The list of Anthropic message params to apply fallback to.
            request_scoped_texts: The texts of the request's trailing request-scoped blocks.
        """
        if resolved_ttl and (not self.profile.get('supports_auto_cache', False) or request_scoped_texts):
            self._apply_message_cache_control(anthropic_messages, resolved_ttl, request_scoped_texts)

    def _apply_explicit_message_caching(
        self,
        model_settings: AnthropicModelSettings,
        anthropic_messages: list[BetaMessageParam],
        request_scoped_texts: tuple[str, ...] = (),
    ) -> None:
        """Apply per-block message caching when `anthropic_cache_messages` is enabled.

        Mutually exclusive with `anthropic_cache` (enforced by `_build_automatic_cache_control`).
        """
        if cache_messages := model_settings.get('anthropic_cache_messages'):
            ttl: Literal['5m', '1h'] = '5m' if cache_messages is True else cache_messages
            self._apply_message_cache_control(anthropic_messages, ttl, request_scoped_texts)

    def _apply_message_cache_control(
        self,
        anthropic_messages: list[BetaMessageParam],
        ttl: Literal['5m', '1h'],
        request_scoped_texts: tuple[str, ...] = (),
    ) -> None:
        """Apply per-block `cache_control` to the last content block the next request will still send.

        That's the last block of the last message, skipping request-scoped content: a native request-scoped
        entry (`cache_control` on one is a 400) and trailing text blocks rendering a request-scoped prompt
        on a model that can't clear it. A breakpoint on either would write a cache entry the next request
        can't read, since the content is cleared or no longer sent by then.

        If the chosen block already has `cache_control` (e.g. from an explicit `CachePoint`),
        it is left unchanged to preserve the user's chosen TTL. On Bedrock, when the previous
        request's breakpoint is further back than the lookback reaches (a wide fan-out of parallel
        tool calls), the end of that request gets a breakpoint too. The Claude API collapses runs
        of tool blocks into one position, so it doesn't need one.
        """
        if isinstance(self.client, AsyncAnthropicBedrock):
            # A request-scoped entry can't carry a breakpoint, so the previous request put its own before it:
            # it's left out of the candidates, its blocks counted with the message ahead of it.
            lasting: list[BetaMessageParam] = []
            block_counts: list[int] = []
            for message in anthropic_messages:
                content = message['content']
                count = 1 if isinstance(content, str) else len(cast(list[BetaContentBlockParam], content))
                if (message.get('clear_at') == 'next_user_message' or is_effort_entry(message)) and lasting:
                    block_counts[-1] += count
                else:
                    lasting.append(message)
                    block_counts.append(count)
            previous_tail = previous_tail_needing_breakpoint([message['role'] for message in lasting], block_counts)
            if previous_tail is not None:
                self._add_message_cache_control(lasting[previous_tail], ttl)
        remaining = request_scoped_texts
        for message in reversed(anthropic_messages):
            if message.get('clear_at') == 'next_user_message':
                continue
            blocks, remaining = blocks_before_request_scoped_text(message, remaining)
            if blocks:
                self._add_cache_control_to_last_cacheable_param(blocks, ttl)
                return

    def _add_message_cache_control(self, message: BetaMessageParam, ttl: Literal['5m', '1h']) -> None:
        """Add `cache_control` to the last cacheable content block of `message`, unless it already has one."""
        content = message['content']
        if isinstance(content, str):  # pragma: no cover
            message['content'] = [
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
            if param['type'] in ANTHROPIC_CACHEABLE_PARAM_TYPES:
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
        add_cache_control_param(params, self._build_cache_control(ttl))

    @staticmethod
    def _map_binary_data(
        data: bytes, media_type: str, *, include_citations: bool = False
    ) -> BetaImageBlockParam | BetaRequestDocumentBlockParam:
        if media_type.startswith('image/'):
            return BetaImageBlockParam(
                source={'data': base64.b64encode(data).decode(), 'media_type': media_type, 'type': 'base64'},  # pyright: ignore[reportArgumentType]
                type='image',
            )
        elif media_type == 'application/pdf':
            return with_document_citations(
                BetaRequestDocumentBlockParam(
                    source=BetaBase64PDFSourceParam(
                        data=base64.b64encode(data).decode(),
                        media_type='application/pdf',
                        type='base64',
                    ),
                    type='document',
                ),
                include_citations,
            )
        elif media_type == 'text/plain':
            return with_document_citations(
                BetaRequestDocumentBlockParam(
                    source=BetaPlainTextSourceParam(data=data.decode('utf-8'), media_type=media_type, type='text'),
                    type='document',
                ),
                include_citations,
            )
        else:
            raise RuntimeError(f'Unsupported binary content media type for Anthropic: {media_type}')

    @staticmethod
    async def _map_image_url(item: ImageUrl, *, include_citations: bool = False) -> BetaImageBlockParam:
        if item.force_download:
            downloaded = await download_item(item, data_format='bytes')
            return AnthropicModel._map_binary_data(  # pyright: ignore[reportReturnType]
                downloaded['data'], item.media_type, include_citations=include_citations
            )
        return BetaImageBlockParam(source={'type': 'url', 'url': item.url}, type='image')

    @staticmethod
    async def _map_document_url(item: DocumentUrl, *, include_citations: bool = False) -> BetaRequestDocumentBlockParam:
        if item.media_type == 'application/pdf':
            if item.force_download:
                downloaded = await download_item(item, data_format='bytes')
                return AnthropicModel._map_binary_data(  # pyright: ignore[reportReturnType]
                    downloaded['data'], item.media_type, include_citations=include_citations
                )
            return with_document_citations(
                BetaRequestDocumentBlockParam(source={'url': item.url, 'type': 'url'}, type='document'),
                include_citations,
            )
        elif item.media_type == 'text/plain':
            downloaded_item = await download_item(item, data_format='text')
            return with_document_citations(
                BetaRequestDocumentBlockParam(
                    source=BetaPlainTextSourceParam(
                        data=downloaded_item['data'], media_type=item.media_type, type='text'
                    ),
                    type='document',
                ),
                include_citations,
            )
        else:
            raise RuntimeError(f'Unsupported document media type: {item.media_type}')

    @staticmethod
    async def _map_file_to_content_block(
        item: BinaryContent | ImageUrl | DocumentUrl | AudioUrl | VideoUrl,
        context: str,
        *,
        include_citations: bool = False,
    ) -> BetaImageBlockParam | BetaRequestDocumentBlockParam:
        """Map a multimodal file item to its Anthropic API content block."""
        if isinstance(item, BinaryContent):
            if item.is_image or item.is_document:
                return AnthropicModel._map_binary_data(item.data, item.media_type, include_citations=include_citations)
            raise NotImplementedError(f'Unsupported binary content type in Anthropic {context}: {item.media_type}')
        elif isinstance(item, ImageUrl):
            return await AnthropicModel._map_image_url(item, include_citations=include_citations)
        elif isinstance(item, DocumentUrl):
            return await AnthropicModel._map_document_url(item, include_citations=include_citations)
        elif isinstance(item, AudioUrl):
            raise NotImplementedError(f'AudioUrl is not supported in Anthropic {context}')
        else:
            raise NotImplementedError(f'VideoUrl is not supported in Anthropic {context}')

    async def _map_user_prompt(
        self,
        part: UserPromptPart,
        *,
        include_citations: bool = False,
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
                        yield with_document_citations(
                            BetaRequestDocumentBlockParam(
                                source=BetaFileDocumentSourceParam(file_id=item.file_id, type='file'),
                                type='document',
                            ),
                            include_citations,
                        )
                    else:
                        raise UserError(
                            f'Unsupported media type {item.media_type!r} for Anthropic file upload. '
                            'Only image and document (text/application) types are supported.'
                        )
                elif is_multi_modal_content(item):
                    yield await AnthropicModel._map_file_to_content_block(
                        item,  # pyright: ignore[reportArgumentType]
                        'user prompts',
                        include_citations=include_citations,
                    )
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

    def _effort_delta_request(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelRequestParameters]:
        """Decide whether this request sends effort changes as per-message `output_config` entries.

        It does when the history holds a mid-conversation change (a `ModelSettingsDeltaPart` that
        `prepare_messages` kept for this model) and nothing rules the entry out: an explicit `anthropic_effort`,
        which is sent as is, and thinking other than adaptive, under which the API rejects a per-message effort
        that differs from the level in effect (`between_tools` on Claude Sonnet 5.5, `disabled` on Claude Haiku
        5.5). The request then keeps the window's opening effort as its `output_config.effort`, so its prefix
        matches the earlier requests'. Otherwise the parts are removed and the current effort goes out as
        `output_config.effort`, as it would without them.
        """
        baseline, has_changes = _effort_window_baseline(messages)
        thinking = _effective_thinking(
            model_settings, self._translate_thinking(model_settings, model_request_parameters)
        )
        if (
            has_changes
            # `prepare_messages` already dropped the parts on a model without the channel, but a direct `request`
            # call with stored history doesn't run it.
            and self.profile.get('supports_mid_conversation_effort', False)
            and not self._overrides_unified_effort(model_settings)
            and not isinstance(thinking, Omit)
            and thinking.get('type') == 'adaptive'
        ):
            if baseline is not None and isinstance(model_request_parameters.thinking, str):
                model_request_parameters = replace(model_request_parameters, thinking=baseline)
            return messages, model_request_parameters
        return _without_model_settings_deltas(messages), model_request_parameters

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
        thinking = model_request_parameters.thinking
        if effort is None and self.profile.get('anthropic_supports_effort', False):
            if isinstance(thinking, str):
                effort = resolve_anthropic_effort(
                    thinking,
                    supports_xhigh=self.profile.get('anthropic_supports_xhigh_effort', False),
                    supports_max=self.profile.get('anthropic_supports_max_effort', True),
                )
            elif thinking is False and self.profile.get('thinking_always_enabled', False):
                # The least thinking a model that can't turn it off accepts.
                effort = 'low'

        if effort is not None:
            # Validate what reaches the wire, where a caller's `extra_body` thinking wins.
            self._validate_effort_vs_disabled_thinking(
                effort,
                _effective_thinking(model_settings, self._translate_thinking(model_settings, model_request_parameters)),
            )

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
        self, effort: AnthropicEffort, thinking: dict[str, object] | Omit
    ) -> None:
        """Reject `xhigh`/`max` effort combined with explicitly disabled thinking.

        Claude Opus 5 and Haiku 5.5 cap effort at `high` once thinking is disabled, as does Claude Sonnet 5.5 with
        `between_tools`, while Claude Opus 4.8 accepts every effort level in that combination. Fail fast with a
        helpful message rather than letting the API return an opaque 400.
        """
        if effort not in ('xhigh', 'max'):
            return
        if not self.profile.get('anthropic_disallows_top_effort_when_thinking_disabled', False):
            return
        if isinstance(thinking, Omit) or (thinking_type := thinking.get('type')) not in ('disabled', 'between_tools'):
            return
        raise UserError(
            f'Model {self.model_name!r} does not support `anthropic_effort={effort!r}` while thinking is '
            f"disabled (`thinking=False` or `anthropic_thinking={{'type': {thinking_type!r}}}`). "
            "Use an effort of 'high' or below, or enable thinking."
        )

    def _get_task_budget(self, model_settings: AnthropicModelSettings) -> AnthropicTaskBudget | None:
        task_budget = model_settings.get('anthropic_task_budget')
        if task_budget is None:
            return None

        if not self.profile.get('anthropic_supports_task_budgets', False):
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


@dataclass
class _MapMessagesContext:
    """The request-wide facts `AnthropicModel._map_message` maps each message against, and what it builds across them."""

    model_request_parameters: ModelRequestParameters
    # Anthropic requires citations on all of a request's documents or none.
    include_citations: bool
    deferred_tools_active: bool
    advisor_active: bool
    bm25_requested: bool
    declared_tool_names: set[str]
    tool_defs_by_name: dict[str, ToolDefinition]
    system_prompt_parts: list[str] = field(default_factory=list[str])
    anthropic_messages: list[BetaMessageParam] = field(default_factory=list[BetaMessageParam])
    rendered_tool_additions: set[str] = field(default_factory=set[str])


@dataclass
class _MapModelRequestContext:
    """What `AnthropicModel._map_model_request` collects from one `ModelRequest`'s parts before it emits the turns."""

    user_content_params: list[BetaContentBlockParam] = field(default_factory=list[BetaContentBlockParam])
    mid_conversation_system_prompts: list[str] = field(default_factory=list[str])
    request_scoped_prompts: list[str] = field(default_factory=list[str])
    tool_availability_blocks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    effort_change: AnthropicEffort | None = None
    """The effort a mid-conversation `ModelSettingsDeltaPart` changes to, sent as an effort-only `system` entry."""
    # `CachePoint`s authored after a mid-conversation instruction or a tool availability
    # change, as the number of user blocks that preceded each one. They can't be placed
    # while mapping: both render after this request's user blocks in the `system` entry
    # despite being authored before the marker, so whether they fall inside the boundary
    # depends on whether anything else follows the marker.
    deferred_cache_points: list[tuple[int, Literal['5m', '1h']]] = field(
        default_factory=list[tuple[int, Literal['5m', '1h']]]
    )


def _effective_thinking_type(
    anthropic_thinking: BetaThinkingConfigParam | None,
    unified_thinking: ThinkingLevel | None,
    profile: AnthropicModelProfile,
) -> Literal['enabled', 'adaptive'] | None:
    """Resolve whether a request will think, and how, for the output-mode and tool-forcing decisions.

    Extended thinking (`{'type': 'enabled'}`) rejects a forced `tool_choice`; adaptive thinking accepts it, but
    the model then answers without thinking. Unified thinking maps to `adaptive` when the profile advertises it
    and to `enabled` otherwise, the same mapping `_translate_thinking` uses to build the wire payload. With no
    thinking setting, models that think by default resolve to `adaptive`, as does `thinking=False` on a model
    that can't turn thinking off, which then thinks at its lowest effort. Returns `None` when the request won't
    think, including Claude Sonnet 5.5's `between_tools`, which doesn't think before responding.
    """
    if anthropic_thinking:
        thinking_type = anthropic_thinking.get('type')
        return thinking_type if thinking_type in ('enabled', 'adaptive') else None
    if unified_thinking:
        return 'adaptive' if profile.get('anthropic_supports_adaptive_thinking', False) else 'enabled'
    if unified_thinking is False and not profile.get('thinking_always_enabled', False):
        return None
    return 'adaptive' if profile.get('thinking_enabled_by_default', False) else None


def _request_thinking_type(
    profile: AnthropicModelProfile,
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> Literal['enabled', 'adaptive'] | None:
    """`_effective_thinking_type` for a request, before or after `Model.prepare_request` runs.

    `params.thinking` is checked first since `Model.prepare_request` moves unified `thinking` from `model_settings`
    into it, but `AnthropicModel.prepare_request` also asks before that happens.
    """
    anthropic_settings = cast(AnthropicModelSettings, model_settings or {})
    unified_thinking = model_request_parameters.thinking
    if unified_thinking is None:
        unified_thinking = anthropic_settings.get('thinking')
    return _effective_thinking_type(anthropic_settings.get('anthropic_thinking'), unified_thinking, profile)


def _support_tool_forcing(
    model_name: str,
    profile: AnthropicModelProfile,
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> bool:
    """Whether to send a forced `tool_choice` ('any'/specific tool), raising if explicitly requested but unavailable.

    On top of the profile's forcing flags, extended thinking rejects forcing, and adaptive thinking accepts it
    but answers without thinking, so only an explicit forcing `tool_choice` is sent then.
    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#forcing-tool-use
    """
    thinking_type = _request_thinking_type(profile, model_settings, model_request_parameters)
    unavailable_reason = tool_forcing_unavailable_reason(
        profile,
        thinking=thinking_type is not None,
        thinking_remedy="Disable thinking with `thinking=False` or `anthropic_thinking={'type': 'disabled'}`",
    )
    if unavailable_reason is None and thinking_type == 'enabled':
        unavailable_reason = (
            "Extended thinking doesn't support forcing tool use. Disable thinking or use `tool_choice='auto'`."
        )
        if profile.get('anthropic_supports_adaptive_thinking', False):
            unavailable_reason += " Alternatively, `anthropic_thinking={'type': 'adaptive'}` supports forcing."
    return support_tool_forcing(
        model_name,
        model_settings,
        unavailable_reason,
        disables_thinking=thinking_type == 'adaptive' and profile.get('forced_tool_choice_disables_thinking', False),
    )
