"""The parameters of one `messages.create` or `count_tokens` request, and the client that sends it."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from anthropic import (
    AsyncAnthropicBedrock,  # pyright: ignore[reportPrivateImportUsage]
    NotGiven,
    Omit,
    Timeout,
    omit as OMIT,
)
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaCompactionConfigParam,
    BetaContainerParams,
    BetaContextManagementConfigParam,
    BetaDiagnosticsParam,
    BetaMessageParam,
    BetaMetadataParam,
    BetaOutputConfigParam,
    BetaRequestMCPServerURLDefinitionParam,
    BetaTextBlockParam,
    BetaThinkingConfigParam,
    BetaToolChoiceParam,
    BetaToolUnionParam,
)
from typing_extensions import TypedDict

from ..._utils import is_str_mapping
from ...profiles.anthropic import ANTHROPIC_SAMPLING_PARAMS, AnthropicModelProfile
from ...providers.anthropic import AsyncAnthropicClient
from ...settings import ModelSettings
from .._sdk_retries import with_max_retries
from ._bedrock_count_tokens import BEDROCK_ANTHROPIC_VERSION

if TYPE_CHECKING:
    from . import AnthropicModelSettings


def build_extra_body(
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
    if not is_str_mapping(extra_body):
        return fields
    merged = {**fields, **extra_body}
    if thinking_override is not None:
        # The override was built from the effective wire object, including any caller-provided
        # `extra_body['thinking']`, so it must replace that original object to add `block_binding`.
        merged['thinking'] = thinking_override
    return merged


def extra_body_thinking(model_settings: AnthropicModelSettings) -> dict[str, object] | None:
    """The `thinking` object a caller hand-rolled in `extra_body`, if any.

    `extra_body` is how a caller reaches a field the installed SDK cannot type yet, so it can carry
    `thinking` — including a `block_binding` of their own, which must be honored rather than
    retried over.
    """
    extra_body = model_settings.get('extra_body')
    if not is_str_mapping(extra_body):
        return None
    thinking = extra_body.get('thinking')
    if not is_str_mapping(thinking):
        return None
    return dict(thinking)


def resolve_effective_thinking(
    model_settings: AnthropicModelSettings, thinking: BetaThinkingConfigParam | Omit
) -> dict[str, object] | Omit:
    """Resolve the thinking object that reaches the wire after `extra_body` precedence."""
    if (caller_thinking := extra_body_thinking(model_settings)) is not None:
        return caller_thinking
    return OMIT if isinstance(thinking, Omit) else dict(thinking)


_DEFAULT_MAX_TOKENS = 16384
"""The `max_tokens` sent when the request doesn't set one and the model's maximum output is unknown.

It stays under the SDK's non-streaming limit (`MAX_NON_STREAMING_TOKENS`).
"""

MAX_NON_STREAMING_TOKENS = 21_333
"""The largest `max_tokens` the Anthropic SDK sends without streaming with its default timeout.

The SDK expects a response to take up to an hour per 128,000 tokens, and requires streaming past 10 minutes.
"""

_LEGACY_DEFAULT_MAX_TOKENS = 4096
"""The default `max_tokens` for models that reject input plus `max_tokens` beyond the context window."""

_MIN_TOKENS_AFTER_THINKING_BUDGET = 4096
"""The room the default `max_tokens` leaves beyond an extended thinking `budget_tokens`."""


class MessagesCreateArgs(TypedDict):
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
    inference_geo: str | Omit
    diagnostics: BetaDiagnosticsParam | Omit
    extra_body: object | None


class CountTokensArgs(TypedDict):
    """The parameters a `count_tokens` request sends on every client. Bedrock's also sends `max_tokens`."""

    system: str | list[BetaTextBlockParam] | Omit
    messages: list[BetaMessageParam]
    model: str
    tools: list[BetaToolUnionParam] | Omit
    tool_choice: BetaToolChoiceParam | Omit
    mcp_servers: list[BetaRequestMCPServerURLDefinitionParam] | Omit
    betas: list[AnthropicBetaParam] | Omit
    output_config: BetaOutputConfigParam | Omit
    cache_control: BetaCacheControlEphemeralParam | Omit
    thinking: BetaThinkingConfigParam | Omit
    context_management: BetaContextManagementConfigParam | Omit
    timeout: float | Timeout | NotGiven
    speed: Literal['standard', 'fast'] | Omit
    extra_headers: dict[str, str]
    extra_body: object | None


@dataclass(kw_only=True)
class PreparedMessagesCreate:
    """The `messages.create` parameters of one request, before any recovery retry changes them.

    `_messages_create` sends them, retrying with a different container or thinking object when the API
    rejects one; a batch request encodes the initial values once, as it gets no retry. `_messages_count_tokens`
    counts the ones `_prepare_messages_create` builds with `count_tokens`.
    """

    system_prompt: str | list[BetaTextBlockParam]
    anthropic_messages: list[BetaMessageParam]
    tools: list[BetaToolUnionParam]
    tool_choice: BetaToolChoiceParam | None
    mcp_servers: list[BetaRequestMCPServerURLDefinitionParam]
    output_config: BetaOutputConfigParam | None
    auto_cache_control: BetaCacheControlEphemeralParam | None
    cache_breakpoints: tuple[list[str], list[str]] | None
    """The cache breakpoints the request asks for and the ones it keeps, for each attempt to record as it's sent.

    `None` when there are none, or when `_prepare_messages_create` recorded them already.
    """
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
    inference_geo: str | Omit


def default_max_tokens(thinking: dict[str, object] | Omit, profile: AnthropicModelProfile) -> int:
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


def resolve_anthropic_service_tier(
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


def with_max_retries_keeping_aws_profile(
    client: AsyncAnthropicClient, model_settings: ModelSettings
) -> AsyncAnthropicClient:
    """The client to send one request with, carrying `ModelSettings['max_retries']` when it's set."""
    if model_settings.get('max_retries') is None or not isinstance(client, AsyncAnthropicBedrock):
        return with_max_retries(client, model_settings)
    copy = with_max_retries(client, model_settings)
    if copy.aws_profile == client.aws_profile:
        return copy
    # Below anthropic 2.6, `AsyncAnthropicBedrock.with_options()` doesn't copy `aws_profile`, so its copy would sign
    # requests with the default AWS credential chain instead of the profile the client was built with. From 2.6 it
    # copies the profile itself, and raises a `TypeError` if `_extra_kwargs` passes it again.
    return with_max_retries(client, model_settings, carry_over={'aws_profile': client.aws_profile})
