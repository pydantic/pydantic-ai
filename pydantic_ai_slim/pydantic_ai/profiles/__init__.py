from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

from typing_extensions import TypedDict

from .._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from ..exceptions import PydanticAIDeprecationWarning
from ..messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..output import StructuredOutputMode

if TYPE_CHECKING:
    from ..messages import ModelMessage

__all__ = [
    'ModelProfile',
    'ModelProfileSpec',
    'ToolAdditionMode',
    'ToolDeferralMode',
    'DEFAULT_PROFILE',
    'DEFAULT_PROMPTED_OUTPUT_TEMPLATE',
    'DEFAULT_THINKING_TAGS',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
    'merge_profile',
    'PromptCacheOutlook',
    'prompt_cache_outlook',
]

ToolDeferralMode: TypeAlias = Literal['standalone', 'with_tool_search']
ToolAdditionMode: TypeAlias = Literal['by_reference', 'with_definitions']


DEFAULT_PROMPTED_OUTPUT_TEMPLATE = dedent(
    """
    Always respond with a JSON object that's compatible with this schema:

    {schema}

    Don't include any text or Markdown fencing before or after.
    """
)
"""Default instructions template for prompted structured output. The `{schema}` placeholder is replaced with the JSON schema for the output."""

DEFAULT_THINKING_TAGS: tuple[str, str] = ('<think>', '</think>')
"""Default `(start_tag, end_tag)` pair for parsing thinking content out of text responses."""


class ModelProfile(TypedDict, total=False):
    """Describes how requests to and responses from specific models or families of models need to be constructed and processed to get the best results, independent of the model and provider classes used.

    All fields are optional; absent keys mean "use the documented default" (defaults are documented per field below and applied at access sites).

    Subclasses (`OpenAIModelProfile`, `AnthropicModelProfile`, ...) add provider-specific keys; cross-class merging via dict-spread is supported.
    """

    supports_tools: bool
    """Whether the model supports tools. Default: `True`."""

    supports_text_output: bool
    """Whether the model supports text output. Default: `True`."""

    supports_tool_return_schema: bool
    """Whether the model natively supports tool return schemas. Default: `False`.

    When True, the model's API accepts a structured return schema alongside each tool definition.
    When False, return schemas are injected as JSON text into tool descriptions as a fallback.
    """

    supports_json_schema_output: bool
    """Whether the model supports JSON schema output. Default: `False`.

    This is also referred to as 'native' support for structured output.
    Relates to the `NativeOutput` output type.
    """

    supports_json_object_output: bool
    """Whether the model supports a dedicated mode to enforce JSON output, without necessarily sending a schema. Default: `False`.

    E.g. [OpenAI's JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode)
    Relates to the `PromptedOutput` output type.
    """

    supports_image_output: bool
    """Whether the model supports image output. Default: `False`."""

    supports_audio_input: bool
    """Whether the model supports audio in user messages. Default: `False`.

    Used when converting `SpeechPart`s from realtime session history in
    `Model.prepare_messages`: if `True`, retained audio is sent to the model as `BinaryContent`;
    otherwise the transcript text is used.

    No shipping profile sets this to `True` yet, so retained realtime audio is currently always
    forwarded as transcript text on handoff; enabling it needs per-model-family verification that the
    provider accepts audio in user messages.
    """

    supports_inline_system_prompts: bool
    """Whether the provider's API accepts `SystemPromptPart`s inline at any position. Default: `False`.

    When `False`, non-leading `SystemPromptPart`s are wrapped as `UserPromptPart`s with
    `<system>...</system>` content in `Model.prepare_messages`. Leading ones still hoist to the
    provider's top-level system parameter.

    APIs that only accept an inline system prompt in certain positions (e.g. Anthropic requires it
    to follow a user turn) still set this to `True`; it's on their model adapters to make the
    positions the API rejects legal. Preserving the part's authority is worth more than preserving
    the exact position it was authored at — an instruction only governs the generation that follows
    it, and that's the same generation either way — so prefer adjusting placement over falling back
    to the `<system>...</system>` rendering, which the model reads as user-authored. Anthropic slides
    the entry past intervening user turns and gives it a minimal user turn to follow when nothing
    legal precedes it.

    `Provider.model_profile` is resolved from the model name alone, so when support also turns on
    something it can't see — which SDK client the provider was built with, say — the adapter narrows
    this in its own `Model.profile` override, as Anthropic does for Microsoft Foundry. Narrowing the
    flag rather than special-casing the adapter's own rendering keeps `Model.prepare_messages` the
    only place that knows the `<system>...</system>` fallback.
    """

    default_structured_output_mode: StructuredOutputMode
    """The default structured output mode to use for the model. Default: `'tool'`."""

    prompted_output_template: str
    """The instructions template to use for prompted structured output. The `{schema}` placeholder will be replaced with the JSON schema for the output. Default: `DEFAULT_PROMPTED_OUTPUT_TEMPLATE`."""

    native_output_requires_schema_in_instructions: bool
    """Whether to add prompted output template in native structured output mode. Default: `False`."""

    json_schema_transformer: type[JsonSchemaTransformer] | None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model. Default: `None`."""

    default_cache_retention: timedelta | None
    """How long the provider keeps a cached prompt prefix when the request doesn't ask for a specific retention. Default: `None` (unknown).

    Measured from the last request that used the prefix. Only documented values are populated. When a
    provider documents a range, the higher end is used: consumers of a `'cold'` outlook are about to pay
    for a full prefix re-write, so a false `'cold'` sacrifices a live cache hit while a false `'warm'`
    merely defers maintenance. Because retention is provider infrastructure, providers populate this
    field; model-family profile functions must never set it. Providers without an honest documented
    expectation boundary leave it `None`.

    Retention requested through model settings, such as `anthropic_cache='1h'` or
    `openai_prompt_cache_retention='24h'`, is resolved by
    [`Model.resolve_cache_retention`][pydantic_ai.models.Model.resolve_cache_retention]; this field is
    what applies when the settings request nothing.

    Consumed by [`prompt_cache_outlook`][pydantic_ai.profiles.prompt_cache_outlook] to classify, from a
    message history alone, whether the next request is likely to hit a warm cache. A `'cold'` outlook is
    a free moment to run history-mutating maintenance (compaction, pruning, repair): the next request
    pays a full prefix re-write either way, so the marginal cache cost of the mutation is ~zero.
    """

    supports_thinking: bool
    """Whether the model supports thinking/reasoning configuration. Default: `False`.

    When False, the unified `thinking` setting in `ModelSettings` is silently ignored.
    """

    thinking_always_enabled: bool
    """Whether the model always uses thinking/reasoning (e.g., OpenAI o-series, DeepSeek R1). Default: `False`.

    When True, `thinking=False` is silently ignored since the model cannot disable thinking.
    Implies `supports_thinking=True`.
    """

    thinking_enabled_by_default: bool
    """Whether the model thinks when the request doesn't configure thinking. Default: `False`.

    True for models that think unless told not to, such as Claude Opus 5, DeepSeek V4 and the OpenAI o-series. Pydantic AI
    uses it to tell whether a request without a thinking setting will think, for example to decide whether a
    tool call can be forced. Unlike `thinking_always_enabled`, it doesn't mean thinking can't be turned off.
    """

    supports_forced_tool_choice: bool
    """Whether the model accepts a forced tool choice: `tool_choice='required'` or a specific tool. Default: `True`.

    Some models reject forcing on every request, such as Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1,
    as do some OpenAI-compatible providers, such as Moonshot AI. When False, a forced tool choice that Pydantic AI
    resolved itself (such as an output tool's) falls back to `'auto'`, with the tools filtered to the requested
    ones where the API can't restrict the choice. An explicit forcing
    [`tool_choice`][pydantic_ai.settings.ModelSettings.tool_choice] raises a `UserError`.
    """

    supports_forced_tool_choice_with_thinking: bool
    """Whether the model accepts a forced tool choice while it thinks. Default: `True`.

    DeepSeek's V4 models, for example, only accept forcing while thinking is off. When False and the request
    thinks, a forced tool choice is handled as if `supports_forced_tool_choice` were False. Whether the request
    thinks accounts for `thinking_enabled_by_default`.
    """

    forced_tool_choice_disables_thinking: bool
    """Whether the model answers a forced tool choice without thinking. Default: `False`.

    Claude models accept a forced tool choice alongside adaptive thinking, but return no thinking for that
    request. When True and the request thinks, Pydantic AI doesn't force a tool choice it resolved itself (such
    as an output tool's): it falls back to `'auto'`, and a structured `output_type` defaults to
    [Native Output](../output.md#native-output) where the model supports it. An explicit forcing
    [`tool_choice`][pydantic_ai.settings.ModelSettings.tool_choice] is still sent.
    """

    thinking_tags: tuple[str, str]
    """The tags used to indicate thinking parts in the model's output. Default: [`DEFAULT_THINKING_TAGS`][pydantic_ai.profiles.DEFAULT_THINKING_TAGS]."""

    ignore_streamed_leading_whitespace: bool
    """Whether to ignore leading whitespace when streaming a response. Default: `False`.

    This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
    which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.

    This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, `GroqModel`, and `BedrockConverseModel`.
    """

    supported_native_tools: frozenset[type[AbstractNativeTool]]
    """The set of native tool types that this model/profile supports. Default: `SUPPORTED_NATIVE_TOOLS` (all)."""

    context_window: int | None
    """The maximum number of tokens the model can handle in a single request, input and output combined. Default: `None` (unknown).

    When no profile layer sets this, `Model.profile` fills it in from
    [genai-prices](https://github.com/pydantic/genai-prices) data if the model is known there.
    Set it explicitly for custom or local models, e.g. `profile={'context_window': 128_000}`.
    """

    tool_deferral_mode: ToolDeferralMode | None
    """When the provider permits a `tools` entry whose schema is withheld. Default: `None`.

    `'standalone'` permits the deferral flag on its own. `'with_tool_search'` permits it only when a
    tool-search tool is present in the same request. `None` means hidden tools can only be withheld
    from the wire. Unsupported deferral is handled on a best-effort basis by withholding the tool.
    """

    tool_addition_mode: ToolAdditionMode | None
    """How the model natively expresses tools added mid-conversation. Default: `None`.

    `'by_reference'` reveals a tool already declared in the request's tool definitions (Anthropic
    `tool_addition` blocks referencing a `defer_loading` entry); `'with_definitions'` carries the full
    newly available definitions in the reveal (OpenAI Responses `additional_tools` items). `None` means
    no native channel: `Model.prepare_messages` projects the change into messages. Additions only —
    tool removal (#6985) is not modeled yet and will get its own field.
    """

    tool_additions: ToolAdditionMode | None
    """Deprecated: use `tool_addition_mode` instead.

    Translated (with a deprecation warning) whenever profiles are merged; an explicit
    `tool_addition_mode` in the same profile wins.
    """

    deferred_tools_require_tool_search: bool
    """Deprecated: use `tool_deferral_mode` instead.

    `True` translates to `tool_deferral_mode='with_tool_search'` (with a deprecation warning)
    whenever profiles are merged. `False` carried no signal on its own — deferral capability came
    from native tool-search support — so it is dropped; an explicit `tool_deferral_mode` in the
    same profile wins.
    """


_LEGACY_PROVIDER_PROFILE_KEYS: dict[str, str] = {
    'openai_supports_tool_choice_required': 'supports_forced_tool_choice',
    'grok_supports_tool_choice_required': 'supports_forced_tool_choice',
    'anthropic_supports_forced_tool_choice': 'supports_forced_tool_choice',
    'openai_supports_forced_tool_choice_with_thinking': 'supports_forced_tool_choice_with_thinking',
    'openrouter_supports_forced_tool_choice_with_thinking': 'supports_forced_tool_choice_with_thinking',
    'openai_reasoning_enabled_by_default': 'thinking_enabled_by_default',
}
"""Provider-prefixed profile keys that moved to `ModelProfile`, mapped to their current spelling."""


def _translate_legacy_profile_keys(profile: ModelProfile, base: ModelProfile | None = None) -> ModelProfile:
    """Translate keys renamed after their release into their current spellings, warning.

    A current spelling in the same profile wins, unless `profile` was derived from `base` (as a callable
    `ModelProfileSpec`'s result is) and it only carries `base`'s value over.
    """
    if (
        'tool_additions' not in profile
        and 'deferred_tools_require_tool_search' not in profile
        and _LEGACY_PROVIDER_PROFILE_KEYS.keys().isdisjoint(profile)
    ):
        return profile
    translated: dict[str, object] = dict(profile)

    def set_translated(key: str, value: object) -> None:
        if key not in translated or (base is not None and translated[key] == base.get(key)):
            translated[key] = value

    legacy_values: dict[str, bool] = {}
    for legacy_key, key in _LEGACY_PROVIDER_PROFILE_KEYS.items():
        if legacy_key in translated:
            warnings.warn(
                f'`ModelProfile` key `{legacy_key}` is deprecated, use `{key}` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=3,
            )
            # Two legacy spellings of the same capability in one profile both have to allow it.
            legacy_values[key] = legacy_values.get(key, True) and bool(translated.pop(legacy_key))
    for key, value in legacy_values.items():
        set_translated(key, value)
    if 'tool_additions' in translated:
        warnings.warn(
            '`ModelProfile` key `tool_additions` is deprecated, use `tool_addition_mode` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=3,
        )
        set_translated('tool_addition_mode', translated.pop('tool_additions'))
    if 'deferred_tools_require_tool_search' in translated:
        warnings.warn(
            '`ModelProfile` key `deferred_tools_require_tool_search` is deprecated, use '
            "`tool_deferral_mode='with_tool_search'` instead.",
            PydanticAIDeprecationWarning,
            stacklevel=3,
        )
        if translated.pop('deferred_tools_require_tool_search'):
            set_translated('tool_deferral_mode', 'with_tool_search')
    return cast('ModelProfile', translated)


DEFAULT_PROFILE: ModelProfile = {
    'supports_tools': True,
    'supports_text_output': True,
    'supports_tool_return_schema': False,
    'supports_json_schema_output': False,
    'supports_json_object_output': False,
    'supports_image_output': False,
    'supports_audio_input': False,
    'default_structured_output_mode': 'tool',
    'prompted_output_template': DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
    'native_output_requires_schema_in_instructions': False,
    'json_schema_transformer': None,
    'default_cache_retention': None,
    'supports_thinking': False,
    'thinking_always_enabled': False,
    'thinking_enabled_by_default': False,
    'supports_forced_tool_choice': True,
    'supports_forced_tool_choice_with_thinking': True,
    'forced_tool_choice_disables_thinking': False,
    'thinking_tags': DEFAULT_THINKING_TAGS,
    'ignore_streamed_leading_whitespace': False,
    'supported_native_tools': SUPPORTED_NATIVE_TOOLS,
    'context_window': None,
    'tool_deferral_mode': None,
    'tool_addition_mode': None,
}
"""Fully populated default `ModelProfile`. Used as the base layer when resolving a model's effective profile."""


ModelProfileSpec: TypeAlias = ModelProfile | Callable[['ModelProfile'], 'ModelProfile']
"""Acceptable shapes for the `profile=` argument on a `Model`.

- A `ModelProfile` dict — a partial profile, merged on top of the provider's resolved default.
- A `Callable[[ModelProfile], ModelProfile]` — receives the provider's resolved default (with `DEFAULT_PROFILE` already merged in) and returns the final profile (full control: replace, derive, ignore the default).

Provider classes still expose `Provider.model_profile(model_name)` (`Callable[[str], ModelProfile | None]`) — that's a separate concept used internally by `Model.profile` to resolve the provider's default for a given model name.
"""


def merge_profile(base: ModelProfile | None, *overrides: ModelProfile | None) -> ModelProfile:
    """Merge profiles via dict-spread. Later arguments override earlier ones; `None` is treated as empty.

    This is the canonical way to layer profiles in providers and tests; replaces the old `ModelProfile.update()` method.
    Deprecated key spellings are translated per input before spreading, so a legacy key in an
    override still overrides the base.
    """
    result: ModelProfile = {}
    if base:
        result = {**result, **_translate_legacy_profile_keys(base)}
    for override in overrides:
        if override:
            result = {**result, **_translate_legacy_profile_keys(override)}
    return result


PromptCacheOutlook: TypeAlias = Literal['warm', 'cold', 'unknown']
"""Predicted state of the provider's prompt cache for the *next* request built on a message history.

- `'warm'`: the last request happened within the provider's documented expectation boundary, so the cached
  prefix is likely still available and the next request should hit it.
- `'cold'`: the last request happened longer ago than the retention window, so the prefix has likely
  been evicted and the next request will pay full input price regardless — a free moment to mutate history.
- `'unknown'`: there's no retention figure for the model, or the history has no usable timestamp, so no
  prediction can be made. Treat like `'warm'` for scheduling (never mutate on a guess).
"""


def prompt_cache_outlook(
    messages: Sequence[ModelMessage],
    *,
    profile: ModelProfile | None = None,
    retention: timedelta | None = None,
    now: datetime | None = None,
) -> PromptCacheOutlook:
    """Predict whether the provider's prompt cache is still warm for the next request on this history.

    This is a pure function of the message history and an expectation boundary — it holds no state and makes no
    requests, so a history processor, capability, or plain application code can call it with just a
    message history to decide whether the next turn is a cheap moment for history-mutating maintenance
    (compaction, pruning, repair). When the outlook is `'cold'` the next request pays a full prefix
    re-write anyway, so the marginal cache cost of mutating history right now is ~zero.

    The prediction compares the most recent [`ModelResponse.timestamp`][pydantic_ai.messages.ModelResponse.timestamp]
    in `messages` against `now`: an idle gap within the retention window is `'warm'`, a larger gap is `'cold'`.
    Responses are the anchor because they mark the provider's last confirmed use of the cache — a request that
    has no response after it (like the just-appended request a [history processor](../message-history.md#processing-message-history)
    sees, which hasn't been sent yet) never touched the cache, so its timestamp must not reset the idle clock.

    Cache points in the history extend whichever boundary applies to their largest TTL, assuming they
    were honored by the provider that served the requests.

    Args:
        messages: The message history the next request would be built on, oldest first.
        profile: The model profile whose [`default_cache_retention`][pydantic_ai.profiles.ModelProfile.default_cache_retention]
            is used as the expectation boundary when `retention` is `None`.
        retention: The retention requested for these requests, replacing the profile's default. With a model
            and its settings in hand, pass
            [`model.resolve_cache_retention(model_settings)`][pydantic_ai.models.Model.resolve_cache_retention]:
            it returns `None` when the settings request nothing, so the profile's default still applies.
        now: The reference time to measure idleness against. Defaults to the current UTC time; inject a
            fixed value for deterministic tests.

    Returns:
        `'warm'`, `'cold'`, or `'unknown'` (see [`PromptCacheOutlook`][pydantic_ai.profiles.PromptCacheOutlook]).
    """
    retention = _expected_cache_retention(messages, profile=profile, retention=retention)
    if retention is None:
        return 'unknown'

    last_timestamp = _last_response_timestamp(messages)
    if last_timestamp is None:
        return 'unknown'

    if now is None:
        now = datetime.now(timezone.utc)
    # Historical messages may carry naive timestamps; assume UTC so the subtraction is well-defined.
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    idle = now - last_timestamp
    return 'warm' if idle <= retention else 'cold'


def _expected_cache_retention(
    messages: Sequence[ModelMessage], *, profile: ModelProfile | None, retention: timedelta | None
) -> timedelta | None:
    """How long the provider is expected to keep this history's cached prefix, or `None` if unknown.

    `retention` (the retention requested by settings) replaces the profile's default, and cache points in
    `messages` extend either to their largest TTL. Cache points alone never produce a boundary: without a
    known base, whether the provider honored them at all is unknown.
    """
    if retention is None and profile is not None:
        retention = profile.get('default_cache_retention')
    if retention is not None and (cache_point_ttl := _max_cache_point_ttl(messages)) is not None:
        retention = max(retention, cache_point_ttl)
    return retention


_CACHE_POINT_TTLS: dict[str, timedelta] = {'5m': timedelta(minutes=5), '1h': timedelta(hours=1)}


def _max_cache_point_ttl(messages: Sequence[ModelMessage]) -> timedelta | None:
    """The largest [`CachePoint`][pydantic_ai.messages.CachePoint] TTL in the served history, or `None` if there are none.

    Like the idle clock, this only counts requests with a response after them: a cache point on a
    request that hasn't been sent yet hasn't written anything to the provider's cache.
    """
    served = next((index for index in range(len(messages), 0, -1) if isinstance(messages[index - 1], ModelResponse)), 0)
    ttls = [
        _CACHE_POINT_TTLS[content.ttl]
        for message in messages[:served]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for content in part.content
        if isinstance(content, CachePoint)
    ]
    return max(ttls) if ttls else None


def _last_response_timestamp(messages: Sequence[ModelMessage]) -> datetime | None:
    """The most recent `ModelResponse` timestamp in the history, scanning from the end.

    Responses mark the provider's last confirmed use of the cache. Requests are deliberately not
    considered: inside an agent run, the just-appended `ModelRequest` is timestamped *before* history
    processors see it, so anchoring on it would make the history look permanently warm — and a request
    with no response after it never reached the provider's cache in the first place.
    """
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return message.timestamp
    return None
