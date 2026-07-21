from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from textwrap import dedent
from typing import TYPE_CHECKING, Literal, TypeAlias

from typing_extensions import TypedDict

from .._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from ..messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..output import StructuredOutputMode

if TYPE_CHECKING:
    from ..messages import ModelMessage

__all__ = [
    'ModelProfile',
    'ModelProfileSpec',
    'DEFAULT_PROFILE',
    'DEFAULT_PROMPTED_OUTPUT_TEMPLATE',
    'DEFAULT_THINKING_TAGS',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
    'merge_profile',
    'PromptCacheOutlook',
    'prompt_cache_outlook',
]


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

    supports_inline_system_prompts: bool
    """Whether the provider's API accepts `SystemPromptPart`s inline at any position. Default: `False`.

    When `False`, non-leading `SystemPromptPart`s are wrapped as `UserPromptPart`s with
    `<system>...</system>` content in `Model.prepare_messages`. Leading ones still hoist to the
    provider's top-level system parameter.
    """

    default_structured_output_mode: StructuredOutputMode
    """The default structured output mode to use for the model. Default: `'tool'`."""

    prompted_output_template: str
    """The instructions template to use for prompted structured output. The `{schema}` placeholder will be replaced with the JSON schema for the output. Default: `DEFAULT_PROMPTED_OUTPUT_TEMPLATE`."""

    native_output_requires_schema_in_instructions: bool
    """Whether to add prompted output template in native structured output mode. Default: `False`."""

    json_schema_transformer: type[JsonSchemaTransformer] | None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model. Default: `None`."""

    supports_thinking: bool
    """Whether the model supports thinking/reasoning configuration. Default: `False`.

    When False, the unified `thinking` setting in `ModelSettings` is silently ignored.
    """

    thinking_always_enabled: bool
    """Whether the model always uses thinking/reasoning (e.g., OpenAI o-series, DeepSeek R1). Default: `False`.

    When True, `thinking=False` is silently ignored since the model cannot disable thinking.
    Implies `supports_thinking=True`.
    """

    thinking_tags: tuple[str, str]
    """The tags used to indicate thinking parts in the model's output. Default: [`DEFAULT_THINKING_TAGS`][pydantic_ai.profiles.DEFAULT_THINKING_TAGS]."""

    ignore_streamed_leading_whitespace: bool
    """Whether to ignore leading whitespace when streaming a response. Default: `False`.

    This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
    which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.

    This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, and `GroqModel`.
    """

    supported_native_tools: frozenset[type[AbstractNativeTool]]
    """The set of native tool types that this model/profile supports. Default: `SUPPORTED_NATIVE_TOOLS` (all)."""

    prompt_cache_retention: timedelta | None
    """How long after the last request the provider can still be expected to have the cached prefix available. Default: `None`.

    Only documented values are populated. When a provider documents a range, the higher end is used:
    consumers of a `'cold'` outlook are about to pay for a full prefix re-write, so a false `'cold'`
    sacrifices a live cache hit while a false `'warm'` merely defers maintenance. Because retention is
    provider infrastructure, providers populate this field; model-family profile functions must never
    set it. Providers without an honest documented expectation boundary leave it `None`.

    OpenAI's Responses API also has a `prompt_cache_retention` request parameter (`'in_memory' | '24h'`)
    that requests a retention policy; this profile field describes the resulting provider behavior. If
    you request extended retention, pass `retention=` to `prompt_cache_outlook` explicitly.

    Consumed by [`prompt_cache_outlook`][pydantic_ai.profiles.prompt_cache_outlook] to classify, from a
    message history alone, whether the next request is likely to hit a warm cache. A `'cold'` outlook is
    a free moment to run history-mutating maintenance (compaction, pruning, repair): the next request
    pays a full prefix re-write either way, so the marginal cache cost of the mutation is ~zero.
    """


DEFAULT_PROFILE: ModelProfile = {
    'supports_tools': True,
    'supports_tool_return_schema': False,
    'supports_json_schema_output': False,
    'supports_json_object_output': False,
    'supports_image_output': False,
    'default_structured_output_mode': 'tool',
    'prompted_output_template': DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
    'native_output_requires_schema_in_instructions': False,
    'json_schema_transformer': None,
    'supports_thinking': False,
    'thinking_always_enabled': False,
    'thinking_tags': DEFAULT_THINKING_TAGS,
    'ignore_streamed_leading_whitespace': False,
    'supported_native_tools': SUPPORTED_NATIVE_TOOLS,
    'prompt_cache_retention': None,
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
    """
    result: ModelProfile = {}
    if base:
        result = {**result, **base}
    for override in overrides:
        if override:
            result = {**result, **override}
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

    Args:
        messages: The message history the next request would be built on, oldest first.
        profile: The model profile whose [`prompt_cache_retention`][pydantic_ai.profiles.ModelProfile.prompt_cache_retention]
            is used as the expectation boundary. When present, cache points in the history extend this
            boundary to their largest TTL, assuming they were honored by the provider that served the requests.
        retention: An explicit expectation boundary, overriding both the profile and cache-point TTLs.
            Use this for settings-based extended retention that is not reflected in the profile.
        now: The reference time to measure idleness against. Defaults to the current UTC time; inject a
            fixed value for deterministic tests.

    Returns:
        `'warm'`, `'cold'`, or `'unknown'` (see [`PromptCacheOutlook`][pydantic_ai.profiles.PromptCacheOutlook]).
    """
    if retention is None and profile is not None:
        retention = profile.get('prompt_cache_retention')
        if retention is not None and (cache_point_ttl := _max_cache_point_ttl(messages)) is not None:
            retention = max(retention, cache_point_ttl)
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


_CACHE_POINT_TTLS: dict[str, timedelta] = {'5m': timedelta(minutes=5), '1h': timedelta(hours=1)}


def _max_cache_point_ttl(messages: Sequence[ModelMessage]) -> timedelta | None:
    """The largest [`CachePoint`][pydantic_ai.messages.CachePoint] TTL in the history, or `None` if there are none."""
    ttls = [
        _CACHE_POINT_TTLS[content.ttl]
        for message in messages
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
