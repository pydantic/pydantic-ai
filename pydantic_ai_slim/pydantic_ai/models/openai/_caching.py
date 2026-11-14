"""Prompt caching: breakpoint placement, instruction counts and the unified `cache` setting."""

from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any, Literal, TypeAlias, cast

from openai.types import chat, responses
from openai.types.chat import ChatCompletionContentPartParam
from typing_extensions import TypedDict

from ...exceptions import UserError
from ...messages import InstructionPart, ModelMessage, ModelRequest, SystemPromptPart
from ...profiles.openai import OpenAIModelProfile, OpenAISystemPromptRole
from ...settings import CacheSetting, ModelSettings, merge_model_settings
from .. import ModelRequestParameters
from .._prompt_cache import split_cache_setting
from ._settings import OpenAIChatModelSettings, OpenAIPromptCacheOptions

_OpenAIPromptCacheTTL: TypeAlias = Literal['30m', '12h']
"""The prompt cache TTLs OpenAI accepts on GPT-5.6 and later."""


class _OpenAIPromptCacheBreakpoint(TypedDict):
    mode: Literal['explicit']


_LEADING_CACHE_POINT_ERROR = (
    'CachePoint cannot be the first content in a user message - '
    'there must be previous content to attach the cache breakpoint to.'
)


_CHAT_BREAKPOINT_PART_TYPES = frozenset({'text', 'image_url', 'input_audio', 'file'})
"""Chat Completions content part types that accept a `prompt_cache_breakpoint`."""


_RESPONSES_BREAKPOINT_PART_TYPES = frozenset({'input_text', 'input_image', 'input_file'})
"""Responses input content types that accept a `prompt_cache_breakpoint`."""


def add_openai_prompt_cache_breakpoint(
    content: Sequence[ChatCompletionContentPartParam | responses.ResponseInputContentParam],
) -> None:
    """Mark the last content part as a cache breakpoint.

    A `CachePoint` that opens a user message has nothing in that message to attach to. The caller
    then appends an empty text part for it to sit on, which `move_leading_cache_breakpoints` moves
    onto the previous message once the whole request is mapped: "cache everything up to here" means
    the same there, and it's what Anthropic and Bedrock do.
    """
    cache_breakpoint: _OpenAIPromptCacheBreakpoint = {'mode': 'explicit'}
    content[-1]['prompt_cache_breakpoint'] = cache_breakpoint


def cacheable_instruction_count(instruction_parts: Sequence[InstructionPart]) -> int:
    """Number of leading static instruction parts, which the instruction breakpoint goes after.

    Agent runs sort static parts first, but a direct `Model.request` caller may not, and a dynamic
    part must never end up inside the cached prefix.

    Unlike Anthropic and Bedrock, this places no second breakpoint after the `on_change='append'` blocks that follow
    the static parts. OpenAI keeps only a request's last four breakpoints, so with a second one a `CachePoint` or two
    would push out the breakpoint after the static instructions first, which is the one every conversation shares
    even when its append blocks differ.
    """
    return next((i for i, part in enumerate(instruction_parts) if part.dynamic), len(instruction_parts))


def leading_system_message_count(
    messages: Sequence[Mapping[str, Any]], system_prompt_role: OpenAISystemPromptRole
) -> int:
    """Number of leading messages holding system prompts, which is where instructions belong."""
    return next((i for i, message in enumerate(messages) if message.get('role') != system_prompt_role), len(messages))


def has_dynamic_system_prompt(messages: Sequence[ModelMessage]) -> bool:
    """Whether any system prompt is dynamic, i.e. its content can change between requests.

    A dynamic system prompt renders ahead of the instructions in the cached prefix, so its changing
    content would silently invalidate the cache; the breakpoint is skipped when one is present.
    """
    return any(
        isinstance(part, SystemPromptPart) and part.dynamic_ref is not None
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def leading_cache_breakpoint(item: dict[str, Any]) -> _OpenAIPromptCacheBreakpoint | None:
    """The breakpoint a leading `CachePoint` left on an empty placeholder opening this user message, if any."""
    content = item.get('content')
    if item.get('role') != 'user' or not isinstance(content, list) or not content:
        return None
    first = cast('dict[str, Any]', content[0])
    if first.get('type') in ('text', 'input_text') and first.get('text') == '' and 'prompt_cache_breakpoint' in first:
        return cast(_OpenAIPromptCacheBreakpoint, first['prompt_cache_breakpoint'])
    return None


def move_leading_cache_breakpoints(
    items: list[chat.ChatCompletionMessageParam] | list[responses.ResponseInputItemParam],
    *,
    text_type: Literal['text', 'input_text'],
) -> None:
    """Move the breakpoint a leading `CachePoint` left on an empty placeholder onto the previous item.

    The breakpoint goes on the last content part of the closest earlier item that can carry it: a
    tool result, a system, user or (on Chat Completions) assistant message, or a function call
    output. A string body becomes a single text part so it can carry the marker; the text is
    unchanged. Responses assistant output (`output_text`, `refusal`) can't carry a breakpoint, so
    those items are skipped. Raises `UserError` when there's no earlier content at all, as Anthropic does.

    `OpenRouterModel._move_leading_cache_controls` does the same for OpenRouter's `cache_control`, with
    different rules; check it when changing this.
    """
    index = 0
    while index < len(items):
        item = cast('dict[str, Any]', items[index])
        breakpoint_value = leading_cache_breakpoint(item)
        if breakpoint_value is not None:
            if not attach_cache_breakpoint_before(items, index, breakpoint_value, text_type=text_type):
                raise UserError(_LEADING_CACHE_POINT_ERROR)
            content = cast('list[Any]', item['content'])
            if len(content) == 1:
                del items[index]
                continue
            item['content'] = content[1:]
        index += 1


def attach_cache_breakpoint_before(
    items: list[chat.ChatCompletionMessageParam] | list[responses.ResponseInputItemParam],
    index: int,
    breakpoint_value: _OpenAIPromptCacheBreakpoint,
    *,
    text_type: Literal['text', 'input_text'],
) -> bool:
    breakpoint_part_types = _CHAT_BREAKPOINT_PART_TYPES if text_type == 'text' else _RESPONSES_BREAKPOINT_PART_TYPES
    for previous in reversed(items[:index]):
        item = cast('dict[str, Any]', previous)
        key = 'output' if item.get('type') == 'function_call_output' else 'content'
        body = item.get(key)
        # A Responses assistant message's text is output (`output_text`), which can't carry a breakpoint.
        if isinstance(body, str) and body and not (text_type == 'input_text' and item.get('role') == 'assistant'):
            item[key] = [{'type': text_type, 'text': body, 'prompt_cache_breakpoint': breakpoint_value}]
            return True
        if isinstance(body, list) and body and cast('dict[str, Any]', body[-1]).get('type') in breakpoint_part_types:
            cast('dict[str, Any]', body[-1])['prompt_cache_breakpoint'] = breakpoint_value
            return True
    return False


CACHE_SETTINGS_KEYS = ('openai_prompt_cache_options', 'openai_cache_instructions')
"""The OpenAI settings that take precedence over the unified `cache` setting.

`openai_prompt_cache_retention` isn't one of them: it's a maximum-retention policy that OpenAI keeps
independent of the GPT-5.6 cache options, so it combines with the unified setting instead.
"""


def openai_effective_cache_settings(settings: OpenAIChatModelSettings) -> tuple[CacheSetting | None, ...]:
    """The cache settings that take effect when any of `CACHE_SETTINGS_KEYS` is present.

    The implicit breakpoint (`mode='implicit'`, the default) caches at the request-wide `ttl` (30 minutes by
    default), as does an instruction breakpoint; with `mode='explicit'` and no instruction breakpoint, only
    `CachePoint`s cache.
    """
    options = settings.get('openai_prompt_cache_options') or {}
    ttl = options.get('ttl', '30m')
    return (
        ttl if options.get('mode', 'implicit') == 'implicit' else None,
        ttl if settings.get('openai_cache_instructions') else None,
    )


def translate_openai_cache(
    profile: OpenAIModelProfile, model_settings: ModelSettings | None, params: ModelRequestParameters
) -> ModelSettings | None:
    """Map the unified `cache` setting onto GPT-5.6's prompt cache options and an instruction breakpoint.

    Only applies to models with explicit prompt cache breakpoints, and only when no explicit OpenAI
    cache setting (`CACHE_SETTINGS_KEYS`) is present, since those take precedence. `openai_cache_instructions`
    keeps its own gates, so requests that continue server-side state still get no instruction breakpoint.

    Caching only the stable prefix (`messages=False`) uses `mode='explicit'`, so OpenAI creates no implicit
    breakpoint and writes only the instruction breakpoint. `keep_implicit_cache_without_breakpoints` restores
    the implicit breakpoint on requests that end up with no breakpoint.
    """
    if (
        not params.cache
        or not profile.get('openai_supports_prompt_cache_breakpoints', False)
        or any(key in (model_settings or {}) for key in CACHE_SETTINGS_KEYS)
    ):
        return model_settings
    translated = cast(OpenAIChatModelSettings, {**(model_settings or {})})
    # The retention has been snapped to the profile's `supported_cache_retentions`, which OpenAI's TTLs make up.
    retention, messages = split_cache_setting(params.cache)
    ttl: _OpenAIPromptCacheTTL = '12h' if retention == '12h' else '30m'
    translated['openai_prompt_cache_options'] = (
        {'mode': 'implicit', 'ttl': ttl} if messages else _PREFIX_ONLY_CACHE_OPTIONS[ttl]
    )
    translated['openai_cache_instructions'] = True
    return translated


_PREFIX_ONLY_CACHE_OPTIONS: dict[_OpenAIPromptCacheTTL, OpenAIPromptCacheOptions] = {
    '30m': {'mode': 'explicit', 'ttl': '30m'},
    '12h': {'mode': 'explicit', 'ttl': '12h'},
}
"""The prompt cache options the unified `cache` setting maps `messages=False` to, per TTL.

Never mutated: `keep_implicit_cache_without_breakpoints` checks for these exact objects, so that options a
caller set with the same values, which take precedence over the unified setting, are left alone.
"""


def has_prompt_cache_breakpoint(
    items: Sequence[chat.ChatCompletionMessageParam] | Sequence[responses.ResponseInputItemParam],
) -> bool:
    """Whether any message or tool result in the request carries a prompt cache breakpoint."""
    for item in items:
        item_dict = cast('dict[str, Any]', item)
        body = item_dict.get('output' if item_dict.get('type') == 'function_call_output' else 'content')
        if isinstance(body, list) and any(
            'prompt_cache_breakpoint' in cast('dict[str, Any]', part) for part in cast('list[Any]', body)
        ):
            return True
    return False


def keep_implicit_cache_without_breakpoints(
    settings: OpenAIChatModelSettings,
    items: Sequence[chat.ChatCompletionMessageParam] | Sequence[responses.ResponseInputItemParam],
) -> None:
    """Keep OpenAI's implicit breakpoint when caching only the stable prefix leaves the request without a breakpoint.

    The unified `cache` setting with `messages=False` turns the implicit breakpoint off and relies on the
    instruction breakpoint, which `openai_cache_instructions` doesn't place on some requests (server-side
    state, merged or `'user'` system prompts, dynamic system prompts, no static instructions). With
    `mode='explicit'` and no breakpoint, the request would cache nothing, so it caches like `cache=True`
    instead. Checking the mapped request rather than repeating those gates keeps the two from drifting.
    Mutates `settings`.
    """
    options = settings.get('openai_prompt_cache_options')
    if (
        options is not None
        and any(options is prefix_only for prefix_only in _PREFIX_ONLY_CACHE_OPTIONS.values())
        and not has_prompt_cache_breakpoint(items)
    ):
        settings['openai_prompt_cache_options'] = {'mode': 'implicit', 'ttl': options.get('ttl', '30m')}


def resolve_legacy_cache_retention(
    profile: OpenAIModelProfile, default_settings: ModelSettings | None, model_settings: ModelSettings | None
) -> timedelta | None:
    """The 24-hour extended retention `openai_prompt_cache_retention` requests on models before GPT-5.6.

    On GPT-5.6 and later, `prompt_cache_retention` is a deprecated *maximum* that doesn't extend the
    30-minute minimum OpenAI guarantees, so it says nothing about how long the prefix stays cached.
    https://developers.openai.com/api/docs/guides/prompt-caching
    """
    if profile.get('openai_supports_prompt_cache_breakpoints', False):
        return None
    settings = merge_model_settings(default_settings, model_settings) or {}
    return timedelta(hours=24) if settings.get('openai_prompt_cache_retention') == '24h' else None
