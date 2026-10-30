from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from . import ModelProfile

if TYPE_CHECKING:
    from ..realtime.profiles import RealtimeModelProfile

GrokReasoningEffort: TypeAlias = Literal['none', 'low', 'medium', 'high']
"""Native xAI `reasoning_effort` values."""

_GROK_BASIC_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('low', 'high'))
_GROK_43_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('none', 'low', 'medium', 'high'))
# Grok 3 Mini keeps its documented `low`/`high` set for the effort levels, plus `none` to turn reasoning off.
_GROK_3_MINI_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('none', 'low', 'high'))
# Grok 4.5 and 4.6 accept `low`/`medium`/`high` but reject `none` (unlike Grok 4.3), so they always
# reason. Grok 4.7 also supports `xhigh`; its mapping is tracked separately in issue #9744, so this
# profile keeps the existing high fallback. https://docs.x.ai/developers/model-capabilities/text/reasoning
_GROK_45_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('low', 'medium', 'high'))
_GROK_47_REASONING_MODELS = frozenset(('grok-4.7',))
_GROK_43_REASONING_MODELS = frozenset(
    (
        'grok-4.3',
        'grok-4.3-latest',
        # `grok-latest` is xAI's floating alias for the newest Grok model. It now serves Grok 4.6, which
        # rejects `reasoning_effort='none'` under its own id, but the alias still takes all four values:
        # xAI routes a `none` request on it to Grok 4.3 (live-verified). https://docs.x.ai/developers/models
        'grok-latest',
        # Retired text slugs that xAI redirects to Grok 4.3, so they accept its `reasoning_effort`
        # values. These exact six are the only slugs the retirement guide maps to Grok 4.3
        # (`grok-code-fast-1` redirects to `grok-build-0.1` instead, so it is excluded).
        # https://docs.x.ai/developers/migration/may-15-retirement
        'grok-4-0709',
        'grok-4-1-fast-reasoning',
        'grok-4-1-fast-non-reasoning',
        'grok-4-fast-reasoning',
        'grok-4-fast-non-reasoning',
        'grok-3',
    )
)
# Grok 3 Mini accepts `reasoning_effort='none'`, though xAI's retirement guide doesn't list it: verified live
# 2026-10-22, `grok-3-mini` and `grok-3-mini-fast` return no reasoning tokens for `none`, so `thinking=False`
# turns reasoning off. The other levels keep mapping onto its documented `low`/`high`. They stay out of
# `_GROK_43_REASONING_MODELS`, which also grants builtin tools. Other `grok-3-mini` ids keep the `low`/`high` set.
_GROK_3_MINI_DISABLEABLE_MODELS = frozenset(('grok-3-mini', 'grok-3-mini-fast'))
_GROK_45_REASONING_MODELS = (
    frozenset(
        (
            'grok-4.5',
            'grok-4.5-latest',
            # Grok 4.6 takes the same `reasoning_effort` values as 4.5, so it shares the set rather than
            # getting a branch of its own. `grok-4.6-latest` is deliberately absent: xAI doesn't serve one.
            'grok-4.6',
            # `grok-build-latest` is xAI's floating alias for the newest Grok build model, currently Grok 4.5,
            # so it accepts the same `reasoning_effort` values. https://docs.x.ai/developers/models
            'grok-build-latest',
        )
    )
    | _GROK_47_REASONING_MODELS
)


class GrokModelProfile(ModelProfile, total=False):
    """Profile for Grok models (used with XaiProvider and various OpenAI-compatible providers).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool
    """Whether the model supports builtin tools (web_search, x_search, code_execution, mcp). Default: `False`."""

    grok_supports_tool_choice_required: bool
    """Deprecated: use [`supports_forced_tool_choice`][pydantic_ai.profiles.ModelProfile.supports_forced_tool_choice] instead.

    Translated (with a deprecation warning) whenever profiles are merged.
    """

    grok_reasoning_efforts: frozenset[GrokReasoningEffort]
    """Native `reasoning_effort` values supported by the Grok model. Default: empty (`frozenset()`)."""

    grok_default_effort_omitted: bool
    """Whether a default thinking request omits `reasoning_effort` so xAI applies its model default. Default: `False`."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    # The retirement-redirect slugs in `_GROK_43_REASONING_MODELS` (e.g. `grok-3`) route to Grok 4.3,
    # which supports builtin tools, so they're builtin-capable too even when the name doesn't match the
    # `grok-4`/`code`/`build` patterns (the `code`/`build` coding models also support builtin tools).
    # Kept as its own flag rather than folded into reasoning-effort support: the two gate different
    # behaviors and shouldn't be derived from a single predicate.
    grok_supports_builtin_tools = (
        model_name.startswith('grok-4')
        or 'code' in model_name
        or 'build' in model_name
        or model_name in _GROK_43_REASONING_MODELS
    )
    grok_reasoning_efforts: frozenset[GrokReasoningEffort]
    if model_name in _GROK_43_REASONING_MODELS:
        grok_reasoning_efforts = _GROK_43_REASONING_EFFORTS
    elif model_name in _GROK_3_MINI_DISABLEABLE_MODELS:
        grok_reasoning_efforts = _GROK_3_MINI_REASONING_EFFORTS
    elif model_name in _GROK_45_REASONING_MODELS:
        grok_reasoning_efforts = _GROK_45_REASONING_EFFORTS
    elif model_name.startswith('grok-3-mini'):
        grok_reasoning_efforts = _GROK_BASIC_REASONING_EFFORTS
    else:
        grok_reasoning_efforts = frozenset()

    supported_native_tools: frozenset[type[AbstractNativeTool]] = (
        SUPPORTED_NATIVE_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    profile: GrokModelProfile = GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=bool(grok_reasoning_efforts),
        # A reasoning model whose `reasoning_effort` set lacks `'none'` (e.g. Grok 4.5) reasons by
        # default and can't be disabled, so it's always-on; Grok 4.3 supports `'none'`, so it's not.
        thinking_always_enabled=bool(grok_reasoning_efforts) and 'none' not in grok_reasoning_efforts,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        grok_reasoning_efforts=grok_reasoning_efforts,
        supported_native_tools=supported_native_tools,
    )
    if model_name in _GROK_43_REASONING_MODELS or model_name in _GROK_47_REASONING_MODELS:
        profile['grok_default_effort_omitted'] = True
    if model_name in _GROK_3_MINI_DISABLEABLE_MODELS:
        # Grok 3 Mini reasons unless told not to.
        profile['thinking_enabled_by_default'] = True
    return profile


def grok_realtime_model_profile(model_name: str) -> RealtimeModelProfile:
    """Get the realtime model profile for an xAI Grok Voice model."""
    return {
        'supports_manual_turn_control': True,
        'supports_interruption': True,
        # Grok Voice always speaks: the API accepts a response-modality setting but ignores it (checked
        # live), so an `output_modality='text'` session would silently come back as audio.
        'supports_text_output': False,
        'supports_session_seeding': True,
        # xAI accepts a mid-session `system` message but doesn't answer it as instructions (checked live: asked to
        # greet the newly verified caller by name, Grok did in none of six runs with a `system` message, and in all
        # three runs with `<system>`-tagged user text).
        'supports_system_messages': False,
        'supports_seeding_images': False,
        'supports_seeding_audio': False,
        # xAI puts `think` in the name of the voice models that take `reasoning.effort`, so match on
        # that rather than pinning versions: `grok-voice-think-fast-2.0` shipped a week after 1.0,
        # and a pinned list would have silently dropped reasoning for anyone who moved to it.
        'supports_thinking': model_name == 'grok-voice-latest' or model_name.startswith('grok-voice-think-'),
        # Like the OpenAI Realtime API it mirrors, Grok Voice answers the user while a tool call is
        # outstanding: verified live 2026-09-25, a question sent 3 seconds into a 15-second tool was
        # answered right away. The protocol has no setting that makes it wait.
        'async_tool_call_mode': 'always',
        'emits_input_speech_events': True,
        # A response's `input_tokens` counts only the input added since the previous response, not the
        # whole conversation, so it can't measure how full the context window is.
        'response_usage_covers_context': False,
        'audio_input_sample_rate': 24000,
        'audio_output_sample_rate': 24000,
    }
