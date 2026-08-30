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
# Grok 4.5 and 4.6 accept `low`/`medium`/`high` but reject `none` (unlike Grok 4.3), so they always
# reason. Verified against the xAI API for both: `reasoning_effort='none'` returns 400 `This model does
# not support 'reasoning_effort' value 'none'`. https://docs.x.ai/developers/models
_GROK_45_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('low', 'medium', 'high'))
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
_GROK_45_REASONING_MODELS = frozenset(
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


class GrokModelProfile(ModelProfile, total=False):
    """Profile for Grok models (used with XaiProvider and various OpenAI-compatible providers).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool
    """Whether the model supports builtin tools (web_search, x_search, code_execution, mcp). Default: `False`."""

    grok_supports_tool_choice_required: bool
    """Whether the provider accepts the value `tool_choice='required'` in the request payload. Default: `True`."""

    grok_reasoning_efforts: frozenset[GrokReasoningEffort]
    """Native `reasoning_effort` values supported by the Grok model. Default: empty (`frozenset()`)."""


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
    elif model_name in _GROK_45_REASONING_MODELS:
        grok_reasoning_efforts = _GROK_45_REASONING_EFFORTS
    elif model_name.startswith('grok-3-mini'):
        grok_reasoning_efforts = _GROK_BASIC_REASONING_EFFORTS
    else:
        grok_reasoning_efforts = frozenset()

    supported_native_tools: frozenset[type[AbstractNativeTool]] = (
        SUPPORTED_NATIVE_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=bool(grok_reasoning_efforts),
        # A reasoning model whose `reasoning_effort` set lacks `'none'` (e.g. grok-3-mini) reasons by
        # default and can't be disabled, so it's always-on; Grok 4.3 supports `'none'`, so it's not.
        thinking_always_enabled=bool(grok_reasoning_efforts) and 'none' not in grok_reasoning_efforts,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        grok_reasoning_efforts=grok_reasoning_efforts,
        supported_native_tools=supported_native_tools,
    )


def grok_realtime_model_profile(model_name: str) -> RealtimeModelProfile:
    """Get the realtime model profile for an xAI Grok Voice model."""
    return {
        'supports_manual_turn_control': True,
        'supports_interruption': True,
        # Grok Voice always speaks: the API has no response-modality control, so an
        # `output_modality='text'` session would silently come back as audio.
        'supports_text_output': False,
        'supports_session_seeding': True,
        'supports_seeding_images': False,
        'supports_seeding_audio': False,
        # xAI puts `think` in the name of the voice models that take `reasoning.effort`, so match on
        # that rather than pinning versions: `grok-voice-think-fast-2.0` shipped a week after 1.0,
        # and a pinned list would have silently dropped reasoning for anyone who moved to it.
        'supports_thinking': model_name == 'grok-voice-latest' or model_name.startswith('grok-voice-think-'),
        'emits_input_speech_events': True,
        'audio_input_sample_rate': 24000,
        'audio_output_sample_rate': 24000,
    }
