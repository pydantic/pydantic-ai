from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from . import ModelProfile

GrokReasoningEffort: TypeAlias = Literal['none', 'low', 'medium', 'high']
"""Native xAI `reasoning_effort` values."""

_GROK_BASIC_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('low', 'high'))
_GROK_43_REASONING_EFFORTS: frozenset[GrokReasoningEffort] = frozenset(('none', 'low', 'medium', 'high'))
_GROK_43_REASONING_MODELS = frozenset(
    (
        'grok-4.3',
        'grok-4.3-latest',
        # `grok-latest` is xAI's floating alias for the newest Grok model, which is currently Grok 4.3,
        # so it accepts the same `reasoning_effort` values. https://docs.x.ai/developers/models
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


@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for Grok models (used with both GrokProvider and XaiProvider).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool = False
    """Whether the model supports builtin tools (web_search, x_search, code_execution, mcp)."""

    grok_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value `tool_choice='required'` in the request payload."""

    grok_reasoning_efforts: frozenset[GrokReasoningEffort] = frozenset()
    """Native `reasoning_effort` values supported by the Grok model."""


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
