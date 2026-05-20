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
        # Retired Grok 4/Grok 3 text slugs and their SDK aliases route to Grok 4.3.
        'grok-4',
        'grok-4-latest',
        'grok-4-0709',
        'grok-4-1-fast',
        'grok-4-1-fast-reasoning',
        'grok-4-1-fast-reasoning-latest',
        'grok-4-1-fast-non-reasoning',
        'grok-4-1-fast-non-reasoning-latest',
        'grok-4-fast',
        'grok-4-fast-reasoning',
        'grok-4-fast-reasoning-latest',
        'grok-4-fast-non-reasoning',
        'grok-4-fast-non-reasoning-latest',
        'grok-code-fast-1',
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
    grok_supports_builtin_tools = model_name.startswith('grok-4') or 'code' in model_name
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
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        grok_reasoning_efforts=grok_reasoning_efforts,
        supported_native_tools=supported_native_tools,
    )
