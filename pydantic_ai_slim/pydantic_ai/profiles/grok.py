from __future__ import annotations as _annotations

from dataclasses import dataclass

from ..builtin_tools import SUPPORTED_BUILTIN_TOOLS, AbstractBuiltinTool
from . import ModelProfile


@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for Grok models (used with both GrokProvider and XaiProvider).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool = False
    """Whether the model supports builtin tools (web_search, code_execution, mcp)."""

    grok_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the request payload."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    model_lower = model_name.lower()

    # Grok-4 models support builtin tools
    grok_supports_builtin_tools = 'grok-4' in model_lower or 'code' in model_lower

    # Set supported builtin tools based on model capability
    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]] = (
        SUPPORTED_BUILTIN_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    # Reasoning model detection:
    # - grok-3-mini: reasoning with effort control (low/high)
    # - grok-4 reasoning variants: always-on reasoning, no effort control
    # - non-reasoning variants (e.g., grok-4-fast-non-reasoning): no thinking
    is_non_reasoning = 'non-reasoning' in model_lower
    is_grok_3_mini = 'grok-3-mini' in model_lower
    is_grok_4 = 'grok-4' in model_lower
    supports_thinking = (is_grok_3_mini or is_grok_4) and not is_non_reasoning
    thinking_always_enabled = supports_thinking and not is_grok_3_mini

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_builtin_tools=supported_builtin_tools,
        supports_thinking=supports_thinking,
        thinking_always_enabled=thinking_always_enabled,
    )
