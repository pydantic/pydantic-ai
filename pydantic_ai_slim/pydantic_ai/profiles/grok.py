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
    """Whether the model supports builtin tools (web_search, x_search, code_execution, mcp)."""

    grok_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the request payload."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    grok_supports_builtin_tools = model_name.startswith('grok-4') or 'code' in model_name
    is_always_reasoning = 'reasoning' in model_name and 'non-reasoning' not in model_name
    supports_thinking_effort = model_name.startswith('grok-3-mini')

    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]] = (
        SUPPORTED_BUILTIN_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=is_always_reasoning or supports_thinking_effort,
        thinking_always_enabled=is_always_reasoning,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_builtin_tools=supported_builtin_tools,
    )
