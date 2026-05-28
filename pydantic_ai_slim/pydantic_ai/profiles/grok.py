from __future__ import annotations as _annotations

from dataclasses import dataclass

from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from . import ModelProfile


@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for Grok models (used with both GrokProvider and XaiProvider).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool = False
    """Whether the model supports builtin tools (web_search, x_search, code_execution, mcp)."""

    grok_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value `tool_choice='required'` in the request payload."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    grok_supports_builtin_tools = (
        model_name.startswith('grok-4')
        or model_name.startswith('grok-build')
        or model_name == 'grok-latest'
        or 'code' in model_name
    )
    supports_thinking_effort = (
        model_name.startswith('grok-4.3') or model_name.startswith('grok-3-mini') or model_name == 'grok-latest'
    )
    thinking_always_enabled = model_name.startswith('grok-3-mini')

    supported_native_tools: frozenset[type[AbstractNativeTool]] = (
        SUPPORTED_NATIVE_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=supports_thinking_effort,
        # grok-3-mini reasons by default; unlike grok-4.3, it has no `'none'`.
        thinking_always_enabled=thinking_always_enabled,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_native_tools=supported_native_tools,
    )
