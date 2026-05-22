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
    grok_supports_builtin_tools = model_name.startswith('grok-4') or 'code' in model_name
    # Only grok-3-mini accepts the `reasoning_effort` parameter. grok-4 reasoning models
    # always reason but reject the parameter, so we treat thinking as unsupported for them
    # to avoid forwarding an argument the API will error on.
    # See https://docs.x.ai/docs/guides/reasoning
    supports_thinking_effort = model_name.startswith('grok-3-mini')

    # grok-3-mini always reasons; the API has no `'none'` value for `reasoning_effort`,
    # so route `thinking=False` through the standard always-on silent-drop path. This
    # binding to `supports_thinking_effort` is coincidental — grok-3-mini happens to be
    # the only grok variant that both supports `reasoning_effort` and lacks a disable
    # value. Newer grok models that accept `reasoning_effort='none'` should set the two
    # flags independently.
    thinking_always_enabled = supports_thinking_effort

    supported_native_tools: frozenset[type[AbstractNativeTool]] = (
        SUPPORTED_NATIVE_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=supports_thinking_effort,
        thinking_always_enabled=thinking_always_enabled,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_native_tools=supported_native_tools,
    )
