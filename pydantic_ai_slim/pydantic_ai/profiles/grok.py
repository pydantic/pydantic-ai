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
    # Only grok-3-mini accepts the `reasoning_effort` parameter. grok-4 reasoning models
    # always reason but reject the parameter, so we treat thinking as unsupported for them
    # to avoid forwarding an argument the API will error on.
    # See https://docs.x.ai/docs/guides/reasoning
    supports_thinking_effort = model_name.startswith('grok-3-mini')

    supported_builtin_tools: frozenset[type[AbstractBuiltinTool]] = (
        SUPPORTED_BUILTIN_TOOLS if grok_supports_builtin_tools else frozenset()
    )

    return GrokModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        supports_thinking=supports_thinking_effort,
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_builtin_tools=supported_builtin_tools,
    )
