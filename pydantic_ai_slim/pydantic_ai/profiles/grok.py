from __future__ import annotations as _annotations

from dataclasses import dataclass

from ..builtin_tools import SUPPORTED_BUILTIN_TOOLS
from . import ModelProfile


@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for Grok models (used with both GrokProvider and XaiProvider).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool = False
    """Whether the model supports builtin tools (web_search, code_execution, mcp)."""

    grok_supports_encrypted_reasoning_content: bool = False
    """Whether the model is a reasoning model (supports extended thinking/reasoning)."""

    grok_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the request payload."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    # Grok-4 models support builtin tools
    grok_supports_builtin_tools = model_name.startswith('grok-4')

    # Reasoning models have 'reasoning' in their name but not 'non-reasoning'
    is_reasoning_model = model_name == 'grok-4' or 'reasoning' in model_name and 'non-reasoning' not in model_name

    # Set supported builtin tools based on model capability
    supported_builtin_tools = SUPPORTED_BUILTIN_TOOLS if grok_supports_builtin_tools else frozenset()

    return GrokModelProfile(
        # xAI supports tool calling
        supports_tools=True,
        # xAI supports JSON schema output for structured responses
        supports_json_schema_output=True,
        # xAI supports JSON object output
        supports_json_object_output=True,
        # Default to 'native' for structured output since xAI supports it well
        default_structured_output_mode='native',
        # Support for builtin tools (web_search, code_execution, mcp)
        grok_supports_builtin_tools=grok_supports_builtin_tools,
        supported_builtin_tools=supported_builtin_tools,
        # Whether this is a reasoning model
        grok_supports_encrypted_reasoning_content=is_reasoning_model,
    )
