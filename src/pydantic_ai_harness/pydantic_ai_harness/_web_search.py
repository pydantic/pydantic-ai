"""Shared preparation for provider-backed web search tools."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.models import Model
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import RealtimeModel
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


def prefer_native_web_search(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
    """Omit this search tool when the model supports native web search."""
    # A run context rehydrated across a durable boundary may not carry the live model and raises on
    # attribute access. In that case, leave the provider-backed search definition unchanged.
    model = ctx.__dict__.get('model')
    if isinstance(model, (Model, RealtimeModel)) and WebSearchTool in model.profile.get(
        'supported_native_tools', frozenset()
    ):
        return replace(tool_def, unless_native='web_search')
    return tool_def
