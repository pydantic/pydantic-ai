"""Shared preparation for provider-backed web search tools."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


def prefer_native_web_search(_ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
    """Use this search tool only when the model has no native web search."""
    return replace(tool_def, unless_native='web_search')
