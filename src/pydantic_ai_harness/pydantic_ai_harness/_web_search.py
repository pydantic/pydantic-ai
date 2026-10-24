"""Shared support for search capabilities that can defer to the model's native web search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition


def native_web_search(include_domains: Sequence[str], exclude_domains: Sequence[str]) -> WebSearchTool:
    """The native web search tool, restricted to the same domains as the provider-backed search."""
    return WebSearchTool(allowed_domains=list(include_domains) or None, blocked_domains=list(exclude_domains) or None)


def defer_to_native_web_search(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
    """Mark a provider-backed `web_search` as the local fallback for the native web search tool.

    Pydantic AI then leaves it out of requests to models that support native web search, and sends
    it, instead of raising, to models that do not.
    """
    return replace(tool_def, unless_native=WebSearchTool.kind)
