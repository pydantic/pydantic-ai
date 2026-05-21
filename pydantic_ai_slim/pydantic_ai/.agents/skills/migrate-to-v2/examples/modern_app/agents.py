"""v2: same Agent, all kwargs merged into a single capabilities=[...] list."""
from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    Instrumentation,
    ProcessHistory,
    PrepareTools,
    ProcessEventStream,
)
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport


def _strip_pii(messages):
    return messages


def _tweak_tools(ctx, defs):
    return defs


async def _on_event(ctx, ev):
    return None


def build() -> Agent:
    return Agent(
        'openai-chat:gpt-4o',
        capabilities=[
            Instrumentation(),
            ProcessHistory(_strip_pii),
            PrepareTools(_tweak_tools),
            ProcessEventStream(_on_event),
        ],
        retries={'tools': 3, 'output': 2},
        toolsets=[MCPToolset(StdioTransport(command='python', args=['-c', 'pass']))],
    )
