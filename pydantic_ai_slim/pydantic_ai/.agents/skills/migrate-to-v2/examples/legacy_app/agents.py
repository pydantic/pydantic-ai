"""v1: an Agent with three deprecated kwargs that all merge into capabilities=[...].

This is the highest-risk migration: a naive find/replace produces multiple
`capabilities=` kwargs, which Python rejects as a duplicate-kwarg error. The
correct fix is to MERGE into a single list.
"""
from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


def _strip_pii(messages):  # DEPRECATION: A2_history_processors
    return messages


def _tweak_tools(ctx, defs):  # DEPRECATION: A3_prepare_tools
    return defs


async def _on_event(ctx, ev):  # DEPRECATION: A5_event_stream_handler
    return None


def build() -> Agent:
    # DEPRECATION: A1_instrument
    # DEPRECATION: A2_history_processors
    # DEPRECATION: A3_prepare_tools
    # DEPRECATION: A5_event_stream_handler
    # DEPRECATION: A6_tool_retries
    # DEPRECATION: A7_mcp_servers
    # DEPRECATION: B1b_openai_string  (warning advisory in v1, behavior flips in v2)
    return Agent(
        'openai:gpt-4o',
        instrument=True,
        history_processors=[_strip_pii],
        prepare_tools=_tweak_tools,
        event_stream_handler=_on_event,
        tool_retries=3,
        output_retries=2,
        mcp_servers=[MCPServerStdio('python', args=['-c', 'pass'])],
    )
