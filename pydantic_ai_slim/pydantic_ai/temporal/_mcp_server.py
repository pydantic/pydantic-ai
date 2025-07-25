from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mcp import types as mcp_types
from pydantic import ConfigDict, with_config
from temporalio import activity, workflow

from pydantic_ai.mcp import MCPServer, ToolResult

from ._settings import TemporalSettings


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CallToolParams:
    name: str
    tool_args: dict[str, Any]
    metadata: dict[str, Any] | None = None


def temporalize_mcp_server(
    server: MCPServer,
    settings: TemporalSettings | None = None,
) -> list[Callable[..., Any]]:
    """Temporalize an MCP server.

    Args:
        server: The MCP server to temporalize.
        settings: The temporal settings to use.
    """
    if activities := getattr(server, '__temporal_activities', None):
        return activities

    id = server.id
    if not id:
        raise ValueError(
            "An MCP server needs to have an ID in order to be used in a durable execution environment like Temporal. The ID will be used to identify the server's activities within the workflow."
        )

    settings = settings or TemporalSettings()

    original_list_tools = server.list_tools
    original_direct_call_tool = server.direct_call_tool

    @activity.defn(name=f'mcp_server__{id}__list_tools')
    async def list_tools_activity() -> list[mcp_types.Tool]:
        return await original_list_tools()

    @activity.defn(name=f'mcp_server__{id}__call_tool')
    async def call_tool_activity(params: _CallToolParams) -> ToolResult:
        return await original_direct_call_tool(params.name, params.tool_args, params.metadata)

    async def list_tools() -> list[mcp_types.Tool]:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
            activity=list_tools_activity,
            **settings.execute_activity_options,
        )

    async def direct_call_tool(
        name: str,
        args: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return await workflow.execute_activity(  # pyright: ignore[reportUnknownMemberType]
            activity=call_tool_activity,
            arg=_CallToolParams(name=name, tool_args=args, metadata=metadata),
            **settings.for_tool(id, name).execute_activity_options,
        )

    server.list_tools = list_tools
    server.direct_call_tool = direct_call_tool

    activities = [list_tools_activity, call_tool_activity]
    setattr(server, '__temporal_activities', activities)
    return activities
