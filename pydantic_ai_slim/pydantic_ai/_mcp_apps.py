"""The records `MCPToolset` attaches to a tool return, and the MCP Apps (SEP-1865) UI wire payload.

This module has no `mcp` dependency, so the UI adapters can read the record without the `mcp` extra.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from ._utils import is_str_dict

MCP_TOOL_METADATA_KEY: Final = 'mcp_tool'
"""Key under which the `MCPToolInfo` record sits in `ToolReturnPart.metadata`."""

MCP_APP_METADATA_KEY: Final = 'mcp_app'
"""Key under which the `MCPAppToolResult` record sits in `ToolReturnPart.metadata`."""

MCP_APPS_UI_TYPE: Final = 'mcp-apps'
"""Name of the UI event that carries the record: the AG-UI activity type and the Vercel AI `data-*` suffix."""

MCP_APP_MIME_TYPE: Final = 'text/html;profile=mcp-app'
"""MIME type of an MCP Apps page."""


class MCPToolInfo(TypedDict):
    """Which server and tool one MCP tool call went to.

    [`MCPToolset`][pydantic_ai.mcp.MCPToolset] attaches this to every tool call it runs, under the `'mcp_tool'` key
    of [`ToolReturnPart.metadata`][pydantic_ai.messages.ToolReturnPart.metadata], so a frontend can label the call.
    It is never sent to the model.
    """

    toolset_id: str | None
    """The [`id`][pydantic_ai.mcp.MCPToolset.id] of the toolset that ran the tool."""

    title: str | None
    """The tool's title, if the server set one."""

    annotations: dict[str, Any] | None
    """The tool's [annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-annotations)
    in wire format (camelCase keys), if the server set any."""


class MCPAppToolResult(TypedDict):
    """What a UI needs to render an [MCP App](https://modelcontextprotocol.io/extensions/apps/overview) for one tool call.

    [`MCPToolset`][pydantic_ai.mcp.MCPToolset] attaches this to calls to tools that name a UI resource, under the `'mcp_app'`
    key of [`ToolReturnPart.metadata`][pydantic_ai.messages.ToolReturnPart.metadata], and the UI adapters forward
    it to the frontend. It is never sent to the model.
    """

    resource_uri: str
    """The `ui://` URI of the app's HTML resource, read with
    [`MCPToolset.read_resource_contents`][pydantic_ai.mcp.MCPToolset.read_resource_contents]."""

    tool_input: dict[str, Any]
    """The arguments the model called the tool with."""

    result: dict[str, Any]
    """The full MCP `CallToolResult` in wire format (camelCase keys), including `structuredContent`, `_meta` and `isError`."""

    toolset_id: str | None
    """The [`id`][pydantic_ai.mcp.MCPToolset.id] of the toolset that ran the tool, to route resource reads."""


_MCP_TOOL_INFO_ADAPTER: TypeAdapter[MCPToolInfo] = TypeAdapter(MCPToolInfo)
_MCP_APP_TOOL_RESULT_ADAPTER: TypeAdapter[MCPAppToolResult] = TypeAdapter(MCPAppToolResult)


def get_mcp_tool_info(metadata: Any) -> MCPToolInfo | None:
    """Return the MCP tool record from a tool return's metadata, or `None` if there is no valid one."""
    if not is_str_dict(metadata) or (value := metadata.get(MCP_TOOL_METADATA_KEY)) is None:
        return None
    try:
        return _MCP_TOOL_INFO_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def get_mcp_app_tool_result(metadata: Any) -> MCPAppToolResult | None:
    """Return the MCP Apps record from a tool return's metadata, or `None` if there is no valid one."""
    if not is_str_dict(metadata) or (value := metadata.get(MCP_APP_METADATA_KEY)) is None:
        return None
    try:
        return _MCP_APP_TOOL_RESULT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def mcp_app_ui_payload(app: MCPAppToolResult) -> dict[str, Any]:
    """Build the payload UI adapters send for an MCP App tool call.

    The keys match the `mcp-apps` activity content of the AG-UI MCP Apps middleware, so frontends can
    share one renderer across protocols. The middleware identifies the server by both `serverId` and
    `serverHash` (a digest of its connection config), and renderers such as CopilotKit's require
    `serverHash`, so the toolset ID fills both.
    """
    payload: dict[str, Any] = {
        'resourceUri': app['resource_uri'],
        'toolInput': app['tool_input'],
        'result': app['result'],
    }
    if app['toolset_id'] is not None:
        payload['serverId'] = payload['serverHash'] = app['toolset_id']
    return payload
