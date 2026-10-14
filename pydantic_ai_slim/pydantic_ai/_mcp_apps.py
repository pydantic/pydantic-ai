"""The MCP Apps (SEP-1865) record `MCPToolset` attaches to a tool return, and its AG-UI activity content.

This module has no `mcp` dependency, so the UI adapters can read the record without the `mcp` extra.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from ._utils import is_str_dict

MCP_APP_METADATA_KEY: Final = 'mcp_app'
"""Key under which the record sits in `ToolReturnPart.metadata`."""

MCP_APPS_UI_TYPE: Final = 'mcp-apps'
"""Name of the UI event that carries the app data: the AG-UI activity type and the Vercel AI `data-*` suffix."""


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


_MCP_APP_TOOL_RESULT_ADAPTER: TypeAdapter[MCPAppToolResult] = TypeAdapter(MCPAppToolResult)


def get_mcp_app_tool_result(metadata: Any) -> MCPAppToolResult | None:
    """Return the MCP Apps record from a tool return's metadata, or `None` if there is no valid one."""
    if not is_str_dict(metadata) or (value := metadata.get(MCP_APP_METADATA_KEY)) is None:
        return None
    try:
        return _MCP_APP_TOOL_RESULT_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def mcp_app_activity_content(app: MCPAppToolResult) -> dict[str, Any]:
    """Build the AG-UI `mcp-apps` activity content for an MCP App tool call.

    The keys match the activity the AG-UI MCP Apps middleware emits, which renderers such as CopilotKit's
    validate. The middleware identifies the server by both `serverId` and `serverHash` (a digest of its
    connection config), and CopilotKit requires `serverHash`, so the toolset ID fills both.
    """
    payload: dict[str, Any] = {
        'resourceUri': app['resource_uri'],
        'toolInput': app['tool_input'],
        'result': app['result'],
    }
    if app['toolset_id'] is not None:
        payload['serverId'] = payload['serverHash'] = app['toolset_id']
    return payload
