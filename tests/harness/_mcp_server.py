"""The MCP SDK's own server, for tests that stand in for a provider's MCP endpoint.

These tests also run in the harness-only install, which has FastMCP's client (`fastmcp-slim[client]`) but not its
server, so the stand-ins are built on the MCP SDK's server instead. MCP SDK v1 calls it `mcp.server.fastmcp.FastMCP`;
SDK v2 renamed it to `mcp.server.mcpserver.MCPServer`. FastMCP's client accepts either as an in-process server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mcp.types import ToolAnnotations

if TYPE_CHECKING:
    # Type-checked against the SDK generation the workspace lock installs.
    from mcp.server.mcpserver import MCPServer
else:
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # MCP SDK v1
        from mcp.server.fastmcp.server import FastMCP as MCPServer, Settings

        # SDK v1 leaves the `lifespan` annotation on `Settings` as an unresolved forward reference, and
        # `pydantic-settings` >= 2.15 warns about that when a server is constructed, which this suite's `error`
        # warnings policy escalates.
        Settings.model_rebuild()

# Built from the wire key, which both SDK generations validate: SDK v2 renamed the field to `read_only_hint`.
READ_ONLY = ToolAnnotations.model_validate({'readOnlyHint': True})
NOT_READ_ONLY = ToolAnnotations.model_validate({'readOnlyHint': False})

__all__ = ['NOT_READ_ONLY', 'READ_ONLY', 'MCPServer']
