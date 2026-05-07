"""Internal helpers shared by [`MCPServer`][pydantic_ai.mcp.MCPServer] and [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset].

Both MCP integrations need to translate `mcp.types.CallToolResult` into a Pydantic AI tool return,
so the audience-partition and meta-merge primitives live here for reuse.
"""

from __future__ import annotations as _annotations

from typing import Any

from mcp import types as mcp_types


def is_for_assistant(part: mcp_types.ContentBlock) -> bool:
    """Return whether the given MCP content block should be forwarded to the model.

    Per the MCP specification, content blocks may carry `annotations.audience` listing the intended recipients
    (`'user'` and/or `'assistant'`). If the field is absent the block is intended for all audiences.

    See https://modelcontextprotocol.io/specification/2025-11-25/server/tools#tool-result.
    """
    if part.annotations is None or part.annotations.audience is None:
        return True
    return 'assistant' in part.annotations.audience


def partition_content(
    content: list[mcp_types.ContentBlock],
) -> tuple[list[mcp_types.ContentBlock], list[mcp_types.ContentBlock]]:
    """Split MCP content blocks by [`is_for_assistant`][pydantic_ai._mcp_util.is_for_assistant] into `(assistant_visible, user_only)` lists."""
    assistant_visible: list[mcp_types.ContentBlock] = []
    user_only: list[mcp_types.ContentBlock] = []
    for part in content:
        (assistant_visible if is_for_assistant(part) else user_only).append(part)
    return assistant_visible, user_only


def merge_meta(*metas: dict[str, Any] | None) -> dict[str, Any] | None:
    """Shallow-merge any number of MCP `_meta` dicts, with later arguments overriding earlier ones.

    Returns `None` when every argument is `None` or empty, so callers can use the result directly as a
    `metadata` field default.
    """
    merged: dict[str, Any] = {}
    for meta in metas:
        if meta:
            merged.update(meta)
    return merged or None
