"""Internal helpers shared by [`MCPServer`][pydantic_ai.mcp.MCPServer] and [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset].

The two MCP integrations both need to translate `mcp.types.CallToolResult` into a Pydantic AI tool return,
so the audience-filtering and metadata-shaping logic lives here.
"""

from __future__ import annotations as _annotations

from typing import Any

from mcp import types as mcp_types

USER_ONLY_PLACEHOLDER_TEXT = 'Tool executed successfully without producing model-visible content.'
"""Sent to the model as `ToolReturn.return_value` when every content block in the tool result was filtered out by audience annotations."""


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


def build_tool_return_metadata(
    result_meta: dict[str, Any] | None,
    user_only: list[mcp_types.ContentBlock],
) -> dict[str, Any] | None:
    """Build the `ToolReturn.metadata` dict for an MCP tool result.

    Returns `None` when there's nothing to attach, in which case the caller should return the bare value
    rather than wrapping it in a `ToolReturn`. Otherwise the result is a dict with optional keys:

    - `meta`: the top-level [`CallToolResult._meta`](https://modelcontextprotocol.io/specification/2025-11-25/basic#meta) dict from the tool result.
    - `user_content`: a list of MCP content blocks (as `model_dump()` dicts) that were annotated `audience=['user']`
      and so withheld from the model. Apps can render or log these out of band.
    """
    metadata: dict[str, Any] = {}
    if result_meta:
        metadata['meta'] = result_meta
    if user_only:
        metadata['user_content'] = [part.model_dump() for part in user_only]
    return metadata or None


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
