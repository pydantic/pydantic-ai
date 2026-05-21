"""Internal helpers shared by the MCP integrations.

Used by [`MCPToolset`][pydantic_ai.mcp.MCPToolset] and the deprecated
[`MCPServer`][pydantic_ai.mcp.MCPServer] / [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset]
to translate `mcp.types.CallToolResult` into a Pydantic AI tool return — audience filtering,
per-part `_meta` propagation, and `ToolReturn` synthesis.
"""

from __future__ import annotations as _annotations

import base64
from typing import Any

import pydantic_core
from mcp import types as mcp_types
from typing_extensions import assert_never

from . import messages


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


def clean_meta(meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip transport-internal keys from an MCP `_meta` payload.

    FastMCP servers tag every result with a `'fastmcp'` namespace (e.g. `{'wrap_result': True}`)
    that's an implementation detail of the server SDK, not application metadata. Filter it out
    before surfacing `_meta` to the caller. Returns `None` when nothing is left.
    """
    if not meta:
        return None
    cleaned = {k: v for k, v in meta.items() if k != 'fastmcp'}
    return cleaned or None


def map_tool_result_part(part: mcp_types.ContentBlock) -> Any:
    """Map an MCP content block to a Pydantic AI value, preserving any `_meta` payload.

    Used for the assistant-visible return value, so JSON-encoded text is decoded into Python
    objects for ergonomic consumption — except when a `_meta` payload would otherwise be lost,
    in which case the part is returned as a [`TextContent`][pydantic_ai.messages.TextContent]
    so the metadata is reachable programmatically.
    """
    if isinstance(part, mcp_types.TextContent):
        if part.meta:
            return messages.TextContent(content=part.text, metadata=part.meta)
        text = part.text
        if text.startswith(('[', '{')):
            try:
                return pydantic_core.from_json(text)
            except ValueError:
                pass
        return text
    if isinstance(part, mcp_types.ImageContent):
        return messages.BinaryImage(
            data=base64.b64decode(part.data), media_type=part.mimeType, metadata=part.meta or None
        )
    if isinstance(part, mcp_types.AudioContent):
        return messages.BinaryContent(
            data=base64.b64decode(part.data), media_type=part.mimeType, metadata=part.meta or None
        )
    if isinstance(part, mcp_types.EmbeddedResource):
        return _embedded_resource_to_pai(part)
    if isinstance(part, mcp_types.ResourceLink):
        uri = str(part.uri)
        if part.meta:
            return messages.TextContent(content=uri, metadata=part.meta)
        return uri
    assert_never(part)


def map_tool_result_user_content(part: mcp_types.ContentBlock) -> messages.UserContent:
    """Map an MCP content block to a [`UserContent`][pydantic_ai.messages.UserContent] value.

    Used for the user-audience side of a tool result, which is forwarded as a separate
    `UserPromptPart`. Unlike [`map_tool_result_part`][pydantic_ai._mcp_util.map_tool_result_part],
    JSON-encoded text is left as-is because `UserContent` doesn't admit dict/list payloads.
    """
    if isinstance(part, mcp_types.TextContent):
        if part.meta:
            return messages.TextContent(content=part.text, metadata=part.meta)
        return part.text
    if isinstance(part, mcp_types.ImageContent):
        return messages.BinaryImage(
            data=base64.b64decode(part.data), media_type=part.mimeType, metadata=part.meta or None
        )
    if isinstance(part, mcp_types.AudioContent):
        return messages.BinaryContent(
            data=base64.b64decode(part.data), media_type=part.mimeType, metadata=part.meta or None
        )
    if isinstance(part, mcp_types.EmbeddedResource):
        resource = _embedded_resource_to_pai(part)
        # Embedded text resources come back as `str | TextContent`, both of which are valid
        # `UserContent`. Binary resources come back as `BinaryContent`, also valid.
        return resource
    if isinstance(part, mcp_types.ResourceLink):
        uri = str(part.uri)
        if part.meta:
            return messages.TextContent(content=uri, metadata=part.meta)
        return uri
    assert_never(part)


def _embedded_resource_to_pai(
    part: mcp_types.EmbeddedResource,
) -> str | messages.TextContent | messages.BinaryContent:
    """Map an MCP embedded resource to a Pydantic AI value, merging the outer and inner `_meta`."""
    resource = part.resource
    meta = merge_meta(part.meta, resource.meta)
    if isinstance(resource, mcp_types.TextResourceContents):
        if meta:
            return messages.TextContent(content=resource.text, metadata=meta)
        return resource.text
    if isinstance(resource, mcp_types.BlobResourceContents):
        return messages.BinaryContent.narrow_type(
            messages.BinaryContent(
                data=base64.b64decode(resource.blob),
                media_type=resource.mimeType or 'application/octet-stream',
                metadata=meta,
            )
        )
    assert_never(resource)
