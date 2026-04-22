"""Internal helpers for MCP audience filtering.

Per the MCP specification, content blocks may carry ``annotations.audience``
that lists their intended recipients. When an MCP tool returns content
annotated for ``user`` only, the model should not see it (or its JSON-encoded
equivalent via ``structured_content``); instead the application receives it
out-of-band via ``ToolReturnPart.metadata['mcp_user_content']``.

This module is private (leading underscore on the filename) because the
helpers are shared implementation details between :mod:`pydantic_ai.mcp` and
:mod:`pydantic_ai.toolsets.fastmcp`. They are not part of the public API.
Refer to ``agent_docs/index.md`` rule:176 for the scoping policy.
"""

from __future__ import annotations as _annotations

from typing import Any, cast

from mcp import types as mcp_types

from . import messages

__all__ = (
    'audience_include',
    'partition_content',
    'user_only_placeholder',
    'wrap_with_user_metadata',
    'USER_ONLY_PLACEHOLDER_TEXT',
)


USER_ONLY_PLACEHOLDER_TEXT = 'Tool executed successfully. (No model-visible content in result.)'


def audience_include(part: mcp_types.ContentBlock) -> bool:
    """Return True if this content block should be forwarded to the model.

    Per the MCP spec, content blocks may carry ``annotations.audience`` which
    lists the intended recipients. When the list is absent (``None``) the
    content is intended for *all* audiences. When it is present, the content
    should only be forwarded to the audiences listed.

    See: https://modelcontextprotocol.io/specification/2025-11-25/server/tools#tool-result
    """
    annotations = part.annotations
    if annotations is None:
        return True
    audience = annotations.audience
    if audience is None:
        return True
    return 'assistant' in audience


def partition_content(
    content: list[mcp_types.ContentBlock],
) -> tuple[list[mcp_types.ContentBlock], list[mcp_types.ContentBlock]]:
    """Partition MCP content blocks into ``(assistant_visible, user_only)`` lists."""
    filtered = [p for p in content if audience_include(p)]
    user_only = [p for p in content if not audience_include(p)]
    return filtered, user_only


def user_only_placeholder(
    user_only: list[mcp_types.ContentBlock],
) -> messages.ToolReturn:
    """Return a placeholder :class:`ToolReturn` for tools whose entire output is user-only.

    The model is told the tool ran successfully; the actual content is preserved in
    ``ToolReturn.metadata['mcp_user_content']`` for the application.
    """
    return messages.ToolReturn(
        return_value=USER_ONLY_PLACEHOLDER_TEXT,
        metadata={'mcp_user_content': [p.model_dump() for p in user_only]},
    )


def wrap_with_user_metadata(
    assistant_content: Any,
    user_only: list[mcp_types.ContentBlock],
) -> messages.ToolReturn:
    """Wrap *assistant_content* in a :class:`ToolReturn` plus user-only metadata.

    Used when a tool result contains a mix of assistant-visible and user-only
    content blocks. The assistant-visible content is returned via
    ``return_value``; the user-only content is preserved in
    ``metadata['mcp_user_content']`` for the application to consume out-of-band.

    Multi-modal items inside *assistant_content* (``BinaryContent``, image and
    audio URLs, uploaded files, and so on) cannot live in ``return_value``: the
    agent graph rejects them with a UserError because they are required to flow
    through ``ToolReturn.content`` as a ``UserPromptPart``. This helper detects
    those items, replaces each one in ``return_value`` with a
    ``"See file <identifier>"`` placeholder, and routes the original objects
    into ``content`` instead, mirroring the convention used by the agent graph
    when it normalises plain tool results.
    """
    return_value, content = _split_multimodal(assistant_content)
    return messages.ToolReturn(
        return_value=return_value,
        content=content if content else None,
        metadata={'mcp_user_content': [p.model_dump() for p in user_only]},
    )


def _split_multimodal(
    assistant_content: Any,
) -> tuple[Any, list[str | messages.UserContent]]:
    """Replace multi-modal items in *assistant_content* with text placeholders.

    Returns ``(return_value, content)`` where ``return_value`` is safe to place
    in ``ToolReturn.return_value`` and ``content`` carries the original
    multi-modal items (with descriptive headers) for ``ToolReturn.content``.

    The shape of ``return_value`` mirrors the shape of the input: a single
    multi-modal item becomes a single placeholder string, a list becomes a
    list with each multi-modal item replaced by its placeholder, and content
    that contains no multi-modal items is returned unchanged so existing
    callers see no behavioural difference.
    """
    no_user_content: list[str | messages.UserContent] = []

    if isinstance(assistant_content, messages.MULTI_MODAL_CONTENT_TYPES):
        identifier = assistant_content.identifier
        single_user_content: list[str | messages.UserContent] = [
            f'This is file {identifier}:',
            assistant_content,
        ]
        single_return: Any = f'See file {identifier}'
        return single_return, single_user_content

    if isinstance(assistant_content, list):
        # Pyright cannot infer the element type of a list narrowed from Any, but
        # mypy considers a same-type cast on Any redundant. Casting to
        # ``list[object]`` keeps both type checkers quiet without suppressions.
        items = cast('list[object]', assistant_content)
        return_values: list[Any] = []
        list_user_content: list[str | messages.UserContent] = []
        any_multimodal = False
        for item in items:
            if isinstance(item, messages.MULTI_MODAL_CONTENT_TYPES):
                any_multimodal = True
                identifier = item.identifier
                return_values.append(f'See file {identifier}')
                list_user_content.extend([f'This is file {identifier}:', item])
            else:
                return_values.append(item)
        if any_multimodal:
            list_return: Any = return_values
            return list_return, list_user_content

    fallthrough_return: Any = cast('Any', assistant_content)
    return fallthrough_return, no_user_content
