"""Shared span-tree helpers for the durable execution suites.

Each engine's test module builds its own `BasicSpan` from the exported spans, so these helpers are
typed against the shape rather than any one of those classes.
"""

from __future__ import annotations

from typing import Any, Protocol


class _SpanNode(Protocol):
    content: str
    # `list[Any]`, not `list[_SpanNode]`: a mutable attribute is invariant, so each engine's own
    # `BasicSpan` — whose `children` is a list of itself — would not satisfy the narrower spelling.
    children: list[Any]


def drop_fastmcp_client_spans(span: _SpanNode) -> None:
    """Drop the spans FastMCP's own client instrumentation emits, in place.

    FastMCP 4 traces every message it sends as an `MCP send <method>` span; FastMCP 3 emits none.
    They sit inside our spans, so leaving them in would pin a dependency's instrumentation in a
    snapshot whose job is the Pydantic AI span hierarchy, and break it on one generation or
    the other.
    """
    span.children = [child for child in span.children if not child.content.startswith('MCP send ')]
    for child in span.children:
        drop_fastmcp_client_spans(child)
