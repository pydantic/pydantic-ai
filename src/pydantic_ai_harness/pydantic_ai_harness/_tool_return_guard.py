"""A `TypeGuard` for `ToolReturn`, kept out of `_typeguards` so light modules that use those don't import `pydantic_ai`."""

from __future__ import annotations

from typing import TypeGuard

from pydantic_ai.messages import ToolReturn


def is_tool_return(value: object) -> TypeGuard[ToolReturn[object]]:
    """Narrow `value` to a `ToolReturn` with `object` parameters, so pyright doesn't infer `Unknown`."""
    return isinstance(value, ToolReturn)
