from __future__ import annotations

from types import ModuleType

from pydantic import BaseModel

_MISSING = object()


def is_mcp_sdk_v2(mcp_types: ModuleType) -> bool:
    """Whether the imported MCP wire types come from the standalone v2 package."""
    return mcp_types.__name__ == 'mcp_types'


def get_mcp_field(value: BaseModel, v1_name: str, v2_name: str, default: object = _MISSING) -> object:
    """Read an MCP model field across SDK v1 and v2 naming conventions.

    The return type stays opaque so every caller must narrow the versioned SDK boundary explicitly.
    """
    field_name = v2_name if v2_name in type(value).model_fields else v1_name
    if default is _MISSING:
        return getattr(value, field_name)
    return getattr(value, field_name, default)
