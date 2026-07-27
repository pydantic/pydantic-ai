from __future__ import annotations

from types import ModuleType
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')


def is_mcp_sdk_v2(mcp_types: ModuleType) -> bool:
    """Whether the imported MCP wire types come from the standalone v2 package."""
    return mcp_types.__name__ == 'mcp_types'


def mcp_field_value(value: BaseModel, v1_name: str, v2_name: str) -> object:
    """Read an MCP model field by whichever spelling the installed SDK uses.

    SDK v2 renamed every wire field from camelCase to snake_case, keeping the v1 spelling as a
    validation alias — so the attribute name depends on which SDK is installed, the value never does.
    Reads a field the installed SDK doesn't define as `None`, so a field added in a later spec
    revision is picked up as soon as the SDK catches up.
    """
    return getattr(value, v2_name if v2_name in type(value).model_fields else v1_name, None)


def mcp_field(value: BaseModel, v1_name: str, v2_name: str, expected: type[T]) -> T:
    """Read a required MCP model field of a non-generic type."""
    result = mcp_field_value(value, v1_name, v2_name)
    assert isinstance(result, expected), f'Expected MCP field to be {expected.__name__}, got {type(result).__name__}'
    return result


def mcp_optional_field(value: BaseModel, v1_name: str, v2_name: str, expected: type[T]) -> T | None:
    """Read an optional MCP model field of a non-generic type."""
    result = mcp_field_value(value, v1_name, v2_name)
    return result if isinstance(result, expected) else None


def mcp_validated_field(value: BaseModel, v1_name: str, v2_name: str, adapter: TypeAdapter[T]) -> T | None:
    """Read an optional MCP model field of a generic type.

    `isinstance` can't narrow a parameterized type like `dict[str, Any]`, so these fields validate
    through a `TypeAdapter` rather than the plain `isinstance` check the readers above use.
    """
    result = mcp_field_value(value, v1_name, v2_name)
    return None if result is None else adapter.validate_python(result)
