from __future__ import annotations

from types import ModuleType
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')


def import_mcp_types(feature: str) -> ModuleType:
    """Import the MCP wire types from whichever SDK generation is installed.

    SDK v1 ships them as `mcp.types`. SDK v2 moved them to a standalone `mcp_types` distribution
    but kept `mcp.types` as an exact re-export of it — so this import yields either generation,
    and only [`is_mcp_sdk_v2`][pydantic_ai._mcp_compat.is_mcp_sdk_v2] tells them apart.
    `feature` names the caller in the error raised when the SDK is not installed.
    """
    try:
        from mcp import types
    except ImportError as import_error:
        raise ImportError(
            f'Please install the `mcp` package to use {feature}, '
            'you can use the `mcp` optional group — `pip install "pydantic-ai-slim[mcp]"`'
        ) from import_error
    return types


def is_mcp_sdk_v2(mcp_types: ModuleType) -> bool:
    """Whether the imported MCP wire types are the SDK v2 generation.

    Detected from the v2 field rename rather than the module name: SDK v2.0.0 restored `mcp.types`
    as an exact re-export of the standalone package, so both spellings resolve to the same v2
    classes and the module name says nothing about the generation.
    """
    return 'input_schema' in mcp_types.Tool.model_fields


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
