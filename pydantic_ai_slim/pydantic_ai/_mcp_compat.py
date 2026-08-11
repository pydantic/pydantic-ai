from __future__ import annotations

from importlib.metadata import version
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')


def is_mcp_sdk_v2() -> bool:
    """Whether the installed MCP SDK is the v2 generation.

    Read from the installed `mcp` distribution version: SDK v2.0.0 restored `mcp.types` as an
    exact re-export of the standalone package, so the module name says nothing about the
    generation, and shape differences like the camelCase → snake_case field rename are symptoms
    of the generation rather than a contract to detect it by.
    """
    return int(version('mcp').split('.')[0]) >= 2


def wire_name(name: str) -> str:
    """The camelCase (wire and SDK v1) spelling of a snake_case field name.

    Every field the MCP SDK v2 renamed is a mechanical snake_case ↔ camelCase pair — the test
    suite checks this against the real SDK v2 models for every field read through this module.
    """
    first, *rest = name.split('_')
    return first + ''.join(word.title() for word in rest)


def mcp_field_value(value: BaseModel, name: str) -> object:
    """Read the MCP model field `name` (snake_case) by whichever spelling the installed SDK uses.

    SDK v2 renamed every wire field from camelCase to snake_case, keeping the camel spelling as a
    validation alias — so the attribute name depends on which SDK is installed, the value never
    does. Reads a field the installed SDK doesn't define as `None`, so a field added in a later
    spec revision is picked up as soon as the SDK catches up.
    """
    return getattr(value, name if name in type(value).model_fields else wire_name(name), None)


def mcp_field(value: BaseModel, name: str, expected: type[T]) -> T:
    """Read a required MCP model field of a non-generic type."""
    result = mcp_field_value(value, name)
    assert isinstance(result, expected), f'Expected MCP field to be {expected.__name__}, got {type(result).__name__}'
    return result


def mcp_optional_field(value: BaseModel, name: str, expected: type[T]) -> T | None:
    """Read an optional MCP model field of a non-generic type."""
    result = mcp_field_value(value, name)
    return result if isinstance(result, expected) else None


def mcp_validated_field(value: BaseModel, name: str, adapter: TypeAdapter[T]) -> T | None:
    """Read an optional MCP model field of a generic type.

    `isinstance` can't narrow a parameterized type like `dict[str, Any]`, so these fields validate
    through a `TypeAdapter` rather than the plain `isinstance` check the readers above use.
    """
    result = mcp_field_value(value, name)
    return None if result is None else adapter.validate_python(result)
