"""`TypeGuard`s that narrow a container or generic class to `object` parameters, so pyright doesn't infer `Unknown`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

from pydantic_ai.messages import ToolReturn


def is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def is_str_keyed_dict(value: object) -> TypeGuard[dict[str, object]]:
    return is_object_dict(value) and all(isinstance(key, str) for key in value)


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def is_object_tuple(value: object) -> TypeGuard[tuple[object, ...]]:
    return isinstance(value, tuple)


def is_object_set(value: object) -> TypeGuard[set[object]]:
    return isinstance(value, set)


def is_object_frozenset(value: object) -> TypeGuard[frozenset[object]]:
    return isinstance(value, frozenset)


def is_object_list_or_tuple(value: object) -> TypeGuard[list[object] | tuple[object, ...]]:
    return isinstance(value, list | tuple)


def is_tool_return(value: object) -> TypeGuard[ToolReturn[object]]:
    return isinstance(value, ToolReturn)
