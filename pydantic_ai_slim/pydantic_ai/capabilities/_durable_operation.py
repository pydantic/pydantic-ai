from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

R = TypeVar('R')


@dataclass(frozen=True)
class DurableOperationMarker:
    name: str
    function: Callable[..., Awaitable[Any]]
    tier_one: bool = False


def operation_name(function: Callable[..., Any], name: str | None) -> str:
    return name or function.__name__.removeprefix('_')


def tier_one_durable_operation(function: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
    """Mark a base hook whose overrides are inherently durable."""
    setattr(
        function,
        '__pydantic_ai_durable_operation__',
        DurableOperationMarker(operation_name(function, None), cast(Callable[..., Awaitable[Any]], function), True),
    )
    return function
