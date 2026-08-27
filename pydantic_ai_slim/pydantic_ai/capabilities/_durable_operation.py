from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast

if TYPE_CHECKING:
    from pydantic_ai.tools import RunContext

    from .abstract import AbstractCapability

R = TypeVar('R')

DurableOperationDispatcher: TypeAlias = Callable[
    ['RunContext[object]', tuple[object, ...], dict[str, object]], Awaitable[object]
]


@dataclass(frozen=True)
class DurableOperationBinding:
    dispatcher: DurableOperationDispatcher
    in_durable_context: Callable[[], bool]


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


async def invoke_durable_operation(
    capability: AbstractCapability[Any],
    operation_name: str,
    ctx: RunContext[Any],
    handler: Callable[..., Awaitable[Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Invoke a capability operation through the active durability binding when present."""
    operation = None
    if capability.id is not None:
        operations = cast(
            dict[tuple[str, str], Callable[..., Awaitable[Any]]] | None,
            ctx.__dict__.get('_durable_operations'),
        )
        if operations is not None:
            operation = operations.get((capability.id, operation_name))
    if operation is not None:
        return await operation(*args, **kwargs)

    binding = (
        capability._get_durable_operation_bindings().get(id(ctx.agent), {}).get(operation_name)  # pyright: ignore[reportPrivateUsage]
        if ctx.agent is not None
        else None
    )
    if binding is not None and binding.in_durable_context():
        return await binding.dispatcher(ctx, cast(tuple[object, ...], args), cast(dict[str, object], kwargs))
    return await handler(*args, **kwargs)
