from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from pydantic_ai.tools import RunContext

    from .abstract import AbstractCapability

ResultT = TypeVar('ResultT')


@dataclass(frozen=True)
class DurableOperationMarker:
    name: str
    function: Callable[..., Awaitable[Any]]
    base_hook: bool = False


_MARKER_ATTRIBUTE = '__pydantic_ai_durable_operation__'


def get_durable_operation_marker(obj: object) -> DurableOperationMarker | None:
    """Return the durable-operation marker attached to `obj`, if present."""
    return cast(DurableOperationMarker | None, getattr(obj, _MARKER_ATTRIBUTE, None))


def set_durable_operation_marker(obj: object, marker: DurableOperationMarker) -> None:
    """Attach a durable-operation `marker` to `obj`."""
    setattr(obj, _MARKER_ATTRIBUTE, marker)


def operation_name(function: Callable[..., Any], name: str | None) -> str:
    return name or function.__name__.removeprefix('_')


def base_hook_durable_operation(function: Callable[..., Awaitable[ResultT]]) -> Callable[..., Awaitable[ResultT]]:
    """Mark a base hook so every override inherits durable execution automatically."""
    set_durable_operation_marker(
        function,
        DurableOperationMarker(
            name=operation_name(function, None),
            function=cast(Callable[..., Awaitable[Any]], function),
            base_hook=True,
        ),
    )
    return function


async def invoke_durable_operation(
    capability: AbstractCapability[Any],
    operation_name: str,
    ctx: RunContext[Any],
    handler: Callable[..., Awaitable[ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ResultT:
    """Invoke a capability operation through the active durability binding when present."""
    operation = active_durable_operation(capability, operation_name, ctx)
    if operation is not None:
        return cast(ResultT, await operation(*args, **kwargs))
    return await handler(*args, **kwargs)


def active_durable_operation(
    capability: AbstractCapability[Any], operation_name: str, ctx: RunContext[Any]
) -> Callable[..., Awaitable[object]] | None:
    """Return the dispatcher when the operation is bound to the active durable run."""
    operation = (
        ctx._durable_operations.get((capability.id, operation_name))  # pyright: ignore[reportPrivateUsage]
        if ctx._durable_operations is not None and capability.id is not None  # pyright: ignore[reportPrivateUsage]
        else None
    )
    if operation is not None:
        return operation
    dispatcher = (
        capability._get_durable_operation_bindings().get(ctx.agent, {}).get(operation_name)  # pyright: ignore[reportPrivateUsage]
        if ctx.agent is not None
        else None
    )
    if dispatcher is None:
        return None

    async def dispatch(*args: object, **kwargs: object) -> object:
        return await dispatcher(ctx, args, kwargs)

    return dispatch
