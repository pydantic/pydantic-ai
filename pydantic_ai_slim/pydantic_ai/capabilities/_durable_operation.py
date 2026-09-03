"""Durable-operation markers and dispatch, importable by `capabilities` without a `durable_exec` import cycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from pydantic_ai.tools import RunContext

    from .abstract import AbstractCapability

ResultT = TypeVar('ResultT')
P = ParamSpec('P')


@dataclass(frozen=True)
class DurableOperationMarker:
    name: str
    _: KW_ONLY
    function: Callable[..., Any]
    base_hook: bool = False


_MARKER_ATTRIBUTE = '__pydantic_ai_durable_operation__'


def get_durable_operation_marker(obj: object) -> DurableOperationMarker | None:
    """Return the durable-operation marker attached to `obj`, if present."""
    return cast(DurableOperationMarker | None, getattr(obj, _MARKER_ATTRIBUTE, None))


def set_durable_operation_marker(obj: object, marker: DurableOperationMarker) -> None:
    """Attach a durable-operation `marker` to `obj`."""
    setattr(obj, _MARKER_ATTRIBUTE, marker)


def validate_operation_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(f'`durable_operation` name must be a string, got {type(name).__name__}')
    if not name:
        raise ValueError('`durable_operation` name must not be empty')
    return name


def base_hook_durable_operation(
    name: str,
) -> Callable[[Callable[P, ResultT]], Callable[P, ResultT]]:
    """Mark a base hook so every override inherits durable execution automatically."""
    name = validate_operation_name(name)

    def decorate(function: Callable[P, ResultT]) -> Callable[P, ResultT]:
        set_durable_operation_marker(
            function,
            DurableOperationMarker(
                name=name,
                function=function,
                base_hook=True,
            ),
        )
        return function

    return decorate


async def invoke_durable_operation(
    capability: AbstractCapability[Any],
    operation_name: str,
    ctx: RunContext[Any],
    handler: Callable[..., Awaitable[ResultT]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ResultT:
    """Run a capability operation through the durability binding active for `ctx`, else through `handler`.

    A per-run operation registered on the context wins; otherwise the dispatcher the durability
    capability bound for the agent at `for_agent` time is used; otherwise the plain handler runs.
    """
    operations = ctx._durable_operations  # pyright: ignore[reportPrivateUsage]
    operation = (
        operations.get((capability.id, operation_name))
        if operations is not None and capability.id is not None
        else None
    )
    if operation is not None:
        return cast(ResultT, await operation(*args, **kwargs))
    dispatcher = (
        capability._get_durable_operation_bindings().get(ctx.agent, {}).get(operation_name)  # pyright: ignore[reportPrivateUsage]
        if ctx.agent is not None
        else None
    )
    if dispatcher is None:
        return await handler(*args, **kwargs)
    return cast(ResultT, await dispatcher(ctx, args, kwargs))
