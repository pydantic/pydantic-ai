from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import lru_cache
from types import MethodType
from typing import Any, Generic, Protocol, TypeVar, overload

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import RunContext

CapabilityT = TypeVar('CapabilityT')
EventT = TypeVar('EventT', bound=AgentStreamEvent)
BoundEventT = TypeVar('BoundEventT', bound=AgentStreamEvent, contravariant=True)

_EventMethod = Callable[[CapabilityT, RunContext[Any], EventT], Awaitable[None]]


class _BoundEventMethod(Protocol[BoundEventT]):
    def __call__(self, ctx: RunContext[Any], event: BoundEventT) -> Awaitable[None]: ...


class _OnEventMethod(Generic[CapabilityT, EventT]):
    """Descriptor created by [`on_event`][pydantic_ai.capabilities.on_event]."""

    def __init__(self, func: _EventMethod[CapabilityT, EventT], event_types: tuple[type[EventT], ...]):
        self.func = func
        self.event_types = event_types

    def __set_name__(self, owner: type[Any], name: str) -> None:
        if name == 'on_event':
            raise UserError(
                "`@on_event` cannot decorate a method named 'on_event': that name is the dispatcher that "
                'invokes the marked listeners. Rename the method, e.g. `_on_event`.'
            )

    @overload
    def __get__(self, instance: None, owner: type[CapabilityT]) -> _OnEventMethod[CapabilityT, EventT]: ...

    @overload
    def __get__(self, instance: CapabilityT, owner: type[CapabilityT]) -> _BoundEventMethod[EventT]: ...

    def __get__(
        self, instance: CapabilityT | None, owner: type[CapabilityT]
    ) -> _OnEventMethod[CapabilityT, EventT] | _BoundEventMethod[EventT]:
        if instance is None:
            return self
        return MethodType(self.func, instance)


@overload
def on_event(func: _EventMethod[CapabilityT, AgentStreamEvent], /) -> _OnEventMethod[CapabilityT, AgentStreamEvent]: ...


@overload
def on_event(
    *event_types: type[EventT],
) -> Callable[[_EventMethod[CapabilityT, EventT]], _OnEventMethod[CapabilityT, EventT]]: ...


def on_event(
    *event_types: Any,
) -> (
    _OnEventMethod[CapabilityT, AgentStreamEvent]
    | Callable[[_EventMethod[CapabilityT, EventT]], _OnEventMethod[CapabilityT, EventT]]
):
    """Mark an async capability method as an event listener.

    Pass event classes to filter with `isinstance`, or use the decorator bare to receive every
    [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent].

    Naming the classes is also what lets dispatch skip the capability entirely for events it doesn't
    listen to, so prefer it over a bare marker when you know the types.
    """
    if len(event_types) == 1 and inspect.isfunction(func := event_types[0]):
        return _OnEventMethod(func, ())

    def decorator(func: _EventMethod[CapabilityT, EventT]) -> _OnEventMethod[CapabilityT, EventT]:
        return _OnEventMethod(func, event_types)

    return decorator


@lru_cache
def collect_on_event_methods(cls: type[Any]) -> tuple[_OnEventMethod[Any, Any], ...]:
    """Collect marked methods in definition order, including inherited methods."""
    methods: dict[str, _OnEventMethod[Any, Any]] = {}
    for base in reversed(cls.__mro__):
        for name, value in base.__dict__.items():
            if isinstance(value, _OnEventMethod):
                methods[name] = value
            elif name in methods:
                del methods[name]
    return tuple(methods.values())


def marked_listens_to(cls: type[Any], event: AgentStreamEvent) -> bool:
    """Whether any method on `cls` marked with `@on_event` accepts `event`."""
    return any(
        not method.event_types or isinstance(event, method.event_types) for method in collect_on_event_methods(cls)
    )
