from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from functools import lru_cache
from types import MethodType
from typing import Any, Generic, Protocol, TypeVar, overload

from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.tools import RunContext

CapabilityT = TypeVar('CapabilityT')
EventT = TypeVar('EventT', bound=AgentStreamEvent)
BoundEventT = TypeVar('BoundEventT', bound=AgentStreamEvent, contravariant=True)

_EventMethod = Callable[[CapabilityT, RunContext[Any], EventT], Awaitable[None]]


class _BoundEventMethod(Protocol[BoundEventT]):
    def __call__(self, ctx: RunContext[Any], event: BoundEventT) -> Awaitable[None]: ...


class OnEventMethod(Generic[CapabilityT, EventT]):
    """Descriptor created by [`on_event`][pydantic_ai.capabilities.on_event]."""

    def __init__(self, func: _EventMethod[CapabilityT, EventT], event_types: tuple[type[EventT], ...]):
        self.func = func
        self.event_types = event_types

    @overload
    def __get__(self, instance: None, owner: type[CapabilityT]) -> OnEventMethod[CapabilityT, EventT]: ...

    @overload
    def __get__(self, instance: CapabilityT, owner: type[CapabilityT]) -> _BoundEventMethod[EventT]: ...

    def __get__(
        self, instance: CapabilityT | None, owner: type[CapabilityT]
    ) -> OnEventMethod[CapabilityT, EventT] | _BoundEventMethod[EventT]:
        if instance is None:
            return self
        return MethodType(self.func, instance)


@overload
def on_event(func: _EventMethod[CapabilityT, AgentStreamEvent], /) -> OnEventMethod[CapabilityT, AgentStreamEvent]: ...


@overload
def on_event(
    *event_types: type[EventT],
) -> Callable[[_EventMethod[CapabilityT, EventT]], OnEventMethod[CapabilityT, EventT]]: ...


def on_event(
    *event_types: Any,
) -> (
    OnEventMethod[CapabilityT, AgentStreamEvent]
    | Callable[[_EventMethod[CapabilityT, EventT]], OnEventMethod[CapabilityT, EventT]]
):
    """Mark an async capability method as an event listener.

    Pass event classes to filter with `isinstance`, or use the decorator bare to receive every
    [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent].
    """
    if len(event_types) == 1 and inspect.isfunction(func := event_types[0]):
        if not inspect.iscoroutinefunction(func):
            raise TypeError('`@on_event` can only decorate async methods')
        return OnEventMethod(func, ())

    def decorator(func: _EventMethod[CapabilityT, EventT]) -> OnEventMethod[CapabilityT, EventT]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError('`@on_event` can only decorate async methods')
        return OnEventMethod(func, event_types)

    return decorator


@lru_cache
def collect_on_event_methods(cls: type[Any]) -> tuple[OnEventMethod[Any, Any], ...]:
    """Collect marked methods in definition order, including inherited methods."""
    methods: dict[str, OnEventMethod[Any, Any]] = {}
    for base in reversed(cls.__mro__):
        for name, value in base.__dict__.items():
            if isinstance(value, OnEventMethod):
                methods[name] = value
            elif name in methods:
                del methods[name]
    return tuple(methods.values())
