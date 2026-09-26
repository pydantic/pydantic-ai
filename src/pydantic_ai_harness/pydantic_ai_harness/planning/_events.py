"""Typed plan events and the deprecated callback event system.

`Planning` emits `CapabilityEvent` subclasses for mutations made through its tools. Direct store
mutations have no run context and emit no run events. The callback-based `PlanEventEmitter` remains
for compatibility with stores configured to use it.
"""

import inspect
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel
from pydantic_ai import CapabilityEvent

from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.planning._types import PlanItem

PLANNING_EVENTS = 'planning'


@dataclass(kw_only=True)
class PlanCreatedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='created'):
    """A plan step was created through a planning tool."""

    item: PlanItem
    previous_state: PlanItem | None = None


@dataclass(kw_only=True)
class PlanUpdatedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='updated'):
    """A plan step was updated through a planning tool."""

    item: PlanItem
    previous_state: PlanItem | None = None


@dataclass(kw_only=True)
class PlanStatusChangedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='status_changed'):
    """A plan step changed status through a planning tool."""

    item: PlanItem
    previous_state: PlanItem | None = None


@dataclass(kw_only=True)
class PlanCompletedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='completed'):
    """A plan step transitioned to `completed` through a planning tool."""

    item: PlanItem
    previous_state: PlanItem | None = None


@dataclass(kw_only=True)
class PlanDeletedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='deleted'):
    """A plan step was deleted through a planning tool."""

    item: PlanItem
    previous_state: PlanItem | None = None


class PlanEventType(str, Enum):
    """The kinds of change a store can emit.

    Attributes:
        created: A new step was added.
        updated: A step's fields changed (any field).
        status_changed: A step's status changed.
        deleted: A step was removed.
        completed: A step transitioned to `completed`.
    """

    created = 'created'
    updated = 'updated'
    status_changed = 'status_changed'
    deleted = 'deleted'
    completed = 'completed'


class PlanEvent(BaseModel):
    """A single plan change delivered to registered listeners.

    Attributes:
        event_type: What happened.
        item: The affected step (post-change for updates).
        previous_state: The step before the change, when the emitter captured it.
    """

    event_type: PlanEventType
    item: PlanItem
    previous_state: PlanItem | None = None


EventCallback = Callable[[PlanEvent], None | Awaitable[None]]
"""Deprecated callback type for `PlanEventEmitter`. Subscribe to typed plan events instead."""


class PlanEventEmitter:
    """Deprecated callback dispatcher. Subscribe to typed plan events instead.

    ```python
    from pydantic_ai_harness.planning import PlanEventEmitter

    emitter = PlanEventEmitter()

    @emitter.on_completed
    async def announce(event):
        print('done:', event.item.content)
    ```
    """

    def __init__(self) -> None:
        warnings.warn(
            '`PlanEventEmitter` is deprecated; subscribe with `@agent.on_event` to the events `Planning` emits '
            'instead -- `PlanCreatedEvent`, `PlanUpdatedEvent`, `PlanStatusChangedEvent`, '
            '`PlanCompletedEvent` and `PlanDeletedEvent`, from `pydantic_ai_harness.planning`, one per '
            '`on_*` method here. They are emitted from the planning tools, so a mutation your own code '
            'makes on the store directly has no run context and reaches no listener.',
            HarnessDeprecationWarning,
            stacklevel=2,
        )
        self._listeners: dict[PlanEventType, list[EventCallback]] = {kind: [] for kind in PlanEventType}

    def on(self, event_type: PlanEventType, callback: EventCallback) -> EventCallback:
        """Register `callback` for `event_type` and return it (usable as a decorator)."""
        self._listeners[event_type].append(callback)
        return callback

    def off(self, event_type: PlanEventType, callback: EventCallback) -> bool:
        """Remove `callback` from `event_type`; return whether it was registered."""
        try:
            self._listeners[event_type].remove(callback)
        except ValueError:
            return False
        return True

    async def emit(self, event: PlanEvent) -> None:
        """Invoke every listener registered for `event.event_type`, awaiting async ones."""
        for callback in self._listeners[event.event_type]:
            result = callback(event)
            if inspect.isawaitable(result):
                await result

    def on_created(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `created` events."""
        return self.on(PlanEventType.created, callback)

    def on_updated(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `updated` events."""
        return self.on(PlanEventType.updated, callback)

    def on_status_changed(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `status_changed` events."""
        return self.on(PlanEventType.status_changed, callback)

    def on_completed(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `completed` events."""
        return self.on(PlanEventType.completed, callback)

    def on_deleted(self, callback: EventCallback) -> EventCallback:
        """Register a listener for `deleted` events."""
        return self.on(PlanEventType.deleted, callback)
