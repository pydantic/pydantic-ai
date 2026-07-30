"""Event system for plan mutations.

Stores can be given a `PlanEventEmitter` so an application reacts to plan
changes -- surface progress in a UI, mirror steps to an external tracker, notify
a channel on completion. Callbacks may be sync or async.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from enum import Enum

from pydantic import BaseModel

from pydantic_ai_harness.planning._types import PlanItem


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
"""A sync or async listener invoked with each emitted `PlanEvent`."""


class PlanEventEmitter:
    """Register listeners and dispatch `PlanEvent`s to them.

    ```python
    from pydantic_ai_harness.planning import PlanEventEmitter

    emitter = PlanEventEmitter()

    @emitter.on_completed
    async def announce(event):
        print('done:', event.item.content)
    ```
    """

    def __init__(self) -> None:
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
