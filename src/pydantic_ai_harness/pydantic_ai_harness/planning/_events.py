"""Typed plan events.

`Planning` emits `CapabilityEvent` subclasses for mutations made through its tools. Direct store
mutations have no run context and emit no run events.
"""

from dataclasses import dataclass
from typing import Annotated

from pydantic import AliasChoices, Field

from pydantic_ai import CapabilityEvent
from pydantic_ai_harness.planning._types import PlanItem

PLANNING_EVENTS = 'planning'

_PreviousItem = Annotated[PlanItem | None, Field(validation_alias=AliasChoices('previous_item', 'previous_state'))]


@dataclass(kw_only=True)
class PlanCreatedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='created'):
    """A plan step was created through a planning tool."""

    item: PlanItem
    previous_item: _PreviousItem = None


@dataclass(kw_only=True)
class PlanUpdatedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='updated'):
    """A plan step was updated through a planning tool."""

    item: PlanItem
    previous_item: _PreviousItem = None


@dataclass(kw_only=True)
class PlanStatusChangedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='status_changed'):
    """A plan step changed status through a planning tool."""

    item: PlanItem
    previous_item: _PreviousItem = None


@dataclass(kw_only=True)
class PlanCompletedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='completed'):
    """A plan step transitioned to `completed` through a planning tool."""

    item: PlanItem
    previous_item: _PreviousItem = None


@dataclass(kw_only=True)
class PlanDeletedEvent(CapabilityEvent, namespace=PLANNING_EVENTS, name='deleted'):
    """A plan step was deleted through a planning tool."""

    item: PlanItem
    previous_item: _PreviousItem = None
