"""Typed events for agent teams."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent
from pydantic_ai_harness.planning import OwnedPlanItem

TEAMS_EVENTS = 'teams'


@dataclass(kw_only=True)
class BoardItemChangedEvent(CapabilityEvent, namespace=TEAMS_EVENTS, name='board_item_changed'):
    """A step on a team's shared board changed.

    Emitted once per step that a `ClaimablePlanStore.claim` or `release` returns, so a
    UI can keep a board in sync by reducing these events. It carries `OwnedPlanItem`,
    unlike the `planning` events, whose `item` is declared as `PlanItem` and would lose
    `owner` when serialized.
    """

    item: OwnedPlanItem
    """The step as it is now."""

    previous: OwnedPlanItem | None
    """The step before the change, or `None` when it was just added."""
