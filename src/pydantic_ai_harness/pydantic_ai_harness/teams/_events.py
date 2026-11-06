"""Events of the `teams` family: board changes and gates.

Each gate is dispatched immediately, so its listeners run before `RunContext.emit` returns and can
reject what the event announces with feedback. Several listeners can each add feedback; the emitter
passes all of it on, in the order the listeners ran.

`TeammateIdleEvent.final_output` is cut at `MAX_EVENT_TEXT_CHARS` with a `truncated` flag, like the
text of the `sub_agents` events, so an event stream that is persisted or forwarded to a UI cannot be
flooded by one verbose member.

As with any capability event, a listener that raises aborts the run that emitted it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import CapabilityEvent
from pydantic_ai_harness.planning import OwnedPlanItem, PlanItem

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


@dataclass(kw_only=True)
class TeammateIdleEvent(CapabilityEvent, namespace=TEAMS_EVENTS, name='teammate_idle', dispatch='immediate'):
    """A team member's run produced its final output and is about to end, leaving the member idle.

    Emitted in the member's own run. A listener that calls `keep_working` keeps the run going: the
    feedback is enqueued as the member's next user message, and the member answers it instead of
    going idle. The gate is asked again the next time the member is about to go idle.
    """

    member: str
    """The member's name."""

    final_output: str
    """The member's final output, as text, cut at `MAX_EVENT_TEXT_CHARS`."""

    truncated: bool
    """Whether `final_output` was cut."""

    feedback: list[str] = field(default_factory=list[str])
    """What listeners told the member to do before it goes idle. Empty when it may go idle."""

    def keep_working(self, feedback: str) -> None:
        """Keep the member working, with `feedback` as its next user message."""
        self.feedback.append(feedback)


@dataclass(kw_only=True)
class _TaskGateEvent(CapabilityEvent, namespace=TEAMS_EVENTS, dispatch='immediate', abstract=True):
    """A team board change a listener can reject, turning the tool call that asked for it into a retry."""

    item: PlanItem
    """The board item the change is about."""

    feedback: list[str] = field(default_factory=list[str])
    """Why listeners rejected the change, sent to the model as a tool retry. Empty when it may go ahead."""

    @property
    def rejected(self) -> bool:
        """Whether a listener rejected the change."""
        return bool(self.feedback)

    def reject(self, feedback: str) -> None:
        """Stop the change, and tell the model why with `feedback`."""
        self.feedback.append(feedback)


@dataclass(kw_only=True)
class TeamTaskCreatingEvent(_TaskGateEvent, name='task_creating'):
    """A team member or the lead is adding an item to the team board through a tool call.

    `item` is the item as it would be added. A rejection turns the tool call into a retry carrying
    the feedback, and the item is not added.
    """


@dataclass(kw_only=True)
class TeamTaskCompletingEvent(_TaskGateEvent, name='task_completing'):
    """A team member or the lead is marking a board item `completed` through a tool call.

    `item` is the item as it is on the board, before the change. A rejection turns the tool call into
    a retry carrying the feedback, for example to require tests before the item counts as done, and
    the item keeps its status.
    """

    owner: str
    """The name of the member completing the item."""
