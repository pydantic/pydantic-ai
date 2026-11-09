"""Events emitted by the `computer` tool, so an application can follow the display without parsing results.

Each is emitted once a call has finished, outside its durable unit, so it is emitted once even under durable
execution. The same facts are in the call's [`ComputerResult`][pydantic_ai.computer_use.ComputerResult], which the
history keeps. They add no telemetry spans; core already traces the tool call.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

COMPUTER_USE_EVENTS = 'computer_use'


@dataclass(kw_only=True)
class ActionsPerformedEvent(CapabilityEvent, namespace=COMPUTER_USE_EVENTS, name='actions_performed'):
    """A `computer` call finished and returned a screenshot."""

    performed: int
    """How many actions ran: all of them, or those before the one that failed."""

    error: str | None = None
    """Why the action after the performed ones failed, or `None` when all of them ran."""

    screenshot_size: tuple[int, int]
    """The `(width, height)` of the screenshot the call returned."""


@dataclass(kw_only=True)
class DisplayRestartedEvent(CapabilityEvent, namespace=COMPUTER_USE_EVENTS, name='display_restarted'):
    """The display restarted during a `computer` call, such as after its process stopped.

    What was on the screen before may be gone. The model isn't told; the call's screenshot shows the screen as it is.
    Only displays that report restarts emit it; see
    [`SupportsRestartCount`][pydantic_ai.workspaces.SupportsRestartCount].
    """
