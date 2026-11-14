"""Events the `computer` tool emits, so an application can follow the screen without parsing results.

Each is emitted once per finished call, outside its durable unit, so durable execution doesn't repeat it. They add no
telemetry spans, since core already traces the tool call.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent
from pydantic_ai.computer_use import ComputerResult

COMPUTER_USE_EVENTS = 'computer_use'


@dataclass(kw_only=True)
class ActionsPerformedEvent(CapabilityEvent, namespace=COMPUTER_USE_EVENTS, name='actions_performed'):
    """A `computer` call finished and returned a screenshot; `tool_call_id` names the call."""

    result: ComputerResult
    """What the call did, the same record the result's `metadata` keeps in the history."""
