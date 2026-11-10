"""Events emitted by the `computer` tool, so an application can follow the screen without parsing results.

Each is emitted once a call has finished, outside its durable unit, so it is emitted once even under durable
execution. They add no telemetry spans; core already traces the tool call.
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
