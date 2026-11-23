"""Events the `computer` tool sends, so an application can follow the screen without reading tool results.

One event is sent for each call that finishes with a screenshot. It's sent from workflow code, not from the call's
durable step, so a retried step doesn't send it twice. The events add no tracing spans, because the tool call is
already traced.
"""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent
from pydantic_ai.computer_use import ComputerResult

COMPUTER_USE_EVENTS = 'computer_use'


@dataclass(kw_only=True)
class ActionsPerformedEvent(CapabilityEvent, namespace=COMPUTER_USE_EVENTS, name='actions_performed'):
    """A `computer` call finished and returned a screenshot. `tool_call_id` says which call."""

    result: ComputerResult
    """What the call did. The tool result's `metadata` keeps the same record in the message history."""
