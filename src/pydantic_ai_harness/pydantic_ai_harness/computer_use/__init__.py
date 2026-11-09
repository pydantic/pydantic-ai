"""Let an agent see and control the display of its workspace."""

from pydantic_ai_harness.computer_use._capability import ComputerUse
from pydantic_ai_harness.computer_use._events import (
    COMPUTER_USE_EVENTS,
    ActionsPerformedEvent,
    DisplayRestartedEvent,
)

__all__ = ['COMPUTER_USE_EVENTS', 'ActionsPerformedEvent', 'ComputerUse', 'DisplayRestartedEvent']
