"""Let an agent see and control a screen: the `ComputerUse` capability and the protocols a screen implements."""

from pydantic_ai_harness.computer_use._capability import ComputerUse, ComputerUseActionLimits
from pydantic_ai_harness.computer_use._events import COMPUTER_USE_EVENTS, ActionsPerformedEvent
from pydantic_ai_harness.computer_use._protocol import (
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsDisplay,
    SupportsScaledScreenshot,
    SupportsZoom,
    UnsupportedActionError,
)

__all__ = [
    'COMPUTER_USE_EVENTS',
    'ActionsPerformedEvent',
    'ComputerUse',
    'ComputerUseActionLimits',
    'Display',
    'DisplayAction',
    'SupportsCursorPosition',
    'SupportsDisplay',
    'SupportsScaledScreenshot',
    'SupportsZoom',
    'UnsupportedActionError',
]
