"""ComputerUse capability: see and control a computer's screen, mouse, and keyboard."""

from typing import TYPE_CHECKING

from pydantic_ai_harness.computer_use._capability import ComputerUse
from pydantic_ai_harness.computer_use._computer import (
    COMPUTER_USE_EVENTS,
    ClickAction,
    Computer,
    ComputerAction,
    ComputerActionsEvent,
    ComputerError,
    DragAction,
    KeypressAction,
    MouseButton,
    MoveAction,
    Point,
    ScreenshotAction,
    ScrollAction,
    ScrollDirection,
    TypeAction,
    WaitAction,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.computer_use._local import LocalComputer

__all__ = [
    'COMPUTER_USE_EVENTS',
    'ClickAction',
    'Computer',
    'ComputerAction',
    'ComputerActionsEvent',
    'ComputerError',
    'ComputerUse',
    'DragAction',
    'KeypressAction',
    'LocalComputer',
    'MouseButton',
    'MoveAction',
    'Point',
    'ScreenshotAction',
    'ScrollAction',
    'ScrollDirection',
    'TypeAction',
    'WaitAction',
]


def __getattr__(name: str) -> object:
    """Import `LocalComputer` on first use, so the protocol and capability work without the `computer-use` extra."""
    if name == 'LocalComputer':
        from pydantic_ai_harness.computer_use._local import LocalComputer

        return LocalComputer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
