"""The `Computer` protocol, the actions the model can request, and the event a call emits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import CapabilityEvent

MouseButton = Literal['left', 'right', 'middle']
"""A mouse button."""

ScrollDirection = Literal['up', 'down', 'left', 'right']
"""The direction to scroll the content under the pointer."""


class ComputerError(Exception):
    """An action the computer could not perform.

    Raise it from a `Computer` method for a failure the model can act on, such as an
    unknown key name or a coordinate off the screen. `ComputerUse` reports the message to
    the model with a fresh screenshot and skips the call's remaining actions. Any other
    exception propagates and fails the tool call.
    """


class Computer(Protocol):
    """A screen, mouse, and keyboard for the `computer` tool to drive.

    Coordinates are pixels in the space of the latest screenshot, origin top left. An
    implementation that scales its screenshots maps coordinates back itself. Methods are
    async; an implementation that blocks should hand the work to a thread.
    """

    async def screenshot(self) -> bytes:
        """Capture the screen as PNG bytes."""
        ...  # pragma: no cover

    async def click(
        self, x: int, y: int, *, button: MouseButton = 'left', count: int = 1, modifiers: Sequence[str] = ()
    ) -> None:
        """Click `count` times at (`x`, `y`) while holding `modifiers`."""
        ...  # pragma: no cover

    async def move(self, x: int, y: int) -> None:
        """Move the pointer to (`x`, `y`) without clicking."""
        ...  # pragma: no cover

    async def drag(self, path: Sequence[tuple[int, int]]) -> None:
        """Press the left button at the first point, move through the rest, and release at the last."""
        ...  # pragma: no cover

    async def scroll(self, x: int, y: int, *, direction: ScrollDirection, amount: int) -> None:
        """Scroll `amount` wheel clicks in `direction` with the pointer at (`x`, `y`)."""
        ...  # pragma: no cover

    async def type_text(self, text: str) -> None:
        """Type `text` at the keyboard focus."""
        ...  # pragma: no cover

    async def press_keys(self, keys: Sequence[str]) -> None:
        """Press `keys` together as one chord, such as `['ctrl', 'c']`, then release them."""
        ...  # pragma: no cover


_Coordinate = Annotated[int, Field(ge=0)]


class _Action(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Point(_Action):
    """A screen position in screenshot pixels."""

    x: _Coordinate
    y: _Coordinate


class ClickAction(_Action):
    """Click at a position. `count=2` double-clicks; `modifiers` are held during the click, e.g. `['shift']`."""

    type: Literal['click']
    x: _Coordinate
    y: _Coordinate
    button: MouseButton = 'left'
    count: Annotated[int, Field(ge=1, le=3)] = 1
    modifiers: list[str] = Field(default_factory=list[str])


class MoveAction(_Action):
    """Move the pointer, for example to reveal a hover menu or tooltip."""

    type: Literal['move']
    x: _Coordinate
    y: _Coordinate


class DragAction(_Action):
    """Hold the left button from the first point through the last, then release."""

    type: Literal['drag']
    path: Annotated[list[Point], Field(min_length=2)]


class ScrollAction(_Action):
    """Scroll with the pointer over a position. `amount` is in mouse-wheel clicks."""

    type: Literal['scroll']
    x: _Coordinate
    y: _Coordinate
    direction: ScrollDirection
    amount: Annotated[int, Field(ge=1, le=50)] = 3


class TypeAction(_Action):
    """Type text at the keyboard focus. Use `keypress` for shortcuts and special keys."""

    type: Literal['type']
    text: Annotated[str, Field(min_length=1)]


class KeypressAction(_Action):
    """Press keys together as one chord, e.g. `['enter']`, `['ctrl', 'a']`, `['cmd', 'shift', 't']`."""

    type: Literal['keypress']
    keys: Annotated[list[str], Field(min_length=1)]


class WaitAction(_Action):
    """Pause, for example while a page or application loads."""

    type: Literal['wait']
    seconds: Annotated[float, Field(gt=0, le=10)] = 1


class ScreenshotAction(_Action):
    """Look at the screen without acting. Every call already ends with a screenshot."""

    type: Literal['screenshot']


ComputerAction = Annotated[
    ClickAction | MoveAction | DragAction | ScrollAction | TypeAction | KeypressAction | WaitAction | ScreenshotAction,
    Field(discriminator='type'),
]
"""One step of a `computer` tool call."""

OBSERVING_ACTIONS: frozenset[str] = frozenset({'screenshot', 'wait'})
"""Action types that change nothing on the computer, so `require_approval` lets them through."""

COMPUTER_USE_EVENTS = 'computer_use'


@dataclass(kw_only=True)
class ComputerActionsEvent(CapabilityEvent, namespace=COMPUTER_USE_EVENTS, name='computer_actions'):
    """A `computer` tool call finished: its actions ran and the closing screenshot was attempted.

    `performed` counts the actions that ran. When one raised `ComputerError`, `error`
    holds its message and the actions after it did not run. `screenshot_size` is the
    closing screenshot's width and height, or `None` when the screenshot failed or its
    PNG header could not be read.
    The actions are model-chosen and typed text can be sensitive, so a UI that renders
    them must treat the content as untrusted.
    """

    actions: tuple[ComputerAction, ...]
    performed: int
    error: str | None
    screenshot_size: tuple[int, int] | None
