"""The actions a [`SupportsComputerUse`][pydantic_ai.workspaces.SupportsComputerUse] backend performs on a display.

Coordinates are display pixels with the origin at the top left. A click or scroll without coordinates happens
at the current pointer position. Key names are normalized on construction to one lowercase vocabulary
(`ctrl`, `shift`, `alt`, `super`, `enter`, `escape`, `tab`, `backspace`, `delete`, `up`, `pageup`, `f1`, `a`,
...), so a backend reads one spelling whichever model produced the action.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

__all__ = (
    'Click',
    'ComputerAction',
    'Drag',
    'HoldKey',
    'KeyPress',
    'MouseButton',
    'MouseDown',
    'MouseUp',
    'MoveMouse',
    'Screenshot',
    'Scroll',
    'TypeText',
    'Wait',
)

MouseButton: TypeAlias = Literal['left', 'middle', 'right', 'back', 'forward']
"""A mouse button; `back` and `forward` are the side buttons."""

_KEY_ALIASES = {
    'return': 'enter',
    'esc': 'escape',
    'control': 'ctrl',
    'cmd': 'super',
    'command': 'super',
    'meta': 'super',
    'win': 'super',
    'windows': 'super',
    'option': 'alt',
    'arrowup': 'up',
    'arrowdown': 'down',
    'arrowleft': 'left',
    'arrowright': 'right',
    'pgup': 'pageup',
    'page_up': 'pageup',
    'prior': 'pageup',
    'pgdn': 'pagedown',
    'page_down': 'pagedown',
    'next': 'pagedown',
    'del': 'delete',
    'back_space': 'backspace',
    'spacebar': 'space',
    ' ': 'space',
}


def _normalize_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_KEY_ALIASES.get(key.lower(), key.lower()) for key in keys)


@dataclass(frozen=True, kw_only=True)
class _Keyed:
    keys: tuple[str, ...] = ()
    """Keys held down during the action, such as `('ctrl',)` or `('shift', 'alt')`."""

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', _normalize_keys(self.keys))


@dataclass(frozen=True, kw_only=True)
class Click(_Keyed):
    """Click a mouse button, optionally moving the pointer first."""

    x: int | None = None
    """Horizontal position; `None` with `y` clicks where the pointer is."""
    y: int | None = None
    """Vertical position; `None` with `x` clicks where the pointer is."""
    button: MouseButton = 'left'
    count: Annotated[int, Field(ge=1, le=3)] = 1
    """1 for a click, 2 for a double click, 3 for a triple click."""
    kind: Literal['click'] = 'click'


@dataclass(frozen=True, kw_only=True)
class MoveMouse(_Keyed):
    """Move the pointer."""

    x: int
    y: int
    kind: Literal['move_mouse'] = 'move_mouse'


@dataclass(frozen=True, kw_only=True)
class Drag(_Keyed):
    """Press the left button at the first point, move through the rest, and release at the last."""

    path: Annotated[tuple[tuple[int, int], ...], Field(min_length=2)]
    """The `(x, y)` points to drag through, at least a start and an end."""
    kind: Literal['drag'] = 'drag'


@dataclass(frozen=True, kw_only=True)
class MouseDown:
    """Press a mouse button without releasing it."""

    button: MouseButton = 'left'
    kind: Literal['mouse_down'] = 'mouse_down'


@dataclass(frozen=True, kw_only=True)
class MouseUp:
    """Release a mouse button."""

    button: MouseButton = 'left'
    kind: Literal['mouse_up'] = 'mouse_up'


@dataclass(frozen=True, kw_only=True)
class Scroll(_Keyed):
    """Turn the scroll wheel, optionally moving the pointer first."""

    x: int | None = None
    y: int | None = None
    dx: int = 0
    """Wheel clicks to the right; negative scrolls left."""
    dy: int = 0
    """Wheel clicks down; negative scrolls up."""
    kind: Literal['scroll'] = 'scroll'


@dataclass(frozen=True, kw_only=True)
class TypeText:
    """Type text as key presses."""

    text: str
    kind: Literal['type_text'] = 'type_text'


@dataclass(frozen=True, kw_only=True)
class KeyPress:
    """Press one key or key combination, such as `('ctrl', 's')`."""

    keys: Annotated[tuple[str, ...], Field(min_length=1)]
    repeat: Annotated[int, Field(ge=1)] = 1
    kind: Literal['key_press'] = 'key_press'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', _normalize_keys(self.keys))


@dataclass(frozen=True, kw_only=True)
class HoldKey:
    """Hold keys down for a while, then release them."""

    keys: Annotated[tuple[str, ...], Field(min_length=1)]
    seconds: Annotated[float, Field(gt=0)]
    kind: Literal['hold_key'] = 'hold_key'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', _normalize_keys(self.keys))


@dataclass(frozen=True, kw_only=True)
class Wait:
    """Do nothing for a while, letting the screen settle."""

    seconds: Annotated[float, Field(ge=0)] = 1.0
    kind: Literal['wait'] = 'wait'


@dataclass(frozen=True, kw_only=True)
class Screenshot:
    """Ask for a screenshot; backends do nothing, since the caller takes one after the actions."""

    kind: Literal['screenshot'] = 'screenshot'


ComputerAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | KeyPress | HoldKey | Wait | Screenshot,
    Field(discriminator='kind'),
]
"""An action on a display. New kinds may be added, so a backend must refuse a kind it doesn't know with
[`UnsupportedComputerActionError`][pydantic_ai.workspaces.UnsupportedComputerActionError]."""
