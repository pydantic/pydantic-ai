"""The actions a [`SupportsComputerUse`][pydantic_ai.workspaces.SupportsComputerUse] backend performs on a display.

Actions are validated on construction. Coordinates are display pixels with the origin at the top left; a click
or scroll without coordinates happens
at the current pointer position. Common key names are normalized on construction to one lowercase spelling
(`ctrl`, `shift`, `alt`, `super`, `enter`, `escape`, `tab`, `backspace`, `delete`, `insert`, `home`, `end`,
`pageup`, `pagedown`, `up`, `down`, `left`, `right`, `space`, `f1` to `f24`, and single characters), so a backend
reads one spelling whichever model produced the action. Other names, such as the X keysym `KP_Enter`, are kept
as written.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass

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
    'Scroll',
    'TypeText',
)

MouseButton: TypeAlias = Literal['left', 'middle', 'right', 'back', 'forward']
"""A mouse button; `back` and `forward` are the side buttons."""

_Coordinate: TypeAlias = Annotated[int, Field(ge=0)]
_Seconds: TypeAlias = Annotated[float, Field(gt=0, allow_inf_nan=False)]

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


_KEY_NAMES = frozenset(
    {
        'ctrl',
        'shift',
        'alt',
        'super',
        'enter',
        'escape',
        'tab',
        'backspace',
        'delete',
        'insert',
        'home',
        'end',
        'pageup',
        'pagedown',
        'up',
        'down',
        'left',
        'right',
        'space',
        *(f'f{n}' for n in range(1, 25)),
    }
)


def _normalize_key(key: str) -> str:
    lower = key.lower()
    if lower in _KEY_ALIASES:
        return _KEY_ALIASES[lower]
    if lower in _KEY_NAMES or len(key) == 1:
        return lower
    # Anything else may be a case-sensitive keysym, such as `KP_Enter` or `XF86AudioMute`.
    return key


def _normalize_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_key(key) for key in keys)


def _check_position(x: int | None, y: int | None) -> None:
    if (x is None) != (y is None):
        raise ValueError('give both `x` and `y`, or neither to act at the pointer')


@dataclass(frozen=True, kw_only=True)
class _Keyed:
    keys: tuple[str, ...] = ()
    """Keys held down during the action, such as `('ctrl',)` or `('shift', 'alt')`."""

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', _normalize_keys(self.keys))


@dataclass(frozen=True, kw_only=True)
class Click(_Keyed):
    """Click a mouse button, optionally moving the pointer first."""

    x: _Coordinate | None = None
    """Horizontal position; with `y` `None` too, clicks where the pointer is."""
    y: _Coordinate | None = None
    """Vertical position; with `x` `None` too, clicks where the pointer is."""
    button: MouseButton = 'left'
    count: Annotated[int, Field(ge=1, le=3)] = 1
    """1 for a click, 2 for a double click, 3 for a triple click."""
    kind: Literal['click'] = 'click'

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True)
class MoveMouse(_Keyed):
    """Move the pointer."""

    x: _Coordinate
    y: _Coordinate
    kind: Literal['move_mouse'] = 'move_mouse'


@dataclass(frozen=True, kw_only=True)
class Drag(_Keyed):
    """Press the left button at the first point, move through the rest, and release at the last."""

    path: Annotated[tuple[tuple[_Coordinate, _Coordinate], ...], Field(min_length=2)]
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
    """Turn the scroll wheel, optionally moving the pointer first.

    Amounts are wheel clicks, not pixels: one click scrolls about three lines in most applications.
    """

    x: _Coordinate | None = None
    y: _Coordinate | None = None
    dx: int = 0
    """Wheel clicks to the right; negative scrolls left."""
    dy: int = 0
    """Wheel clicks down; negative scrolls up."""
    kind: Literal['scroll'] = 'scroll'

    def __post_init__(self) -> None:
        super().__post_init__()
        _check_position(self.x, self.y)
        if not self.dx and not self.dy:
            raise ValueError('give a non-zero `dx` or `dy`')


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
    seconds: _Seconds
    kind: Literal['hold_key'] = 'hold_key'

    def __post_init__(self) -> None:
        object.__setattr__(self, 'keys', _normalize_keys(self.keys))


ComputerAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | KeyPress | HoldKey,
    Field(discriminator='kind'),
]
"""A mouse or keyboard action on a display. New kinds may be added, so a backend must refuse a kind it doesn't know with
[`UnsupportedComputerActionError`][pydantic_ai.workspaces.UnsupportedComputerActionError]."""
