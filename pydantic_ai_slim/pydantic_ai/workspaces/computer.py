"""The actions a [`SupportsDisplay`][pydantic_ai.workspaces.SupportsDisplay] backend performs on a display.

Coordinates are pixels of the backend's screenshot, with the origin at the top left; a click or scroll without
coordinates happens at the pointer.

Common key names are normalized to one lowercase spelling, and a combination written as one name, such as `ctrl+s`,
is split into its keys. Other names, such as the X keysym `KP_Enter`, are kept as written; a backend refuses names it
can't press. The set of normalized names may grow.

How these actions change, so backends written today keep doing what the model asked:

- New kinds of action are new types in [`ComputerAction`][pydantic_ai.workspaces.ComputerAction].
- An existing action only gains fields whose default is today's behaviour. Anything a backend must honour, such as
  scrolling by pixels, is a new type instead.
- The values a field accepts, such as the mouse buttons, don't grow.

A backend refuses an action, or a value, it can't perform exactly with
[`UnsupportedComputerActionError`][pydantic_ai.workspaces.UnsupportedComputerActionError]; it never ignores one.
"""

from __future__ import annotations as _annotations

import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass

__all__ = (
    'Click',
    'ComputerAction',
    'Drag',
    'HoldKeys',
    'MouseButton',
    'MouseDown',
    'MouseUp',
    'MoveMouse',
    'Point',
    'PressKeys',
    'Region',
    'Scroll',
    'TypeText',
)

MouseButton: TypeAlias = Literal['left', 'middle', 'right', 'back', 'forward']
"""A mouse button; `back` and `forward` are the side buttons."""

_Coordinate: TypeAlias = Annotated[int, Field(ge=0)]


def _require_type(schema: dict[str, Any]) -> None:
    # `type` has a default so Python callers can omit it, but a model must always send it: it picks the action.
    if 'type' in schema.get('properties', {}):
        schema['required'] = ['type', *(name for name in schema.get('required', []) if name != 'type')]


_CONFIG = ConfigDict(use_attribute_docstrings=True, json_schema_extra=_require_type)

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


_COMBINATION = re.compile(r'(?<=.)\+(?=.)')
"""A `+` between two characters joins keys, as in `ctrl+s`; a lone `+`, or the last one in `ctrl++`, is the plus key."""


def _normalize_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if '' in keys:
        raise ValueError('a key name must not be empty')
    normalized = tuple(_normalize_key(part) for key in keys for part in _COMBINATION.split(key))
    if len(normalized) > 8:
        raise ValueError('at most 8 keys, counting each key of a combination such as `ctrl+s`')
    return normalized


_MODIFIERS = frozenset({'ctrl', 'shift', 'alt', 'super'})


def _normalize_modifiers(modifiers: tuple[str, ...]) -> tuple[str, ...]:
    # Checked here, before anything runs: a display that skipped an unknown one would still act without it.
    normalized = _normalize_keys(modifiers)
    if not _MODIFIERS.issuperset(normalized):
        raise ValueError('modifiers must be ctrl, shift, alt or super')
    return normalized


_Keys: TypeAlias = Annotated[tuple[str, ...], Field(min_length=1, max_length=8), AfterValidator(_normalize_keys)]
_Modifiers: TypeAlias = Annotated[tuple[str, ...], Field(max_length=8), AfterValidator(_normalize_modifiers)]


def _check_position(x: int | None, y: int | None) -> None:
    if (x is None) != (y is None):
        raise ValueError('give both `x` and `y`, or neither to act at the pointer')


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Point:
    """A position on the screenshot."""

    x: _Coordinate
    y: _Coordinate


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Region:
    """A rectangle on the screenshot, from its top left corner to its bottom right one."""

    left: _Coordinate
    top: _Coordinate
    right: _Coordinate
    bottom: _Coordinate

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError('a region needs `right` greater than `left` and `bottom` greater than `top`')


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Click:
    """Click a mouse button, at a position or where the pointer is."""

    type: Literal['click'] = 'click'
    x: _Coordinate | None = None
    """From the left edge of the screenshot. Omit both `x` and `y` to click where the pointer is."""
    y: _Coordinate | None = None
    """From the top edge of the screenshot."""
    button: MouseButton = 'left'
    count: Annotated[int, Field(ge=1, le=3)] = 1
    """1 for a click, 2 for a double click, 3 for a triple click."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the click: `ctrl`, `shift`, `alt` or `super`."""

    def __post_init__(self) -> None:
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class MoveMouse:
    """Move the pointer, for example to reveal a hover menu."""

    type: Literal['move'] = 'move'
    x: _Coordinate
    """From the left edge of the screenshot."""
    y: _Coordinate
    """From the top edge of the screenshot."""
    modifiers: _Modifiers = ()
    """Modifier keys held while moving: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Drag:
    """Press the left button at the first point, move through the rest, and release at the last."""

    type: Literal['drag'] = 'drag'
    path: Annotated[tuple[Point, ...], Field(min_length=2, max_length=50)]
    """The points to drag through: at least a start and an end."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the drag: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class MouseDown:
    """Press a mouse button and keep it pressed until a `mouse_up`."""

    type: Literal['mouse_down'] = 'mouse_down'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class MouseUp:
    """Release a mouse button pressed with `mouse_down`."""

    type: Literal['mouse_up'] = 'mouse_up'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Scroll:
    """Scroll the content under a position, or under the pointer."""

    type: Literal['scroll'] = 'scroll'
    direction: Literal['up', 'down', 'left', 'right']
    """Which way the content moves into view: `down` shows what is below."""
    amount: Annotated[int, Field(ge=1, le=50)] = 3
    """Mouse-wheel clicks; one click scrolls about three lines."""
    x: _Coordinate | None = None
    """From the left edge of the screenshot. Omit both `x` and `y` to scroll where the pointer is."""
    y: _Coordinate | None = None
    """From the top edge of the screenshot."""
    modifiers: _Modifiers = ()
    """Modifier keys held while scrolling: `ctrl`, `shift`, `alt` or `super`; `shift` scrolls sideways in some applications."""

    def __post_init__(self) -> None:
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class TypeText:
    """Type text at the keyboard focus. Use `keypress` for shortcuts and special keys."""

    type: Literal['type'] = 'type'
    text: Annotated[str, Field(min_length=1, max_length=2_000, pattern=r'^[^\x00]*$')]
    """Up to 2,000 characters, without NUL; type longer text in several actions."""


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class PressKeys:
    """Press keys together as one combination, then release them."""

    type: Literal['keypress'] = 'keypress'
    keys: _Keys
    """One key per entry, pressed together: `["enter"]`, `["ctrl", "a"]`, `["super", "shift", "t"]`."""
    repeat: Annotated[int, Field(ge=1, le=100)] = 1
    """How many times to press the combination."""


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class HoldKeys:
    """Hold keys down for a while, then release them."""

    type: Literal['hold_key'] = 'hold_key'
    keys: _Keys
    """One key per entry, held together."""
    seconds: Annotated[float, Field(gt=0, le=30)]
    """How long to hold them, up to 30 seconds."""


ComputerAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | PressKeys | HoldKeys,
    Field(discriminator='type'),
]
"""A mouse or keyboard action on a display. New types may be added, so a backend must refuse one it doesn't know with
[`UnsupportedComputerActionError`][pydantic_ai.workspaces.UnsupportedComputerActionError]."""
