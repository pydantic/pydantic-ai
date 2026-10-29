"""The actions a [`Display`][pydantic_ai.workspaces.Display] performs.

Coordinates are pixels of the backend's screenshot, with the origin at the top left; a click or scroll without
coordinates happens at the pointer.

The model writes these as the `computer` tool's arguments, so they follow the providers' computer tools: each is told
apart by `type`, not the `kind` Pydantic AI's own message types use, and named as OpenAI's are (`click`, `keypress`).

A key is a single character or one of a fixed set of names; other names are refused when the action is validated.
Common spellings, including X keysyms such as `KP_Enter` or `minus` and browser names such as `NumpadAdd`, are
normalized to one lowercase name or character, and a combination written as one name, such as `ctrl+s`, is split into
its keys. A character presses whatever the keyboard layout needs to type it, shift or AltGr included; an upper-case
letter is its lower-case key. Names may be added only for keys no existing name covers.

How these actions change, so backends written today keep doing what the model asked:

- New kinds of action are new types in [`ComputerAction`][pydantic_ai.workspaces.ComputerAction].
- An action refuses fields it doesn't have, so a field added later is never silently dropped by an older version.
- An existing action only gains fields whose default is today's behaviour. Anything a backend must honour, such as
  scrolling by pixels, is a new type instead.
- The values a field accepts, such as the mouse buttons, don't grow.
- A field added later is left out of serialized data while it holds its default (`Field(exclude_if=...)`), so an
  older version can still read every action that doesn't use it.
- A new field, here or in a result, never changes what an existing field means.

A backend refuses an action, or a value, it can't perform exactly with
[`WorkspaceUnsupportedActionError`][pydantic_ai.workspaces.WorkspaceUnsupportedActionError]; it never ignores one.
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
        schema['required'] = ['type', *schema.get('required', [])]


ACTION_CONFIG = ConfigDict(extra='forbid', use_attribute_docstrings=True, json_schema_extra=_require_type)

_MODIFIER_NAMES = ('ctrl', 'shift', 'alt', 'super')
_MODIFIERS = frozenset(_MODIFIER_NAMES)

_NAMED_KEYS = (
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
    'printscreen',
    'pause',
    'menu',
    'capslock',
    'numlock',
    'scrolllock',
)
_NUMPAD_OPERATORS = ('add', 'subtract', 'multiply', 'divide', 'decimal')
_MEDIA_KEYS = ('volumemute', 'volumeup', 'volumedown', 'mediaplay', 'medianext', 'mediaprev')

_KEY_NAMES = frozenset(
    {
        *_MODIFIER_NAMES,
        *_NAMED_KEYS,
        *(f'f{n}' for n in range(1, 25)),
        *(f'numpad{n}' for n in range(10)),
        'numpad_enter',
        *(f'numpad_{operator}' for operator in _NUMPAD_OPERATORS),
        *_MEDIA_KEYS,
    }
)

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
    'prtsc': 'printscreen',
    'apps': 'menu',
    'contextmenu': 'menu',
    # X keysyms, as Claude's computer tool writes keys, matched without case. A left-hand modifier is the modifier;
    # right-hand ones are refused, left free to name keys of their own, such as AltGr.
    'control_l': 'ctrl',
    'shift_l': 'shift',
    'alt_l': 'alt',
    'super_l': 'super',
    'meta_l': 'super',
    'print': 'printscreen',
    'caps_lock': 'capslock',
    'num_lock': 'numlock',
    'scroll_lock': 'scrolllock',
    **{f'kp_{n}': f'numpad{n}' for n in range(10)},
    'kp_enter': 'numpad_enter',
    **{f'kp_{operator}': f'numpad_{operator}' for operator in _NUMPAD_OPERATORS},
    'xf86audiomute': 'volumemute',
    'xf86audioraisevolume': 'volumeup',
    'xf86audiolowervolume': 'volumedown',
    'xf86audioplay': 'mediaplay',
    'xf86audionext': 'medianext',
    'xf86audioprev': 'mediaprev',
    # X keysyms for punctuation, as in `ctrl+plus`.
    'plus': '+',
    'minus': '-',
    'equal': '=',
    'comma': ',',
    'period': '.',
    'slash': '/',
    'backslash': '\\',
    'semicolon': ';',
    'apostrophe': "'",
    'grave': '`',
    'bracketleft': '[',
    'bracketright': ']',
    'less': '<',
    'greater': '>',
    'question': '?',
    'exclam': '!',
    'at': '@',
    'numbersign': '#',
    'dollar': '$',
    'percent': '%',
    'asciicircum': '^',
    'ampersand': '&',
    'asterisk': '*',
    'parenleft': '(',
    'parenright': ')',
    'underscore': '_',
    'colon': ':',
    'quotedbl': '"',
    'braceleft': '{',
    'braceright': '}',
    'bar': '|',
    'asciitilde': '~',
    # DOM `key` and `code` spellings, as browser automation writes keys.
    'controlleft': 'ctrl',
    'shiftleft': 'shift',
    'altleft': 'alt',
    'metaleft': 'super',
    'numpadenter': 'numpad_enter',
    **{f'numpad{operator}': f'numpad_{operator}' for operator in _NUMPAD_OPERATORS},
    'audiovolumemute': 'volumemute',
    'audiovolumeup': 'volumeup',
    'audiovolumedown': 'volumedown',
    'mediaplaypause': 'mediaplay',
    'mediatracknext': 'medianext',
    'mediatrackprevious': 'mediaprev',
}

_UNKNOWN_KEY = 'is not a key: use a single character or one of ' + ', '.join(
    (
        *_MODIFIER_NAMES,
        *_NAMED_KEYS,
        'f1 to f24',
        'numpad0 to numpad9',
        'numpad_enter',
        *(f'numpad_{operator}' for operator in _NUMPAD_OPERATORS),
        *_MEDIA_KEYS,
    )
)


def _normalize_key(key: str) -> str:
    lower = key.lower()
    name = _KEY_ALIASES.get(lower, lower)
    if len(name) > 1 and name not in _KEY_NAMES:
        raise ValueError(f'{key!r} {_UNKNOWN_KEY}')
    return name


_COMBINATION = re.compile(r'(?<=.)\+(?=.)')
"""A `+` between two characters joins keys, as in `ctrl+s`; a lone `+`, or the last one in `ctrl++`, is the plus key."""


def _normalize_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if '' in keys:
        raise ValueError('a key name must not be empty')
    normalized = tuple(_normalize_key(part) for key in keys for part in _COMBINATION.split(key))
    if len(normalized) > 8:
        raise ValueError('at most 8 keys, counting each key of a combination such as `ctrl+s`')
    return normalized


def _normalize_modifiers(modifiers: tuple[str, ...]) -> tuple[str, ...]:
    # Checked here, before anything runs: a display that skipped an unknown one would still act without it.
    normalized = _normalize_keys(modifiers)
    if not _MODIFIERS.issuperset(normalized):
        raise ValueError('modifiers must be ctrl, shift, alt or super')
    return normalized


_Keys: TypeAlias = Annotated[tuple[str, ...], Field(min_length=1, max_length=8), AfterValidator(_normalize_keys)]
_Modifiers: TypeAlias = Annotated[tuple[str, ...], Field(max_length=8), AfterValidator(_normalize_modifiers)]

_CONTROL_CHARACTER = re.compile('[\x00-\x08\x0b-\x1f\x7f-\x9f]')


def _check_text(text: str) -> str:
    if match := _CONTROL_CHARACTER.search(text):
        raise ValueError(
            f'text must not contain the control character U+{ord(match.group()):04X}: only `\\n` (Enter) and '
            '`\\t` (Tab) are typed as keys; write a line break as `\\n`, and press other keys with `keypress`'
        )
    return text


def _check_position(x: int | None, y: int | None) -> None:
    if (x is None) != (y is None):
        raise ValueError('give both `x` and `y`, or neither to act at the pointer')


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Point:
    """A position on the screenshot."""

    x: _Coordinate
    y: _Coordinate


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Region:
    """A rectangle on the screenshot, from its top left corner to its bottom right one."""

    left: _Coordinate
    top: _Coordinate
    right: _Coordinate
    bottom: _Coordinate

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError('a region needs `right` greater than `left` and `bottom` greater than `top`')


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
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


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class MoveMouse:
    """Move the pointer, for example to reveal a hover menu."""

    type: Literal['move'] = 'move'
    x: _Coordinate
    """From the left edge of the screenshot."""
    y: _Coordinate
    """From the top edge of the screenshot."""
    modifiers: _Modifiers = ()
    """Modifier keys held while moving: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Drag:
    """Press the left button at the first point, move through the rest, and release at the last."""

    type: Literal['drag'] = 'drag'
    path: Annotated[tuple[Point, ...], Field(min_length=2, max_length=50)]
    """The points to drag through: at least a start and an end."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the drag: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class MouseDown:
    """Press a mouse button and keep it pressed until a `mouse_up`."""

    type: Literal['mouse_down'] = 'mouse_down'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class MouseUp:
    """Release a mouse button pressed with `mouse_down`."""

    type: Literal['mouse_up'] = 'mouse_up'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
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


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class TypeText:
    r"""Type text at the keyboard focus. Use `keypress` for shortcuts and special keys.

    `\n` presses Enter and `\t` presses Tab. Other control characters, including `\r`, are refused when the action
    is validated, so every backend types the same keys.
    """

    type: Literal['type'] = 'type'
    text: Annotated[str, Field(min_length=1, max_length=2_000), AfterValidator(_check_text)]
    """Up to 2,000 characters; type longer text in several actions."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class PressKeys:
    """Press keys together as one combination, then release them."""

    type: Literal['keypress'] = 'keypress'
    keys: _Keys
    """One key per entry, pressed together: `["enter"]`, `["ctrl", "a"]`, `["super", "shift", "t"]`."""
    repeat: Annotated[int, Field(ge=1, le=100)] = 1
    """How many times to press the combination."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
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
[`WorkspaceUnsupportedActionError`][pydantic_ai.workspaces.WorkspaceUnsupportedActionError]."""
