"""The `computer` tool: the actions a model can ask for, and the shapes of the tool's arguments and result.

A capability that gives an agent a screen adds a tool of kind
[`COMPUTER_KIND`][pydantic_ai.computer_use.COMPUTER_KIND] and performs its actions on that screen. For a provider
with a computer tool of its own, the model adapter sends that tool instead and converts its calls and results.

Positions (`x` and `y`) are pixels of the latest screenshot, from its top left corner, unless the call's
[`coordinate_grid`][pydantic_ai.computer_use.ComputerCallPart.coordinate_grid] is set. A click or scroll without a
position happens at the pointer.

The `type` field picks the action: the class name in snake case, such as `click` or `press_keys`. A key is a single
character or one of a fixed set of names. Common other spellings, such as the X11 `KP_Enter` or the browser
`NumpadAdd`, are converted, and `ctrl+s` is split into two keys. A character presses whatever keys the keyboard
layout needs to type it. An upper-case letter presses the same key as its lower-case one.

The actions change only in ways that keep an existing tool doing what the model asked:

- A new action gets a new `type` in [`ComputerToolAction`][pydantic_ai.computer_use.ComputerToolAction], so an
  older tool rejects it instead of ignoring it. Actions reject unknown fields for the same reason.
- An existing action only gains fields whose default keeps today's behaviour, left out of saved data while at the
  default. A change a tool must act on, such as scrolling by pixels, is a new action.
- The values a field accepts, such as the mouse buttons, never grow. A key name is added only for a key no existing
  name covers.
- A new field, on an action or a result, never changes what an existing field means.

A result's content is a short summary, what each action returned, then a screenshot of the screen after the call.
The summary's wording can change between releases;
[`ComputerResult`][pydantic_ai.computer_use.ComputerResult], in the result's `metadata`, does not.

In the message history, the tool's calls and results become
[`ComputerCallPart`][pydantic_ai.computer_use.ComputerCallPart] and
[`ComputerReturnPart`][pydantic_ai.computer_use.ComputerReturnPart], whatever the tool's name.
"""

from __future__ import annotations as _annotations

import re
from typing import Annotated, Any, Literal, NotRequired, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from ._computer_use import COMPUTER_KIND_ID, COORDINATE_GRID
from .messages import BinaryImage, ToolCallPart, ToolKind, ToolReturnPart

__all__ = (
    'COMPUTER_KIND',
    'Click',
    'ComputerArgs',
    'ComputerCallPart',
    'ComputerContent',
    'ComputerResult',
    'ComputerReturnPart',
    'ComputerToolAction',
    'CursorPosition',
    'Drag',
    'HoldKeys',
    'MouseButton',
    'MouseDown',
    'MouseMove',
    'MouseUp',
    'Point',
    'PressKeys',
    'Region',
    'Screenshot',
    'Scroll',
    'TypeText',
    'Wait',
    'Zoom',
)

MouseButton: TypeAlias = Literal['left', 'middle', 'right', 'back', 'forward']
"""A mouse button; `back` and `forward` are the side buttons."""

_Coordinate: TypeAlias = Annotated[int, Field(ge=0)]


def _require_type(schema: dict[str, Any]) -> None:
    # `type` has a default so Python callers can omit it, but a model must always send it: it picks the action.
    if 'type' in schema.get('properties', {}):
        schema['required'] = ['type', *schema.get('required', [])]


_ACTION_CONFIG = ConfigDict(extra='forbid', use_attribute_docstrings=True, json_schema_extra=_require_type)

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
    # X11 key names (keysyms), as Claude sends them. Right-hand modifiers such as `Control_R` are rejected, so those
    # names stay free for keys such as AltGr.
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
    # X11 names of punctuation keys, as in `ctrl+plus`.
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
    # Browser key names (the DOM `key` and `code` values), which browser automation tools use.
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
    if not key:
        raise ValueError('a key name must not be empty')
    lower = key.lower()
    name = _KEY_ALIASES.get(lower, lower)
    if len(name) > 1 and name not in _KEY_NAMES:
        raise ValueError(f'{key!r} {_UNKNOWN_KEY}')
    return name


_COMBINATION = re.compile(r'(?<=.)\+(?=.)')
"""A `+` between two characters joins keys, as in `ctrl+s`; a lone `+`, or the last one in `ctrl++`, is the plus key."""


def _normalize_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_key(part) for key in keys for part in _COMBINATION.split(key))


def _normalize_modifiers(modifiers: tuple[str, ...]) -> tuple[str, ...]:
    # Checked before any action runs, so a screen can't skip a modifier it doesn't know and act without it.
    normalized = _normalize_keys(modifiers)
    if not _MODIFIERS.issuperset(normalized):
        raise ValueError('modifiers must be ctrl, shift, alt or super')
    return normalized


_Keys: TypeAlias = Annotated[tuple[str, ...], Field(min_length=1), AfterValidator(_normalize_keys)]
_Modifiers: TypeAlias = Annotated[tuple[str, ...], AfterValidator(_normalize_modifiers)]


def _check_position(x: int | None, y: int | None) -> None:
    if (x is None) != (y is None):
        raise ValueError('give both `x` and `y`, or neither to act at the pointer')


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Point:
    """A position on the screenshot."""

    x: _Coordinate
    y: _Coordinate


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Region:
    """A rectangle on the screenshot, from its top left corner to its bottom right one.

    `right` and `bottom` are just outside it: `left=0, right=10` is 10 pixels wide.
    """

    left: _Coordinate
    top: _Coordinate
    right: _Coordinate
    bottom: _Coordinate

    def __post_init__(self) -> None:
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError('a region needs `right` greater than `left` and `bottom` greater than `top`')


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Click:
    """Click a mouse button, at a position or where the pointer is."""

    type: Literal['click'] = 'click'
    x: _Coordinate | None = None
    """From the left edge of the screenshot. Omit both `x` and `y` to click where the pointer is."""
    y: _Coordinate | None = None
    """From the top edge of the screenshot."""
    button: MouseButton = 'left'
    count: Annotated[int, Field(ge=1)] = 1
    """1 for a click, 2 for a double click, 3 for a triple click."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the click: `ctrl`, `shift`, `alt` or `super`."""

    def __post_init__(self) -> None:
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class MouseMove:
    """Move the pointer, for example to reveal a hover menu."""

    type: Literal['mouse_move'] = 'mouse_move'
    x: _Coordinate
    """From the left edge of the screenshot."""
    y: _Coordinate
    """From the top edge of the screenshot."""
    modifiers: _Modifiers = ()
    """Modifier keys held while moving: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Drag:
    """Press the left button at the first point, move through the rest, and release at the last."""

    type: Literal['drag'] = 'drag'
    path: Annotated[tuple[Point, ...], Field(min_length=2)]
    """The points to drag through: at least a start and an end."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the drag: `ctrl`, `shift`, `alt` or `super`."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class MouseDown:
    """Press a mouse button and keep it pressed until a `mouse_up`."""

    type: Literal['mouse_down'] = 'mouse_down'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class MouseUp:
    """Release a mouse button pressed with `mouse_down`."""

    type: Literal['mouse_up'] = 'mouse_up'
    button: MouseButton = 'left'


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Scroll:
    """Scroll the content under a position, or under the pointer."""

    type: Literal['scroll'] = 'scroll'
    direction: Literal['up', 'down', 'left', 'right']
    """Which way the content moves into view: `down` shows what is below."""
    amount: Annotated[int, Field(ge=1)] = 3
    """Mouse-wheel clicks; one click scrolls about three lines."""
    x: _Coordinate | None = None
    """From the left edge of the screenshot. Omit both `x` and `y` to scroll where the pointer is."""
    y: _Coordinate | None = None
    """From the top edge of the screenshot."""
    modifiers: _Modifiers = ()
    """Modifier keys held while scrolling: `ctrl`, `shift`, `alt` or `super`.

    `shift` scrolls sideways in some applications.
    """

    def __post_init__(self) -> None:
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class TypeText:
    """Type text at the keyboard focus. Use `press_keys` for shortcuts and special keys."""

    type: Literal['type_text'] = 'type_text'
    text: Annotated[str, Field(min_length=1)]


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class PressKeys:
    """Press keys together as one combination, then release them."""

    type: Literal['press_keys'] = 'press_keys'
    keys: _Keys
    """One key per entry, pressed together: `["enter"]`, `["ctrl", "a"]`, `["super", "shift", "t"]`."""
    repeat: Annotated[int, Field(ge=1)] = 1
    """How many times to press the combination."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class HoldKeys:
    """Hold keys down for a while, then release them."""

    type: Literal['hold_keys'] = 'hold_keys'
    keys: _Keys
    """One key per entry, held together."""
    seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    """How long to hold them."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Wait:
    """Pause, for example while a page or application loads."""

    type: Literal['wait'] = 'wait'
    seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1
    """How long to wait."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Screenshot:
    """Look at the screen without acting. A call's result already ends with a screenshot."""

    type: Literal['screenshot'] = 'screenshot'


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Zoom:
    """Look closely at part of the screen, to read small text or find a small target."""

    type: Literal['zoom'] = 'zoom'
    region: Region
    """The part of the screen to look at. Positions in later actions are still on the full screen."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class CursorPosition:
    """Report where the pointer is."""

    type: Literal['cursor_position'] = 'cursor_position'


ComputerToolAction: TypeAlias = Annotated[
    Click
    | MouseMove
    | Drag
    | MouseDown
    | MouseUp
    | Scroll
    | TypeText
    | PressKeys
    | HoldKeys
    | Wait
    | Screenshot
    | Zoom
    | CursorPosition,
    Field(discriminator='type'),
]
"""What the `computer` tool accepts: mouse and keyboard actions, plus waiting and looking."""


class ComputerArgs(TypedDict):
    """The `computer` tool's arguments."""

    actions: list[ComputerToolAction]
    """The actions to perform, in order."""


ComputerContent: TypeAlias = list[str | BinaryImage]
"""What the model sees from a `computer` call: a summary, what each action returned, then the screenshot."""


class ComputerResult(TypedDict):
    """A `computer` call's outcome, returned as its `metadata`."""

    performed: int
    """How many actions ran: all of them, or those before the one that failed."""

    error: NotRequired[str]
    """Why an action failed. It is the action right after the `performed` ones. The actions after it did not run."""

    changed: NotRequired[bool]
    """`True` when the actions that ran differ from those the model asked for, for example because an approval
    changed them. A model adapter then sends the summary instead of reporting the model's actions as the ones that
    ran."""


COMPUTER_KIND = ToolKind(
    'computer_use', 'computer', args=ComputerArgs, content=ComputerContent, metadata=ComputerResult
)
"""The tool kind that marks a tool as the `computer` tool.

When a request includes a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool], the provider's own computer tool is
sent to the model in place of the tool of this kind, which still runs the calls."""
assert COMPUTER_KIND.id == COMPUTER_KIND_ID


class ComputerCallPart(ToolCallPart[ComputerArgs], kind=COMPUTER_KIND):
    """A call of the `computer` tool."""

    @property
    def actions(self) -> list[ComputerToolAction] | None:
        """The actions the model asked for, or `None` if the arguments are incomplete (still streaming) or malformed."""
        typed = self.typed_args
        return typed['actions'] if typed is not None else None

    @property
    def coordinate_grid(self) -> int | None:
        """The size of the grid the call's positions are on, such as `1000` for positions from 0 to 999.

        `None` means pixels. The model that made the call records it in `provider_details`, from its profile's
        `computer_use_coordinate_grid`. A history that lost `provider_details`, such as one a UI rebuilt, gives `None`.
        """
        grid = (self.provider_details or {}).get(COORDINATE_GRID)
        return grid if type(grid) is int and grid >= 2 else None


class ComputerReturnPart(ToolReturnPart[ComputerContent, ComputerResult], kind=COMPUTER_KIND):
    """A successful result of the `computer` tool."""

    @property
    def result(self) -> ComputerResult | None:
        """The call's outcome. `None` when the result has no metadata, such as a result a UI rebuilt."""
        return self.typed_metadata

    @property
    def screenshot(self) -> BinaryImage | None:
        """The screen after the call. `None` if the result doesn't end with an image, such as after it was dropped."""
        content = self.typed_content
        last = content[-1] if content else None
        return last if isinstance(last, BinaryImage) else None
