"""The `computer` tool: its kind, its actions, and the shapes of its arguments and result.

A capability that gives an agent a screen, such as Pydantic AI Harness's `ComputerUse`, defines a tool of
[`COMPUTER_KIND`][pydantic_ai.computer_use.COMPUTER_KIND]. A model adapter whose provider has a native computer tool
sends that tool as the native one, translates the provider's calls into these arguments, and reads the result back
from them. Driving a screen is not part of Pydantic AI: the capability does that.

Coordinates are pixels of the latest screenshot the model saw, with the origin at the top left, unless the call's
[`coordinate_grid`][pydantic_ai.computer_use.ComputerCallPart.coordinate_grid] names a grid they are on instead. The
agent sets it on the calls of a model whose profile sets `computer_use_coordinate_grid`, such as Gemini, after telling
that model in the tool's description. A click or scroll without coordinates happens at the pointer.

The actions follow the providers' computer tools: each is told apart by `type`, not the `kind` Pydantic AI's own
message types use, and named as OpenAI's are (`click`, `keypress`). A key is a single character or one of a fixed set
of names; other names are refused when the action is validated. Common spellings, including X keysyms such as
`KP_Enter` or `minus` and browser names such as `NumpadAdd`, are normalized to one lowercase name or character, and a
combination written as one name, such as `ctrl+s`, is split into its keys. A character presses whatever the keyboard
layout needs to type it, shift or AltGr included; an upper-case letter is its lower-case key.

How the actions change, so a tool written today keeps doing what the model asked:

- New kinds of action are new types in [`ComputerToolAction`][pydantic_ai.computer_use.ComputerToolAction], so a
  tool of this kind fails a call with a type it doesn't know rather than ignoring it.
- An action refuses fields it doesn't have, so a field added later is never silently dropped by an older version.
- An existing action only gains fields whose default is today's behaviour. Anything a tool must honour, such as
  scrolling by pixels, is a new type instead.
- The values a field accepts, such as the mouse buttons, don't grow, and key names are added only for keys no
  existing name covers.
- A field added later is left out of serialized data while it holds its default (`Field(exclude_if=...)`), so an
  older version can still read every action that doesn't use it.
- A new field, here or in a result, never changes what an existing field means.

A result's content is text and images: a short summary, then what each action that returns something returned, in
action order, then the screenshot. The last image is always the screen after the call. The summary's wording isn't
stable; [`ComputerResult`][pydantic_ai.computer_use.ComputerResult] in the result's `metadata` is.

The tool's parts are promoted to [`ComputerCallPart`][pydantic_ai.computer_use.ComputerCallPart] and
[`ComputerReturnPart`][pydantic_ai.computer_use.ComputerReturnPart], so code reads them with `isinstance` and typed
properties, whatever the tool is called.
"""

from __future__ import annotations as _annotations

import re
from typing import Annotated, Any, Literal, NotRequired, TypeAlias

import pydantic
from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

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
    'MouseUp',
    'MoveMouse',
    'Point',
    'PressKeys',
    'Region',
    'SafetyCheck',
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
    normalized = tuple(_normalize_key(part) for key in keys for part in _COMBINATION.split(key))
    if len(normalized) > 8:
        raise ValueError('at most 8 keys, counting each key of a combination such as `ctrl+s`')
    return normalized


def _normalize_modifiers(modifiers: tuple[str, ...]) -> tuple[str, ...]:
    # Checked here, before anything runs: a screen that skipped an unknown one would still act without it.
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
    count: Annotated[int, Field(ge=1, le=3)] = 1
    """1 for a click, 2 for a double click, 3 for a triple click."""
    modifiers: _Modifiers = ()
    """Modifier keys held during the click: `ctrl`, `shift`, `alt` or `super`."""

    def __post_init__(self) -> None:
        _check_position(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class MoveMouse:
    """Move the pointer, for example to reveal a hover menu."""

    type: Literal['move'] = 'move'
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
    path: Annotated[tuple[Point, ...], Field(min_length=2, max_length=50)]
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
    amount: Annotated[int, Field(ge=1, le=50)] = 3
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
    r"""Type text at the keyboard focus. Use `keypress` for shortcuts and special keys.

    `\n` presses Enter and `\t` presses Tab. Other control characters, including `\r`, are refused when the action
    is validated, so every backend types the same keys.
    """

    type: Literal['type'] = 'type'
    text: Annotated[str, Field(min_length=1, max_length=2_000), AfterValidator(_check_text)]
    """Up to 2,000 characters; type longer text in several actions."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class PressKeys:
    """Press keys together as one combination, then release them."""

    type: Literal['keypress'] = 'keypress'
    keys: _Keys
    """One key per entry, pressed together: `["enter"]`, `["ctrl", "a"]`, `["super", "shift", "t"]`."""
    repeat: Annotated[int, Field(ge=1, le=100)] = 1
    """How many times to press the combination."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class HoldKeys:
    """Hold keys down for a while, then release them."""

    type: Literal['hold_key'] = 'hold_key'
    keys: _Keys
    """One key per entry, held together."""
    seconds: Annotated[float, Field(gt=0, le=30)]
    """How long to hold them, up to 30 seconds."""


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class Wait:
    """Pause, for example while a page or application loads."""

    type: Literal['wait'] = 'wait'
    seconds: Annotated[float, Field(gt=0, le=30)] = 1
    """How long to wait, up to 30 seconds."""


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
    | MoveMouse
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
"""What the model sees from a `computer` call; see the module docstring for its order."""


class ComputerResult(TypedDict):
    """A `computer` call's outcome, returned as its `metadata` for model adapters, UIs and later calls."""

    performed: int
    """How many actions ran: all of them, or those before the one that failed."""

    error: NotRequired[str]
    """Why the action after the performed ones failed; the rest did not run."""

    screenshot_size: NotRequired[tuple[int, int]]
    """The `(width, height)` of the screenshot in this result, as the model saw it: the space of the model's next
    coordinates, which may be smaller than the display."""

    changed: NotRequired[bool]
    """Set when the actions that ran differ from the ones the model asked for, such as after an approval changed
    them: `performed` counts the ones that ran, and the summary says what they were. A model adapter then sends the
    summary rather than reporting the model's own actions as run."""


@dataclass(frozen=True, kw_only=True)
class SafetyCheck:
    """A check a provider asks a person to make before a `computer` call runs, such as OpenAI's `pending_safety_checks`.

    A call with safety checks waits for approval whichever tool runs it, as a
    [deferred tool](../deferred-tools.md#human-in-the-loop-tool-approval) does, and the model adapter acknowledges the
    checks once the approved call has a result; see
    [`ComputerCallPart.safety_checks`][pydantic_ai.computer_use.ComputerCallPart.safety_checks]. A call with safety
    checks for an external or deferred tool raises a `UserError`, since approval can't gate a tool run elsewhere.
    """

    message: str | None = None
    """What the person should check, in the provider's words."""

    code: str | None = None
    """The provider's identifier for the kind of check, such as `'malicious_instructions'`."""

    id: str | None = None
    """The provider's identifier for this check, which it needs back when the call is approved."""


_SAFETY_CHECKS = 'safety_checks'
"""The `provider_details` key under which a model adapter keeps a call's safety checks, as `SafetyCheck` dicts."""

_SAFETY_CHECKS_TA = pydantic.TypeAdapter(list[SafetyCheck])

_COORDINATE_GRID = 'coordinate_grid'
"""The `provider_details` key under which the agent records the grid a call's positions are on."""

_SCREENSHOT_MAX_SIDE = 'screenshot_max_side'
"""The `provider_details` key under which the agent records the longest screenshot side the call's model takes."""

_SCREENSHOT_MAX_PIXELS = 'screenshot_max_pixels'
"""The `provider_details` key under which the agent records the most screenshot pixels the call's model takes."""


_KEYS_ADDED_BY_AGENT = frozenset({_COORDINATE_GRID, _SCREENSHOT_MAX_SIDE, _SCREENSHOT_MAX_PIXELS})
"""The `provider_details` keys the agent adds to a `computer` call after the model returned it, which the provider
never saw."""


def _int_at_least(value: object, minimum: int = 1) -> int | None:
    """`value` when it's an `int` (not a `bool`) of at least `minimum`, otherwise `None`."""
    return value if type(value) is int and value >= minimum else None


COMPUTER_KIND = ToolKind(
    'computer_use', 'computer', args=ComputerArgs, content=ComputerContent, metadata=ComputerResult
)
"""The kind of the `computer` tool, whose calls a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool] sends to a
provider's native computer tool."""


class ComputerCallPart(ToolCallPart[ComputerArgs], kind=COMPUTER_KIND):
    """A call of the `computer` tool."""

    @property
    def actions(self) -> list[ComputerToolAction] | None:
        """The actions the model asked for, or `None` if the arguments are incomplete (still streaming) or malformed."""
        typed = self.typed_args
        return typed['actions'] if typed is not None else None

    @property
    def safety_checks(self) -> list[SafetyCheck]:
        """The checks the provider asks a person to make before this call runs; empty when there are none."""
        checks = (self.provider_details or {}).get(_SAFETY_CHECKS)
        return _SAFETY_CHECKS_TA.validate_python(checks) if checks else []

    @property
    def shared_coordinate_grid(self) -> int | None:
        """The size of the grid the call's positions are on, such as `1000` for 0 to 999, or `None` for pixels.

        Pixels are of the latest screenshot before the call. The agent records the grid on the calls of a model whose
        profile sets `computer_use_coordinate_grid`; a history that lost the call's `provider_details`, such as one a
        UI rebuilt, reads `None`.
        """
        return _int_at_least((self.provider_details or {}).get(_COORDINATE_GRID), minimum=2)

    @property
    def screenshot_max_side(self) -> int | None:
        """The longest side, in pixels, of a screenshot the model that made the call takes, or `None` for no limit.

        The agent records it from the profile's `computer_use_screenshot_max_side`, so the screenshot the call returns
        can be taken at a size that model can see; a call without it, such as in a history a UI rebuilt, reads `None`.
        """
        return _int_at_least((self.provider_details or {}).get(_SCREENSHOT_MAX_SIDE))

    @property
    def screenshot_max_pixels(self) -> int | None:
        """The most pixels in a screenshot the model that made the call takes, or `None` for no limit.

        Recorded from the profile's `computer_use_screenshot_max_pixels`; see
        [`screenshot_max_side`][pydantic_ai.computer_use.ComputerCallPart.screenshot_max_side].
        """
        return _int_at_least((self.provider_details or {}).get(_SCREENSHOT_MAX_PIXELS))


class ComputerReturnPart(ToolReturnPart[ComputerContent, ComputerResult], kind=COMPUTER_KIND):
    """A successful result of the `computer` tool."""

    @property
    def result(self) -> ComputerResult | None:
        """The call's outcome, or `None` when the result has none, such as one a UI rebuilt without its metadata."""
        return self.typed_metadata

    @property
    def screenshot(self) -> BinaryImage | None:
        """The screen after the call, or `None` if the result doesn't end with one, such as once it was dropped."""
        content = self.typed_content
        last = content[-1] if content else None
        return last if isinstance(last, BinaryImage) else None
