"""The `computer` tool: the actions a model can ask for, and the shapes of the tool's arguments and result.

A capability that gives an agent a screen adds a tool of kind
[`COMPUTER_KIND`][pydantic_ai.computer_use.COMPUTER_KIND] and performs its actions on that screen. For a provider
with a computer tool of its own, the model adapter sends that tool instead and converts its calls and results.

Positions (`x` and `y`) are pixels of the latest screenshot, from its top left corner. A click or scroll without a
position happens at the pointer.

The `type` field picks the action: the class name in snake case, such as `click` or `press_keys`. A key is a
[`NamedKey`][pydantic_ai.computer_use.NamedKey] or a single character, in any case. A character presses whatever keys
the keyboard layout needs to type it. An upper-case letter presses the same key as its lower-case one. A model adapter
turns its provider's key names, such as the X11 `Page_Down`, into these.

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

A call with a [`confirmation`][pydantic_ai.computer_use.ComputerCallPart.confirmation] must not run until a person
approves it. The runner enforces this, as `ComputerUse` does: a custom runner that skips the check runs flagged
actions without approval. A runner that ran it after a person approved it sets
[`confirmed`][pydantic_ai.computer_use.ComputerResult.confirmed] on the result. A model adapter tells the provider that
a person confirmed the call only when it is set, and otherwise says nobody did.

In the message history, the tool's calls and results become
[`ComputerCallPart`][pydantic_ai.computer_use.ComputerCallPart] and
[`ComputerReturnPart`][pydantic_ai.computer_use.ComputerReturnPart], whatever the tool's name.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Any, Literal, NotRequired, TypeAlias

from pydantic import BeforeValidator, ConfigDict, Field
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
    'Key',
    'Modifier',
    'MouseButton',
    'MouseDown',
    'MouseMove',
    'MouseUp',
    'NamedKey',
    'Point',
    'PressKeys',
    'Region',
    'Screenshot',
    'Scroll',
    'TypeText',
    'Wait',
    'Zoom',
)

MouseButton: TypeAlias = Literal['left', 'middle', 'right']
"""A mouse button."""

_Coordinate: TypeAlias = Annotated[int, Field(ge=0)]


def _mark_type_required(schema: dict[str, Any]) -> None:
    # `type` has a default so Python callers can omit it, but a model must always send it: it picks the action.
    if 'type' in schema.get('properties', {}):
        schema['required'] = ['type', *schema.get('required', [])]


_ACTION_CONFIG = ConfigDict(extra='forbid', use_attribute_docstrings=True, json_schema_extra=_mark_type_required)

Modifier: TypeAlias = Literal['ctrl', 'shift', 'alt', 'super']
"""A modifier key, held during a click, move, drag or scroll."""

NamedKey: TypeAlias = Literal[
    Modifier,
    'enter', 'escape', 'tab', 'backspace', 'delete', 'insert', 'home', 'end', 'pageup', 'pagedown',
    'up', 'down', 'left', 'right', 'space', 'printscreen', 'pause', 'menu', 'capslock', 'numlock', 'scrolllock',
    'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
]  # fmt: skip
"""A key with a name: a modifier, a navigation or editing key, or `f1` to `f12`."""

Key: TypeAlias = NamedKey | Annotated[str, Field(min_length=1, max_length=1)]
"""A named key, or a single character such as `a` or `+`."""


def _lowercase_key(key: object) -> object:
    # Models write names in any case, such as `Home` or `F5`. An upper-case letter presses the same key as its
    # lower-case one.
    return key.lower() if isinstance(key, str) else key


_Keys: TypeAlias = Annotated[tuple[Annotated[Key, BeforeValidator(_lowercase_key)], ...], Field(min_length=1)]
_Modifiers: TypeAlias = tuple[Annotated[Modifier, BeforeValidator(_lowercase_key)], ...]


def _check_x_and_y_given_together(x: int | None, y: int | None) -> None:
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
        _check_x_and_y_given_together(self.x, self.y)


@dataclass(frozen=True, kw_only=True, config=_ACTION_CONFIG)
class MouseMove:
    """Move the pointer, for example to reveal a hover menu."""

    type: Literal['mouse_move'] = 'mouse_move'
    x: _Coordinate
    """From the left edge of the screenshot."""
    y: _Coordinate
    """From the top edge of the screenshot."""


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
    amount: Annotated[int, Field(ge=1)]
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
        _check_x_and_y_given_together(self.x, self.y)


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

    actions_changed: NotRequired[bool]
    """`True` when the actions that ran differ from those the model asked for, for example because an approval
    changed them. A model adapter then sends the summary instead of reporting the model's actions as the ones that
    ran."""

    confirmed: NotRequired[Literal[True]]
    """Set by the runner when a person approved this call; providers that need an acknowledgement send one only when it
    is set."""


COMPUTER_KIND = ToolKind(
    'computer_use', 'computer', args=ComputerArgs, content=ComputerContent, metadata=ComputerResult
)
"""The tool kind that marks a tool as the `computer` tool.

When a request includes a [`ComputerTool`][pydantic_ai.native_tools.ComputerTool], the provider's own computer tool is
sent to the model in place of the tool of this kind, which still runs the calls.

A runner must set [`confirmed`][pydantic_ai.computer_use.ComputerResult.confirmed] on the result of a call with a
[`confirmation`][pydantic_ai.computer_use.ComputerCallPart.confirmation] that a person approved."""


class ComputerCallPart(ToolCallPart[ComputerArgs], kind=COMPUTER_KIND):
    """A call of the `computer` tool."""

    @property
    def actions(self) -> list[ComputerToolAction] | None:
        """The actions the model asked for, or `None` if the arguments are incomplete (still streaming) or malformed."""
        typed = self.typed_args
        return typed['actions'] if typed is not None else None

    @property
    def confirmation(self) -> str | None:
        """The provider's reason this call must be approved by a person before it runs, or `None`.

        A provider sets it on actions its safety checks flag, such as accepting a cookie banner or sending a message. A
        model adapter sets it by putting the reason, as a string, under the `'confirmation'` key of the call's
        `provider_details`.
        """
        reason = (self.provider_details or {}).get('confirmation')
        return reason if isinstance(reason, str) else None


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
