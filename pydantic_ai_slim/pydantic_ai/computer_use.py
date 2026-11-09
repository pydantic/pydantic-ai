"""The `computer` tool: its kind and the shapes of its arguments and result.

A capability that gives an agent a screen, such as Pydantic AI Harness's `ComputerUse`, defines a tool of
[`COMPUTER_KIND`][pydantic_ai.computer_use.COMPUTER_KIND]. A model adapter whose provider has a native computer tool
sends that tool as the native one, translates the provider's calls into these arguments, and reads the result back
from them.

Coordinates are pixels of the latest screenshot the model saw, or, when the model's profile sets
`computer_use_coordinate_grid`, points on that grid. New action types may be added, so a tool of this kind fails a
call with a type it doesn't know.

A result's content is text and images: a short summary, then what each action that returns something returned, in
action order, then the screenshot. The last image is always the screen after the call. The summary's wording isn't
stable; [`ComputerResult`][pydantic_ai.computer_use.ComputerResult] in the result's `metadata` is.

The tool's parts are promoted to [`ComputerCallPart`][pydantic_ai.computer_use.ComputerCallPart] and
[`ComputerReturnPart`][pydantic_ai.computer_use.ComputerReturnPart], so code reads them with `isinstance` and typed
properties, whatever the tool is called.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Literal, NotRequired, TypeAlias

import pydantic
from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .messages import BinaryImage, ToolCallPart, ToolKind, ToolReturnPart
from .workspaces import Click, Drag, HoldKeys, MouseDown, MouseUp, MoveMouse, PressKeys, Region, Scroll, TypeText
from .workspaces.computer import ACTION_CONFIG

__all__ = (
    'COMPUTER_KIND',
    'ComputerArgs',
    'ComputerCallPart',
    'ComputerContent',
    'ComputerResult',
    'ComputerReturnPart',
    'ComputerToolAction',
    'CursorPosition',
    'SafetyCheck',
    'Screenshot',
    'Wait',
    'Zoom',
)


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Wait:
    """Pause, for example while a page or application loads."""

    type: Literal['wait'] = 'wait'
    seconds: Annotated[float, Field(gt=0, le=30)] = 1
    """How long to wait, up to 30 seconds."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Screenshot:
    """Look at the screen without acting. A call's result already ends with a screenshot."""

    type: Literal['screenshot'] = 'screenshot'


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
class Zoom:
    """Look closely at part of the screen, to read small text or find a small target."""

    type: Literal['zoom'] = 'zoom'
    region: Region
    """The part of the screen to look at. Positions in later actions are still on the full screen."""


@dataclass(frozen=True, kw_only=True, config=ACTION_CONFIG)
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
"""What the `computer` tool accepts: the workspace's display actions, plus waiting and looking."""


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

    display_restarted: NotRequired[bool]
    """Set when the display restarted during the call, such as after its process stopped, so what was on the screen
    before may be gone. The model isn't told; the screenshot shows the screen as it is now."""


@dataclass(frozen=True, kw_only=True)
class SafetyCheck:
    """A check a provider asks a person to make before a `computer` call runs, such as OpenAI's `pending_safety_checks`.

    A call with safety checks waits for approval; see
    [`ComputerCallPart.safety_checks`][pydantic_ai.computer_use.ComputerCallPart.safety_checks].
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
