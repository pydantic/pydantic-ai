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
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Annotated, Literal, NotRequired, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .messages import BinaryImage, ToolKind
from .workspaces import Click, Drag, HoldKeys, MouseDown, MouseUp, MoveMouse, PressKeys, Region, Scroll, TypeText
from .workspaces.computer import ACTION_CONFIG

if TYPE_CHECKING:
    from .models import ModelRequestParameters

__all__ = (
    'COMPUTER_KIND',
    'DISPLAY_SIZE',
    'SAFETY_CHECKS',
    'ComputerResult',
    'ComputerToolAction',
    'CursorPosition',
    'Screenshot',
    'Wait',
    'Zoom',
    'computer_tool_name',
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


class _ComputerArgs(TypedDict):
    """The `computer` tool's arguments."""

    actions: list[ComputerToolAction]


class ComputerResult(TypedDict):
    """A `computer` call's outcome, kept in the tool return's `metadata` for model adapters and UIs."""

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


_ComputerContent: TypeAlias = list[str | BinaryImage]
"""What the model sees from a `computer` call; see the module docstring for its order."""

SAFETY_CHECKS = 'safety_checks'
"""The `provider_details` key under which a model adapter puts the checks a provider asks a person to approve.

Its value is a list of dicts, each with at least a `message` for the person; other keys are the provider's own, kept
to send back when the call is approved.
"""

DISPLAY_SIZE = 'display_size'
"""The `ToolDefinition.metadata` key under which the `computer` tool gives the `(width, height)` of the screenshots
the model sees, which may be smaller than the display.

A provider whose native tool declares a display size, such as Claude's `computer_20251124`, sends this one.
"""

COMPUTER_KIND = ToolKind._core('computer', args=_ComputerArgs, content=_ComputerContent)  # pyright: ignore[reportPrivateUsage]
"""The kind of the `computer` tool, which model adapters send as their provider's native computer tool."""


def computer_tool_name(model_request_parameters: ModelRequestParameters) -> str | None:
    """The name of the request's `computer` tool, which a model adapter may send as its provider's native tool.

    `None` unless the tool is fully visible in this request: a deferred or withheld one stays a function tool.
    """
    tool = next((t for t in model_request_parameters.function_tools if t.tool_kind == COMPUTER_KIND.id), None)
    if tool is None or model_request_parameters.visibility_of(tool.name) != 'visible':
        return None
    return tool.name
