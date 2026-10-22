"""The `computer` tool's kind and shapes, shared by the `ComputerUse` capability and the model adapters."""

from __future__ import annotations as _annotations

from typing import Annotated, Literal, NotRequired, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .messages import BinaryImage, ToolKind
from .workspaces import Click, Drag, HoldKey, KeyPress, MouseDown, MouseUp, MoveMouse, Scroll, TypeText

__all__ = ('COMPUTER_KIND', 'ComputerArgs', 'ComputerContent', 'ComputerResult', 'ComputerToolAction', 'Screenshot', 'Wait')


@dataclass(frozen=True, kw_only=True)
class Wait:
    """Do nothing for a while, letting the screen settle."""

    seconds: Annotated[float, Field(ge=0, le=60, allow_inf_nan=False)] = 1.0
    kind: Literal['wait'] = 'wait'


@dataclass(frozen=True, kw_only=True)
class Screenshot:
    """Ask for a screenshot; the `computer` tool returns one after every call anyway."""

    kind: Literal['screenshot'] = 'screenshot'


ComputerToolAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | KeyPress | HoldKey | Wait | Screenshot,
    Field(discriminator='kind'),
]
"""What the `computer` tool accepts: the workspace's display actions, plus waiting and asking for a screenshot."""


class ComputerArgs(TypedDict):
    """The `computer` tool's arguments."""

    actions: list[ComputerToolAction]
    """The actions to perform, in order."""


class ComputerResult(TypedDict):
    """What the `computer` tool reports next to the screenshot it returns."""

    executed: int
    """How many of the actions were performed: all of them, or those before the one that failed."""

    error: NotRequired[str]
    """Why the action after the performed ones failed; the rest were not attempted."""

    screen_size: list[int]
    """The screenshot's `[width, height]` in pixels, the coordinate space of the actions."""


ComputerContent: TypeAlias = list[ComputerResult | BinaryImage]
"""What a `computer` call returns: the result, then the screenshot taken after the actions."""

COMPUTER_KIND = ToolKind._core('computer', args=ComputerArgs, content=ComputerContent)  # pyright: ignore[reportPrivateUsage]
"""The kind of the `computer` tool, which model adapters send as their provider's native computer tool."""
