"""The `computer` tool: its kind and the shapes of its arguments and result.

A capability that gives an agent a screen, such as Pydantic AI Harness's `ComputerUse`, defines a tool of
[`COMPUTER_KIND`][pydantic_ai.computer_use.COMPUTER_KIND]. A model adapter whose provider has a native computer tool
sends that tool as the native one, translates the provider's calls into these arguments, and reads the result back
from them.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Literal, NotRequired, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .messages import BinaryImage, ToolKind
from .workspaces import Click, Drag, HoldKeys, MouseDown, MouseUp, MoveMouse, PressKeys, Scroll, TypeText
from .workspaces.computer import _CONFIG  # pyright: ignore[reportPrivateUsage]

__all__ = (
    'COMPUTER_KIND',
    'SAFETY_CHECKS',
    'ComputerResult',
    'ComputerToolAction',
    'Screenshot',
    'Wait',
)


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Wait:
    """Pause, for example while a page or application loads."""

    type: Literal['wait'] = 'wait'
    seconds: Annotated[float, Field(gt=0, le=10, allow_inf_nan=False)] = 1
    """How long to wait, up to 10 seconds."""


@dataclass(frozen=True, kw_only=True, config=_CONFIG)
class Screenshot:
    """Look at the screen without acting. Every call already ends with a screenshot."""

    type: Literal['screenshot'] = 'screenshot'


ComputerToolAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | PressKeys | HoldKeys | Wait | Screenshot,
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


_ComputerContent: TypeAlias = list[str | BinaryImage]
"""What the model sees from a `computer` call: a short summary, then the screenshot."""

SAFETY_CHECKS = 'safety_checks'
"""The `provider_details` key under which a model adapter puts the checks a provider asks a person to approve."""

COMPUTER_KIND = ToolKind._core('computer', args=_ComputerArgs, content=_ComputerContent)  # pyright: ignore[reportPrivateUsage]
"""The kind of the `computer` tool, which model adapters send as their provider's native computer tool."""
