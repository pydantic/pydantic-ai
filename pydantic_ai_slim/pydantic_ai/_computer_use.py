"""The `computer` tool's kind and shapes, shared by the `ComputerUse` capability and the model adapters."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Annotated, Literal, NotRequired, TypeAlias

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from .messages import BinaryImage, ToolKind
from .workspaces import Click, Drag, HoldKeys, MouseDown, MouseUp, MoveMouse, PressKeys, Scroll, TypeText
from .workspaces.computer import _CONFIG  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from .models import ModelRequestParameters
    from .tools import ToolDefinition

__all__ = (
    'COMPUTER_KIND',
    'MAX_ACTIONS_PER_CALL',
    'OBSERVING_ACTIONS',
    'ComputerArgs',
    'ComputerContent',
    'ComputerResult',
    'ComputerToolAction',
    'Screenshot',
    'Wait',
    'computer_tool',
)

MAX_ACTIONS_PER_CALL = 20


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

OBSERVING_ACTIONS = frozenset({'wait', 'screenshot'})
"""Action types that change nothing on the screen, so `require_approval` lets them through."""


class ComputerArgs(TypedDict):
    """The `computer` tool's arguments."""

    actions: list[ComputerToolAction]


class ComputerResult(TypedDict):
    """A `computer` call's outcome, kept in the tool return's `metadata` for model adapters and UIs."""

    performed: int
    """How many actions ran: all of them, or those before the one that failed."""

    error: NotRequired[str]
    """Why the action after the performed ones failed; the rest did not run."""

    screenshot_size: NotRequired[tuple[int, int]]
    """The closing screenshot's width and height, absent when it failed."""


ComputerContent: TypeAlias = list[str | BinaryImage]
"""What the model sees from a `computer` call: a short summary, then the screenshot."""

COMPUTER_KIND = ToolKind._core('computer', args=ComputerArgs, content=ComputerContent)  # pyright: ignore[reportPrivateUsage]
"""The kind of the `computer` tool, which model adapters send as their provider's native computer tool."""


def computer_tool(model_request_parameters: ModelRequestParameters) -> ToolDefinition | None:
    """The `computer` tool among a request's tools, which a model adapter may send as its provider's native tool."""
    return next((t for t in model_request_parameters.function_tools if t.tool_kind == COMPUTER_KIND.id), None)
