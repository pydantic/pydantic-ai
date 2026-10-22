"""`ComputerUse`: lets an agent see and control the display of its workspace."""

from __future__ import annotations as _annotations

import struct
from dataclasses import KW_ONLY, dataclass, field
from typing import Any

import anyio

from pydantic_ai._computer_use import COMPUTER_KIND, ComputerResult, ComputerToolAction, Screenshot, Wait
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import WorkspaceError, WorkspaceUnavailableError

from .abstract import AbstractCapability

__all__ = ('ComputerUse',)

_DESCRIPTION = """\
See and control the computer's screen with the mouse and keyboard.

Performs the actions in order, stops at the first one that fails, then returns a screenshot of the whole \
screen. Coordinates are pixels of that screenshot, with the origin at the top left. Start with a \
`screenshot` action to see the screen, and check the screenshot after each step before continuing."""


async def computer(ctx: RunContext[Any], actions: list[ComputerToolAction]) -> ToolReturn:
    executed = 0
    error: str | None = None
    for action in actions:
        try:
            if isinstance(action, Wait):
                await anyio.sleep(action.seconds)
            elif not isinstance(action, Screenshot):
                await ctx.workspace.act(action)
        except WorkspaceUnavailableError:
            raise
        except WorkspaceError as e:
            error = f'{action.kind} failed: {e}'
            break
        executed += 1
    try:
        screenshot = await ctx.workspace.screenshot()
    except WorkspaceUnavailableError:
        raise
    except WorkspaceError as e:
        raise ToolFailed(f'Taking the screenshot failed: {e}') from e
    # A PNG's header chunk starts with its width and height.
    width, height = struct.unpack('>II', screenshot[16:24])
    result = ComputerResult(executed=executed, screen_size=[width, height])
    if error is not None:
        result['error'] = error
    return ToolReturn([result, BinaryImage(data=screenshot, media_type='image/png')])


@dataclass
class ComputerUse(AbstractCapability[AgentDepsT]):
    """Lets the agent see and control the display of its workspace with a `computer` tool.

    The workspace must have a display: a sandbox capability started with a desktop, or any backend that
    implements [`SupportsComputerUse`][pydantic_ai.workspaces.SupportsComputerUse]. The model must accept
    images, since every call returns a screenshot.
    """

    _: KW_ONLY

    id: str | None = 'computer'
    """One-off: an agent has a single display to control, so the id is fixed by default."""

    _toolset: FunctionToolset[AgentDepsT] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        tool = Tool[AgentDepsT](
            computer, name='computer', description=_DESCRIPTION, sequential=True, tool_kind=COMPUTER_KIND
        )
        self._toolset = FunctionToolset([tool], id=self.id)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when it has no workspace or its model can't see screenshots."""
        if not ctx.workspace.attached or not ctx.workspace.has_display:
            raise UserError(
                '`ComputerUse` needs a workspace with a display. Attach a sandbox capability that starts a '
                'desktop, or a capability whose workspace backend implements `SupportsComputerUse`.'
            )
        model = ctx.model
        if isinstance(model, Model) and not model.profile.get('supports_image_input', True):
            raise UserError(
                f'`ComputerUse` needs a model that accepts images, and {model.model_name!r} does not: '
                'every `computer` call returns a screenshot.'
            )

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        return self._toolset
