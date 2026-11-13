"""A fake `computer` tool for testing model adapters, without driving a screen.

It keeps the tool's contract: it takes `ComputerToolAction`s, records the ones that act, and returns a summary and
a screenshot with the outcome in the result's metadata, the way Pydantic AI Harness's `ComputerUse` does.
"""

from __future__ import annotations as _annotations

import struct
import zlib
from typing import Any

from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.computer_use import (
    COMPUTER_KIND,
    ComputerResult,
    ComputerToolAction,
    CursorPosition,
    HoldKeys,
    MoveMouse,
    Screenshot,
    Wait,
    Zoom,
)
from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.native_tools import ComputerTool
from pydantic_ai.toolsets import FunctionToolset


def png(width: int, height: int) -> bytes:
    """A complete black RGB PNG of the given size."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))

    rows = b''.join(b'\x00' + bytes(width * 3) for _ in range(height))
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


FAKE_DISPLAY_SIZE = (1280, 800)
FAKE_SCREENSHOT = png(*FAKE_DISPLAY_SIZE)
"""The screenshot the fake `computer` tool returns."""

_LOOKING = (Wait, Screenshot)


class FakeComputer(AbstractCapability[Any]):
    """A `computer` tool that records the actions it runs, refusing to act on a read-only workspace.

    It refuses `hold_key`, so tests can see an action refused, and returns what zoom and cursor-position actions
    read before its screenshot. With `native`, it also adds a `ComputerTool` declaring `display_size`, as
    `ComputerUse` does.
    """

    def __init__(
        self,
        *,
        requires_approval: bool = False,
        native: bool = True,
        display_size: tuple[int, int] = FAKE_DISPLAY_SIZE,
    ) -> None:
        self.native = native
        self.display_size = display_size
        self.actions: list[ComputerToolAction] = []
        tool = Tool(self._computer, name='computer', tool_kind=COMPUTER_KIND, requires_approval=requires_approval)
        self._toolset = FunctionToolset[Any]([tool])

    async def _computer(self, ctx: RunContext[Any], actions: list[ComputerToolAction]) -> ToolReturn:
        result = ComputerResult(performed=0)
        outputs: list[str | BinaryImage] = []
        for action in actions:
            if isinstance(action, Zoom):
                region = action.region
                zoomed = png(region.right - region.left, region.bottom - region.top)
                outputs.append(BinaryImage(data=zoomed, media_type='image/png'))
            elif isinstance(action, CursorPosition):
                moves = [done for done in self.actions if isinstance(done, MoveMouse)]
                x, y = (moves[-1].x, moves[-1].y) if moves else (640, 400)
                outputs.append(f'The pointer is at ({x}, {y}).')
            elif not isinstance(action, _LOOKING):
                if ctx.workspace.read_only:
                    result['error'] = 'This workspace is read-only.'
                    break
                if isinstance(action, HoldKeys):
                    result['error'] = "the fake display cannot perform 'hold_key'"
                    break
                self.actions.append(action)
            result['performed'] += 1
        summary = f'Ran {result["performed"]} of {len(actions)} actions.'
        if 'error' in result:
            summary += f' The next one failed: {result["error"]}'
        screenshot = BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')
        return ToolReturn([summary, *outputs, screenshot], metadata=result)

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def get_native_tools(self) -> list[ComputerTool]:
        return [ComputerTool(display_size=self.display_size, optional=True)] if self.native else []
