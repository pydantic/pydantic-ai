"""A fake `computer` tool for testing model adapters without a screen. It returns results in the shape Pydantic AI Harness's tool does."""

from __future__ import annotations as _annotations

import struct
import zlib
from typing import Any

from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.computer_use import COMPUTER_KIND, ComputerResult, ComputerToolAction, Screenshot, Wait
from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.native_tools import ComputerTool
from pydantic_ai.toolsets import FunctionToolset


def png(
    width: int, height: int, *, background: bytes = b'\x00\x00\x00', red_box: tuple[int, int, int, int] | None = None
) -> bytes:
    """A complete RGB PNG of the given size in `background`, with an optional red `(left, top, right, bottom)` box."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data))

    left, top, right, bottom = red_box or (0, 0, 0, 0)
    plain = b'\x00' + background * width
    marked = b'\x00' + background * left + b'\xff\x00\x00' * (right - left) + background * (width - right)
    rows = b''.join(marked if top <= y < bottom else plain for y in range(height))
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


FAKE_DISPLAY_SIZE = (1280, 800)
FAKE_SCREENSHOT = png(*FAKE_DISPLAY_SIZE)
"""The screenshot the fake `computer` tool returns."""

_LOOKING = (Wait, Screenshot)


class FakeComputer(AbstractCapability[Any]):
    """A `computer` tool that records the actions it runs.

    With `fail_acting=True`, the first action that isn't a screenshot or a wait fails, which gives tests a failed call.
    With `native=True`, it also adds a `ComputerTool`, as `ComputerUse` does.
    """

    def __init__(self, *, requires_approval: bool = False, native: bool = True, fail_acting: bool = False) -> None:
        self.native = native
        self.fail_acting = fail_acting
        self.actions: list[ComputerToolAction] = []
        tool = Tool(self._computer, name='computer', tool_kind=COMPUTER_KIND, requires_approval=requires_approval)
        self._toolset = FunctionToolset[Any]([tool])

    async def _computer(self, actions: list[ComputerToolAction]) -> ToolReturn:
        result = ComputerResult(performed=0)
        for action in actions:
            if not isinstance(action, _LOOKING):
                if self.fail_acting:
                    result['error'] = 'The action failed.'
                    break
                self.actions.append(action)
            result['performed'] += 1
        summary = f'Ran {result["performed"]} of {len(actions)} actions.'
        if 'error' in result:
            summary += f' The next one failed: {result["error"]}'
        return ToolReturn([summary, BinaryImage(data=FAKE_SCREENSHOT, media_type='image/png')], metadata=result)

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def get_native_tools(self) -> list[ComputerTool]:
        return [ComputerTool(display_size=FAKE_DISPLAY_SIZE, optional=True)] if self.native else []
