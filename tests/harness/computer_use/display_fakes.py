"""Fake workspaces with a screen, for `ComputerUse` tests.

They build on core's `InMemoryProvider`, whose environments a backend built anywhere can reattach to, and add the
harness display protocols to its backends.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from typing import Protocol, Self, runtime_checkable

from pydantic_ai.computer_use import ComputerReturnPart, MouseMove, Point, Region
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.computer_use import (
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsDisplay,
    SupportsZoom,
    UnsupportedActionError,
)

from ...computer_fakes import FAKE_DISPLAY_SIZE, FAKE_SCREENSHOT, png
from ...workspace_fakes import FilesystemOnlyWorkspaceBackend, InMemoryProvider, ProviderBackend


class ScreenProvider(InMemoryProvider):
    """An `InMemoryProvider` whose environments have a screen, recording every action performed on it."""

    def __init__(self, name: str = 'fake', *, in_unit: Callable[[], bool] | None = None) -> None:
        super().__init__(name, in_unit=in_unit)
        self.actions: list[DisplayAction] = []
        # Set to lose the worker just after the next action takes effect, on whichever backend serves it.
        self.lose_next_action = False

    def reset(self) -> None:
        super().reset()
        self.actions.clear()
        self.lose_next_action = False

    def backend(self, ref: WorkspaceRef | None) -> ScreenBackend:
        return ScreenBackend(self, ref)


class ScreenBackend(
    ProviderBackend,
    SupportsDisplay,
    SupportsZoom,
    SupportsCursorPosition,
):
    """A provider backend that is its own display, with every optional ability."""

    def __init__(self, provider: ScreenProvider, ref: WorkspaceRef | None) -> None:
        super().__init__(provider, ref)
        self.screens = provider

    @property
    def display(self) -> Self:
        return self

    async def display_size(self) -> tuple[int, int]:
        await self._files()
        return FAKE_DISPLAY_SIZE

    async def screenshot(self) -> bytes:
        await self._files()
        return FAKE_SCREENSHOT

    async def act(self, action: DisplayAction) -> None:
        await self._files()
        # Everything but `hold_keys`, so tests can see an action kind refused.
        if action.type not in {
            'click',
            'mouse_move',
            'drag',
            'mouse_down',
            'mouse_up',
            'scroll',
            'type_text',
            'press_keys',
        }:
            raise UnsupportedActionError(f'the fake display cannot perform {action.type!r}')
        self.screens.actions.append(action)
        if self.screens.lose_next_action:
            self.screens.lose_next_action = False
            raise RuntimeError('the worker was lost')

    async def zoom(self, region: Region) -> bytes:
        await self._files()
        return png(region.right - region.left, region.bottom - region.top)

    async def cursor_position(self) -> Point:
        await self._files()
        moves = [action for action in self.screens.actions if isinstance(action, MouseMove)]
        return Point(x=moves[-1].x, y=moves[-1].y) if moves else Point(x=640, y=400)


class DisplayOnlyBackend(FilesystemOnlyWorkspaceBackend, SupportsDisplay):
    """A screen backend's files and display, without zoom or the pointer position."""

    def __init__(self, inner: ScreenBackend) -> None:
        super().__init__(inner)
        self._display = inner

    @property
    def display(self) -> Self:
        return self

    async def display_size(self) -> tuple[int, int]:
        return await self._display.display_size()

    async def screenshot(self) -> bytes:
        return await self._display.screenshot()

    async def act(self, action: DisplayAction) -> None:
        await self._display.act(action)


@runtime_checkable
class FullDisplay(Display, SupportsZoom, SupportsCursorPosition, Protocol):
    """A display with every optional ability, as every harness sandbox's is."""


def full_display(backend: WorkspaceBackend) -> FullDisplay:
    """A sandbox backend's display, checked to have every optional ability."""
    assert isinstance(backend, SupportsDisplay)
    display = backend.display
    assert isinstance(display, FullDisplay)
    return display


def look_click_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A `FunctionModel` function: takes a screenshot, clicks on what it showed, then answers."""
    steps = [[{'type': 'screenshot'}], [{'type': 'click', 'x': 3, 'y': 4}]]
    done = sum(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts)
    if done == len(steps):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': steps[done]})])
