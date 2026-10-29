"""Tests for `ManagedDisplay`, the start-and-retry policy every harness display shares."""

from __future__ import annotations

import anyio
import pytest

from pydantic_ai.workspaces import (
    Click,
    ComputerAction,
    Point,
    PressKeys,
    Region,
    WorkspaceError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness._display import ManagedDisplay

from ..workspace_fakes import FAKE_SCREENSHOT

pytestmark = pytest.mark.anyio


class FakeDriver:
    """A display that counts its starts, can fail its next calls, and refuses one kind of action."""

    def __init__(self, refuse: type[ComputerAction] | None = None) -> None:
        self.starts = 0
        self.failures: list[Exception] = []
        """Errors the next calls raise, in order."""
        self.acted: list[ComputerAction] = []
        self._refuse = refuse

    async def start(self) -> FakeDriver:
        self.starts += 1
        await anyio.sleep(0)  # a start takes a moment, so concurrent first uses overlap
        return self

    def _maybe_fail(self) -> None:
        if self.failures:
            raise self.failures.pop(0)

    async def display_size(self) -> tuple[int, int]:
        self._maybe_fail()
        return (1280, 800)

    async def screenshot(self) -> bytes:
        self._maybe_fail()
        return FAKE_SCREENSHOT

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        self._maybe_fail()
        return FAKE_SCREENSHOT

    async def act(self, action: ComputerAction) -> None:
        self._maybe_fail()
        if self._refuse is not None and isinstance(action, self._refuse):
            raise WorkspaceUnsupportedActionError('the fake display refuses it')
        self.acted.append(action)

    async def zoom(self, region: Region) -> bytes:
        self._maybe_fail()
        return FAKE_SCREENSHOT

    async def cursor_position(self) -> Point:
        self._maybe_fail()
        return Point(x=0, y=0)


async def test_concurrent_first_uses_start_one_display() -> None:
    driver = FakeDriver()
    display = ManagedDisplay(driver.start)
    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(display.display_size)
    assert driver.starts == 1


async def test_a_failed_look_starts_the_display_again_and_looks_again() -> None:
    driver = FakeDriver()
    display = ManagedDisplay(driver.start)
    await display.display_size()

    driver.failures = [WorkspaceError('the display went away')]
    assert await display.screenshot() == FAKE_SCREENSHOT
    assert driver.starts == 2


async def test_a_failed_action_is_not_retried_and_the_next_call_starts_the_display() -> None:
    driver = FakeDriver()
    display = ManagedDisplay(driver.start)
    await display.display_size()

    driver.failures = [WorkspaceError('the display went away')]
    with pytest.raises(WorkspaceError, match='went away'):
        await display.act(Click(x=1, y=1))
    assert driver.acted == []  # not done twice

    await display.act(Click(x=2, y=2))
    assert (driver.starts, driver.acted) == (2, [Click(x=2, y=2)])


class Screen:
    """An environment's X server as the start script sees it: up or down."""

    def __init__(self, *, up: bool = False) -> None:
        self.up = up
        self.driver = FakeDriver()

    def display(self) -> ManagedDisplay:
        """A display object on this screen, as a backend built for one durable unit has."""

        async def start() -> FakeDriver:
            if not self.up:
                self.up = True
                display.started_fresh()
            return await self.driver.start()

        display = ManagedDisplay(start)
        return display


async def test_an_action_does_not_run_on_a_display_that_started_for_it() -> None:
    """A backend rebuilt for one action finds the display down: the screen the action was meant for is gone."""
    screen = Screen()
    display = screen.display()
    with pytest.raises(WorkspaceError, match='The display was not running and started for this action'):
        await display.act(Click(x=1, y=1))
    assert screen.driver.acted == []
    await display.act(Click(x=2, y=2))
    # The next action uses the display that started, without starting it again.
    assert (screen.driver.acted, screen.driver.starts) == ([Click(x=2, y=2)], 1)


async def test_an_action_runs_on_a_display_that_was_already_up() -> None:
    screen = Screen(up=True)
    await screen.display().act(Click(x=1, y=1))
    assert screen.driver.acted == [Click(x=1, y=1)]


async def test_a_refused_action_keeps_the_display() -> None:
    driver = FakeDriver(refuse=PressKeys)
    display = ManagedDisplay(driver.start)
    with pytest.raises(WorkspaceUnsupportedActionError):
        await display.act(PressKeys(keys=('a',)))
    await display.act(Click(x=1, y=1))
    assert driver.starts == 1


@pytest.mark.parametrize(
    'error', [WorkspaceUnavailableError('the sandbox is gone'), WorkspaceTimeoutError('the display is stuck')]
)
async def test_an_unavailable_or_stuck_environment_is_not_retried(error: WorkspaceError) -> None:
    driver = FakeDriver()
    display = ManagedDisplay(driver.start)
    await display.display_size()

    driver.failures = [error]
    with pytest.raises(type(error)):
        await display.screenshot()
    assert driver.starts == 1
