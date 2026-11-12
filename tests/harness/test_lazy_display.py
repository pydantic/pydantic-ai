"""Tests for `LazyDisplay`, the start-and-retry policy every harness display shares.

It is private, but each backend reaches only some of its rules cheaply, so they are pinned here once.
"""

from __future__ import annotations

import anyio
import pytest

from pydantic_ai.computer_use import Click, Point, PressKeys, Region
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness._display import LazyDisplay
from pydantic_ai_harness.computer_use import (
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsZoom,
    UnsupportedActionError,
)

from .display_fakes import FAKE_SCREENSHOT


class FakeDriver:
    """A display that counts its starts, can fail its next calls, and refuses one kind of action."""

    def __init__(self, refuse: type[DisplayAction] | None = None) -> None:
        self.starts = 0
        self.failures: list[Exception] = []
        """Errors the next calls raise, in order."""
        self.acted: list[DisplayAction] = []
        self._refuse = refuse

    async def start(self) -> None:
        self.starts += 1
        await anyio.sleep(0)  # a start takes a moment, so concurrent first uses overlap

    def _maybe_fail(self) -> None:
        if self.failures:
            raise self.failures.pop(0)

    async def display_size(self) -> tuple[int, int]:
        self._maybe_fail()
        return (1280, 800)

    async def screenshot(self) -> bytes:
        self._maybe_fail()
        return FAKE_SCREENSHOT

    async def act(self, action: DisplayAction) -> None:
        self._maybe_fail()
        if self._refuse is not None and isinstance(action, self._refuse):
            raise UnsupportedActionError('the fake display refuses it')
        self.acted.append(action)

    async def zoom(self, region: Region) -> bytes:
        self._maybe_fail()
        return FAKE_SCREENSHOT

    async def cursor_position(self) -> Point:
        self._maybe_fail()
        return Point(x=0, y=0)


async def test_concurrent_first_uses_start_one_display() -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(display.display_size)
    assert driver.starts == 1


async def test_every_call_reaches_the_started_display() -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    assert await display.screenshot() == FAKE_SCREENSHOT
    assert await display.zoom(Region(left=0, top=0, right=10, bottom=10)) == FAKE_SCREENSHOT
    assert await display.cursor_position() == Point(x=0, y=0)
    assert driver.starts == 1


async def test_a_failed_look_starts_the_display_again_and_looks_again() -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    await display.display_size()

    driver.failures = [WorkspaceError('the display went away')]
    assert await display.screenshot() == FAKE_SCREENSHOT
    assert driver.starts == 2


async def test_a_failed_action_is_not_retried_and_the_next_call_starts_the_display() -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    await display.display_size()

    driver.failures = [WorkspaceError('the display went away')]
    with pytest.raises(WorkspaceError, match='went away'):
        await display.act(Click(x=1, y=1))
    assert driver.acted == []  # not done twice

    await display.act(Click(x=2, y=2))
    assert (driver.starts, driver.acted) == (2, [Click(x=2, y=2)])


def test_it_is_a_display_with_every_optional_ability() -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    assert all(isinstance(display, protocol) for protocol in (Display, SupportsZoom, SupportsCursorPosition))


async def test_a_refused_action_keeps_the_display() -> None:
    driver = FakeDriver(refuse=PressKeys)
    display = LazyDisplay(driver, driver.start)
    with pytest.raises(UnsupportedActionError):
        await display.act(PressKeys(keys=('a',)))
    await display.act(Click(x=1, y=1))
    assert driver.starts == 1


@pytest.mark.parametrize(
    'error', [WorkspaceUnavailableError('the sandbox is gone'), WorkspaceTimeoutError('the display is stuck')]
)
async def test_an_unavailable_or_stuck_environment_is_not_retried(error: WorkspaceError) -> None:
    driver = FakeDriver()
    display = LazyDisplay(driver, driver.start)
    await display.display_size()

    driver.failures = [error]
    with pytest.raises(type(error)):
        await display.screenshot()
    assert driver.starts == 1
