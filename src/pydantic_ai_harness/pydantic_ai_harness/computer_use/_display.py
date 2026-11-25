"""`LazyDisplay`, which starts a backend's display on first use and again after a failure, plus shared helpers."""

from __future__ import annotations as _annotations

import base64
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Protocol, TypeVar

import anyio

from pydantic_ai.computer_use import Point, Region
from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use._png import PNG_SIGNATURE
from pydantic_ai_harness.computer_use._protocol import (
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsZoom,
    UnsupportedActionError,
)

T = TypeVar('T')


def checked_desktop_size(desktop: bool | tuple[int, int]) -> tuple[int, int] | None:
    """The display size `desktop=` asks for, or `None` for no display, refusing a size that isn't one."""
    if desktop is True:
        return (1280, 800)
    if not desktop:
        return None
    if len(desktop) != 2 or min(desktop) < 1:
        raise UserError(f'`desktop` must be a `(width, height)` of at least 1 pixel each, got {desktop!r}.')
    return desktop


RELEASE_TIMEOUT = 5
"""Maximum seconds a desktop driver spends releasing held keys and buttons after a failed or cancelled action. The
slowest release measured took under a second."""

DEFAULT_DESKTOP_BASE_IMAGE = 'debian:bookworm-slim'
"""The image a new sandbox with a desktop and no `image` runs, where the backend's own default has no `apt-get` to
install the display's programs with."""


def png_from_base64(encoded: str | None, source: str) -> bytes:
    """The PNG a display sent as base64, or a `WorkspaceError` naming `source` when it isn't one."""
    try:
        data = base64.b64decode(''.join((encoded or '').split()), validate=True)
    except ValueError:
        data = b''
    if not data.startswith(PNG_SIGNATURE):
        raise WorkspaceError(f'{source} did not return a PNG screenshot.')
    return data


class DisplayDriver(Display, SupportsZoom, SupportsCursorPosition, Protocol):
    """Controls one running display, such as an X server through commands, or a provider's API."""


class LazyDisplay:
    """The display a backend hands out for its whole life. It starts on first use, and again after a failure.

    `operation` wraps every call, including the start, for example to turn a provider's errors into workspace errors.
    The backend also calls `start` before each shell command, so a program the command opens shows on the screen.

    A call that only looks, such as a screenshot, is retried once after a failure, and the retry restarts a lost
    display. An action is never retried, so nothing is done twice.

    It implements [`Display`][pydantic_ai_harness.computer_use.Display], `SupportsZoom` and `SupportsCursorPosition`.
    """

    def __init__(
        self,
        driver: DisplayDriver,
        start_display: Callable[[], Awaitable[None]],
        *,
        operation: Callable[[], AbstractAsyncContextManager[object]] = nullcontext,
    ):
        self._driver = driver
        self._start_display = start_display
        self._operation = operation
        self._running = False
        self._start_lock = anyio.Lock()

    def mark_stopped(self) -> None:
        """Forget that the display runs, so the next call starts it, such as after the environment restarted."""
        self._running = False

    async def start(self) -> None:
        """Start the display, unless it is known to be running."""
        async with self._operation():
            await self._start_unless_running()

    async def _start_unless_running(self) -> None:
        # One start at a time: two would both find no display and start two.
        async with self._start_lock:
            if not self._running:
                await self._start_display()
                self._running = True

    async def _run_on_driver(self, call: Callable[[DisplayDriver], Awaitable[T]], *, retry_once: bool) -> T:
        """Start the display unless it runs, then make `call`. With `retry_once`, a failed call is made once more."""
        may_retry = retry_once
        while True:
            try:
                async with self._operation():
                    await self._start_unless_running()
                    return await call(self._driver)
            except UnsupportedActionError:
                raise  # a refused action, such as an unknown key: the display is fine
            except Exception as error:
                # The display may have stopped or the environment restarted, so the next call starts it again. Starting
                # one that still runs does nothing.
                self._running = False
                # Retrying can't help an unavailable environment, and would double a timeout.
                if isinstance(error, WorkspaceUnavailableError | WorkspaceTimeoutError):
                    raise
                if not may_retry:
                    raise
                may_retry = False

    async def display_size(self) -> tuple[int, int]:
        return await self._run_on_driver(lambda driver: driver.display_size(), retry_once=True)

    async def screenshot(self) -> bytes:
        return await self._run_on_driver(lambda driver: driver.screenshot(), retry_once=True)

    async def act(self, action: DisplayAction) -> None:
        await self._run_on_driver(lambda driver: driver.act(action), retry_once=False)

    async def zoom(self, region: Region) -> bytes:
        return await self._run_on_driver(lambda driver: driver.zoom(region), retry_once=True)

    async def cursor_position(self) -> Point:
        return await self._run_on_driver(lambda driver: driver.cursor_position(), retry_once=True)


class LiveViewDisplay(LazyDisplay):
    """A `LazyDisplay` that also implements `SupportsLiveView`, for a backend whose provider can show its screen.

    `live_view` makes the URL. It runs inside `operation`, after the display has started, so the page shows a screen.
    """

    def __init__(
        self,
        driver: DisplayDriver,
        start_display: Callable[[], Awaitable[None]],
        *,
        live_view: Callable[[], Awaitable[str]],
        operation: Callable[[], AbstractAsyncContextManager[object]] = nullcontext,
    ):
        super().__init__(driver, start_display, operation=operation)
        self._live_view = live_view

    async def live_view_url(self) -> str:
        try:
            async with self._operation():
                await self._start_unless_running()
                return await self._live_view()
        except Exception:
            # As in `_run_on_driver`: the display may have stopped, so the next call starts it again.
            self._running = False
            raise
