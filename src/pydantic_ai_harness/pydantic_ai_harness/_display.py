"""`ManagedDisplay`, a backend's display started on first use and again after a failure, and helpers the displays share."""

from __future__ import annotations as _annotations

import base64
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Protocol, TypeVar

import anyio

from pydantic_ai.workspaces import (
    ComputerAction,
    Point,
    Region,
    WorkspaceError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)

__all__ = ('DisplayDriver', 'ManagedDisplay')

T = TypeVar('T')

DEFAULT_DESKTOP_SIZE = (1280, 800)
"""The display size `desktop=True` starts."""

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def desktop_size(desktop: bool | tuple[int, int]) -> tuple[int, int] | None:
    """The display size `desktop=` asks for, or `None` for no display."""
    return DEFAULT_DESKTOP_SIZE if desktop is True else desktop or None


def png_size(png: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's header chunk, which the signature puts at bytes 16 to 24."""
    if len(png) < 24 or not png.startswith(PNG_SIGNATURE) or png[12:16] != b'IHDR':
        return None
    return int.from_bytes(png[16:20], 'big'), int.from_bytes(png[20:24], 'big')


def png_from_base64(encoded: str | None, source: str) -> bytes:
    """The PNG a display sent as base64, or a `WorkspaceError` naming `source` when it isn't one."""
    try:
        data = base64.b64decode(''.join((encoded or '').split()), validate=True)
    except ValueError:
        data = b''
    if not data.startswith(PNG_SIGNATURE):
        raise WorkspaceError(f'{source} did not return a PNG screenshot.')
    return data


class DisplayDriver(Protocol):
    """Drives one running display, such as an X server through commands or a provider's API."""

    async def display_size(self) -> tuple[int, int]: ...
    async def screenshot(self) -> bytes: ...
    async def scaled_screenshot(self, width: int, height: int) -> bytes: ...
    async def act(self, action: ComputerAction) -> None: ...
    async def zoom(self, region: Region) -> bytes: ...
    async def cursor_position(self) -> Point: ...


class ManagedDisplay:
    """The display a backend hands out for its lifetime: started on first use, and again after a failure.

    `start` brings the display up, doing nothing to one that's running, and returns its driver. `scope` wraps every
    call, start included, such as to map a provider's errors.

    A call that only looks, such as a screenshot, is retried once after a failure, which starts a display that was
    lost; an action isn't, so nothing is done twice. The backend calls `started_fresh` when `start` launched a new
    display, and an action whose own call launched one refuses to run: the windows it was meant for are gone.

    `screen_key` names the screen it shows, so `ComputerUse` runs in one process on one screen take turns at it,
    whichever workspace they reach it through, such as a sandbox's ref.
    """

    def __init__(
        self,
        start: Callable[[], Awaitable[DisplayDriver]],
        *,
        scope: Callable[[], AbstractAsyncContextManager[object]] = nullcontext,
        screen_key: Callable[[], object | None] = lambda: None,
    ):
        self._start = start
        self._scope = scope
        self._screen_key = screen_key
        self._driver: DisplayDriver | None = None
        self._lock = anyio.Lock()
        self._starts = 0

    @property
    def screen_key(self) -> object | None:
        """A hashable name for the screen this display shows, or `None` until it's known."""
        return self._screen_key()

    def started_fresh(self) -> None:
        """Record that `start` launched a new display, without the windows that were open before."""
        self._starts += 1

    def reset(self) -> None:
        """Forget the running display, such as when the environment may have restarted: the next call starts it."""
        self._driver = None

    async def driver(self) -> DisplayDriver:
        """The running display's driver, starting the display unless it's running."""
        # One start at a time: two would both find no display and start two.
        async with self._lock:
            if self._driver is None:
                self._driver = await self._start()
            return self._driver

    async def _call(self, call: Callable[[DisplayDriver], Awaitable[T]], *, observe: bool) -> T:
        retry = observe
        while True:
            driver = None
            try:
                async with self._scope():
                    driver = await self.driver()
                    result = await call(driver)
            except WorkspaceUnsupportedActionError:
                raise  # a refused action, such as an unknown key: the display is fine
            except Exception as error:
                # The display may have stopped, or the environment restarted: the next call starts it again.
                if driver is not None and self._driver is driver:
                    self._driver = None
                # Retrying can't help an unavailable environment, and would double a timeout.
                if isinstance(error, WorkspaceUnavailableError | WorkspaceTimeoutError):
                    raise
                if not retry:
                    raise
                retry = False
            else:
                return result

    async def display_size(self) -> tuple[int, int]:
        return await self._call(lambda driver: driver.display_size(), observe=True)

    async def screenshot(self) -> bytes:
        return await self._call(lambda driver: driver.screenshot(), observe=True)

    async def scaled_screenshot(self, width: int, height: int) -> bytes:
        """The whole display scaled to `width` by `height`, for a model that can't take it at full size.

        This is [`SupportsScaledScreenshot`][pydantic_ai.workspaces.SupportsScaledScreenshot].
        """
        return await self._call(lambda driver: driver.scaled_screenshot(width, height), observe=True)

    async def act(self, action: ComputerAction) -> None:
        starts = self._starts

        async def act(driver: DisplayDriver) -> bool:
            if self._starts != starts:
                return False
            await driver.act(action)
            return True

        if not await self._call(act, observe=False):
            # The display was down and started for this action: the windows it was meant for are gone. Refused outside
            # the call, so the display that just started stays and the next action doesn't start it again.
            raise WorkspaceError('The display was not running and started for this action, so it did not run.')

    async def zoom(self, region: Region) -> bytes:
        return await self._call(lambda driver: driver.zoom(region), observe=True)

    async def cursor_position(self) -> Point:
        return await self._call(lambda driver: driver.cursor_position(), observe=True)
