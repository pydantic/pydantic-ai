"""`ManagedDisplay`, a backend's display started on first use and again after a failure, and helpers the displays share."""

from __future__ import annotations as _annotations

import base64
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

import anyio

from pydantic_ai.computer_use import Point, Region
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError, WorkspaceUnavailableError

from ._png import PNG_SIGNATURE
from .computer_use import DisplayAction, UnsupportedActionError

__all__ = ('DisplayDriver', 'ManagedDisplay')

T = TypeVar('T')

DEFAULT_DESKTOP_SIZE = (1280, 800)
"""The display size `desktop=True` starts."""


def desktop_size(desktop: bool | tuple[int, int]) -> tuple[int, int] | None:
    """The display size `desktop=` asks for, or `None` for no display."""
    return DEFAULT_DESKTOP_SIZE if desktop is True else desktop or None


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
    async def act(self, action: DisplayAction) -> None: ...
    async def zoom(self, region: Region) -> bytes: ...
    async def cursor_position(self) -> Point: ...


class ManagedDisplay:
    """The display a backend hands out for its lifetime: started on first use, and again after a failure.

    `start` brings the display up, doing nothing to one that's running, and returns its driver.

    A call that only looks, such as a screenshot, is retried once after a failure, which starts a display that was
    lost; an action isn't, so nothing is done twice. The backend calls `started_fresh` when `start` launched a new
    display. Until a screenshot shows the new display, an action refuses to run: the windows it was meant for are
    gone, whichever call launched it.

    It implements [`Display`][pydantic_ai_harness.computer_use.Display] and every optional ability.
    """

    def __init__(self, start: Callable[[], Awaitable[DisplayDriver]]):
        self._start = start
        self._driver: DisplayDriver | None = None
        self._lock = anyio.Lock()
        self._unseen = False
        """Whether a display launched since the last screenshot, so no one has seen what is on it."""

    def started_fresh(self) -> None:
        """Record that `start` launched a new display, without the windows that were open before."""
        self._unseen = True

    async def _look(self, call: Callable[[DisplayDriver], Awaitable[bytes]]) -> bytes:
        image = await self._call(call, observe=True)
        self._unseen = False
        return image

    async def _driver_or_start(self) -> DisplayDriver:
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
                driver = await self._driver_or_start()
                result = await call(driver)
            except UnsupportedActionError:
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
        return await self._look(lambda driver: driver.screenshot())

    async def act(self, action: DisplayAction) -> None:
        async def act(driver: DisplayDriver) -> bool:
            if self._unseen:
                return False
            await driver.act(action)
            return True

        if not await self._call(act, observe=False):
            # The display was down and started since the last screenshot, for this action or for an earlier look such
            # as reading its size: the windows the action was meant for are gone. Refused outside the call, so the
            # display that started stays and the next screenshot shows it.
            raise WorkspaceError(
                'The display was not running and started since the last screenshot, so the action did not run.'
            )

    async def zoom(self, region: Region) -> bytes:
        return await self._call(lambda driver: driver.zoom(region), observe=True)

    async def cursor_position(self) -> Point:
        return await self._call(lambda driver: driver.cursor_position(), observe=True)
