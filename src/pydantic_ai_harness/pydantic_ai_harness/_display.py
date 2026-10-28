"""`ManagedDisplay`: a backend's display, started on first use and again after a failure that may have lost it."""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from typing import Protocol, TypeVar

import anyio

from pydantic_ai.workspaces import (
    ComputerAction,
    Point,
    Region,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)

__all__ = ('DisplayDriver', 'ManagedDisplay')

T = TypeVar('T')


class DisplayDriver(Protocol):
    """Drives one running display, such as an X server through commands or a provider's API."""

    async def display_size(self) -> tuple[int, int]: ...
    async def screenshot(self) -> bytes: ...
    async def act(self, action: ComputerAction) -> None: ...
    async def zoom(self, region: Region) -> bytes: ...
    async def cursor_position(self) -> Point: ...


class ManagedDisplay:
    """The display a backend hands out for its lifetime: started on first use, and again after a failure.

    `start` brings the display up, doing nothing to one that's running, and returns its driver. `scope` wraps every
    call, start included, such as to map a provider's errors.

    A call that only looks, such as a screenshot, is retried once after a failure, which starts a display that was
    lost; an action isn't, so nothing is done twice.
    """

    def __init__(
        self,
        start: Callable[[], Awaitable[DisplayDriver]],
        *,
        scope: Callable[[], AbstractAsyncContextManager[object]] = nullcontext,
    ):
        self._start = start
        self._scope = scope
        self._driver: DisplayDriver | None = None
        self._lock = anyio.Lock()

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
                    return await call(driver)
            except WorkspaceUnsupportedActionError:
                raise  # a refused action, such as an unknown key: the display is fine
            except Exception as error:
                # The display may have stopped, or the environment restarted: the next call starts it again.
                if driver is not None and self._driver is driver:
                    self._driver = None
                # Retrying can't help an unavailable environment, and would double a timeout.
                if not retry or isinstance(error, WorkspaceUnavailableError | WorkspaceTimeoutError):
                    raise
                retry = False

    async def display_size(self) -> tuple[int, int]:
        return await self._call(lambda driver: driver.display_size(), observe=True)

    async def screenshot(self) -> bytes:
        return await self._call(lambda driver: driver.screenshot(), observe=True)

    async def act(self, action: ComputerAction) -> None:
        await self._call(lambda driver: driver.act(action), observe=False)

    async def zoom(self, region: Region) -> bytes:
        return await self._call(lambda driver: driver.zoom(region), observe=True)

    async def cursor_position(self) -> Point:
        return await self._call(lambda driver: driver.cursor_position(), observe=True)
