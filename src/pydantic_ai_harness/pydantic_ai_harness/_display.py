"""`LazyDisplay`, a backend's display started on first use and again after a failure, and helpers the displays share."""

from __future__ import annotations as _annotations

import base64
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

import anyio

from pydantic_ai.computer_use import Point, Region
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError, WorkspaceUnavailableError

from ._png import PNG_SIGNATURE
from .computer_use import DisplayAction, UnsupportedActionError

__all__ = ('DisplayDriver', 'LazyDisplay')

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


class LazyDisplay:
    """The display a backend hands out for its lifetime: started on first use, and again after a failure.

    Like the backend, it does nothing when built. `driver` drives the display once it runs. `start` brings the
    display up, doing nothing to one that's running.

    A call that only looks, such as a screenshot, is retried once after a failure, which starts a display that was
    lost; an action isn't, so nothing is done twice.

    It implements [`Display`][pydantic_ai_harness.computer_use.Display] and every optional ability.
    """

    def __init__(self, driver: DisplayDriver, start: Callable[[], Awaitable[None]]):
        self._driver = driver
        self._start = start
        self._running = False
        """Whether the display is known to be running, so a call needn't start it first."""
        self._lock = anyio.Lock()

    async def _start_unless_running(self) -> None:
        # One start at a time: two would both find no display and start two.
        async with self._lock:
            if not self._running:
                await self._start()
                self._running = True

    async def _call(self, call: Callable[[DisplayDriver], Awaitable[T]], *, observe: bool) -> T:
        retry = observe
        while True:
            try:
                await self._start_unless_running()
                return await call(self._driver)
            except UnsupportedActionError:
                raise  # a refused action, such as an unknown key: the display is fine
            except Exception as error:
                # The display may have stopped, or the environment restarted: the next call starts it again, which
                # does nothing to a display that is still running.
                self._running = False
                # Retrying can't help an unavailable environment, and would double a timeout.
                if isinstance(error, WorkspaceUnavailableError | WorkspaceTimeoutError):
                    raise
                if not retry:
                    raise
                retry = False

    async def display_size(self) -> tuple[int, int]:
        return await self._call(lambda driver: driver.display_size(), observe=True)

    async def screenshot(self) -> bytes:
        return await self._call(lambda driver: driver.screenshot(), observe=True)

    async def act(self, action: DisplayAction) -> None:
        await self._call(lambda driver: driver.act(action), observe=False)

    async def zoom(self, region: Region) -> bytes:
        return await self._call(lambda driver: driver.zoom(region), observe=True)

    async def cursor_position(self) -> Point:
        return await self._call(lambda driver: driver.cursor_position(), observe=True)
