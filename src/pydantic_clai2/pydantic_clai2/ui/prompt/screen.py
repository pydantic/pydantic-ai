"""Who has to step aside when a plugin takes the whole terminal mid-run."""

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import anyio

from pydantic_clai2.plugins import FullScreen, bare_screen

_OVERLAY: ContextVar[object | None] = ContextVar('_OVERLAY', default=None)
"""The overlay the current context runs inside, inherited by its menu threads and their actions."""


class Screen:
    """The shell's `FullScreen`: bound to the live renderer and status row for the length of one prompt.

    Plugin hosts are created once at load time, but what has to stop before a widget can draw
    changes every prompt. Hosts hold `screen.full`; the prompt loop binds what it means.

    One widget owns the screen at a time: a second `full()` (a parallel tool call, say) waits for
    the first to exit. It is not re-entrant; a widget that opens another widget does so inside
    its own block, not through a nested `full()`. The exception is a command run inside
    `overlay()`: it owns the screen already, so plugin code it runs (loading or configuring a
    plugin mid-turn) passes straight through `full()` instead of waiting for itself.
    """

    def __init__(self) -> None:
        """Start without a stream or editor to suspend."""
        self._take: FullScreen = bare_screen
        self._owner = anyio.Lock(fast_acquire=True)
        self._overlay: object | None = None
        self.editor: FullScreen | None = None

    @contextmanager
    def bound(self, take: FullScreen) -> Generator[None]:
        """While active, `full()` defers to `take`; afterwards it is a no-op again."""
        self._take = take
        try:
            yield
        finally:
            self._take = bare_screen

    @asynccontextmanager
    async def full(self) -> AsyncGenerator[None]:
        """Own the terminal until the block exits. Give this to `PluginHost` as its `full_screen`."""
        if self._overlay is not None and _OVERLAY.get() is self._overlay:
            yield
            return
        async with self._owner, self._take(), (self.editor or bare_screen)():
            yield

    @asynccontextmanager
    async def overlay(self) -> AsyncGenerator[None]:
        """Own the terminal for a menu the user opened mid-turn, leaving the stream running.

        Unlike `full()`, the turn is not paused: its output is held by the menu worker
        instead. Widgets still take turns, so a question the agent asks waits for the menu.
        """
        async with self._owner, (self.editor or bare_screen)():
            self._overlay = object()
            token = _OVERLAY.set(self._overlay)
            try:
                yield
            finally:
                _OVERLAY.reset(token)
                self._overlay = None
