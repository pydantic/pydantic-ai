"""A desktop on this machine: a private virtual X screen, or the X screen you're logged in to."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anyio
import anyio.to_thread

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    WorkspaceCommand,
    WorkspaceUnavailableError,
)
from pydantic_ai.workspaces.local import _INHERITED_ENV  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness._display import DEFAULT_DESKTOP_SIZE, ManagedDisplay
from pydantic_ai_harness._x11 import X11Computer

__all__ = ('LocalDesktopBackend', 'VirtualScreen')

_PROGRAMS = ('Xvfb', 'fluxbox', 'xprop', 'xdotool', 'xmodmap', 'scrot', 'setpriv')
_PACKAGES = 'xvfb fluxbox x11-utils xdotool x11-xserver-utils scrot util-linux'
_START_TIMEOUT = 10


@functools.cache
def _starting_thread() -> ThreadPoolExecutor:
    """The one thread that starts screens' processes, which runs until this process exits.

    The kernel stops a process started with `setpriv --pdeathsig` when the thread that started it exits, so a screen
    started from a short-lived thread, such as a thread pool's running `run_sync`, would stop with that thread.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix='pydantic-ai-screen')


class VirtualScreen:
    """An invisible X screen (Xvfb) with a window manager, started on first use and restarted if it stops.

    It stops when this process exits. Its processes aren't tied to an event loop, so each `run_sync` can use it.
    """

    def __init__(self, size: tuple[int, int]):
        self._size = size
        self._lock = anyio.Lock()
        self._display: str | None = None
        # Xvfb, then the window managers started on it.
        self._processes: list[subprocess.Popen[bytes]] = []
        self.starts = 0
        """How many times the screen started, each time without the windows it had."""

    async def display(self) -> str:
        """The screen's `DISPLAY`, such as `:3`, once it's running."""
        async with self._lock:
            if self._display is None or self._processes[0].poll() is not None:
                await self.stop()
                try:
                    self._display = await self._start()
                except BaseException:
                    await self.stop()
                    raise
                self.starts += 1
            elif self._processes[-1].poll() is not None:
                # Only the window manager exited, such as from its menu's Exit: another keeps the screen's windows.
                await self._start_window_manager(self._display)
            return self._display

    async def _start(self) -> str:
        if missing := [program for program in _PROGRAMS if shutil.which(program) is None]:
            raise WorkspaceUnavailableError(
                f'A virtual screen needs {", ".join(missing)}; on Debian or Ubuntu: `apt install {_PACKAGES}`.'
            )
        width, height = self._size
        # Xvfb picks a free display and prints its number once it's ready. `-noreset` keeps the pointer and the key
        # mapping between commands, which a server with no other client resets as each one disconnects.
        xvfb = await self._open(
            ['Xvfb', '-displayfd', '1', '-screen', '0', f'{width}x{height}x24', '-nolisten', 'tcp', '-noreset'],
            stdout=subprocess.PIPE,
        )
        assert xvfb.stdout is not None
        number = b''
        with anyio.move_on_after(_START_TIMEOUT):
            number = await anyio.to_thread.run_sync(xvfb.stdout.readline, abandon_on_cancel=True)
        if not number.strip():
            raise WorkspaceUnavailableError('The virtual screen did not start: Xvfb printed no display number.')
        display = f':{number.decode().strip()}'
        await self._start_window_manager(display)
        return display

    async def _start_window_manager(self, display: str) -> None:
        # Without a window manager, clicking a window doesn't focus or raise it.
        window_manager = await self._open(['fluxbox'], env={'DISPLAY': display})
        with anyio.move_on_after(_START_TIMEOUT):
            while window_manager.poll() is None:
                if await _has_window_manager(display):
                    return
                await anyio.sleep(0.05)
        raise WorkspaceUnavailableError('The virtual screen did not start: its window manager did not start.')

    async def _open(
        self, argv: list[str], *, stdout: int | None = None, env: dict[str, str] | None = None
    ) -> subprocess.Popen[bytes]:
        # The kernel stops it if this process dies without stopping it.
        starting = _starting_thread().submit(
            subprocess.Popen,
            ['setpriv', '--pdeathsig', 'KILL', '--', *argv],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.DEVNULL,
            # What `LocalWorkspaceBackend` commands inherit: an app started from the window manager's menu must not see
            # the program's secrets.
            env={name: os.environ[name] for name in _INHERITED_ENV if name in os.environ} | (env or {}),
        )
        process = await anyio.to_thread.run_sync(starting.result)
        self._processes.append(process)
        return process

    async def stop(self) -> None:
        """Stop the screen and its window manager; the next `display()` starts a new one."""
        processes, self._processes, self._display = self._processes, [], None
        for process in processes:
            process.kill()
        # Shielded, as this also runs when a start is cancelled.
        with anyio.CancelScope(shield=True):
            for process in processes:
                await anyio.to_thread.run_sync(process.wait)
                if process.stdout is not None:
                    process.stdout.close()


async def _has_window_manager(display: str) -> bool:
    result = await anyio.run_process(['xprop', '-display', display, '-root', '_NET_SUPPORTING_WM_CHECK'], check=False)
    return b'window id' in result.stdout


def _session_env() -> dict[str, str]:
    """The `DISPLAY` and `XAUTHORITY` of the X session this process runs in."""
    if os.environ.get('WAYLAND_DISPLAY') or os.environ.get('XDG_SESSION_TYPE') == 'wayland':
        raise WorkspaceUnavailableError(
            'This is a Wayland session, which the desktop can neither see nor control. '
            'Log in with an X11 (Xorg) session, or use a virtual screen.'
        )
    if not os.environ.get('DISPLAY'):
        raise WorkspaceUnavailableError(
            'There is no X screen to use: `DISPLAY` is not set. Use a virtual screen instead.'
        )
    return {name: os.environ[name] for name in ('DISPLAY', 'XAUTHORITY') if name in os.environ}


class LocalDesktopBackend(LocalWorkspaceBackend):
    """A [`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] with a screen.

    Commands run on the screen too, so an app the agent starts opens there.

    Args:
        working_dir: Where commands start and relative paths resolve.
        env: Environment variables for every command, on top of the inherited `PATH`, `HOME` and locale.
        screen: The virtual screen to use, such as one several backends share; by default a new private one.
            `'real'` uses the X screen you're logged in to: the agent then moves your mouse and types into your
            windows.
    """

    def __init__(
        self,
        working_dir: str | Path,
        *,
        env: Mapping[str, str] | None = None,
        screen: VirtualScreen | Literal['real'] | None = None,
    ):
        super().__init__(working_dir, env=env)
        self._screen = None if screen == 'real' else screen or VirtualScreen(DEFAULT_DESKTOP_SIZE)
        self._display = ManagedDisplay(self._start_display, screen_key=self._screen_key)
        # A later start of the screen than this backend has seen started it without the windows it had.
        self._screen_starts = None if self._screen is None else self._screen.starts

    def _screen_key(self) -> object | None:
        # Every workspace on this machine that uses the session's screen shares its `DISPLAY`, whatever its directory.
        if self._screen is None:
            display = os.environ.get('DISPLAY')
            return ('DISPLAY', display) if display else None
        return self._screen

    async def _display_env(self) -> dict[str, str]:
        if sys.platform != 'linux':
            raise WorkspaceUnavailableError(
                'A local desktop needs Linux. Use `DockerSandbox(desktop=True)` for one on this machine.'
            )
        if self._screen is None:
            return _session_env()
        display = await self._screen.display()
        if self._screen.starts != self._screen_starts:
            self._screen_starts = self._screen.starts
            self._display.started_fresh()
        return {'DISPLAY': display}

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        try:
            display_env = await self._display_env()
        except WorkspaceUnavailableError:
            display_env = {}  # a command needs no screen; the display's calls say why there's none
        return await super().run(command, shell=shell, env={**display_env, **(env or {})}, timeout=timeout)

    async def _start_display(self) -> X11Computer:
        await self._display_env()  # says why there's no screen, starting a virtual one
        # `run` gives every display command the screen's `DISPLAY`, starting the screen again if it stopped.
        return X11Computer(self)

    @property
    def display(self) -> ManagedDisplay:
        """The session's screen, or this backend's virtual one."""
        return self._display
