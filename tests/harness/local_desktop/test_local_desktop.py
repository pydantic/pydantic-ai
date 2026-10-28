"""Tests for `LocalDesktop`, with fake X programs on `PATH` in place of Xvfb, fluxbox and xdotool."""

from __future__ import annotations

import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import anyio
import pytest

from pydantic_ai.workspaces import Click, Point, ReadOnlyWorkspace, Region, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness import LocalDesktop, LocalDesktopBackend

T = TypeVar('T')

pytestmark = pytest.mark.anyio
linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='a local desktop needs Linux')

_FAKES = {
    # Prints the display number it "chose", then runs until stopped, as Xvfb does with `-displayfd 1`.
    'Xvfb': 'echo "Xvfb $*" >> "$LOG"; echo $$ >> "$LOG.pids"; echo 7; exec sleep 60',
    'fluxbox': 'echo "fluxbox $DISPLAY ${SECRET-unset}" >> "$LOG"; echo $$ >> "$LOG.pids"; exec sleep 60',
    # Drops `--pdeathsig KILL --` and runs the rest.
    'setpriv': 'echo "setpriv $1 $2" >> "$LOG"; shift 3; exec "$@"',
    # The window manager is ready on the second check.
    'xprop': 'if [ -f "$LOG.wm" ]; then echo "_NET_SUPPORTING_WM_CHECK(WINDOW): window id # 0x400001"; fi; touch "$LOG.wm"',
    'xdotool': """echo "xdotool $DISPLAY $*" >> "$LOG"
case "$1" in
  getdisplaygeometry) echo 1280 800;;
  getmouselocation) printf 'X=3\\nY=4\\nSCREEN=0\\nWINDOW=1\\n';;
esac""",
    'xmodmap': 'true',
    # Writes a PNG signature to the file named last, which is all a screenshot is checked for.
    'scrot': 'for last; do :; done; printf "\\211PNG\\r\\n\\032\\n" > "$last"',
}


_REAL = ('sh', 'sleep', 'mktemp', 'base64', 'rm', 'touch')
"""The real programs the fakes and the display commands use."""


def _install(bin_dir: Path, fakes: dict[str, str]) -> None:
    for name, body in fakes.items():
        path = bin_dir / name
        # The log's path is written in, since the screen's processes don't inherit this process's environment.
        path.write_text(f'#!/bin/sh\nLOG={bin_dir.parent / "log"}\n{body}\n')
        path.chmod(0o755)


@pytest.fixture
def fake_x(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`PATH` with only fake X programs and the few real ones they need; returns the log the fakes append to."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in _REAL:
        (bin_dir / name).symlink_to(shutil.which(name))  # pyright: ignore[reportArgumentType]
    _install(bin_dir, _FAKES)
    log = tmp_path / 'log'
    log.touch()
    monkeypatch.setenv('PATH', str(bin_dir))
    return log


def _lines(log: Path, program: str) -> list[str]:
    return [line for line in log.read_text(encoding='utf-8').splitlines() if line.startswith(program)]


def _backend(desktop: LocalDesktop[None]) -> LocalDesktopBackend:
    backend = desktop.get_workspace(None, ref=None)  # pyright: ignore[reportArgumentType]
    assert isinstance(backend, LocalDesktopBackend)
    return backend


async def _stop(desktop: LocalDesktop[None]) -> None:
    await desktop._screen.stop()  # pyright: ignore[reportOptionalMemberAccess,reportPrivateUsage]


@linux_only
async def test_a_virtual_screen_starts_once_and_every_command_runs_on_it(
    fake_x: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('SECRET', 'sk-not-for-the-screen')
    desktop = LocalDesktop[None](tmp_path, size=(1024, 768))
    try:
        backend = _backend(desktop)
        display = backend.display
        assert (await backend.run('echo $DISPLAY', shell=True)).stdout == ':7\n'
        assert await display.display_size() == (1280, 800)
        assert (await display.screenshot()).startswith(b'\x89PNG')
        assert (await display.zoom(Region(left=0, top=0, right=10, bottom=10))).startswith(b'\x89PNG')
        await display.act(Click(x=5, y=6))
        assert await display.cursor_position() == Point(x=3, y=4)
        # A later run's backend uses the same screen.
        assert (await _backend(desktop).run('echo $DISPLAY', shell=True)).stdout == ':7\n'
    finally:
        await _stop(desktop)

    [xvfb] = _lines(fake_x, 'Xvfb')
    assert xvfb == 'Xvfb -displayfd 1 -screen 0 1024x768x24 -nolisten tcp -noreset'
    # Apps started from the window manager's menu get the workspace's environment, not this process's.
    assert _lines(fake_x, 'fluxbox') == ['fluxbox :7 unset']
    # The kernel stops both if this process dies.
    assert _lines(fake_x, 'setpriv') == ['setpriv --pdeathsig KILL'] * 2
    assert all(line.startswith('xdotool :7 ') for line in _lines(fake_x, 'xdotool'))


def _in_new_event_loop(function: Callable[[], Awaitable[T]]) -> T:
    """Run `function` in an event loop of its own, on a thread of its own, as `run_sync` would."""
    with ThreadPoolExecutor(1) as executor:
        return executor.submit(anyio.run, function).result()


@linux_only
def test_a_screen_that_stopped_starts_again_in_a_later_event_loop(fake_x: Path, tmp_path: Path):
    # Each `run_sync` runs in an event loop of its own, so whether the screen is running can't depend on one.
    desktop = LocalDesktop[None](tmp_path)
    backend = _backend(desktop)
    try:
        _in_new_event_loop(backend.display.display_size)
        for pid in _pids(fake_x):
            _kill(pid)
        assert _in_new_event_loop(backend.display.display_size) == (1280, 800)
    finally:
        _in_new_event_loop(lambda: _stop(desktop))
    assert len(_lines(fake_x, 'Xvfb')) == 2


def _pids(log: Path) -> list[int]:
    """The process ids of the fake Xvfb and window managers started so far, in order."""
    pids = Path(f'{log}.pids')
    return [int(pid) for pid in pids.read_text(encoding='utf-8').split()] if pids.exists() else []


def _kill(pid: int) -> None:
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)  # gone before the next call looks


@linux_only
async def test_a_window_manager_that_exited_is_started_again_on_the_same_screen(fake_x: Path, tmp_path: Path):
    # A misclick on the window manager's Exit must not close every app on the screen.
    desktop = LocalDesktop[None](tmp_path)
    backend = _backend(desktop)
    try:
        await backend.display.display_size()
        assert backend.display.starts == 1
        xvfb, window_manager = _pids(fake_x)
        _kill(window_manager)

        assert (await backend.run('echo $DISPLAY', shell=True)).stdout == ':7\n'
        os.kill(xvfb, 0)  # still running
        assert len(_lines(fake_x, 'Xvfb')) == 1
        assert len(_lines(fake_x, 'fluxbox')) == 2
        assert backend.display.starts == 1

        _kill(xvfb)
        await backend.display.display_size()
        assert backend.display.starts == 2
    finally:
        await _stop(desktop)


@linux_only
def test_a_screen_outlives_the_thread_that_started_it(fake_x: Path, tmp_path: Path):
    # The real `setpriv`, whose `--pdeathsig` fires when the starting thread exits, not only the process.
    bin_dir = Path(os.environ['PATH'].split(':')[0])
    (bin_dir / 'setpriv').unlink()
    (bin_dir / 'setpriv').symlink_to(shutil.which('setpriv', path=os.defpath))  # pyright: ignore[reportArgumentType]
    desktop = LocalDesktop[None](tmp_path)
    display = _backend(desktop).display
    try:
        _in_new_event_loop(display.display_size)  # its thread has exited when this returns
        _in_new_event_loop(display.display_size)
        # A screen that had stopped with the thread would have been started again.
        assert len(_lines(fake_x, 'Xvfb')) == 1
        assert display.starts == 1
    finally:
        _in_new_event_loop(lambda: _stop(desktop))


@linux_only
@pytest.mark.parametrize(
    ('fakes', 'error'),
    [
        ({'fluxbox': None}, 'A virtual screen needs fluxbox; on Debian or Ubuntu: `apt install xvfb fluxbox'),
        ({'Xvfb': 'exit 1'}, 'Xvfb printed no display number'),
        ({'fluxbox': 'exit 1', 'xprop': 'true'}, 'its window manager did not start'),
    ],
    ids=['missing-program', 'xvfb-fails', 'window-manager-fails'],
)
async def test_a_screen_that_cannot_start_says_why(
    fake_x: Path, tmp_path: Path, fakes: dict[str, str | None], error: str
):
    bin_dir = Path(os.environ['PATH'].split(':')[0])
    for name, body in fakes.items():
        if body is None:
            (bin_dir / name).unlink()
        else:
            _install(bin_dir, {name: body})
    desktop = LocalDesktop[None](tmp_path)
    with pytest.raises(WorkspaceUnavailableError, match=error.replace('`', '.')):
        await _backend(desktop).display.screenshot()
    for pid in _pids(fake_x):  # what did start was stopped
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@linux_only
async def test_a_start_that_is_cancelled_stops_what_it_started(fake_x: Path, tmp_path: Path):
    _install(Path(os.environ['PATH'].split(':')[0]), {'xprop': 'true'})  # the window manager never gets ready
    desktop = LocalDesktop[None](tmp_path)
    with anyio.move_on_after(0.5):
        await _backend(desktop).display.screenshot()
    pids = [int(pid) for pid in Path(f'{fake_x}.pids').read_text(encoding='utf-8').split()]
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@linux_only
async def test_the_real_screen_is_the_session_this_process_runs_in(
    fake_x: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setenv('XAUTHORITY', '/run/user/1000/gdm/Xauthority')
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    monkeypatch.delenv('XDG_SESSION_TYPE', raising=False)
    backend = _backend(LocalDesktop[None](tmp_path, real_screen=True))
    output = await backend.run('echo "$DISPLAY $XAUTHORITY"', shell=True)
    assert output.stdout == ':0 /run/user/1000/gdm/Xauthority\n'
    assert await backend.display.display_size() == (1280, 800)
    assert _lines(fake_x, 'Xvfb') == []


@linux_only
@pytest.mark.parametrize(
    ('env', 'error'),
    [
        ({'DISPLAY': ':0', 'XDG_SESSION_TYPE': 'wayland'}, 'This is a Wayland session'),
        ({'DISPLAY': ':0', 'WAYLAND_DISPLAY': 'wayland-0'}, 'This is a Wayland session'),
        ({}, '`DISPLAY` is not set'),
    ],
    ids=['wayland-session', 'wayland-display', 'no-display'],
)
async def test_the_real_screen_is_refused_where_it_cannot_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], error: str
):
    for name in ('DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE'):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(WorkspaceUnavailableError, match=error.replace('`', '.')):
        await _backend(LocalDesktop[None](tmp_path, real_screen=True)).display.screenshot()


async def test_it_needs_linux(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    backend = _backend(LocalDesktop[None](tmp_path))
    with pytest.raises(WorkspaceUnavailableError, match='A local desktop needs Linux'):
        await backend.display.screenshot()
    # Commands need no screen.
    assert (await backend.run(['echo', 'hi'])).stdout == 'hi\n'


def test_it_is_a_local_workspace_with_a_screen(tmp_path: Path):
    desktop = LocalDesktop[None](tmp_path, read_only=True)
    assert isinstance(desktop.get_workspace(None, ref=None), ReadOnlyWorkspace)  # pyright: ignore[reportArgumentType]
    # Like `LocalWorkspace`, a ref for another directory never redirects it.
    assert desktop.get_workspace(None, ref=WorkspaceRef(provider='local', id='/elsewhere')) is None  # pyright: ignore[reportArgumentType]
    assert desktop.id == 'local_workspace'


def test_workspaces_on_one_screen_name_it_alike(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """So `ComputerUse` runs on one screen take turns at it, whichever directory each works in."""
    monkeypatch.setenv('DISPLAY', ':0')
    first, second = (tmp_path / 'a', tmp_path / 'b')
    first.mkdir()
    second.mkdir()
    session = [_backend(LocalDesktop[None](path, real_screen=True)) for path in (first, second)]
    assert session[0].ref != session[1].ref
    assert session[0].display.screen_key == session[1].display.screen_key

    private = LocalDesktop[None](first)
    assert _backend(private).display.screen_key == _backend(private).display.screen_key
    assert _backend(private).display.screen_key != _backend(LocalDesktop[None](first)).display.screen_key
    assert _backend(private).display.screen_key != session[0].display.screen_key
