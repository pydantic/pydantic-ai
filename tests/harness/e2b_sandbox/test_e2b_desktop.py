"""Tests for `E2BSandbox(desktop=True)`, against the fake E2B SDK with a scripted X display."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerUse, SupportsLiveView
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
from .fake_e2b import FakeE2B


@pytest.fixture
def display(fake_e2b: FakeE2B) -> FakeDisplay:
    fake_e2b.responder = display = FakeDisplay()
    return display


@pytest.mark.parametrize(('template', 'created'), [(None, 'desktop'), ('my-desktop', 'my-desktop')])
async def test_a_desktop_sandbox_runs_the_desktop_template_unless_given_one(
    fake_e2b: FakeE2B, display: FakeDisplay, template: str | None, created: str
) -> None:
    await E2BSandboxBackend(template=template, desktop=True).get_sandbox()

    assert [call.template for call in fake_e2b.create_calls] == [created]


async def test_a_shell_command_finds_the_desktop_running(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    # The shell command gets `DISPLAY`, so a program it starts opens on the screen, even before the first screenshot.
    await E2BSandboxBackend(desktop=True, env={'FOO': 'bar'}).run('xterm &', shell=True)

    assert len(display.starts) == 1
    assert fake_e2b.create_calls[0].envs == {'DISPLAY': ':1', 'FOO': 'bar'}
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1', 'FOO': 'bar'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await full_display(E2BSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    # XFCE is the template's desktop.
    assert 'fluxbox' not in start


async def test_a_display_that_does_not_start_names_the_sandbox_and_what_it_needs(fake_e2b: FakeE2B) -> None:
    def responder(command: str, timeout: float | None) -> tuple[str, str, int]:
        needs = "the display needs Xvfb, and there's no apt-get to install them with"
        return ('', needs, 1) if 'Xvfb' in command else ('', '', 0)

    fake_e2b.responder = responder

    with pytest.raises(
        WorkspaceUnavailableError,
        match=r"^Could not start the display in E2B sandbox sbx-1: the display needs Xvfb, and there's no apt-get to "
        r'install them with$',
    ):
        await full_display(E2BSandboxBackend(desktop=True)).display_size()


async def test_a_desktop_sandbox_reattaches_with_its_display(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    ref = WorkspaceRef(provider='e2b', id='sbx-desktop')

    await full_display(E2BSandbox(desktop=(1440, 900)).backend(ref)).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert E2BSandbox().backend(ref).display is None


def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        E2BSandbox(desktop=(0, 800))


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(
    fake_e2b: FakeE2B, display: FakeDisplay
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[E2BSandbox(desktop=True), ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert [call.template for call in fake_e2b.create_calls] == ['desktop']
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)


_STUBS = {
    'setsid': 'exec "$@"',  # macOS has no `setsid`
    'Xvfb': ':',  # the `xdotool` stub plays the X server
    # The start script checks these are installed; the display answers, so they don't run.
    **dict.fromkeys(['xwininfo', 'scrot'], 'exit 1'),
    # The start script runs as root and makes the X socket directory, which these keep off the host's `/tmp`.
    'id': 'echo 0',
    **dict.fromkeys(['mkdir', 'chmod'], ':'),
    'xdotool': '[ "$1" != search ] || [ -e "$bin/xfce-started" ]',  # X answers; XFCE's window shows once started
    # Shows its window, then keeps running past the script's next 0.1-second check, so the start succeeds and no
    # process is left behind. A `startxfce4-fails` flag file makes it exit with an error instead.
    'startxfce4': '[ -e "$bin/startxfce4-fails" ] && { echo no session bus; exit 17; }; '
    'echo started >>"$bin/xfce-starts"; touch "$bin/xfce-started"; exec sleep 1',
}


async def _start_script(fake_e2b: FakeE2B) -> str:
    """Return the desktop start script exactly as the backend sends it."""
    scripts: list[str] = []

    def respond(command: str, timeout: float | None) -> tuple[str, str, int]:
        args = shlex.split(command)
        if args[:2] == ['sh', '-c']:
            scripts.append(args[2])
        return ('  Width: 1280\n  Height: 800\n' if 'xwininfo' in command else ''), '', 0

    fake_e2b.responder = respond
    await full_display(E2BSandboxBackend(desktop=True)).display_size()
    (start,) = scripts
    assert 'Xvfb' in start
    return start


_TOOLS = ('cat', 'grep', 'rm', 'sleep', 'tail', 'touch', 'tr')
"""Host programs the script and the stubs may use. Every other program is a stub or missing, never the host's."""


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """The directory of stub programs, the only directory on the script's `PATH`. Tests put flag files in it."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name, body in _STUBS.items():
        (bin_dir / name).write_text(f'#!/bin/sh\nbin={bin_dir}\n{body}\n')
        (bin_dir / name).chmod(0o755)
    for name in _TOOLS:
        tool = shutil.which(name)
        assert tool is not None, name
        (bin_dir / name).symlink_to(tool)
    return bin_dir


def _run(script: str, stubs: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run `script` under `/bin/sh` with only the stubs on `PATH` and its fixed file paths moved into the stubs."""
    root = stubs.parent
    for old, new in [('/tmp/', f'{root}/'), ('/proc/', f'{root}/proc/')]:
        script = script.replace(old, new)
    env = {'PATH': str(stubs)}
    return subprocess.run(['/bin/sh', '-c', script, 'sh', *args], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')
class TestStartScript:
    """Runs the desktop start script, captured from the fake SDK, in a real shell.

    The other tests answer the scripts by matching their text, so these check that the start script parses and does
    what it should, with stub programs for the sandbox's own.
    """

    async def test_xfce_starts_unless_its_desktop_shows(self, fake_e2b: FakeE2B, stubs: Path) -> None:
        start = await _start_script(fake_e2b)

        for _ in range(2):
            result = _run(start, stubs, '1280x800')
            assert result.returncode == 0, result.stderr

        assert (stubs / 'xfce-starts').read_text() == 'started\n'

    async def test_xfce_exiting_fails_the_start_with_its_output(self, fake_e2b: FakeE2B, stubs: Path) -> None:
        start = await _start_script(fake_e2b)
        (stubs / 'startxfce4-fails').touch()

        result = _run(start, stubs, '1280x800')

        assert result.returncode == 1
        assert result.stderr == 'the XFCE desktop did not start:\nno session bus\n'

    async def test_a_template_without_xfce_starts_with_no_desktop(self, fake_e2b: FakeE2B, stubs: Path) -> None:
        start = await _start_script(fake_e2b)
        (stubs / 'startxfce4').unlink()

        result = _run(start, stubs, '1280x800')

        assert result.returncode == 0, result.stderr
        assert not (stubs / 'xfce-starts').exists()

    async def test_an_x_server_outside_the_sandbox_on_its_display_stops_the_start(
        self, fake_e2b: FakeE2B, stubs: Path
    ) -> None:
        """In a network namespace shared with the host, the host's X server listens on the abstract socket."""
        start = await _start_script(fake_e2b)
        (stubs.parent / 'proc' / 'net').mkdir(parents=True)
        (stubs.parent / 'proc' / 'net' / 'unix').write_text(
            f'0000000000000000: 00000002 00000000 00010000 0001 01 12345 @{stubs.parent}/.X11-unix/X1\n'
        )

        result = _run(start, stubs, '1280x800')

        assert result.returncode == 1
        assert 'another X server, outside this sandbox, already uses display :1' in result.stderr
        assert not (stubs / 'xfce-starts').exists()


async def test_the_live_view_starts_once_behind_a_password(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    screen = E2BSandboxBackend(desktop=True).display
    assert isinstance(screen, SupportsLiveView)

    url = await screen.live_view_url()
    assert await screen.live_view_url() == url

    commands = fake_e2b.sandboxes[0].commands.calls
    (start,) = [call for call in commands if 'x11vnc' in call.command]
    assert start.envs is not None
    password = start.envs['VNC_PASSWORD']
    assert len(password) == 8
    # The password reaches x11vnc through the env, so it isn't in the command line.
    assert password not in start.command
    assert '-viewonly' in start.command and '-passwd "$VNC_PASSWORD"' in start.command
    assert url == (
        'https://6080-sbx-1.e2b.app/vnc.html?autoconnect=true&view_only=true&resize=scale&password=' + password
    )
    # The display started first, so x11vnc has a screen to share.
    assert len(display.starts) == 1


async def test_a_live_view_that_cannot_start_says_why(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    def responder(command: str, timeout: float | None) -> tuple[str, str, int]:
        if 'x11vnc' in command:
            return '', 'the live view needs x11vnc and noVNC in /opt/noVNC\n', 1
        return display(command, timeout)

    fake_e2b.responder = responder
    screen = E2BSandboxBackend(desktop=True).display
    assert isinstance(screen, SupportsLiveView)

    with pytest.raises(
        WorkspaceError,
        match=r'^Could not start the live view in E2B sandbox sbx-1: the live view needs x11vnc and noVNC in /opt/noVNC$',
    ):
        await screen.live_view_url()
