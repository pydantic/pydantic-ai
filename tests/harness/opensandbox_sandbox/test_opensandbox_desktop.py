"""Tests for `OpenSandbox(desktop=True)`, against the fake server with a scripted X display."""

from __future__ import annotations

import os

import pytest
from opensandbox.constants import DEFAULT_EGRESS_PORT, DEFAULT_EXECD_PORT

from pydantic_ai import Agent, UserError
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
from .fake_opensandbox import FakeOpenSandbox

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')


@pytest.fixture
def display(server: FakeOpenSandbox) -> FakeDisplay:
    server.display = display = FakeDisplay()
    return display


def _backend(
    server: FakeOpenSandbox,
    working_dir: str,
    *,
    desktop: bool | tuple[int, int] = False,
    env: dict[str, str] | None = None,
) -> OpenSandboxBackend:
    return OpenSandboxBackend(
        image='fake-image', connection=server.connection, working_dir=working_dir, desktop=desktop, env=env
    )


async def test_a_shell_command_finds_the_desktop_running(
    server: FakeOpenSandbox, display: FakeDisplay, working_dir: str
) -> None:
    # A program the shell starts opens on the screen, so a command starts the display.
    backend = _backend(server, working_dir, desktop=True, env={'FOO': 'bar'})

    # The fake display refuses `xwininfo` until the display has started, so this passes only if it started first.
    assert 'Width: 1280' in (await backend.run('xwininfo -root', shell=True)).stdout
    assert (await backend.run('echo "$DISPLAY $FOO"', shell=True)).stdout == ':1 bar\n'
    assert len(display.starts) == 1
    await backend.aclose()


async def test_a_desktop_size_sets_the_display_it_starts(
    server: FakeOpenSandbox, display: FakeDisplay, working_dir: str
) -> None:
    backend = _backend(server, working_dir, desktop=(1440, 900))

    await full_display(backend).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start
    await backend.aclose()


async def test_a_desktop_sandbox_reattaches_with_its_display(
    server: FakeOpenSandbox, display: FakeDisplay, working_dir: str
) -> None:
    created = _backend(server, working_dir)
    await created.working_dir()
    ref = created.ref
    assert ref is not None
    await created.aclose()

    reattached = OpenSandbox(connection=server.connection, working_dir=working_dir, desktop=(1440, 900)).backend(ref)
    await full_display(reattached).display_size()
    await reattached.aclose()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert OpenSandbox(connection=server.connection, working_dir=working_dir).backend(ref).display is None


async def test_a_desktop_capability_runs_the_computer_tool_without_exposing_a_port(
    server: FakeOpenSandbox, display: FakeDisplay, working_dir: str
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[
            OpenSandbox(image='fake-image', connection=server.connection, working_dir=working_dir, desktop=True),
            ComputerUse(native=False),
        ],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    # The SDK looks up execd's and the egress sidecar's endpoints itself; the desktop asks for no other port.
    looked_up = {int(request.rsplit('/', 1)[1]) for request in server.requests if '/endpoints/' in request}
    assert looked_up <= {DEFAULT_EXECD_PORT, DEFAULT_EGRESS_PORT}


@pytest.mark.parametrize('desktop', [(0, 0), (1280, -1), (1280,)])
def test_a_desktop_size_must_be_a_width_and_height_of_at_least_1(desktop: tuple[int, int]) -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        OpenSandboxBackend(image='fake-image', desktop=desktop)
