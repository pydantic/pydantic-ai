"""Tests for `E2BSandbox(desktop=True)`, against the fake E2B SDK with a scripted X display."""

from __future__ import annotations

import base64

import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    Click,
    Point,
    Region,
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._x11 import STARTED
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from ...workspace_fakes import FAKE_SCREENSHOT
from .fake_e2b import FakeE2B


class FakeDisplay:
    """Answers the start script and `xdotool` like a sandbox whose X server is up or down."""

    def __init__(self, *, up: bool = False, can_start: bool = True) -> None:
        self.up = up
        self.can_start = can_start
        self.starts: list[str] = []
        self.xdotool: list[str] = []
        self.captures: list[str] = []
        self.viewers: list[str] = []

    def __call__(self, command: str, timeout: float | None) -> tuple[str, str, int]:
        if 'Xvfb' in command:
            self.starts.append(command)
            if not self.up and not self.can_start:
                return '', 'the X server did not start:\n(EE) Fatal server error', 1
            started, self.up = not self.up, True
            return (STARTED if started else ''), '', 0
        elif command.startswith('xdotool '):
            self.xdotool.append(command)
            if not self.up:
                return '', "Error: Can't open display: :1", 1
            if 'XF86AudioMute' in command:
                return '', '(symbol) No such key name XF86AudioMute', 0
            if command == 'xdotool getdisplaygeometry':
                return '1280 800\n', '', 0
            if command == 'xdotool getmouselocation --shell':
                return 'X=5\nY=6\nSCREEN=0\n', '', 0
        elif 'novnc_proxy' in command:
            self.viewers.append(command)
            if not self.up:
                return '', 'the viewer did not start:\nno display', 1
            return 'Pass1234', '', 0
        elif 'scrot' in command:
            self.captures.append(command)
            return base64.b64encode(FAKE_SCREENSHOT).decode(), '', 0
        return '', '', 0


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


async def test_the_display_starts_on_first_use_and_actions_go_to_it(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    workspace = Workspace(E2BSandboxBackend(desktop=True))

    assert await workspace.display_size() == (1280, 800)
    await workspace.act(Click(x=3, y=4))

    assert len(display.starts) == 1
    assert display.xdotool[-2:] == ['xdotool getdisplaygeometry', 'xdotool mousemove 3 4 click --repeat 1 1']
    # E2B's locale is POSIX, under which `xdotool type` stops at the first character outside ASCII.
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'}


async def test_screenshots_zooms_and_the_pointer_come_from_the_display(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    workspace = Workspace(E2BSandboxBackend(desktop=True))

    assert await workspace.screenshot() == FAKE_SCREENSHOT
    assert await workspace.zoom(Region(left=2, top=3, right=12, bottom=8)) == FAKE_SCREENSHOT
    assert await workspace.cursor_position() == Point(x=5, y=6)
    assert display.captures[-1].endswith(' sh -a 2,3,10,5')
    assert await workspace.scaled_screenshot(640, 400) == FAKE_SCREENSHOT
    assert display.captures[-1].endswith(' sh 640x400')


async def test_backends_on_one_sandbox_name_one_screen(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    first, other = E2BSandboxBackend(desktop=True), E2BSandboxBackend(desktop=True)
    await Workspace(first).display_size()
    await Workspace(other).display_size()
    assert first.ref is not None and first.display is not None and other.display is not None
    again = E2BSandbox(desktop=True).backend(first.ref).display
    assert again is not None
    assert first.display.screen_key == again.screen_key != other.display.screen_key


async def test_the_live_view_url_opens_the_viewer_with_its_password(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    backend = E2BSandboxBackend(desktop=True)

    url = await backend.live_view_url()

    sandbox_id = fake_e2b.sandboxes[0].sandbox_id
    assert url == f'https://6080-{sandbox_id}.e2b.example/vnc.html?autoconnect=true&resize=scale&password=Pass1234'


async def test_a_viewer_that_does_not_start_is_an_error(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    backend = E2BSandboxBackend(desktop=True)
    await Workspace(backend).display_size()
    display.up = False  # the start script's own failure, as the fake answers it

    with pytest.raises(WorkspaceError, match=r'could not start the desktop viewer.*the viewer did not start'):
        await backend.live_view_url()


async def test_commands_see_the_desktop(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await E2BSandboxBackend(desktop=True).run('true', shell=True)
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await Workspace(E2BSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


async def test_a_display_that_went_away_is_started_again(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    backend = E2BSandboxBackend(desktop=True)
    workspace = Workspace(backend)
    await workspace.display_size()
    assert backend.display is not None and backend.display.starts == 1

    display.up = False
    with pytest.raises(WorkspaceError):
        await workspace.act(Click(x=1, y=1))
    await workspace.act(Click(x=1, y=1))

    assert len(display.starts) == 2
    assert backend.display.starts == 2


async def test_a_template_without_a_working_x_server_says_why(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    display.can_start = False
    workspace = Workspace(E2BSandboxBackend(template='base', desktop=True))

    with pytest.raises(WorkspaceUnavailableError, match='it needs Xvfb, xdotool, xmodmap, scrot and setsid') as error:
        await workspace.screenshot()
    assert 'Fatal server error' in str(error.value)


async def test_a_sandbox_without_a_desktop_has_no_display(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()

    assert Workspace(backend).has_display is False
    with pytest.raises(UserError, match='has no display'):
        await Workspace(backend).screenshot()
    assert fake_e2b.create_calls == []


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='e2b', id='sbx-desktop')

    assert Workspace(E2BSandbox(desktop=True).backend(ref)).has_display is True
    assert Workspace(E2BSandbox().backend(ref)).has_display is False


async def test_a_desktop_capability_gives_a_new_run_a_display(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[E2BSandbox(desktop=True)])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await ctx.workspace.display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
    assert [call.template for call in fake_e2b.create_calls] == ['desktop']
