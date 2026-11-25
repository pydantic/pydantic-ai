"""Tests for `BoxdSandbox(desktop=True)`, against the fake boxd SDK's computer API."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import anyio
import pytest
from boxd import APIConnectionError, NotFoundError, PermissionDeniedError

from pydantic_ai import Agent
from pydantic_ai.computer_use import (
    Click,
    ComputerReturnPart,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.boxd_sandbox._desktop import INSTALL_PANEL, START_PANEL, WRITE_PANEL_SETTINGS
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction, SupportsLiveView, UnsupportedActionError
from tests._timeouts import READINESS_WAIT_TIMEOUT

from ..computer_use.display_fakes import full_display, look_click_then_finish
from .fake_boxd import APIStatusError, FakeBoxd

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host')


async def _started(host: FakeBoxd) -> BoxdSandboxBackend:
    backend = BoxdSandboxBackend(desktop=True)
    # Starts the desktop.
    await full_display(backend).screenshot()
    host.desktop.calls.clear()
    return backend


def _path(points: int) -> tuple[Point, ...]:
    # `y` differs from `x`, so swapped coordinates would fail the test.
    return tuple(Point(x=n, y=2 * n) for n in range(points))


@pytest.mark.parametrize(
    ('action', 'calls'),
    [
        pytest.param(
            Click(x=3, y=4, button='middle', count=3, modifiers=('super', 'shift')),
            ["click 3 4 middle count=3 modifiers='super+shift'"],
            id='click',
        ),
        pytest.param(Click(), ["click pointer left count=1 modifiers=''"], id='click-at-pointer'),
        pytest.param(MouseMove(x=5, y=6), ['mouse_move 5 6'], id='mouse_move'),
        pytest.param(
            Drag(path=(Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)), modifiers=('ctrl',)),
            ["drag 1 2 3 4 5 6 modifiers='ctrl'"],
            id='drag',
        ),
        pytest.param(
            Drag(path=_path(101)),
            # Too long for one call, so the button is pressed, moved along the path and released in separate calls.
            ['mouse_down 0 0 left', *(f'mouse_move {n} {2 * n}' for n in range(1, 101)), 'mouse_up 100 200 left'],
            id='long-drag',
        ),
        pytest.param(MouseDown(button='right'), ['mouse_down pointer right'], id='mouse-down'),
        pytest.param(MouseUp(), ['mouse_up pointer left'], id='mouse-up'),
        pytest.param(
            Scroll(x=1, y=2, direction='left', amount=5, modifiers=('ctrl',)),
            ["scroll 1 2 dx=-5 dy=0 modifiers='ctrl'"],
            id='scroll',
        ),
        pytest.param(
            Scroll(direction='down', amount=3), ["scroll pointer dx=0 dy=3 modifiers=''"], id='scroll-at-pointer'
        ),
        pytest.param(
            Scroll(x=1, y=2, direction='up', amount=201),
            [f"scroll 1 2 dx=0 dy={clicks} modifiers=''" for clicks in (-100, -100, -1)],
            id='scroll-in-parts',
        ),
        pytest.param(TypeText(text='héllo 😀\n'), ["type 8 'héllo 😀\\n'"], id='type-any-unicode'),
        pytest.param(
            TypeText(text='a' * 20_001),
            [f"type 10000 '{'a' * 20}'", f"type 10000 '{'a' * 20}'", "type 1 'a'"],
            id='type-in-parts',
        ),
        pytest.param(
            PressKeys(keys=('ctrl', 's'), repeat=201),
            [
                "key 'ctrl+s' repeat=100 hold_ms=0",
                "key 'ctrl+s' repeat=100 hold_ms=0",
                "key 'ctrl+s' repeat=1 hold_ms=0",
            ],
            id='combination-repeated-in-parts',
        ),
        pytest.param(
            PressKeys(keys=('\n', '\t')), ["key 'Return+Tab' repeat=1 hold_ms=0"], id='newline-and-tab-as-keys'
        ),
        pytest.param(
            PressKeys(keys=('ctrl', '+', 'é', 'pause', 'f12')),
            ["key 'ctrl+U002B+U00E9+Pause+f12' repeat=1 hold_ms=0"],
            id='keysyms',
        ),
        pytest.param(HoldKeys(keys=('shift', 'a'), seconds=1.5), ["key 'shift+a' repeat=0 hold_ms=1500"], id='hold'),
        # boxd itself refuses a hold over 5 minutes, with the error `hold_ms is at most 300000`.
        pytest.param(HoldKeys(keys=('a',), seconds=301), ["key 'a' repeat=0 hold_ms=301000"], id='long-hold'),
        # boxd treats `hold_ms=0` as a plain press, so the shortest hold is 1 ms.
        pytest.param(HoldKeys(keys=('a',), seconds=0.0001), ["key 'a' repeat=0 hold_ms=1"], id='short-hold'),
    ],
)
async def test_each_action_maps_onto_boxd_calls(host: FakeBoxd, action: DisplayAction, calls: list[str]) -> None:
    backend = await _started(host)

    await full_display(backend).act(action)

    assert host.desktop.calls == calls


@pytest.mark.parametrize(
    'action',
    [
        pytest.param(Drag(path=_path(101), modifiers=('shift',)), id='long-drag-holding-a-key'),
        pytest.param(object(), id='unknown-action-type'),
    ],
)
async def test_what_boxd_cannot_do_is_refused_before_anything_runs(host: FakeBoxd, action: DisplayAction) -> None:
    backend = await _started(host)

    with pytest.raises(UnsupportedActionError):
        await full_display(backend).act(action)
    assert host.desktop.calls == []


async def test_a_failed_long_drag_releases_the_button(host: FakeBoxd) -> None:
    backend = await _started(host)
    # The press succeeds. The first move fails.
    host.desktop.errors = [None, APIStatusError('the desktop went away', code='internal', grpc_code=13)]

    with pytest.raises(APIStatusError, match='the desktop went away'):
        await full_display(backend).act(Drag(path=_path(101)))

    assert host.desktop.calls == ['mouse_down 0 0 left', 'mouse_move 1 2', 'mouse_up pointer left']


async def test_a_long_drag_whose_release_fails_says_the_button_may_still_be_held(host: FakeBoxd) -> None:
    backend = await _started(host)
    # The press succeeds. The first move fails, and so does the release after it.
    host.desktop.errors = [
        None,
        APIStatusError('move refused', code='invalid_argument', grpc_code=3),
        APIStatusError('release connection lost', code='unavailable', grpc_code=14),
    ]

    with pytest.raises(
        WorkspaceError,
        match=r'^A long drag failed \(move refused\), and releasing the left mouse button failed too, so it may still be held\.$',
    ):
        await full_display(backend).act(Drag(path=_path(101)))

    assert host.desktop.calls == ['mouse_down 0 0 left', 'mouse_move 1 2', 'mouse_up pointer left']


async def test_rejected_credentials_during_a_long_drag_end_the_run_when_the_release_fails(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.errors = [
        None,
        PermissionDeniedError('not a member of this organization'),
        APIStatusError('release connection lost', code='unavailable', grpc_code=14),
    ]

    with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials') as error:
        await full_display(backend).act(Drag(path=_path(101)))
    assert 'not a member' not in str(error.value)


async def test_a_machine_deleted_during_a_failed_long_drag_ends_the_run(host: FakeBoxd) -> None:
    """A desktop failure on a machine that is gone raises `WorkspaceUnavailableError`, so the model doesn't retry."""
    backend = await _started(host)
    host.desktop.errors = [
        None,
        APIStatusError('move refused', code='invalid_argument', grpc_code=3),
        APIStatusError('release connection lost', code='unavailable', grpc_code=14),
    ]
    # After the failure, the lookup finds that the machine is gone.
    host.get_errors.append(NotFoundError('VM is destroyed', code='not_found', grpc_code=5))

    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await full_display(backend).act(Drag(path=_path(101)))


async def test_a_cancelled_long_drag_releases_the_button(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.hang_on = 'mouse_move 50 100'

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(full_display(backend).act, Drag(path=_path(101)))
        await anyio.wait_all_tasks_blocked()  # the button is down and a move waits for boxd
        tasks.cancel_scope.cancel()

    assert host.desktop.calls == [
        'mouse_down 0 0 left',
        *(f'mouse_move {n} {2 * n}' for n in range(1, 51)),
        'mouse_up pointer left',
    ]


async def test_the_desktop_is_held_at_its_size_when_it_starts(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=(1440, 900))

    await full_display(backend).display_size()

    assert host.desktop.calls == ['set_screen_size 1440x900']


async def test_a_new_desktop_machine_installs_the_panel_from_its_spec(host: FakeBoxd) -> None:
    await BoxdSandboxBackend(desktop=True, vcpu=2).run(['true'])

    machine = host.only
    assert machine.create_kwargs['isolated'] is False
    assert machine.create_kwargs['spec'] == {'version': 1, 'isolated': True, 'size': '2x8', 'packages': [INSTALL_PANEL]}


async def test_the_panel_starts_on_each_connection_after_the_screen_is_held(host: FakeBoxd) -> None:
    backend = await _started(host)
    assert host.exec_commands.count(START_PANEL) == 1

    await backend.aclose()
    await full_display(backend).screenshot()

    assert host.exec_commands.count(START_PANEL) == 2


async def test_a_command_does_not_wait_for_the_panel_but_the_next_look_does(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=True)

    await backend.run(['true'])
    assert START_PANEL not in host.exec_commands

    await full_display(backend).screenshot()
    assert host.exec_commands.count(START_PANEL) == 1


@pytest.mark.parametrize('fails', ['exit', 'error'])
async def test_a_panel_that_does_not_start_leaves_the_screen_working(
    host: FakeBoxd, caplog: pytest.LogCaptureFixture, fails: str
) -> None:
    if fails == 'exit':
        host.panel_exit_code = 1
    else:
        host.panel_error = APIConnectionError('the connection dropped before the reply arrived')
    backend = BoxdSandboxBackend(desktop=True)

    with caplog.at_level(logging.WARNING):
        assert await full_display(backend).screenshot() == host.desktop.png

    assert caplog.messages == [
        'The panel did not start on boxd machine machine-1; see /tmp/pydantic-ai-desktop.log there'
    ]


@pytest.mark.parametrize('script', [START_PANEL, INSTALL_PANEL, WRITE_PANEL_SETTINGS])
def test_the_panel_scripts_parse(script: str) -> None:
    # The fake doesn't run them, so a shell syntax error would show only on a real machine.
    subprocess.run(['/bin/sh', '-n', '-c', script], check=True)


def test_the_scripts_the_panel_buttons_run_parse(tmp_path: Path) -> None:
    subprocess.run(
        ['/bin/sh', '-c', WRITE_PANEL_SETTINGS], check=True, env={'HOME': str(tmp_path), 'PATH': '/usr/bin:/bin'}
    )

    for button in ('terminal', 'browser'):
        subprocess.run(['/bin/sh', '-n', str(tmp_path / '.config/pydantic-ai-desktop' / button)], check=True)


# Stands in for xterm, agent-browser, xdotool and xwininfo on a desktop with one window. As on boxd's desktop, the
# window is unviewable until the next computer call shows it, and showing it puts it back at the top left. Here that
# call comes straight after the first look that finds the window unviewable. xterm runs until it has the keyboard.
_FAKE_DESKTOP = """#!/bin/sh
cd "$DESKTOP_STATE"
shown() { [ -e shown ] || { touch shown; echo 0 0 >position; false; }; }
case ${0##*/} in
  agent-browser) exit;;
  xterm) until [ -e focused ]; do sleep 0.01; done; exit;;
  xwininfo) shown && echo IsViewable; exit 0;;
esac
case $1 in
  search) echo 1; exit;;
  getdisplaygeometry) echo WIDTH=1280 HEIGHT=800; exit;;
esac
while [ $# -gt 0 ]; do
  case $1 in
    windowmove) echo "$3 $4" >position;;
    windowsize) echo "$3 $4" >size;;
    windowfocus) shown || exit 1; touch focused; shift 2; continue;;
  esac
  shift 4
done
"""


@pytest.mark.parametrize('button', ['terminal', 'browser'])
def test_each_panel_button_opens_its_window_below_the_panel_with_the_keyboard(tmp_path: Path, button: str) -> None:
    # The fake boxd doesn't run the panel, so a fake desktop runs the button's script, and the live test
    # `test_the_panel_opens_a_terminal_below_it_that_has_the_keyboard` validates what it encodes.
    bin_dir, state = tmp_path / 'bin', tmp_path / 'state'
    bin_dir.mkdir()
    state.mkdir()
    (bin_dir / 'fake-desktop').write_text(_FAKE_DESKTOP)
    (bin_dir / 'fake-desktop').chmod(0o755)
    for program in ('xterm', 'agent-browser', 'xdotool', 'xwininfo'):
        (bin_dir / program).symlink_to('fake-desktop')
    env = {'HOME': str(tmp_path), 'PATH': f'{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin', 'DESKTOP_STATE': str(state)}
    subprocess.run(['/bin/sh', '-c', WRITE_PANEL_SETTINGS], check=True, env=env, timeout=READINESS_WAIT_TIMEOUT)

    # Bounded, so a script that never gives the window the keyboard fails here instead of hanging the run.
    subprocess.run(
        [str(tmp_path / '.config/pydantic-ai-desktop' / button)], check=True, env=env, timeout=READINESS_WAIT_TIMEOUT
    )

    assert {file.name: file.read_text() for file in state.iterdir()} == {
        'shown': '',
        'position': '0 40\n',
        'size': '1280 760\n',
        'focused': '',
    }


@pytest.mark.parametrize(
    ('has_browser', 'buttons', 'items'),
    [
        (True, ['button_text = Terminal', 'button_text = Chromium'], 'panel_items = PPT'),
        (False, ['button_text = Terminal'], 'panel_items = PT'),
    ],
)
def test_the_panel_has_a_browser_button_only_when_the_machine_has_agent_browser(
    tmp_path: Path, has_browser: bool, buttons: list[str], items: str
) -> None:
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    if has_browser:
        (bin_dir / 'agent-browser').write_text('#!/bin/sh\n')
        (bin_dir / 'agent-browser').chmod(0o755)
    env = {'HOME': str(tmp_path), 'PATH': f'{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin'}

    subprocess.run(['/bin/sh', '-c', WRITE_PANEL_SETTINGS], check=True, env=env)

    settings = (tmp_path / '.config' / 'pydantic-ai-desktop' / 'tint2rc').read_text().splitlines()
    assert [line for line in settings if line.startswith('button_text')] == buttons
    assert items in settings


async def test_looks_come_from_the_desktop(host: FakeBoxd) -> None:
    backend = await _started(host)
    display = full_display(backend)
    # The size boxd reports, not the size that was asked for.
    host.desktop.size = (1000, 700)

    assert await display.screenshot() == host.desktop.png
    assert await display.zoom(Region(left=10, top=20, right=110, bottom=70)) == host.desktop.png
    assert await display.cursor_position() == Point(x=7, y=8)
    assert await display.display_size() == (1000, 700)
    assert host.desktop.calls == ['zoom 10 20 100 50']


async def test_a_reply_that_is_not_a_png_is_a_workspace_error(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.png = b'GIF89a'

    with pytest.raises(WorkspaceError, match='did not return a PNG'):
        await full_display(backend).screenshot()


async def test_a_value_boxd_refuses_is_a_workspace_error(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.errors.append(APIStatusError('unknown key', code='invalid_argument', grpc_code=3))

    with pytest.raises(WorkspaceError, match='The boxd desktop failed: unknown key'):
        await full_display(backend).act(PressKeys(keys=('a',)))


async def test_a_deleted_machine_ends_the_run(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.only.status = 'destroyed'

    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await full_display(backend).act(Click(x=1, y=1))


async def test_after_aclose_the_desktop_is_held_again_before_any_input(host: FakeBoxd) -> None:
    """The machine, and its desktop, may restart before a later run connects to it again."""
    backend = await _started(host)
    await backend.aclose()

    await full_display(backend).act(Click(x=1, y=1))

    assert host.desktop.calls == ['set_screen_size 1280x800', "click 1 1 left count=1 modifiers=''"]


async def test_a_shell_command_finds_the_desktop_running(host: FakeBoxd) -> None:
    # The shell command gets `DISPLAY`, so a program it starts opens on the screen, even before the first screenshot.
    backend = BoxdSandboxBackend(desktop=True, env={'A': '1'})

    await backend.run(['true'])

    assert host.desktop.calls == ['set_screen_size 1280x800']
    assert [line for line in host.lines if line.startswith('exec ')][
        -1
    ] == 'exec env -- DISPLAY=:100 A=1 true </dev/null'


def test_a_machine_without_a_desktop_has_no_display() -> None:
    assert BoxdSandboxBackend().display is None


def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        BoxdSandbox(desktop=(0, 800))


async def test_a_desktop_machine_runs_computer_use(host: FakeBoxd) -> None:
    sandbox = BoxdSandbox(desktop=True)
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    result = await agent.run('Click.')

    *_, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == host.desktop.png
    assert host.desktop.calls == ['set_screen_size 1280x800', "click 3 4 left count=1 modifiers=''"]

    # A backend that connects with the run's ref reaches the same desktop.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == host.desktop.png


async def test_the_live_view_is_the_machines_desktop_page_in_the_console(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=True)
    screen = backend.display
    assert isinstance(screen, SupportsLiveView)

    assert await screen.live_view_url() == 'https://app.boxd.example/desktop/machine'
    # The screen was set up first, so the page shows the desktop at its size.
    assert any(call.startswith('set_screen_size') for call in host.desktop.calls)

    assert backend.ref is not None
    host.machines[backend.ref.id].desktop_url = ''
    with pytest.raises(WorkspaceError, match=r"^boxd did not give a desktop URL for the machine '.+'\.$"):
        await screen.live_view_url()
