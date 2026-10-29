"""Tests for `DaytonaSandbox(desktop=True)`, against the fake Daytona SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from daytona import DaytonaInternalServerError

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    Click,
    ComputerAction,
    Drag,
    HoldKeys,
    MouseDown,
    MouseUp,
    MoveMouse,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .fake_daytona import FakeDaytona, FakeSandbox

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def _started(fake_daytona: FakeDaytona) -> tuple[DaytonaSandboxBackend, FakeSandbox]:
    backend = DaytonaSandboxBackend(desktop=True)
    await Workspace(backend).display_size()
    (sandbox,) = fake_daytona.sandboxes
    sandbox.desktop_calls.clear()
    return backend, sandbox


@pytest.mark.parametrize(
    ('action', 'calls'),
    [
        pytest.param(
            Click(x=3, y=4, button='right', count=3, modifiers=('super', 'shift')),
            ["click 3 4 right clicks=3 modifiers=['cmd', 'shift']"],
            id='click',
        ),
        pytest.param(Click(), ['get_position', 'click 7 8 left clicks=1 modifiers=[]'], id='click-at-pointer'),
        pytest.param(MoveMouse(x=5, y=6), ['move 5 6'], id='move'),
        pytest.param(
            MoveMouse(x=5, y=6, modifiers=('alt',)), ['keydown alt', 'move 5 6', 'keyup alt'], id='move-with-modifier'
        ),
        pytest.param(
            Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('ctrl',)),
            # Each leg is moved along in at least two steps, so a web page's drag-and-drop registers it.
            ['keydown ctrl', 'mousedown 1 1 left', 'move 1 1', 'move 2 2', 'move 2 2', 'move 3 3', 'mouseup 3 3 left']
            + ['keyup ctrl'],
            id='drag',
        ),
        pytest.param(MouseDown(button='middle'), ['mousedown None None middle'], id='mouse-down'),
        pytest.param(MouseUp(), ['mouseup None None left'], id='mouse-up'),
        pytest.param(
            Scroll(x=1, y=2, direction='left', amount=5, modifiers=('ctrl',)),
            ["scroll 1 2 left 5 modifiers=['ctrl']"],
            id='scroll',
        ),
        pytest.param(
            Scroll(direction='down'), ['get_position', 'scroll 7 8 down 3 modifiers=[]'], id='scroll-at-pointer'
        ),
        pytest.param(TypeText(text='héllo'), ['type héllo'], id='type'),
        pytest.param(
            PressKeys(keys=('ctrl+s',), repeat=2),
            ["press s modifiers=['ctrl']", "press s modifiers=['ctrl']"],
            id='combination-repeated',
        ),
        pytest.param(PressKeys(keys=('Return',)), ['press enter modifiers=[]'], id='alias'),
        pytest.param(
            PressKeys(keys=('ctrl', 'KP_Enter', 'numpad_multiply')),
            ['keydown ctrl', 'keydown num_enter', 'keydown num_asterisk', 'keyup num_asterisk', 'keyup num_enter']
            + ['keyup ctrl'],
            id='renamed-keys',
        ),
        pytest.param(
            PressKeys(keys=('shift', 'ctrl', '!')), ["press 1 modifiers=['shift', 'ctrl']"], id='shifted-symbol'
        ),
        pytest.param(PressKeys(keys=('super',)), ['keydown cmd', 'keyup cmd'], id='modifier-alone'),
        pytest.param(
            PressKeys(keys=('a', 'b')), ['keydown a', 'keydown b', 'keyup b', 'keyup a'], id='two-keys-together'
        ),
        pytest.param(
            HoldKeys(keys=('shift', 'a'), seconds=0.01),
            ['keydown shift', 'keydown a', 'keyup a', 'keyup shift'],
            id='hold',
        ),
    ],
)
async def test_each_action_maps_onto_daytona_calls(
    fake_daytona: FakeDaytona, action: ComputerAction, calls: list[str]
) -> None:
    backend, sandbox = await _started(fake_daytona)

    await Workspace(backend).act(action)

    assert sandbox.desktop_calls == calls


@pytest.mark.parametrize(
    'action',
    [
        pytest.param(Click(button='back'), id='side-button'),
        pytest.param(TypeText(text='done 😀'), id='emoji'),
        pytest.param(MouseDown(button='forward'), id='side-button-down'),
        pytest.param(PressKeys(keys=('f13',)), id='key-that-crashes-the-desktop'),
        pytest.param(PressKeys(keys=('printscreen',)), id='key-without-a-daytona-name'),
        pytest.param(HoldKeys(keys=('é',), seconds=1), id='non-ascii-key'),
        pytest.param(object(), id='unknown-action-type'),
    ],
)
async def test_what_daytona_cannot_do_is_refused_before_anything_runs(
    fake_daytona: FakeDaytona, action: ComputerAction
) -> None:
    backend, sandbox = await _started(fake_daytona)

    with pytest.raises(WorkspaceUnsupportedActionError):
        await Workspace(backend).act(action)
    assert sandbox.desktop_calls == []


async def test_a_cancelled_hold_lets_go_of_what_it_pressed(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(Workspace(backend).act, HoldKeys(keys=('ctrl', 'a'), seconds=30))
        await anyio.wait_all_tasks_blocked()  # holding, in its sleep
        tasks.cancel_scope.cancel()

    assert sandbox.desktop_calls == ['keydown ctrl', 'keydown a', 'keyup a', 'keyup ctrl']


async def test_a_failed_release_still_lets_go_of_the_other_keys(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    up = sandbox.computer_use.up
    failed: list[str] = []

    async def up_failing_once(
        x: Any = None, y: int | None = None, button: str = 'left', request_timeout: float | None = None
    ) -> None:
        if x == 'b' and not failed:
            failed.append(x)
            raise DaytonaInternalServerError('unexpected EOF', status_code=500)
        await up(x, y, button, request_timeout)

    sandbox.computer_use.up = up_failing_once
    with pytest.raises(WorkspaceError, match='unexpected EOF'):
        await Workspace(backend).act(PressKeys(keys=('ctrl', 'a', 'b')))

    # `b` failed to come up, so it is tried again along with the keys still down.
    assert sandbox.desktop_calls == [
        'keydown ctrl',
        'keydown a',
        'keydown b',
        'keyup b',
        'keyup a',
        'keyup ctrl',
    ]


async def test_a_failed_drag_releases_the_button_and_keys(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def fail_on_move(x: int, y: int, request_timeout: float | None = None) -> None:
        raise DaytonaInternalServerError('unexpected EOF', status_code=500)

    sandbox.computer_use.move = fail_on_move
    with pytest.raises(WorkspaceError, match='unexpected EOF'):
        await Workspace(backend).act(
            Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('shift',))
        )

    assert sandbox.desktop_calls == [
        'keydown shift',
        'mousedown 1 1 left',
        'mouseup None None left',
        'keyup shift',
    ]


async def test_the_live_view_url_is_a_signed_link_to_the_desktops_viewer(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    url = await backend.live_view_url(expires_in=60)

    assert url == 'https://6080-signed.proxy.example/vnc.html?autoconnect=true&resize=scale'
    assert sandbox.desktop_calls == ['signed_preview 6080 60']


async def test_looks_come_from_the_desktop(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    workspace = Workspace(backend)

    assert await workspace.screenshot() == sandbox.computer_use.png
    assert await workspace.zoom(Region(left=10, top=20, right=110, bottom=70)) == sandbox.computer_use.png
    assert await workspace.cursor_position() == Point(x=7, y=8)
    # Scaled by the ratio that fits; below a tenth, the least Daytona scales by, it would come back larger than asked.
    assert await workspace.scaled_screenshot(640, 400) == sandbox.computer_use.png
    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot scale'):
        await workspace.scaled_screenshot(100, 62)
    assert sandbox.desktop_calls == [
        'take_full_screen',
        'take_region 10 20 100 50',
        'get_position',
        'get_info',
        'take_compressed png 0.5',
        'get_info',
    ]


async def test_replies_the_desktop_cannot_use_are_workspace_errors(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    sandbox.computer_use.png = b'GIF89a'
    with pytest.raises(WorkspaceError, match='did not return a PNG'):
        await Workspace(backend).screenshot()
    sandbox.computer_use.displays = []
    with pytest.raises(WorkspaceError, match='no display size'):
        await Workspace(backend).display_size()
    sandbox.computer_use.position = SimpleNamespace(x=None, y=None)
    with pytest.raises(WorkspaceError, match='no pointer position'):
        await Workspace(backend).cursor_position()


async def test_an_action_after_the_sandbox_restarted_starts_the_desktop_before_any_input(
    fake_daytona: FakeDaytona,
) -> None:
    backend, sandbox = await _started(fake_daytona)
    assert backend.display is not None
    await Workspace(backend).screenshot()
    starts = backend.display.starts

    sandbox.computer_use.stop_with_sandbox()
    with pytest.raises(WorkspaceError):
        await Workspace(backend).act(Click(x=1, y=1))
    # The next action starts the desktop again, and doesn't run on the new screen.
    with pytest.raises(WorkspaceError, match='The display restarted before this action'):
        await Workspace(backend).act(Click(x=1, y=1))
    await Workspace(backend).act(Click(x=1, y=1))

    assert not sandbox.computer_use.broken
    assert sandbox.desktop_calls[-2:] == ['start', 'click 1 1 left clicks=1 modifiers=[]']
    assert backend.display.starts == starts + 1


async def test_a_desktop_that_stopped_answering_ends_the_run(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    sandbox.computer_use.broken = True

    with pytest.raises(WorkspaceUnavailableError, match='stopping and starting the sandbox brings it back'):
        await Workspace(backend).screenshot()


async def test_backends_on_one_sandbox_name_one_screen(fake_daytona: FakeDaytona) -> None:
    first, other = DaytonaSandboxBackend(desktop=True), DaytonaSandboxBackend(desktop=True)
    await first.get_sandbox()
    await other.get_sandbox()
    assert first.ref is not None and first.display is not None and other.display is not None
    again = DaytonaSandbox(desktop=True).backend(first.ref).display
    assert again is not None
    assert first.display.screen_key == again.screen_key != other.display.screen_key


async def test_a_desktop_size_is_set_on_a_new_sandbox(fake_daytona: FakeDaytona) -> None:
    await DaytonaSandboxBackend(desktop=(1440, 900), env={'A': '1'}).get_sandbox()
    await DaytonaSandboxBackend(desktop=True).get_sandbox()
    await DaytonaSandboxBackend().get_sandbox()

    assert [params.env_vars for params in fake_daytona.create_params] == [
        {'DISPLAY': ':0', 'A': '1', 'VNC_RESOLUTION': '1440x900'},
        {'DISPLAY': ':0', 'VNC_RESOLUTION': '1280x800'},
        None,
    ]


@pytest.mark.parametrize('size', [(640, 400), (1280, 700)])
def test_a_desktop_smaller_than_daytona_starts_is_refused(size: tuple[int, int]) -> None:
    with pytest.raises(UserError, match="Daytona's desktop is at least 1024x768"):
        DaytonaSandboxBackend(desktop=size)


async def test_a_sandbox_without_a_desktop_has_no_display(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()

    assert Workspace(backend).has_display is False
    with pytest.raises(UserError, match='has no display'):
        await Workspace(backend).screenshot()
    assert fake_daytona.sandboxes == []


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='daytona', id='sb-1')

    assert Workspace(DaytonaSandbox(desktop=True).backend(ref)).has_display is True
    assert Workspace(DaytonaSandbox().backend(ref)).has_display is False


async def test_a_desktop_capability_gives_a_new_run_a_display(fake_daytona: FakeDaytona) -> None:
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[DaytonaSandbox(desktop=True)])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await ctx.workspace.display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
