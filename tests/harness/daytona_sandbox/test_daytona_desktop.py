"""Tests for `DaytonaSandbox(desktop=True)`, against the fake Daytona SDK."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from daytona import DaytonaInternalServerError, DaytonaTimeoutError

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.computer_use import (
    Click,
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
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerAction, UnsupportedActionError
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from ..display_fakes import screen
from .fake_daytona import FakeDaytona, FakeSandbox


async def _started(fake_daytona: FakeDaytona) -> tuple[DaytonaSandboxBackend, FakeSandbox]:
    backend = DaytonaSandboxBackend(desktop=True)
    await screen(backend).display_size()
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

    await screen(backend).act(action)

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

    with pytest.raises(UnsupportedActionError):
        await screen(backend).act(action)
    assert sandbox.desktop_calls == []


async def test_a_cancelled_hold_lets_go_of_what_it_pressed(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(screen(backend).act, HoldKeys(keys=('ctrl', 'a'), seconds=30))
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
        await screen(backend).act(PressKeys(keys=('ctrl', 'a', 'b')))

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
        await screen(backend).act(Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('shift',)))

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
    workspace = screen(backend)

    assert await workspace.screenshot() == sandbox.computer_use.png
    assert await workspace.zoom(Region(left=10, top=20, right=110, bottom=70)) == sandbox.computer_use.png
    assert await workspace.cursor_position() == Point(x=7, y=8)
    # Scaled by the ratio that fits; below a tenth, the least Daytona scales by, it would come back larger than asked.
    assert await workspace.scaled_screenshot(640, 400) == sandbox.computer_use.png
    with pytest.raises(UnsupportedActionError, match='cannot scale'):
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
        await screen(backend).screenshot()
    sandbox.computer_use.displays = []
    with pytest.raises(WorkspaceError, match='no display size'):
        await screen(backend).display_size()
    sandbox.computer_use.position = SimpleNamespace(x=None, y=None)
    with pytest.raises(WorkspaceError, match='no pointer position'):
        await screen(backend).cursor_position()


async def test_an_action_after_the_sandbox_restarted_starts_the_desktop_before_any_input(
    fake_daytona: FakeDaytona,
) -> None:
    backend, sandbox = await _started(fake_daytona)
    assert backend.display is not None
    await screen(backend).screenshot()

    sandbox.computer_use.stop_with_sandbox()
    with pytest.raises(WorkspaceError):
        await screen(backend).act(Click(x=1, y=1))
    # The next action starts the desktop again, and doesn't run on the new screen.
    with pytest.raises(WorkspaceError, match='The display was not running and started for this action'):
        await screen(backend).act(Click(x=1, y=1))
    await screen(backend).act(Click(x=1, y=1))

    assert not sandbox.computer_use.broken
    assert sandbox.desktop_calls[-2:] == ['start', 'click 1 1 left clicks=1 modifiers=[]']


async def test_actions_in_a_row_check_once_that_the_desktop_runs(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)
    checks = 0
    get_status = sandbox.computer_use.get_status

    async def counted(request_timeout: float | None = None) -> SimpleNamespace:
        nonlocal checks
        checks += 1
        return await get_status(request_timeout=request_timeout)

    monkeypatch.setattr(sandbox.computer_use, 'get_status', counted)
    await screen(backend).act(Click(x=1, y=1))
    await screen(backend).act(Click(x=2, y=2))

    assert checks == 1


async def test_a_desktop_that_does_not_answer_in_time_is_a_timeout(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def slow(request_timeout: float | None = None) -> SimpleNamespace:
        raise DaytonaTimeoutError('request timeout')

    monkeypatch.setattr(sandbox.computer_use, 'take_full_screen', slow)
    with pytest.raises(WorkspaceTimeoutError, match='The Daytona desktop did not answer in time'):
        await screen(backend).screenshot()


async def test_a_desktop_that_stopped_answering_ends_the_run(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    sandbox.computer_use.broken = True

    with pytest.raises(WorkspaceUnavailableError, match='stopping and starting the sandbox brings it back'):
        await screen(backend).screenshot()


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


def test_an_attached_sandbox_keeps_its_own_size_whatever_desktop_says() -> None:
    backend = DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id='sandbox-1'), desktop=(800, 600))
    assert backend.display is not None


async def test_a_sandbox_without_a_desktop_has_no_display(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()

    assert backend.display is None
    with pytest.raises(UserError, match='has no display: create it with `desktop=True`'):
        await backend.live_view_url()
    assert fake_daytona.sandboxes == []


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='daytona', id='sb-1')

    assert DaytonaSandbox(desktop=True).backend(ref).display is not None
    assert DaytonaSandbox().backend(ref).display is None


async def test_a_desktop_capability_gives_a_new_run_a_display(fake_daytona: FakeDaytona) -> None:
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[DaytonaSandbox(desktop=True)])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await screen(ctx.workspace.backend).display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
