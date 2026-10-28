"""Tests for `DaytonaSandbox(desktop=True)`, against the fake Daytona SDK."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import anyio
import pytest
from daytona import DaytonaBadRequestError, DaytonaServiceUnavailableError

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
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from .fake_daytona import FakeDaytona, FakeSandbox

pytestmark = pytest.mark.anyio(backends=['asyncio'])


async def _started(fake_daytona: FakeDaytona) -> tuple[DaytonaSandboxBackend, FakeSandbox]:
    backend = DaytonaSandboxBackend(desktop=True)
    await backend.display_size()
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

    await backend.act(action)

    assert sandbox.desktop_calls == calls


@pytest.mark.parametrize(
    'action',
    [
        pytest.param(Click(button='back'), id='side-button'),
        pytest.param(TypeText(text='done 😀'), id='emoji'),
        pytest.param(MouseDown(button='forward'), id='side-button-down'),
        pytest.param(PressKeys(keys=('f13',)), id='key-that-crashes-the-desktop'),
        pytest.param(PressKeys(keys=('KP_Enter',)), id='keysym'),
        pytest.param(HoldKeys(keys=('é',), seconds=1), id='non-ascii-key'),
    ],
)
async def test_what_daytona_cannot_do_is_refused_before_anything_runs(
    fake_daytona: FakeDaytona, action: ComputerAction
) -> None:
    backend, sandbox = await _started(fake_daytona)

    with pytest.raises(WorkspaceUnsupportedActionError):
        await backend.act(action)
    assert sandbox.desktop_calls == []


async def test_an_unknown_action_type_is_refused(fake_daytona: FakeDaytona) -> None:
    backend, _ = await _started(fake_daytona)

    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot perform'):
        await backend.act(object())  # pyright: ignore[reportArgumentType]


async def test_a_cancelled_hold_lets_go_of_what_it_pressed(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(backend.act, HoldKeys(keys=('ctrl', 'a'), seconds=30))
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
            raise DaytonaServiceUnavailableError('connection is shut down', status_code=503)
        await up(x, y, button, request_timeout)

    sandbox.computer_use.up = up_failing_once
    with pytest.raises(DaytonaServiceUnavailableError):
        await backend.act(PressKeys(keys=('ctrl', 'a', 'b')))

    # `b` failed to come up, so it is tried again along with the keys still down.
    assert sandbox.desktop_calls == ['keydown ctrl', 'keydown a', 'keydown b', 'keyup b', 'keyup a', 'keyup ctrl']


async def test_a_failed_drag_releases_the_button_and_keys(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def fail_on_move(x: int, y: int, request_timeout: float | None = None) -> None:
        raise DaytonaServiceUnavailableError('connection is shut down', status_code=503)

    sandbox.computer_use.move = fail_on_move
    with pytest.raises(DaytonaServiceUnavailableError):
        await backend.act(Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('shift',)))

    assert sandbox.desktop_calls == ['keydown shift', 'mousedown 1 1 left', 'mouseup None None left', 'keyup shift']


async def test_the_live_view_url_is_a_signed_link_to_the_desktops_viewer(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    url = await backend.live_view_url(expires_in=60)

    assert url == 'https://6080-signed.proxy.example/vnc.html?autoconnect=true&resize=scale'
    assert sandbox.desktop_calls == ['signed_preview 6080 60']


async def test_screenshots_zoom_and_pointer(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    assert await backend.screenshot() == sandbox.computer_use.png
    assert await backend.zoom(Region(left=10, top=20, right=110, bottom=70)) == sandbox.computer_use.png
    assert await backend.cursor_position() == Point(x=7, y=8)
    assert sandbox.desktop_calls == ['take_full_screen', 'take_region 10 20 100 50', 'get_position']


async def test_replies_the_desktop_cannot_use_are_workspace_errors(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    sandbox.computer_use.png = b'GIF89a'
    with pytest.raises(WorkspaceError, match='did not return a PNG'):
        await backend.screenshot()
    sandbox.computer_use.displays = []
    with pytest.raises(WorkspaceError, match='no display size'):
        await backend.display_size()
    sandbox.computer_use.position = SimpleNamespace(x=None, y=None)
    with pytest.raises(WorkspaceError, match='no pointer position'):
        await backend.cursor_position()


async def test_the_desktop_starts_once_however_many_first_uses_race(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend(desktop=True)
    workspace = Workspace(backend)
    await backend.get_sandbox()
    (sandbox,) = fake_daytona.sandboxes
    sandbox.computer_use.start_gate = gate = asyncio.Event()

    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(workspace.display_size)
        await anyio.wait_all_tasks_blocked()
        gate.set()

    assert sandbox.computer_use.start_calls == 1
    assert await workspace.display_size() == (1280, 800)
    assert sandbox.computer_use.start_calls == 1


async def test_a_failed_desktop_call_starts_it_again_next_time(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    sandbox.computer_use.error = DaytonaServiceUnavailableError('connection is shut down', status_code=503)
    with pytest.raises(DaytonaServiceUnavailableError):
        await backend.act(Click(x=1, y=1))
    await backend.act(Click(x=1, y=1))

    assert sandbox.computer_use.start_calls == 2


async def test_a_failed_start_is_retried(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend(desktop=True)
    await backend.get_sandbox()
    (sandbox,) = fake_daytona.sandboxes

    sandbox.computer_use.error = DaytonaBadRequestError('no desktop packages', status_code=400)
    with pytest.raises(WorkspaceError, match='The Daytona desktop failed: no desktop packages'):
        await backend.screenshot()
    await backend.screenshot()

    assert sandbox.computer_use.start_calls == 2


async def test_a_refused_action_does_not_restart_the_desktop(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    with pytest.raises(WorkspaceUnsupportedActionError):
        await backend.act(PressKeys(keys=('numpad_enter',)))
    await backend.act(Click(x=1, y=1))

    assert sandbox.computer_use.start_calls == 1


async def test_a_desktop_size_is_set_on_a_new_sandbox(fake_daytona: FakeDaytona) -> None:
    await DaytonaSandboxBackend(desktop=(1440, 900), env={'A': '1'}).get_sandbox()
    await DaytonaSandboxBackend(desktop=True).get_sandbox()
    await DaytonaSandboxBackend().get_sandbox()

    assert [params.env_vars for params in fake_daytona.create_params] == [
        {'DISPLAY': ':0', 'A': '1', 'VNC_RESOLUTION': '1440x900'},
        {'DISPLAY': ':0', 'VNC_RESOLUTION': '1280x800'},
        None,
    ]


async def test_a_sandbox_without_a_desktop_has_no_display(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()

    assert Workspace(backend).has_display is False
    with pytest.raises(UserError, match='has no display: create it with `desktop=True`'):
        await backend.screenshot()
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
    assert fake_daytona.create_params[0].env_vars == {'DISPLAY': ':0', 'VNC_RESOLUTION': '1280x800'}
