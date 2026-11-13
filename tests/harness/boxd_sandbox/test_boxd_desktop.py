"""Tests for `BoxdSandbox(desktop=True)`, against the fake boxd SDK's computer API."""

from __future__ import annotations

import anyio
import pytest
from boxd import NotFoundError

from pydantic_ai import Agent, UserError
from pydantic_ai.computer_use import (
    Click,
    ComputerReturnPart,
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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction, UnsupportedActionError

from ..display_fakes import screen
from .fake_boxd import APIStatusError, FakeBoxd


async def _started(host: FakeBoxd, desktop: bool | tuple[int, int] = True) -> BoxdSandboxBackend:
    backend = BoxdSandboxBackend(desktop=desktop)
    # Starts the desktop.
    await screen(backend).screenshot()
    host.desktop.calls.clear()
    return backend


def _path(points: int) -> tuple[Point, ...]:
    return tuple(Point(x=n, y=n) for n in range(points))


@pytest.mark.parametrize(
    ('action', 'calls'),
    [
        pytest.param(
            Click(x=3, y=4, button='back', count=3, modifiers=('super', 'shift')),
            ["click 3 4 back count=3 modifiers='super+shift'"],
            id='click',
        ),
        pytest.param(Click(), ["click pointer left count=1 modifiers=''"], id='click-at-pointer'),
        pytest.param(MoveMouse(x=5, y=6), ['mouse_move 5 6'], id='move'),
        pytest.param(
            Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('ctrl',)),
            ["drag 3 points from 1 1 modifiers='ctrl'"],
            id='drag',
        ),
        pytest.param(
            Drag(path=_path(101)),
            # Too long for one call: pressed, moved along the path, released.
            ['mouse_down 0 0 left', *(f'mouse_move {n} {n}' for n in range(1, 101)), 'mouse_up 100 100 left'],
            id='long-drag',
        ),
        pytest.param(MouseDown(button='forward'), ['mouse_down pointer forward'], id='mouse-down'),
        pytest.param(MouseUp(), ['mouse_up pointer left'], id='mouse-up'),
        pytest.param(
            Scroll(x=1, y=2, direction='left', amount=5, modifiers=('ctrl',)),
            ["scroll 1 2 dx=-5 dy=0 modifiers='ctrl'"],
            id='scroll',
        ),
        pytest.param(Scroll(direction='down'), ["scroll pointer dx=0 dy=3 modifiers=''"], id='scroll-at-pointer'),
        pytest.param(
            Scroll(x=1, y=2, direction='up', amount=201),
            [f"scroll 1 2 dx=0 dy={clicks} modifiers=''" for clicks in (-100, -100, -1)],
            id='scroll-in-parts',
        ),
        pytest.param(TypeText(text='héllo 😀\n'), ["type 8 'héllo 😀\\n'"], id='type-any-unicode'),
        pytest.param(
            TypeText(text='a' * 4_001),
            [f"type 2000 '{'a' * 20}'", f"type 2000 '{'a' * 20}'", "type 1 'a'"],
            id='type-in-parts',
        ),
        pytest.param(
            PressKeys(keys=('ctrl+s',), repeat=201),
            [
                "key 'ctrl+s' repeat=100 hold_ms=0",
                "key 'ctrl+s' repeat=100 hold_ms=0",
                "key 'ctrl+s' repeat=1 hold_ms=0",
            ],
            id='combination-repeated-in-parts',
        ),
        pytest.param(PressKeys(keys=('Return',)), ["key 'enter' repeat=1 hold_ms=0"], id='alias'),
        pytest.param(
            PressKeys(keys=('\n', '\t')), ["key 'Return+Tab' repeat=1 hold_ms=0"], id='newline-and-tab-as-keys'
        ),
        pytest.param(
            PressKeys(keys=('ctrl', '+', 'é', 'KP_Enter', 'xf86audiomute', 'f24')),
            ["key 'ctrl+U002B+U00E9+KP_Enter+XF86AudioMute+f24' repeat=1 hold_ms=0"],
            id='keysyms',
        ),
        pytest.param(HoldKeys(keys=('shift', 'a'), seconds=1.5), ["key 'shift+a' repeat=0 hold_ms=1500"], id='hold'),
        # boxd reads `hold_ms=0` as a plain press.
        pytest.param(HoldKeys(keys=('a',), seconds=0.0001), ["key 'a' repeat=0 hold_ms=1"], id='short-hold'),
    ],
)
async def test_each_action_maps_onto_boxd_calls(host: FakeBoxd, action: DisplayAction, calls: list[str]) -> None:
    backend = await _started(host)

    await screen(backend).act(action)

    assert host.desktop.calls == calls


@pytest.mark.parametrize(
    'action',
    [
        pytest.param(MoveMouse(x=1, y=1, modifiers=('alt',)), id='move-holding-a-key'),
        pytest.param(Drag(path=_path(101), modifiers=('shift',)), id='long-drag-holding-a-key'),
        pytest.param(HoldKeys(keys=('a',), seconds=300.5), id='hold-longer-than-boxd-allows'),
        pytest.param(object(), id='unknown-action-type'),
    ],
)
async def test_what_boxd_cannot_do_is_refused_before_anything_runs(host: FakeBoxd, action: DisplayAction) -> None:
    backend = await _started(host)

    with pytest.raises(UnsupportedActionError):
        await screen(backend).act(action)
    assert host.desktop.calls == []


@pytest.mark.parametrize(
    ('method', 'args', 'kwargs', 'message'),
    [
        pytest.param('type', ('a' * 10_001,), {}, 'type at most 10000 characters per call', id='type'),
        pytest.param('scroll', (1, 1), {'dy': -101}, 'scroll at most 100 wheel clicks each way', id='scroll'),
        pytest.param('key', ('a',), {'repeat': 101}, 'repeat at most 100 times', id='repeat'),
        pytest.param('drag', ([(1, 1)],), {}, 'a drag path has 2 to 100 points', id='drag'),
    ],
)
async def test_the_fake_refuses_what_boxd_refuses_in_one_call(
    host: FakeBoxd, method: str, args: tuple[object, ...], kwargs: dict[str, int], message: str
) -> None:
    """Pins the per-call limits measured on boxd, which the splitting tests above stay within."""
    await BoxdSandboxBackend(desktop=True).get_sandbox()
    computer = host.client().machines.computer

    with pytest.raises(APIStatusError, match=message):
        await getattr(computer, method)(host.only.id, *args, **kwargs)


async def test_a_failed_long_drag_releases_the_button(host: FakeBoxd) -> None:
    backend = await _started(host)
    # The press goes through; the first move fails.
    host.desktop.errors = [None, APIStatusError('the desktop went away', code='internal', grpc_code=13)]

    with pytest.raises(APIStatusError, match='the desktop went away'):
        await screen(backend).act(Drag(path=_path(101)))

    assert host.desktop.calls == ['mouse_down 0 0 left', 'mouse_move 1 1', 'mouse_up pointer left']


@pytest.mark.parametrize(
    ('machine_deleted', 'error', 'message'),
    [
        (
            False,
            WorkspaceError,
            r'move refused\. Releasing the left mouse button failed too, so it may still be held\.',
        ),
        # A machine deleted mid-drag ends the run rather than leaving the model to retry.
        (True, WorkspaceUnavailableError, 'no longer exists'),
    ],
)
async def test_a_long_drag_whose_release_fails_says_the_button_may_still_be_held(
    host: FakeBoxd, machine_deleted: bool, error: type[WorkspaceError], message: str
) -> None:
    backend = await _started(host)
    # The press goes through; the first move fails, and so does the release after it.
    host.desktop.errors = [
        None,
        APIStatusError('move refused', code='invalid_argument', grpc_code=3),
        APIStatusError('release connection lost', code='unavailable', grpc_code=14),
    ]
    if machine_deleted:
        # The lookup after the failure finds the machine gone.
        host.get_errors.append(NotFoundError('VM is destroyed', code='not_found', grpc_code=5))

    with pytest.raises(error, match=message):
        await screen(backend).act(Drag(path=_path(101)))

    assert host.desktop.calls == ['mouse_down 0 0 left', 'mouse_move 1 1', 'mouse_up pointer left']


async def test_a_cancelled_long_drag_releases_the_button(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.hang_on = 'mouse_move 50 50'

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(screen(backend).act, Drag(path=_path(101)))
        await anyio.wait_all_tasks_blocked()  # pressed, a move waiting on boxd
        tasks.cancel_scope.cancel()

    assert host.desktop.calls[0] == 'mouse_down 0 0 left'
    assert host.desktop.calls[-1] == 'mouse_up pointer left'


async def test_the_desktop_is_held_at_its_size_when_it_starts(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=(1440, 900))

    assert await screen(backend).display_size() == (1440, 900)
    assert host.desktop.calls == ['set_screen_size 1440x900']


async def test_looks_come_from_the_desktop(host: FakeBoxd) -> None:
    backend = await _started(host)
    display = screen(backend)

    assert await display.screenshot() == host.desktop.png
    assert await display.zoom(Region(left=10, top=20, right=110, bottom=70)) == host.desktop.png
    assert await display.cursor_position() == Point(x=7, y=8)
    assert await display.display_size() == (1280, 800)
    assert host.desktop.calls == ['zoom 10 20 100 50']


async def test_a_reply_that_is_not_a_png_is_a_workspace_error(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.png = b'GIF89a'

    with pytest.raises(WorkspaceError, match='did not return a PNG'):
        await screen(backend).screenshot()


async def test_a_value_boxd_refuses_is_a_workspace_error(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.desktop.errors.append(APIStatusError('unknown key', code='invalid_argument', grpc_code=3))

    with pytest.raises(WorkspaceError, match='The boxd desktop failed: unknown key'):
        await screen(backend).act(PressKeys(keys=('a',)))


async def test_a_deleted_machine_ends_the_run(host: FakeBoxd) -> None:
    backend = await _started(host)
    host.only.status = 'destroyed'

    with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
        await screen(backend).act(Click(x=1, y=1))


async def test_after_aclose_the_desktop_is_held_again_before_any_input(host: FakeBoxd) -> None:
    """The machine may restart before a later run attaches to it again, and its desktop with it."""
    backend = await _started(host)
    await backend.aclose()

    await screen(backend).act(Click(x=1, y=1))

    assert host.desktop.calls == ['set_screen_size 1280x800', "click 1 1 left count=1 modifiers=''"]


async def test_commands_run_on_the_desktop(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=True, env={'A': '1'})

    await backend.run(['true'])

    assert [line for line in host.lines if line.startswith('exec ')][
        -1
    ] == 'exec env -- DISPLAY=:100 A=1 true </dev/null'


async def test_the_live_view_url_is_the_desktop_in_the_console(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend(desktop=True)

    url = await backend.live_view_url()

    assert url == f'https://app.boxd.example/desktop/{host.only.name}'
    # The desktop runs, held at its size, before the link is handed out.
    assert host.desktop.calls == ['set_screen_size 1280x800']


async def test_a_machine_without_a_desktop_has_no_display(host: FakeBoxd) -> None:
    backend = BoxdSandboxBackend()

    assert backend.display is None
    with pytest.raises(UserError, match='has no display: create it with `desktop=True`'):
        await backend.live_view_url()
    assert host.machines == {}


async def test_a_desktop_link_boxd_did_not_return_is_a_workspace_error(host: FakeBoxd) -> None:
    """boxd builds the link from the console's address, which a client configured with only an API endpoint lacks."""
    backend = BoxdSandboxBackend(desktop=True)
    await backend.get_sandbox()
    host.only.desktop_link = False

    with pytest.raises(WorkspaceError, match='did not return a desktop link'):
        await backend.live_view_url()


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='boxd', id='machine-1')

    assert BoxdSandbox(desktop=True).backend(ref).display is not None
    assert BoxdSandbox().backend(ref).display is None


def _look_click_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Takes a screenshot, clicks on what it showed, then answers."""
    steps = [[{'type': 'screenshot'}], [{'type': 'click', 'x': 3, 'y': 4}]]
    done = sum(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts)
    if done == len(steps):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': steps[done]})])


async def test_a_desktop_machine_runs_computer_use(host: FakeBoxd) -> None:
    sandbox = BoxdSandbox(desktop=True)
    agent = Agent(FunctionModel(_look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    result = await agent.run('Click.')

    *_, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == host.desktop.png
    assert "click 3 4 left count=1 modifiers=''" in host.desktop.calls

    # A backend attached by the run's ref reaches the same desktop.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == host.desktop.png
