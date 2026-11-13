"""Tests for `DaytonaSandbox(desktop=True)`, against the fake Daytona SDK."""

from __future__ import annotations

from types import SimpleNamespace

import anyio
import anyio.lowlevel
import pytest
from daytona import DaytonaInternalServerError, DaytonaRateLimitError, DaytonaTimeoutError

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
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction, UnsupportedActionError
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from ..display_fakes import screen
from .fake_daytona import FakeDaytona, FakeSandbox


@pytest.fixture
def no_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drags and holds pause between steps; the tests don't wait for real."""

    async def checkpoint(seconds: float) -> None:
        await anyio.lowlevel.checkpoint()

    monkeypatch.setattr(anyio, 'sleep', checkpoint)


async def _started(fake_daytona: FakeDaytona) -> tuple[DaytonaSandboxBackend, FakeSandbox]:
    backend = DaytonaSandboxBackend(desktop=True)
    # Starts the desktop.
    await screen(backend).screenshot()
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
        pytest.param(
            Scroll(x=1, y=2, direction='down', amount=1_001),
            [f'scroll 1 2 down {clicks} modifiers=[]' for clicks in (500, 500, 1)],
            id='scroll-in-parts',
        ),
        pytest.param(TypeText(text='héllo'), ['type héllo'], id='type'),
        pytest.param(
            TypeText(text='a\tb\t\tc'),
            ['type a', 'press tab modifiers=None', 'type b', 'press tab modifiers=None', 'press tab modifiers=None']
            + ['type c'],
            id='type-tabs',
        ),
        pytest.param(
            TypeText(text='a' * 4_001), [f'type {"a" * 2_000}', f'type {"a" * 2_000}', 'type a'], id='type-in-parts'
        ),
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
    fake_daytona: FakeDaytona, no_pauses: None, action: DisplayAction, calls: list[str]
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
    fake_daytona: FakeDaytona, action: DisplayAction
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


async def test_a_cancelled_type_waits_for_the_part_daytona_accepted(fake_daytona: FakeDaytona) -> None:
    """Daytona types a part it accepted even when the caller stops waiting, so the next action can't overtake it."""
    backend, sandbox = await _started(fake_daytona)
    type_part = sandbox.computer_use.type
    accepted, release = anyio.Event(), anyio.Event()

    async def slow_type(text: str, delay: int | None = None, request_timeout: float | None = None) -> None:
        accepted.set()
        await release.wait()
        await type_part(text, delay, request_timeout)

    sandbox.computer_use.type = slow_type
    typing = anyio.CancelScope()

    async def type_text() -> None:
        with typing:
            await screen(backend).act(TypeText(text='a' * 4_001))

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(type_text)
        await accepted.wait()
        typing.cancel()
        await anyio.wait_all_tasks_blocked()
        assert not typing.cancelled_caught  # still waiting for the part Daytona accepted
        release.set()

    assert typing.cancelled_caught
    # The rest was never sent.
    assert sandbox.desktop_calls == [f'type {"a" * 2_000}']


async def test_a_failed_release_still_lets_go_of_the_other_keys(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    up = sandbox.computer_use.up
    failed: list[str] = []

    async def up_failing_once(
        x: int | str | None = None, y: int | None = None, button: str = 'left', request_timeout: float | None = None
    ) -> None:
        if x == 'b' and not failed:
            failed.append('b')
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


async def test_a_failed_drag_releases_the_button_and_keys(fake_daytona: FakeDaytona, no_pauses: None) -> None:
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
    assert sandbox.desktop_calls == [
        'take_full_screen',
        'take_region 10 20 100 50',
        'get_position',
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
    # The next action starts the desktop again and runs on it.
    await screen(backend).act(Click(x=1, y=1))

    assert not sandbox.computer_use.broken
    assert sandbox.desktop_calls[-2:] == ['start', 'click 1 1 left clicks=1 modifiers=[]']


async def test_a_failed_start_of_a_running_desktop_starts_it_again_on_the_next_action(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)
    computer_use = sandbox.computer_use
    start = computer_use.start
    computer_use.stop_with_sandbox()

    async def failing_start(request_timeout: float | None = None) -> None:
        monkeypatch.setattr(computer_use, 'start', start)
        raise DaytonaInternalServerError('the X server did not start', status_code=500)

    monkeypatch.setattr(computer_use, 'start', failing_start)
    # The live view link starts the desktop even when it last ran.
    with pytest.raises(WorkspaceError, match='did not start'):
        await backend.live_view_url()
    await screen(backend).act(Click(x=1, y=1))

    assert sandbox.desktop_calls == ['start', 'click 1 1 left clicks=1 modifiers=[]']


async def test_after_aclose_the_desktop_is_started_again_before_any_input(fake_daytona: FakeDaytona) -> None:
    """The sandbox may stop and start before a later run attaches to it again."""
    backend, sandbox = await _started(fake_daytona)
    await backend.aclose()
    sandbox.computer_use.stop_with_sandbox()

    await screen(backend).act(Click(x=1, y=1))
    assert not sandbox.computer_use.broken
    assert sandbox.desktop_calls == ['start', 'click 1 1 left clicks=1 modifiers=[]']


async def test_every_action_checks_that_the_desktop_runs(fake_daytona: FakeDaytona) -> None:
    """The sandbox can stop and start between two actions with no screenshot between them."""
    backend, sandbox = await _started(fake_daytona)
    await screen(backend).act(Click(x=1, y=1))

    sandbox.computer_use.stop_with_sandbox()
    with pytest.raises(WorkspaceError, match="Daytona's desktop is not running"):
        await screen(backend).act(Click(x=2, y=2))
    # No input reached the stopped desktop.
    assert not sandbox.computer_use.broken
    assert sandbox.desktop_calls == ['click 1 1 left clicks=1 modifiers=[]']


async def test_the_live_view_starts_a_desktop_that_stopped_with_the_sandbox(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    sandbox.computer_use.stop_with_sandbox()

    await backend.live_view_url()

    assert sandbox.computer_use.started
    assert sandbox.desktop_calls == ['start', 'signed_preview 6080 3600']


async def test_a_public_sandbox_desktop_is_refused_before_it_starts(fake_daytona: FakeDaytona) -> None:
    """Daytona serves a public sandbox's preview links without a token, so its desktop would be open to anyone."""
    backend = DaytonaSandboxBackend(desktop=True)
    await backend.get_sandbox()
    (sandbox,) = fake_daytona.sandboxes
    # Made public elsewhere, after this backend fetched it as private.
    sandbox.server_public = True

    with pytest.raises(UserError, match='is public'):
        await screen(backend).screenshot()
    with pytest.raises(UserError, match='is public'):
        await backend.live_view_url()
    assert sandbox.desktop_calls == []


async def test_a_desktop_that_does_not_answer_in_time_is_a_timeout(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def slow(request_timeout: float | None = None) -> SimpleNamespace:
        raise DaytonaTimeoutError('request timeout')

    monkeypatch.setattr(sandbox.computer_use, 'take_full_screen', slow)
    with pytest.raises(WorkspaceTimeoutError, match='The Daytona desktop did not answer in time'):
        await screen(backend).screenshot()


async def test_a_rate_limit_reaches_the_caller_as_it_came(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable engine retries Daytona's own transient errors, so they are not turned into a `WorkspaceError`."""
    backend, sandbox = await _started(fake_daytona)

    async def limited(request_timeout: float | None = None) -> SimpleNamespace:
        raise DaytonaRateLimitError('Too many requests', status_code=429)

    monkeypatch.setattr(sandbox.computer_use, 'take_full_screen', limited)
    with pytest.raises(DaytonaRateLimitError):
        await screen(backend).screenshot()


async def test_a_scroll_the_desktop_reports_as_failed_is_an_error(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def refused(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(sandbox.computer_use, 'scroll', refused)
    with pytest.raises(WorkspaceError, match='did not scroll'):
        await screen(backend).act(Scroll(x=1, y=1, direction='down', amount=3))


async def test_a_desktop_that_stopped_answering_ends_the_run(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    sandbox.computer_use.broken = True

    with pytest.raises(WorkspaceUnavailableError, match='stopping and starting the sandbox brings it back'):
        await screen(backend).screenshot()


async def test_a_proxy_refusing_traffic_for_a_moment_does_not_end_the_run(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend(desktop=True)
    await backend.get_sandbox()
    (sandbox,) = fake_daytona.sandboxes
    sandbox.computer_use.proxy_unavailable = 1

    # The first start is refused by the proxy, which says nothing about the desktop; the retry reaches it.
    assert await screen(backend).screenshot() == sandbox.computer_use.png


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


def _look_click_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Takes a screenshot, clicks on what it showed, then answers."""
    steps = [[{'type': 'screenshot'}], [{'type': 'click', 'x': 3, 'y': 4}]]
    done = sum(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts)
    if done == len(steps):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': steps[done]})])


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(fake_daytona: FakeDaytona) -> None:
    sandbox_capability = DaytonaSandbox(desktop=True)
    agent = Agent(
        FunctionModel(_look_click_then_finish),
        capabilities=[sandbox_capability, ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    (sandbox,) = fake_daytona.sandboxes
    assert looked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == sandbox.computer_use.png
    assert 'click 3 4 left clicks=1 modifiers=[]' in sandbox.desktop_calls

    # A backend attached by ref reaches the same desktop.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox_capability.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == sandbox.computer_use.png
