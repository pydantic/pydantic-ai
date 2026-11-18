"""Tests for `DaytonaSandbox(desktop=True)`, against the fake Daytona SDK."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import anyio
import anyio.lowlevel
import pytest
from daytona import (
    DaytonaAuthenticationError,
    DaytonaBadGatewayError,
    DaytonaBadRequestError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaInternalServerError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
)

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
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerUse, DisplayAction, UnsupportedActionError
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend

from ..display_fakes import look_click_then_finish, screen
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
        # Daytona refuses the side buttons itself.
        pytest.param(Click(x=1, y=2, button='back'), ['click 1 2 back clicks=1 modifiers=[]'], id='side-button'),
        pytest.param(MouseMove(x=5, y=6), ['move 5 6'], id='mouse_move'),
        pytest.param(
            MouseMove(x=5, y=6, modifiers=('alt',)),
            ['keydown alt', 'move 5 6', 'keyup alt'],
            id='mouse_move-with-modifier',
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
        pytest.param(TypeText(text='a\tb'), ['type a\tb'], id='type-tab'),
        pytest.param(
            TypeText(text='a' * 4_001), [f'type {"a" * 2_000}', f'type {"a" * 2_000}', 'type a'], id='type-in-parts'
        ),
        # Daytona types `\r\n` as one Enter only within one request.
        pytest.param(
            TypeText(text='a' * 1_999 + '\r\n' + 'b'), [f'type {"a" * 1_999}\n', 'type b'], id='type-crlf-in-parts'
        ),
        pytest.param(
            PressKeys(keys=('ctrl+s',), repeat=2),
            ["press s modifiers=['ctrl']", "press s modifiers=['ctrl']"],
            id='combination-repeated',
        ),
        pytest.param(PressKeys(keys=('Return',)), ['press enter modifiers=[]'], id='alias'),
        # Daytona refuses a key it has no name for itself, and its desktop keeps working.
        pytest.param(PressKeys(keys=('printscreen',)), ['press printscreen modifiers=[]'], id='key-daytona-refuses'),
        pytest.param(
            PressKeys(keys=('numlock', 'numpad5')),
            ['keydown num_lock', 'keydown num5', 'keyup num5', 'keyup num_lock'],
            id='renamed-keys',
        ),
        pytest.param(
            PressKeys(keys=('shift', 'ctrl', '!')), ["press 1 modifiers=['shift', 'ctrl']"], id='shifted-symbol'
        ),
        pytest.param(PressKeys(keys=('super',)), ['keydown cmd', 'keyup cmd'], id='modifier-alone'),
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
    ('action', 'message'),
    [
        pytest.param(TypeText(text='done 😀'), 'cannot type characters beyond U\\+FFFF', id='emoji'),
        pytest.param(PressKeys(keys=('f13',)), "cannot press the key 'f13'", id='key-that-crashes-the-desktop'),
        pytest.param(PressKeys(keys=('numpad_add',)), "cannot press the key 'numpad_add'", id='numpad-key'),
        pytest.param(object(), 'cannot perform <object', id='unknown-action-type'),
    ],
)
async def test_what_daytona_cannot_do_is_refused_before_anything_runs(
    fake_daytona: FakeDaytona, action: DisplayAction, message: str
) -> None:
    backend, sandbox = await _started(fake_daytona)

    with pytest.raises(UnsupportedActionError, match=message):
        await screen(backend).act(action)
    assert sandbox.desktop_calls == []


async def test_a_cancelled_hold_lets_go_of_what_it_pressed(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(screen(backend).act, HoldKeys(keys=('ctrl', 'a'), seconds=30))
        await anyio.wait_all_tasks_blocked()  # holding, in its sleep
        tasks.cancel_scope.cancel()

    assert sandbox.desktop_calls == ['keydown ctrl', 'keydown a', 'keyup a', 'keyup ctrl']


@pytest.mark.parametrize(
    ('action', 'sent'),
    [
        pytest.param(TypeText(text='a' * 4_001), f'type {"a" * 2_000}', id='type'),
        pytest.param(Scroll(x=1, y=2, direction='down', amount=1_001), 'scroll 1 2 down 500 modifiers=[]', id='scroll'),
        pytest.param(Click(x=1, y=2), 'click 1 2 left clicks=1 modifiers=[]', id='click'),
    ],
)
@pytest.mark.parametrize('native', [False, True], ids=['scope-cancel', 'task-cancel'])
async def test_a_cancelled_action_waits_for_the_input_daytona_accepted(
    fake_daytona: FakeDaytona, action: DisplayAction, sent: str, native: bool
) -> None:
    """Daytona carries out input it accepted even when the caller stops waiting, so the next action can't overtake it.

    Typing and scrolling are sent in parts, and a cancel sends no more of them.
    """
    backend, sandbox = await _started(fake_daytona)
    desktop = sandbox.computer_use
    desktop.request_gate = asyncio.Event()
    scope = anyio.CancelScope()

    async def act() -> None:
        with scope:
            await screen(backend).act(action)

    acting = asyncio.ensure_future(act())
    await desktop.request_accepted.wait()
    acting.cancel() if native else scope.cancel()
    await anyio.wait_all_tasks_blocked()
    assert not acting.done()  # still waiting for the part Daytona accepted
    desktop.request_gate.set()

    with pytest.raises(asyncio.CancelledError) if native else nullcontext():
        await acting
    # The rest was never sent.
    assert sandbox.desktop_calls == [sent]


async def test_a_failed_release_still_lets_go_of_the_other_keys(fake_daytona: FakeDaytona) -> None:
    backend, sandbox = await _started(fake_daytona)
    up = sandbox.computer_use.keyboard.up
    failed: list[str] = []

    async def up_failing_once(key: str, request_timeout: float | None = None) -> None:
        if key == 'b' and not failed:
            failed.append('b')
            raise DaytonaInternalServerError('unexpected EOF', status_code=500)
        await up(key, request_timeout)

    sandbox.computer_use.keyboard.up = up_failing_once
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


async def test_a_key_daytona_refuses_to_hold_is_not_reported_held(fake_daytona: FakeDaytona) -> None:
    """Daytona refuses `keyboard.down` of a key it hasn't with a 500 (observed live 2026-11-16), as it refuses `press`."""
    backend, sandbox = await _started(fake_daytona)
    down = sandbox.computer_use.keyboard.down
    refusal = 'Failed to press key down: internal server error: unsupported key "printscreen"'

    async def down_refusing(key: str, request_timeout: float | None = None) -> None:
        if key == 'printscreen':
            raise DaytonaInternalServerError(refusal, status_code=500)
        await down(key, request_timeout)

    sandbox.computer_use.keyboard.down = down_refusing
    with pytest.raises(WorkspaceError, match=f': {refusal}$'):
        await screen(backend).act(HoldKeys(keys=('ctrl', 'printscreen'), seconds=1))

    assert sandbox.desktop_calls == ['keydown ctrl', 'keyup ctrl']


async def test_a_key_daytona_did_not_confirm_holding_is_let_go(fake_daytona: FakeDaytona) -> None:
    """Daytona may still press a key whose `keyboard.down` timed out, so it is released like one it pressed."""
    backend, sandbox = await _started(fake_daytona)
    down = sandbox.computer_use.keyboard.down

    async def down_unconfirmed(key: str, request_timeout: float | None = None) -> None:
        await down(key, request_timeout)
        if key == 'a':
            raise DaytonaTimeoutError('request timeout')

    sandbox.computer_use.keyboard.down = down_unconfirmed
    with pytest.raises(WorkspaceUnavailableError, match="didn't confirm the input"):
        await screen(backend).act(HoldKeys(keys=('ctrl', 'a'), seconds=1))

    assert sandbox.desktop_calls == ['keydown ctrl', 'keydown a', 'keyup a', 'keyup ctrl']


async def test_a_key_whose_hold_answer_cannot_be_read_is_let_go(fake_daytona: FakeDaytona) -> None:
    """An answer the SDK can't decode comes without an HTTP status, so the key may be down and is released."""
    backend, sandbox = await _started(fake_daytona)
    down = sandbox.computer_use.keyboard.down

    async def down_unreadable(key: str, request_timeout: float | None = None) -> None:
        await down(key, request_timeout)
        raise DaytonaError('Failed to press key down: Expecting property name enclosed in double quotes')

    sandbox.computer_use.keyboard.down = down_unreadable
    with pytest.raises(WorkspaceError, match='Expecting property name'):
        await screen(backend).act(HoldKeys(keys=('ctrl',), seconds=1))

    assert sandbox.desktop_calls == ['keydown ctrl', 'keyup ctrl']


HOLDING_CTRL = MouseMove(x=1, y=1, modifiers=('ctrl',))


@pytest.mark.parametrize(
    ('action', 'held', 'release'),
    [
        pytest.param(HOLDING_CTRL, 'ctrl', 'failed', id='failed-key'),
        pytest.param(
            Drag(path=(Point(x=1, y=1), Point(x=2, y=2))), 'the left mouse button', 'failed', id='failed-drag'
        ),
        pytest.param(HOLDING_CTRL, 'ctrl', 'unconfirmed', id='unconfirmed'),
        pytest.param(HOLDING_CTRL, 'ctrl', 'sandbox-deleted', id='sandbox-deleted'),
    ],
)
async def test_a_failed_release_after_a_failed_action_says_what_may_still_be_held(
    fake_daytona: FakeDaytona, no_pauses: None, action: DisplayAction, held: str, release: str
) -> None:
    """The model must not go on with a key or button held; a lost sandbox or a release that may still run ends the run."""
    backend, sandbox = await _started(fake_daytona)

    async def failing(*args: object, **kwargs: object) -> None:
        if release == 'sandbox-deleted':
            fake_daytona.purge(sandbox)
        raise DaytonaInternalServerError('unexpected EOF', status_code=500)

    async def unconfirmed(*args: object, **kwargs: object) -> None:
        raise DaytonaTimeoutError('request timeout')

    sandbox.computer_use.mouse.move = failing
    sandbox.computer_use.mouse.up = sandbox.computer_use.keyboard.up = (
        unconfirmed if release == 'unconfirmed' else failing
    )
    expected = {
        'failed': pytest.raises(
            WorkspaceError, match=f'^unexpected EOF; releasing {held} failed too, so it may still be held$'
        ),
        'unconfirmed': pytest.raises(WorkspaceUnavailableError, match="didn't confirm the input"),
        'sandbox-deleted': pytest.raises(WorkspaceUnavailableError, match='no longer exists'),
    }[release]
    with expected:
        await screen(backend).act(action)


async def test_rejected_credentials_end_the_run_without_their_text_when_a_release_fails_too(
    fake_daytona: FakeDaytona,
) -> None:
    """The credential error's own text may carry the token, so it isn't passed to the model with the held keys."""
    backend, sandbox = await _started(fake_daytona)

    async def rejected(*args: object, **kwargs: object) -> None:
        raise DaytonaAuthenticationError('expired token secret-value', status_code=401)

    async def failing(*args: object, **kwargs: object) -> None:
        raise DaytonaInternalServerError('unexpected EOF', status_code=500)

    sandbox.computer_use.mouse.move = rejected
    sandbox.computer_use.keyboard.up = failing
    with pytest.raises(WorkspaceUnavailableError) as raised:
        await screen(backend).act(MouseMove(x=1, y=1, modifiers=('ctrl',)))
    assert 'secret-value' not in str(raised.value)


@pytest.mark.parametrize(
    ('error', 'ends_run'),
    [
        pytest.param(DaytonaTimeoutError('request timeout'), True, id='timeout'),
        pytest.param(DaytonaConnectionError('Server disconnected'), True, id='connection-lost'),
        pytest.param(DaytonaBadGatewayError('upstream dropped', status_code=502), True, id='proxy-lost-sandbox'),
        pytest.param(DaytonaBadRequestError('invalid key', status_code=400), False, id='refused'),
    ],
)
async def test_input_daytona_did_not_confirm_ends_the_run(
    fake_daytona: FakeDaytona, error: DaytonaError, ends_run: bool
) -> None:
    """Daytona may still be carrying out input it didn't confirm, so later input must not race it; a refusal didn't run."""
    backend, sandbox = await _started(fake_daytona)

    async def unconfirmed(*args: object, **kwargs: object) -> None:
        raise error

    sandbox.computer_use.mouse.move = unconfirmed
    with pytest.raises(WorkspaceError) as raised:
        await screen(backend).act(MouseMove(x=1, y=1, modifiers=('ctrl',)))

    assert isinstance(raised.value, WorkspaceUnavailableError) is ends_run
    assert sandbox.desktop_calls == ['keydown ctrl', 'keyup ctrl']


async def test_a_failed_drag_releases_the_button_and_keys(fake_daytona: FakeDaytona, no_pauses: None) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def fail_on_move(x: int, y: int, request_timeout: float | None = None) -> None:
        raise DaytonaInternalServerError('unexpected EOF', status_code=500)

    sandbox.computer_use.mouse.move = fail_on_move
    with pytest.raises(WorkspaceError, match='unexpected EOF'):
        await screen(backend).act(Drag(path=(Point(x=1, y=1), Point(x=2, y=2), Point(x=3, y=3)), modifiers=('shift',)))

    assert sandbox.desktop_calls == [
        'keydown shift',
        'mousedown 1 1 left',
        'mouseup None None left',
        'keyup shift',
    ]


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
    with pytest.raises(WorkspaceError, match=r'^Daytona did not return a PNG screenshot\.$'):
        await screen(backend).screenshot()
    sandbox.computer_use.displays = []
    with pytest.raises(WorkspaceError, match='no display size'):
        await screen(backend).display_size()
    sandbox.computer_use.position = SimpleNamespace(x=None, y=None)
    with pytest.raises(WorkspaceError, match='no pointer position'):
        await screen(backend).cursor_position()


async def test_after_aclose_the_desktop_is_started_again_before_any_input(fake_daytona: FakeDaytona) -> None:
    """The sandbox may stop and start before a later run attaches to it again."""
    backend, sandbox = await _started(fake_daytona)
    await backend.aclose()
    sandbox.computer_use.stop_with_sandbox()

    await screen(backend).act(Click(x=1, y=1))
    assert sandbox.desktop_calls == ['start', 'click 1 1 left clicks=1 modifiers=[]']


async def test_a_desktop_that_does_not_answer_in_time_is_a_timeout(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def slow(request_timeout: float | None = None) -> SimpleNamespace:
        raise DaytonaTimeoutError('request timeout')

    monkeypatch.setattr(sandbox.computer_use.screenshot, 'take_full_screen', slow)
    with pytest.raises(WorkspaceTimeoutError, match='The Daytona desktop did not answer in time'):
        await screen(backend).screenshot()


async def test_a_rate_limit_reaches_the_caller_as_it_came(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A durable engine retries Daytona's own transient errors, so they are not turned into a `WorkspaceError`."""
    backend, sandbox = await _started(fake_daytona)

    async def limited(request_timeout: float | None = None) -> SimpleNamespace:
        raise DaytonaRateLimitError('Too many requests', status_code=429)

    monkeypatch.setattr(sandbox.computer_use.screenshot, 'take_full_screen', limited)
    with pytest.raises(DaytonaRateLimitError):
        await screen(backend).screenshot()


async def test_a_scroll_the_desktop_reports_as_failed_is_an_error(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, sandbox = await _started(fake_daytona)

    async def refused(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(sandbox.computer_use.mouse, 'scroll', refused)
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


async def test_a_shell_command_finds_the_desktop_running(fake_daytona: FakeDaytona) -> None:
    # The shell is told a program it starts opens on the screen, before anything has looked at the screen.
    backend = DaytonaSandboxBackend(desktop=True)
    await backend.run('xterm &', shell=True)

    (sandbox,) = fake_daytona.sandboxes
    assert sandbox.desktop_calls == ['start']


async def test_a_desktop_size_is_set_on_a_new_sandbox(fake_daytona: FakeDaytona) -> None:
    await DaytonaSandboxBackend(desktop=(1440, 900), env={'A': '1'}).get_sandbox()
    await DaytonaSandboxBackend(desktop=True).get_sandbox()
    await DaytonaSandboxBackend().get_sandbox()

    assert [params.env_vars for params in fake_daytona.create_params] == [
        {'DISPLAY': ':0', 'A': '1', 'VNC_RESOLUTION': '1440x900'},
        {'DISPLAY': ':0', 'VNC_RESOLUTION': '1280x800'},
        None,
    ]


async def test_a_public_sandbox_gets_no_desktop(fake_daytona: FakeDaytona) -> None:
    """Daytona's VNC preview of the desktop has no password, and a public sandbox's previews need no token."""
    sandbox = fake_daytona.sandbox()
    sandbox.public = True
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[ComputerUse(native=False)])

    with pytest.raises(
        UserError, match=r"^The Daytona sandbox 'sb-existing' is public, so anyone with its preview URL"
    ):
        await agent.run(
            'Click.', workspace=DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id=sandbox.id), desktop=True)
        )
    assert sandbox.desktop_calls == []


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='daytona', id='sb-1')

    assert DaytonaSandbox(desktop=True).backend(ref).display is not None
    assert DaytonaSandbox().backend(ref).display is None


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(fake_daytona: FakeDaytona) -> None:
    sandbox_capability = DaytonaSandbox(desktop=True)
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[sandbox_capability, ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    (sandbox,) = fake_daytona.sandboxes
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == sandbox.computer_use.png
    assert 'click 3 4 left clicks=1 modifiers=[]' in sandbox.desktop_calls

    # A backend attached by ref reaches the same desktop.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox_capability.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == sandbox.computer_use.png
    assert fake_daytona.sandboxes == [sandbox]
    assert sandbox.desktop_calls[-1] == 'take_full_screen'
