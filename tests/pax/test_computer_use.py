"""`/computer-use`: desktop sandboxes, `ComputerUse` activation, rendering, live viewing, and provider wiring."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.computer_use import (
    Click,
    ComputerCallPart,
    ComputerResult,
    ComputerToolAction,
    CursorPosition,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Screenshot,
    Scroll,
    TypeText,
    Wait,
    Zoom,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef
from pydantic_ai_harness.computer_use import (
    ActionsPerformedEvent,
)
from pydantic_ai_pax import Session
from pydantic_ai_pax.builtin_plugins import sandbox
from pydantic_ai_pax.builtin_plugins.sandbox import SandboxPlugin, history_sandbox
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
    COMPUTER_USE,
    ActionMarker,
    action_markers,
    format_action,
    format_actions,
    render_computer_event,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers import (
    GitHubCredential,
)
from pydantic_ai_pax.builtin_plugins.sandbox.viewer import (
    HEADER,
    MAX_FRAMES,
    DesktopFrame,
    DesktopSession,
    DesktopViewer,
    create_viewer_app,
    sandbox_key,
)
from pydantic_ai_pax.builtin_plugins.web import server as web_server
from pydantic_ai_pax.config import PERMISSIONS_SETTING
from pydantic_ai_pax.plugins import SessionEnd, Transcript

# `checkout`, `isolated_git_config`, `provider`, and `upstream` are fixtures shared with `/sandbox`'s tests.
from tests.pax.test_sandbox import (
    PNG_A,
    PNG_B,
    FakeDisplay,
    FakeProvider,
    checkout,
    isolated_git_config,
    load,
    on_screen,
    plugin_of,
    provider,
    upstream,
)

__all__ = ('checkout', 'isolated_git_config', 'provider', 'upstream')


@pytest.fixture
def display(provider: FakeProvider) -> FakeDisplay:
    """The screen of the fake provider's desktop sandboxes."""
    provider.display = screen = FakeDisplay()
    return screen


def stub_viewer_port(monkeypatch: pytest.MonkeyPatch, plugin: SandboxPlugin, port: int) -> None:
    """Skip binding a real port; `test_desktop_viewer_app_end_to_end` serves the viewer for real."""
    monkeypatch.setattr(plugin._viewer, 'ensure_started', lambda: port)  # pyright: ignore[reportPrivateUsage]


def viewer_frames(plugin: SandboxPlugin, ref: WorkspaceRef) -> list[DesktopFrame] | None:
    """The frames the viewer page shows; only that page shows them, and `test_desktop_viewer_app_end_to_end` reads it."""
    session = plugin._viewer.get(sandbox_key(ref))  # pyright: ignore[reportPrivateUsage]
    return None if session is None else list(session.frames)


def on_screen_sandbox(plugin: SandboxPlugin):
    """The sandbox of the conversation `plugin` shows."""
    sandbox = plugin.current()
    assert sandbox is not None
    return sandbox


def render_plain(renderable: object) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=120).print(renderable)
    return buf.getvalue().rstrip('\n')


# ---------------------------------------------------------------------------
# Action formatting & terminal rendering
# ---------------------------------------------------------------------------


def test_action_formatting_markers_and_rendering() -> None:
    actions: list[ComputerToolAction] = [
        Click(x=10, y=20),
        Click(x=30, y=40, button='right', count=2, modifiers=('ctrl',)),
        Click(count=3),
        Click(x=5, y=6, count=4),
        MouseMove(x=50, y=60),
        Drag(path=(Point(x=1, y=2), Point(x=3, y=4), Point(x=9, y=8))),
        MouseDown(button='left'),
        MouseUp(button='middle'),
        Scroll(direction='down', amount=3, x=100, y=200, modifiers=('shift',)),
        Scroll(direction='up', amount=2),
        TypeText(text='hello\x1b[31m'),
        TypeText(text='x' * 50),
        PressKeys(keys=('ctrl', 'c')),
        PressKeys(keys=('enter',), repeat=2),
        HoldKeys(keys=('shift',), seconds=2.0),
        HoldKeys(keys=('alt',), seconds=0.5),
        Screenshot(),
        Wait(seconds=1.5),
        Zoom(region=Region(left=0, top=10, right=100, bottom=200)),
        CursorPosition(),
    ]
    formatted = [format_action(a) for a in actions]
    assert formatted == [
        'click (10, 20)',
        'ctrl+right-double-click (30, 40)',
        'triple-click',
        'click ×4 (5, 6)',
        'move (50, 60)',
        'drag (1, 2) -> (9, 8)',
        'mouse-down left',
        'mouse-up middle',
        'shift+scroll down 3 (100, 200)',
        'scroll up 2',
        'type "hello\\x1b[31m"',
        f'type "{"x" * 39}…"',
        'press ctrl+c',
        'press enter ×2',
        'hold shift (2s)',
        'hold alt (0.5s)',
        'screenshot',
        'wait 1.5s',
        'zoom (0, 10)-(100, 200)',
        'cursor-position',
    ]
    assert format_actions(actions[:2]) == 'click (10, 20) · ctrl+right-double-click (30, 40)'
    assert action_markers(actions[:6]) == (
        ActionMarker(kind='click', points=((10, 20),), label='click (10, 20)'),
        ActionMarker(kind='click', points=((30, 40),), label='ctrl+right-double-click (30, 40)'),
        ActionMarker(kind='click', points=((5, 6),), label='click ×4 (5, 6)'),
        ActionMarker(kind='move', points=((50, 60),), label='move (50, 60)'),
        ActionMarker(kind='drag', points=((1, 2), (3, 4), (9, 8)), label='drag (1, 2) -> (9, 8)'),
    )

    call_part = ComputerCallPart(
        tool_name='computer',
        args={'actions': [{'type': 'click', 'x': 10, 'y': 20}, {'type': 'screenshot'}]},
        tool_call_id='call-1',
    )
    rendered_call = render_computer_event(FunctionToolCallEvent(part=call_part))
    assert rendered_call is not None
    assert 'computer click (10, 20) · screenshot' in render_plain(rendered_call)

    ok_one = render_computer_event(ActionsPerformedEvent(result=ComputerResult(performed=1)))
    assert ok_one is not None and '1 action · screenshot' in render_plain(ok_one)
    ok_many = render_computer_event(
        ActionsPerformedEvent(result=ComputerResult(performed=3)),
        live_url='http://127.0.0.1:9999/?token=t&sandbox=s',
    )
    assert ok_many is not None
    assert '3 actions · screenshot · watch live ↗' in render_plain(ok_many)
    err = render_computer_event(ActionsPerformedEvent(result=ComputerResult(performed=0, error='bad\x1bkey')))
    assert err is not None and 'bad\\x1bkey' in render_plain(err)

    other_call = FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'ls'}))
    assert render_computer_event(other_call) is None


# ---------------------------------------------------------------------------
# End-to-end `/computer-use` toggle, inline live view, resume & `/sandbox delete`
# ---------------------------------------------------------------------------


@pytest.mark.moves_cache_prefix(
    reason='Toggles /computer-use off and back on mid-conversation, intentionally changing offered tools and instructions.'
)
async def test_computer_use_end_to_end_toggle_and_resume(
    checkout: Path, provider: FakeProvider, display: FakeDisplay, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = checkout
    loaded = load(work)
    plugin = plugin_of(loaded)
    stub_viewer_port(monkeypatch, plugin, 43219)

    seen_tools: list[list[str]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        seen_tools.append([tool.name for tool in info.function_tools])
        if len(messages) == 3 and 'computer' in seen_tools[-1]:
            args = json.dumps({'actions': [{'type': 'click', 'x': 120, 'y': 240}, {'type': 'type_text', 'text': 'hi'}]})
            yield {0: DeltaToolCall(name='computer', json_args=args, tool_call_id='cu-1')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None))
    session = Session(agent, deps=None, plugins=loaded.capabilities, workspace=work)
    loaded.host.conversation = session

    # On a local conversation, `computer` is NOT offered.
    await session.prompt('local turn')
    assert 'computer' not in seen_tools[-1]

    # Arm `/computer-use`, then start `/sandbox fake` — `/sandbox` creates it with `desktop=True`.
    session.replace_messages([])
    armed = await plugin.computer_use_command([])
    assert 'Computer use is on' in armed
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use']

    started = await plugin.command(['fake'])
    assert 'Fake sandbox sbx-1' in started
    assert 'Computer use is on' in started
    assert provider.created == [GitHubCredential(login='Ada', token='gho_secret')]
    assert provider.created_desktop == [True]
    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox (computer-use)']
    ref = history_sandbox(session.messages)
    assert ref is not None
    assert ref == WorkspaceRef(provider='fake', id='sbx-1')

    # Now a turn receives `computer`, starts the loopback viewer lazily, executes actions, and records the frame.
    result = await session.prompt('click and type')
    assert 'computer' in seen_tools[-1]
    assert result.response.workspace_ref == ref
    assert result.response.metadata == {'clai2_computer_use': True}
    assert display.actions == [Click(x=120, y=240), TypeText(text='hi')]
    [first_req] = [m for m in result.new_messages() if isinstance(m, ModelRequest)][:1]
    assert 'also has a desktop' in (first_req.instructions or '')

    [frame] = viewer_frames(plugin, ref) or []
    assert frame.png == PNG_A
    assert frame.summary == 'click (120, 240) · type "hi"'
    assert frame.markers == (ActionMarker(kind='click', points=((120, 240),), label='click (120, 240)'),)

    # The `computer` result row links to the loopback viewer.
    rendered = plugin.render(ActionsPerformedEvent(result=ComputerResult(performed=2)))
    assert rendered is not None
    assert '2 actions · screenshot · watch live ↗' in render_plain(rendered)

    # Toggling `/computer-use` off disables the `computer` tool while keeping the sandbox running, and survives resume and compaction.
    disabled = await plugin.computer_use_command([])
    assert 'Computer use is off' in disabled
    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox']
    await session.prompt('without desktop')
    assert 'computer' not in seen_tools[-1]

    # Drop everything before the last turn (simulating compaction discarding the `/computer-use off` marker).
    compacted_messages = session.messages[-2:]
    resumed_off_loaded = load(work, Transcript(messages=compacted_messages))
    resumed_off = plugin_of(resumed_off_loaded)
    assert [segment() for segment in resumed_off.get_status_segments()] == ['fake sandbox']
    resumed_off_session = Session(agent, deps=None, plugins=resumed_off_loaded.capabilities, workspace=work)
    resumed_off_session.replace_messages(list(compacted_messages))
    await resumed_off_session.prompt('still without desktop after compaction and restart')
    assert 'computer' not in seen_tools[-1]
    # The sandbox keeps its desktop, though the model gets no `computer` tool while it is off.
    assert on_screen_sandbox(resumed_off).desktop is True
    await resumed_off.on_session_end(SessionEnd(reason='exit'))

    # Toggling `/computer-use` back on re-enables immediately without recreating the sandbox.
    re_enabled = await plugin.computer_use_command([])
    assert 'Computer use is on' in re_enabled
    assert len(provider.created) == 1
    await session.prompt('back with desktop')
    assert 'computer' in seen_tools[-1]

    # Resuming in a fresh plugin instance reopens with `desktop=True` and seeds the viewer from saved history.
    resumed_loaded = load(work, Transcript(messages=session.messages))
    resumed = plugin_of(resumed_loaded)
    stub_viewer_port(monkeypatch, resumed, 43219)
    assert (await resumed.computer_use_command(['view'])).startswith('Desktop viewer: http://127.0.0.1:43219/')
    assert [frame.png for frame in viewer_frames(resumed, ref) or []] == [PNG_A]
    assert await on_screen_sandbox(resumed).screenshot() == PNG_A
    assert provider.reopened == [ref, ref] and provider.reopened_desktop == [True, True]

    # The history naming the sandbox, not the one on screen, says it has a desktop.
    unrelated = plugin_of(load(work, Transcript(messages=[])))
    assert unrelated.sandbox(ref, session.messages).desktop is True

    # `/sandbox delete` is the single delete command; it removes the viewer session and closes cleanly.
    deleted = await plugin.command(['delete'])
    assert 'Deleted Fake sandbox sbx-1' in deleted
    assert provider.deleted == [ref]
    assert viewer_frames(plugin, ref) is None
    await plugin.on_session_end(SessionEnd(reason='exit'))
    await resumed.on_session_end(SessionEnd(reason='exit'))


async def test_computer_use_toggle_upgrade_and_errors(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    work = checkout
    plugin = plugin_of(load(work))
    computer_use = next(command for command in plugin.get_commands() if command.name == 'computer-use')
    assert computer_use.complete is not None
    assert list(computer_use.complete([])) == ['on', 'off', 'view']
    assert list(computer_use.complete(['on', ''])) == []

    with pytest.raises(ValueError, match=r'^Usage: /computer-use \[on \| off \| view\]$'):
        await plugin.computer_use_command(['unknown'])

    # Without computer use on, a `computer` row has no viewer link.
    rendered = plugin.render(ActionsPerformedEvent(result=ComputerResult(performed=1)))
    assert rendered is not None and 'watch live' not in render_plain(rendered)

    # Local conversation arm/disarm and automatic disarm when switching to another conversation.
    assert await plugin.computer_use_command(['off']) == 'Computer use is off.'
    assert 'Computer use is on' in await plugin.computer_use_command(['on'])
    assert 'Computer use is off' in await plugin.computer_use_command(['off'])
    await plugin.computer_use_command(['on'])
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use']
    plugin.host.conversation = Transcript(conversation_id='c-2')
    assert [segment() for segment in plugin.get_status_segments()] == ['']

    # Retroactive attach (`desktop_at_create=False`): works immediately even after user turns have run, and survives resume.
    await plugin.command(['fake'])
    # A provider missing what a desktop needs says so.
    provider.requirement = 'Fake needs a display package.'
    with pytest.raises(ValueError, match='Fake needs a display package'):
        await plugin.computer_use_command(['on'])
    provider.requirement = None
    await plugin.host.conversation.commit_messages(
        [*plugin.host.conversation.messages, ModelRequest.user_text_prompt('ran a turn')]
    )
    assert 'already off' in await plugin.computer_use_command(['off'])
    enabled = await plugin.computer_use_command(['on'])
    assert 'Computer use is on' in enabled
    assert 'already on' in await plugin.computer_use_command(['on'])
    assert 'computer use on' in await plugin.command([])
    assert await on_screen_sandbox(plugin).screenshot() == PNG_A
    retro_resumed = plugin_of(load(work, Transcript(messages=plugin.host.conversation.messages)))
    assert [segment() for segment in retro_resumed.get_status_segments()] == ['fake sandbox (computer-use)']
    assert await on_screen_sandbox(retro_resumed).screenshot() == PNG_A
    assert provider.reopened[-1] == WorkspaceRef(provider='fake', id='sbx-1') and provider.reopened_desktop[-1]

    # A deleted sandbox refuses `/computer-use` and `/computer-use view`.
    await plugin.command(['delete'])
    with pytest.raises(ValueError, match='was deleted'):
        await plugin.computer_use_command([])
    with pytest.raises(ValueError, match='Computer use is not on here'):
        await plugin.computer_use_command(['view'])

    # A provider that gives a desktop only at create never replaces a sandbox to add one, even an unused one: a fork
    # or copy of the conversation may work in it, and only `/sandbox delete` deletes a sandbox.
    provider.desktop_at_create = True
    refused_plugin = plugin_of(load(work))
    await refused_plugin.command(['fake'])
    deleted = list(provider.deleted)
    with pytest.raises(
        ValueError,
        match=r'gives a desktop only to a new sandbox\. Run /sandbox delete, /new, /computer-use, then /sandbox fake',
    ):
        await refused_plugin.computer_use_command(['on'])
    assert provider.deleted == deleted
    with pytest.raises(ValueError, match='Computer use is not on here'):
        await refused_plugin.computer_use_command(['view'])
    with pytest.raises(UserError, match='has no desktop'):
        await on_screen_sandbox(refused_plugin).screenshot()

    # `/computer-use` first makes the sandbox with its desktop. Off and on again keeps it, also once compaction
    # dropped the markers.
    desktop_plugin = plugin_of(load(work))
    await desktop_plugin.computer_use_command(['on'])
    await desktop_plugin.command(['fake'])
    assert provider.created_desktop[-1] is True
    await desktop_plugin.computer_use_command(['off'])
    desktop_ref = history_sandbox(desktop_plugin.host.conversation.messages)
    await desktop_plugin.host.conversation.commit_messages(
        [
            ModelRequest.user_text_prompt('A summary of the earlier work.'),
            ModelResponse(parts=[TextPart('done')], workspace_ref=desktop_ref, metadata={COMPUTER_USE: False}),
        ]
    )
    created = len(provider.created)
    assert 'Computer use is on' in await desktop_plugin.computer_use_command(['on'])
    assert len(provider.created) == created


@pytest.mark.moves_cache_prefix(reason='`/permissions full` mid-conversation adds the `computer` tool')
async def test_the_computer_tool_needs_full_permissions(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    settings: dict[str, JsonValue] = {PERMISSIONS_SETTING: 'read-only'}
    loaded = load(checkout, settings=settings)
    plugin = plugin_of(loaded)
    planned: list[dict[str, JsonValue]] = []
    offered: list[list[str]] = []
    instructions: list[str] = []
    revoke_mid_turn: list[bool] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        offered.append([tool.name for tool in info.function_tools])
        instructions.append(info.instructions or '')
        if 'computer' not in offered[-1] or isinstance(messages[-1].parts[-1], ToolReturnPart):
            yield 'done'
        else:
            if revoke_mid_turn:
                settings[PERMISSIONS_SETTING] = 'auto'
            args = json.dumps({'actions': planned}) if planned else '{'
            yield {0: DeltaToolCall(name='computer', json_args=args, tool_call_id='cu')}

    session = on_screen(loaded, checkout, model=FunctionModel(stream_function=respond))
    needs_full = 'The model gets the computer tool only under /permissions full.'
    assert needs_full in await plugin.computer_use_command(['on'])
    assert needs_full in await plugin.command(['fake'])

    # The screen can change anything, so outside `full` there is no `computer` tool, as harness offers none on a
    # read-only workspace; the model is told why.
    await session.prompt('click it')
    assert 'computer' not in offered[-1]
    assert 'only once the user runs /permissions full' in instructions[-1]

    settings[PERMISSIONS_SETTING] = 'full'
    assert needs_full not in await plugin.computer_use_command(['on'])

    # Arguments that are not JSON go back to the model as a retry, and the viewer records nothing.
    await session.prompt('garbled')
    assert display.actions == [] and viewer_frames(plugin, WorkspaceRef(provider='fake', id='sbx-1')) is None

    planned[:] = [{'type': 'click', 'x': 1, 'y': 2}]
    await session.prompt('click it now')
    assert display.actions == [Click(x=1, y=2)]

    # `/permissions` can leave `full` while a turn runs: the steps after that no longer offer the tool.
    revoke_mid_turn.append(True)
    await session.prompt('click again')
    assert 'computer' in offered[-2] and 'computer' not in offered[-1]
    await plugin.on_session_end(SessionEnd(reason='exit'))


async def test_a_history_without_the_desktop_opening_the_sandbox_first(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    started_loaded = load(checkout)
    started = plugin_of(started_loaded)
    on_screen(started_loaded, checkout)
    await started.computer_use_command(['on'])
    await started.command(['fake'])
    with_desktop = started.host.conversation.messages
    await started.on_session_end(SessionEnd(reason='exit'))
    ref = WorkspaceRef(provider='fake', id='sbx-1')

    # A fork made before `/computer-use` opens the sandbox first, after a restart.
    loaded = load(checkout, Transcript(messages=list(with_desktop)))
    plugin = plugin_of(loaded)
    fork = [
        ModelRequest.user_text_prompt('before computer use'),
        ModelResponse(parts=[TextPart('ok')], workspace_ref=ref),
    ]
    plugin.sandbox(ref, fork).open()

    tools: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages
        tools.extend(tool.name for tool in info.function_tools)
        yield 'done'

    session = on_screen(loaded, checkout, model=FunctionModel(stream_function=respond))
    session.replace_messages(list(with_desktop))
    await session.prompt('look at the desktop')
    assert 'computer' in tools
    assert provider.reopened_desktop[-2:] == [False, True]
    await plugin.on_session_end(SessionEnd(reason='exit'))
    assert provider.closed[-2:] == [ref, ref]


async def test_view_and_side_panel(
    checkout: Path, provider: FakeProvider, display: FakeDisplay, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = checkout
    opened: list[str] = []
    browser_opens = True

    def fake_open(url: str) -> bool:
        opened.append(url)
        return browser_opens

    monkeypatch.setattr(sandbox, 'open_browser', fake_open)

    # Side panel before enabling computer use returns instructional note.
    plugin = plugin_of(load(work, terminal=True))
    [panel] = plugin.get_side_panels()
    assert panel.id == 'desktop'
    note_view = await panel.view()
    assert note_view.url is None and '/computer-use' in note_view.note

    # `/computer-use view` opens the loopback viewer through a private redirect page, so no token is in argv.
    stub_viewer_port(monkeypatch, plugin, 43210)
    await plugin.computer_use_command(['on'])
    await plugin.command(['fake'])
    msg = await plugin.computer_use_command(['view'])
    assert msg.startswith('Opened the desktop viewer in your browser: http://127.0.0.1:43210/?token=')
    assert len(opened) == 1 and opened[0].startswith('file://') and 'token=' not in opened[0]
    page = anyio.Path(opened[0].removeprefix('file://'))
    assert 'http://127.0.0.1:43210/?token=' in await page.read_text(encoding='utf-8')

    # When no browser opens, it prints the link instead.
    browser_opens = False
    assert (await plugin.computer_use_command(['view'])).startswith('Desktop viewer: http://127.0.0.1:43210/?token=')

    # Session end unlinks the private redirect page.
    await plugin.on_session_end(SessionEnd(reason='exit'))
    assert not await page.exists()

    # Headless view does not call open_browser and tolerates failed initial capture.
    display.fail_screenshot = WorkspaceError('display starting')
    headless = plugin_of(load(work, terminal=False))
    stub_viewer_port(monkeypatch, headless, 43211)
    await headless.computer_use_command(['on'])
    await headless.command(['fake'])
    headless_msg = await headless.computer_use_command(['view'])
    assert headless_msg.startswith('Desktop viewer: http://127.0.0.1:43211/?token=')
    assert len(opened) == 2

    # Once a frame exists, `/computer-use view` reuses it without a live capture.
    display.fail_screenshot = None
    await headless.computer_use_command(['view'])
    display.fail_screenshot = WorkspaceError('not asked for')
    assert (await headless.computer_use_command(['view'])).startswith('Desktop viewer: http://127.0.0.1:43211/?token=')
    display.fail_screenshot = None

    # Side panel now returns the viewer URL for the active desktop sandbox.
    [headless_panel] = headless.get_side_panels()
    active_view = await headless_panel.view()
    assert active_view.url is not None and active_view.url.startswith('http://127.0.0.1:43211/?token=')


# ---------------------------------------------------------------------------
# In-process DesktopViewer ASGI tests (security, frames, SSE, capture)
# ---------------------------------------------------------------------------


async def test_desktop_viewer_app_end_to_end() -> None:
    viewer = DesktopViewer()
    ref = WorkspaceRef(provider='fake-desktop', id='sbx-1')
    captured_png = PNG_B
    fail_capture = False

    async def capture() -> bytes:
        if fail_capture:
            raise WorkspaceError('screen offline')
        return captured_png

    session = viewer.register(DesktopSession(ref=ref, title='Fake', capture=capture))
    key = sandbox_key(ref)
    token = 'secret-token'
    app = create_viewer_app(viewer, token=token, hosts=frozenset({'127.0.0.1:8765'}))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8765') as client:
        assert (await client.get(f'/?token={token}&sandbox={key}', headers={'host': 'evil.example'})).status_code == 403
        assert (await client.get(f'/?token=wrong&sandbox={key}')).status_code == 403
        assert (await client.post(f'/api/capture?sandbox={key}')).status_code == 403

        # Unknown sandbox -> 404.
        assert (await client.get(f'/?token={token}&sandbox=missing')).status_code == 404

        # Valid HTML page with strict CSP and frame-ancestors for Pax web UI.
        page = await client.get(f'/?token={token}&sandbox={key}')
        assert page.status_code == 200
        csp = page.headers['content-security-policy']
        assert "frame-ancestors 'self' http://127.0.0.1:* http://localhost:*" in csp
        assert page.headers['cache-control'] == 'no-store'
        assert page.headers['referrer-policy'] == 'no-referrer'
        assert page.headers['x-content-type-options'] == 'nosniff'

        # Record a frame and fetch its PNG (never cached on disk); missing seq -> 404.
        recorded = session.record(PNG_A, actions=(Click(x=10, y=20),), result=ComputerResult(performed=1))
        img = await client.get(f'/frames/{recorded.seq}.png?token={token}&sandbox={key}')
        assert img.status_code == 200 and img.content == PNG_A and img.headers['content-type'] == 'image/png'
        assert img.headers['cache-control'] == 'no-store'
        assert img.headers['x-content-type-options'] == 'nosniff'
        assert (await client.get(f'/frames/999.png?token={token}&sandbox={key}')).status_code == 404

        # POST /api/capture captures live screenshot; errors return 502.
        cap = await client.post(f'/api/capture?sandbox={key}', headers={HEADER: token})
        assert cap.status_code == 200 and cap.json()['summary'] == 'Live snapshot'
        fail_capture = True
        assert (await client.post(f'/api/capture?sandbox={key}', headers={HEADER: token})).status_code == 502
        fail_capture = False

        # SSE stream replays existing frames on subscribe, pushes new frames, and terminates on session.close().
        sse_lines: list[str] = []
        subscribed = asyncio.Event()
        real_subscribe = session.subscribe

        def notify_subscribe() -> asyncio.Queue[DesktopFrame | None]:
            subscribed.set()
            return real_subscribe()

        session.subscribe = notify_subscribe

        async def read_sse() -> None:
            async with client.stream('GET', f'/events?token={token}&sandbox={key}') as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        sse_lines.append(line.removeprefix('data: '))

        task = asyncio.create_task(read_sse())
        await subscribed.wait()
        session.record(PNG_B, summary='pushed')
        session.close()
        await task
        session.subscribe = real_subscribe
        assert [json.loads(line)['summary'] for line in sse_lines] == ['click (10, 20)', 'Live snapshot', 'pushed']

    # Bounded ring buffer and full subscriber queue eviction.
    q = session.subscribe()
    for i in range(MAX_FRAMES + 5):
        session.record(PNG_A, summary=f'f-{i}')
    assert len(session.frames) == MAX_FRAMES
    session.close()
    assert q.full()
    session.unsubscribe(q)
    await viewer.close()


async def test_viewer_server_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise DesktopViewer.ensure_started(), url(), and close() via a WebServer.serve stub (no bound socket needed).

    stopped: list[int] = []
    built_apps: list[object] = []

    class _FakeServer:
        port = 51234

        async def stop(self) -> None:
            stopped.append(self.port)

    def fake_serve(build: Callable[[int], object], *, host: str, port: int) -> _FakeServer:
        assert host == '127.0.0.1' and port == 0
        built_apps.append(build(51234))
        return _FakeServer()

    monkeypatch.setattr(web_server.WebServer, 'serve', staticmethod(fake_serve))
    viewer = DesktopViewer()
    ref = WorkspaceRef(provider='fake-desktop', id='sbx-1')
    url_1 = viewer.url(ref)
    url_2 = viewer.url(ref)
    assert url_1 == url_2 and url_1.startswith('http://127.0.0.1:51234/?token=')
    assert viewer.try_url(ref) == url_1
    assert len(built_apps) == 1
    viewer.remove(ref)  # Removing an unregistered ref is a no-op.
    await viewer.close()
    assert stopped == [51234]

    def fail_serve(build: Callable[[int], object], *, host: str, port: int) -> _FakeServer:
        del build, host, port
        raise OSError('port blocked')

    monkeypatch.setattr(web_server.WebServer, 'serve', staticmethod(fail_serve))
    assert DesktopViewer().try_url(ref) is None
