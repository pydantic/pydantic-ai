"""`/computer-use`: desktop sandboxes, `ComputerUse` activation, rendering, the desktop link, and provider wiring."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import anyio
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
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.computer_use import ActionsPerformedEvent, Display, SupportsDisplay
from pydantic_ai_pax import Session
from pydantic_ai_pax.builtin_plugins import sandbox
from pydantic_ai_pax.builtin_plugins.sandbox import SandboxPlugin, history_sandbox
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
    COMPUTER_USE,
    format_action,
    format_actions,
    render_computer_event,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers import (
    GitHubCredential,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend
from pydantic_ai_pax.config import PERMISSIONS_SETTING, Settings
from pydantic_ai_pax.plugins import ConversationChanged, SessionEnd, SessionStart, Transcript, TurnStart

# `checkout`, `isolated_git_config`, `provider`, and `upstream` are fixtures shared with `/sandbox`'s tests.
from pydantic_ai_pax.ui.rendering.status import StatusLink
from tests.pax.test_sandbox import (
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


def on_screen_sandbox(plugin: SandboxPlugin):
    """The sandbox of the conversation `plugin` shows."""
    sandbox = plugin.current()
    assert sandbox is not None
    return sandbox


def screen(plugin: SandboxPlugin) -> Display | None:
    """The display of the sandbox `plugin` shows, attached as a run would attach it."""
    backend = on_screen_sandbox(plugin).open()
    return backend.display if isinstance(backend, SupportsDisplay) else None


def render_plain(renderable: object) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=False, width=120).print(renderable)
    return buf.getvalue().rstrip('\n')


# ---------------------------------------------------------------------------
# Action formatting & terminal rendering
# ---------------------------------------------------------------------------


def test_action_formatting_and_rendering() -> None:
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
    ok_many = render_computer_event(ActionsPerformedEvent(result=ComputerResult(performed=3)))
    assert ok_many is not None and render_plain(ok_many) == '  ⎿ 3 actions · screenshot'
    err = render_computer_event(ActionsPerformedEvent(result=ComputerResult(performed=0, error='bad\x1bkey')))
    assert err is not None and 'bad\\x1bkey' in render_plain(err)

    other_call = FunctionToolCallEvent(part=ToolCallPart('shell', {'command': 'ls'}))
    assert render_computer_event(other_call) is None


# ---------------------------------------------------------------------------
# End-to-end `/computer-use` toggle, resume & `/sandbox delete`
# ---------------------------------------------------------------------------


@pytest.mark.moves_cache_prefix(
    reason='Toggles /computer-use off and back on mid-conversation, intentionally changing offered tools and instructions.'
)
async def test_computer_use_end_to_end_toggle_and_resume(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    work = checkout
    loaded = load(work)
    plugin = plugin_of(loaded)

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
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use', '']

    started = await plugin.command(['fake'])
    assert 'Fake sandbox sbx-1' in started
    assert 'Computer use is on' in started
    assert provider.created == [GitHubCredential(login='Ada', token='gho_secret')]
    assert provider.created_desktop == [True]
    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox (computer-use)', '']
    ref = history_sandbox(session.messages)
    assert ref is not None
    assert ref == WorkspaceRef(provider='fake', id='sbx-1')

    # Now a turn receives `computer` and runs its actions on the desktop.
    result = await session.prompt('click and type')
    assert 'computer' in seen_tools[-1]
    assert result.response.workspace_ref == ref
    assert result.response.metadata == {'clai2_computer_use': True}
    assert display.actions == [Click(x=120, y=240), TypeText(text='hi')]
    [first_req] = [m for m in result.new_messages() if isinstance(m, ModelRequest)][:1]
    assert 'also has a desktop' in (first_req.instructions or '')

    rendered = plugin.render(ActionsPerformedEvent(result=ComputerResult(performed=2)))
    assert rendered is not None and render_plain(rendered) == '  ⎿ 2 actions · screenshot'

    # Toggling `/computer-use` off disables the `computer` tool while keeping the sandbox running, and survives resume and compaction.
    disabled = await plugin.computer_use_command([])
    assert 'Computer use is off' in disabled
    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox', '']
    await session.prompt('without desktop')
    assert 'computer' not in seen_tools[-1]

    # Drop everything before the last turn (simulating compaction discarding the `/computer-use off` marker).
    compacted_messages = session.messages[-2:]
    resumed_off_loaded = load(work, Transcript(messages=compacted_messages))
    resumed_off = plugin_of(resumed_off_loaded)
    assert [segment() for segment in resumed_off.get_status_segments()] == ['fake sandbox', '']
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

    # Resuming in a fresh plugin instance reopens with `desktop=True`.
    resumed_loaded = load(work, Transcript(messages=session.messages))
    resumed = plugin_of(resumed_loaded)
    assert screen(resumed) is display
    assert provider.reopened == [ref, ref] and provider.reopened_desktop == [True, True]

    # The history naming the sandbox, not the one on screen, says it has a desktop.
    unrelated = plugin_of(load(work, Transcript(messages=[])))
    assert unrelated.sandbox(ref, session.messages).desktop is True

    # `/sandbox delete` is the single delete command, and closes cleanly.
    deleted = await plugin.command(['delete'])
    assert 'Deleted Fake sandbox sbx-1' in deleted
    assert provider.deleted == [ref]
    await plugin.on_session_end(SessionEnd(reason='exit'))
    await resumed.on_session_end(SessionEnd(reason='exit'))


async def test_computer_use_toggle_upgrade_and_errors(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    work = checkout
    plugin = plugin_of(load(work))
    computer_use = next(command for command in plugin.get_commands() if command.name == 'computer-use')
    assert computer_use.complete is not None
    assert list(computer_use.complete([])) == ['on', 'off']
    assert list(computer_use.complete(['on', ''])) == []

    with pytest.raises(ValueError, match=r'^Usage: /computer-use \[on \| off\]$'):
        await plugin.computer_use_command(['unknown'])

    # Local conversation arm/disarm and automatic disarm when switching to another conversation.
    assert await plugin.computer_use_command(['off']) == 'Computer use is off.'
    assert 'Computer use is on' in await plugin.computer_use_command(['on'])
    assert 'Computer use is off' in await plugin.computer_use_command(['off'])
    await plugin.computer_use_command(['on'])
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use', '']
    plugin.host.conversation = Transcript(conversation_id='c-2')
    assert [segment() for segment in plugin.get_status_segments()] == ['', '']

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
    assert screen(plugin) is display
    retro_resumed = plugin_of(load(work, Transcript(messages=plugin.host.conversation.messages)))
    assert [segment() for segment in retro_resumed.get_status_segments()] == ['fake sandbox (computer-use)', '']
    assert screen(retro_resumed) is display
    assert provider.reopened[-1] == WorkspaceRef(provider='fake', id='sbx-1') and provider.reopened_desktop[-1]

    # A deleted sandbox refuses `/computer-use`.
    await plugin.command(['delete'])
    with pytest.raises(ValueError, match='was deleted'):
        await plugin.computer_use_command([])

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
    assert screen(refused_plugin) is None

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
        native = [type(tool).__name__ for tool in info.model_request_parameters.native_tools]
        offered.append([tool.name for tool in info.function_tools] + native)
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

    # Arguments that are not JSON go back to the model as a retry, and nothing runs on the desktop.
    await session.prompt('garbled')
    assert display.actions == []

    planned[:] = [{'type': 'click', 'x': 1, 'y': 2}]
    await session.prompt('click it now')
    assert display.actions == [Click(x=1, y=2)]

    # `/permissions` can leave `full` while a turn runs: the steps after that offer no computer tool of any kind.
    revoke_mid_turn.append(True)
    await session.prompt('click again')
    assert {'computer', 'ComputerTool'} <= set(offered[-2])
    assert {'computer', 'ComputerTool'}.isdisjoint(offered[-1])
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


@dataclass
class LinkedProvider(FakeProvider):
    """A fake provider whose website shows a sandbox's desktop on a link that expires, as Daytona's does."""

    desktop_link_seconds: int | None = 3600
    links: list[str] = field(default_factory=list[str])
    fail: bool = False

    async def desktop_url(self, backend: SandboxBackend) -> str | None:
        assert isinstance(backend, SupportsDisplay) and backend.display is not None and backend.ref is not None
        if self.fail:
            raise ConnectionError('the provider is down')
        link = f'https://desktop.example/{backend.ref.id}/vnc.html?token={len(self.links)}'
        self.links.append(link)
        return link


def row(plugin: SandboxPlugin) -> list[str | StatusLink]:
    return [segment() for segment in plugin.get_status_segments()]


def watch(token: int) -> StatusLink:
    return StatusLink(text='watch live', url=f'https://desktop.example/sbx-1/vnc.html?token={token}')


@pytest.fixture
def linked(provider: FakeProvider, display: FakeDisplay, monkeypatch: pytest.MonkeyPatch) -> LinkedProvider:
    linked = LinkedProvider(provider.root, provider.upstream, display=display)
    monkeypatch.setattr(sandbox, 'PROVIDERS', {linked.name: linked})
    return linked


async def test_the_status_row_links_to_the_desktop_while_it_is_on(checkout: Path, linked: LinkedProvider) -> None:
    loaded = load(checkout)
    plugin = plugin_of(loaded)
    await plugin.command(['fake'])
    assert row(plugin) == ['fake sandbox', '']

    await plugin.computer_use_command(['on'])
    assert row(plugin) == ['fake sandbox', watch(0)]
    # A turn does not ask for another while the link is fresh.
    await loaded.dispatch(TurnStart(text='go'))
    assert row(plugin) == ['fake sandbox', watch(0)]

    await plugin.computer_use_command(['off'])
    assert row(plugin) == ['fake sandbox', '']

    # A Pax resuming the conversation shows a link at once.
    resumed_loaded = load(checkout, Transcript(messages=[*loaded.host.conversation.messages[:-2]]))
    await resumed_loaded.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))
    assert row(plugin_of(resumed_loaded)) == ['fake sandbox', watch(1)]
    await plugin.on_session_end(SessionEnd(reason='exit'))
    await plugin_of(resumed_loaded).on_session_end(SessionEnd(reason='exit'))


async def test_an_expiring_link_is_renewed_before_it_stops_opening(
    checkout: Path, linked: LinkedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(sandbox, 'monotonic', lambda: now[0])
    loaded = load(checkout)
    plugin = plugin_of(loaded)
    await plugin.computer_use_command(['on'])
    await plugin.command(['fake'])
    assert row(plugin) == ['fake sandbox', watch(0)]

    now[0] += 49 * 60
    await loaded.dispatch(ConversationChanged(conversation_id=loaded.host.conversation.conversation_id, title='t'))
    assert row(plugin) == ['fake sandbox', watch(0)]
    now[0] += 2 * 60
    await loaded.dispatch(TurnStart(text='go'))
    assert row(plugin) == ['fake sandbox', watch(1)]

    # A renewal that fails keeps the link it has while that still opens.
    linked.fail = True
    now[0] += 51 * 60
    await loaded.dispatch(TurnStart(text='go'))
    assert row(plugin) == ['fake sandbox', watch(1)]
    # Past its hour, the row drops the link rather than offer one that no longer opens.
    now[0] += 10 * 60
    assert row(plugin) == ['fake sandbox (computer-use)', '']
    await plugin.on_session_end(SessionEnd(reason='exit'))


async def test_a_link_that_never_expires_is_got_once(
    checkout: Path, linked: LinkedProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # boxd's console link needs the owner's login, not a token, so it never expires.
    linked.desktop_link_seconds = None
    now = [1000.0]
    monkeypatch.setattr(sandbox, 'monotonic', lambda: now[0])
    loaded = load(checkout)
    plugin = plugin_of(loaded)
    await plugin.computer_use_command(['on'])
    await plugin.command(['fake'])
    now[0] += 24 * 3600
    await loaded.dispatch(TurnStart(text='go'))
    assert row(plugin) == ['fake sandbox', watch(0)]
    assert len(linked.links) == 1
    await plugin.on_session_end(SessionEnd(reason='exit'))


async def test_a_link_that_cannot_be_got_leaves_the_row_without_one(
    checkout: Path, linked: LinkedProvider, caplog: pytest.LogCaptureFixture
) -> None:
    linked.fail = True
    plugin = plugin_of(load(checkout))
    await plugin.computer_use_command(['on'])
    assert 'Computer use is on' in await plugin.command(['fake'])
    assert row(plugin) == ['fake sandbox (computer-use)', '']
    assert 'Could not get the link to watch the Fake desktop: the provider is down' in caplog.text
    await plugin.on_session_end(SessionEnd(reason='exit'))


async def test_a_turn_cut_off_mid_stream_keeps_computer_use_on(
    checkout: Path, provider: FakeProvider, display: FakeDisplay
) -> None:
    """Core keeps an interrupted turn's partial response without `after_model_request`, so it has no `COMPUTER_USE`."""
    loaded = load(checkout)
    plugin = plugin_of(loaded)
    offered: list[list[str]] = []
    streaming = anyio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        offered.append([tool.name for tool in info.function_tools])
        yield 'half'
        if not streaming.is_set():
            streaming.set()
            await anyio.sleep_forever()

    session = on_screen(loaded, checkout, model=FunctionModel(stream_function=respond))
    await plugin.computer_use_command(['on'])
    await plugin.command(['fake'])
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(session.prompt, 'click it')
        await streaming.wait()
        tasks.cancel_scope.cancel()
    cut_off = session.messages[-1]
    assert isinstance(cut_off, ModelResponse) and cut_off.parts and COMPUTER_USE not in (cut_off.metadata or {})

    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox (computer-use)', '']
    resumed = plugin_of(load(checkout, Transcript(messages=session.messages)))
    assert on_screen_sandbox(resumed).desktop is True
    await session.prompt('click it again')
    assert 'computer' in offered[-1]
    await plugin.on_session_end(SessionEnd(reason='exit'))
    await resumed.on_session_end(SessionEnd(reason='exit'))
