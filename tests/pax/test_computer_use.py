"""`/computer-use`: desktop sandboxes, `ComputerUse` activation, rendering, live viewing, and provider wiring."""

from __future__ import annotations

import asyncio
import io
import json
import struct
import zlib
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path

import anyio
import httpx
import pytest
from rich.console import Console

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ToolCallPart,
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
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import (
    ActionsPerformedEvent,
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsDisplay,
    SupportsZoom,
    UnsupportedActionError,
)
from pydantic_ai_pax import Session
from pydantic_ai_pax.builtin_plugins import sandbox
from pydantic_ai_pax.builtin_plugins.sandbox import SandboxPlugin, history_sandbox
from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
    ActionMarker,
    action_markers,
    format_action,
    format_actions,
    render_computer_event,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers import (
    GitHubCredential,
    boxd as boxd_provider,
    daytona as daytona_provider,
    e2b as e2b_provider,
    kubernetes as kubernetes_provider,
    local as local_provider,
    modal as modal_provider,
    opensandbox as opensandbox_provider,
    render as render_provider,
    sprites as sprites_provider,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, is_desktop_ref, ref_provider
from pydantic_ai_pax.builtin_plugins.sandbox.viewer import (
    HEADER,
    MAX_FRAMES,
    DesktopFrame,
    DesktopSession,
    DesktopViewer,
    _ViewerGuard,  # pyright: ignore[reportPrivateUsage]
    create_viewer_app,
    sandbox_key,
)
from pydantic_ai_pax.plugins import LoadedPlugin, PluginHost, SessionEnd, Transcript, load_plugin
from tests.pax.test_sandbox import git


def make_png(width: int = 8, height: int = 6, color: tuple[int, int, int] = (20, 40, 80)) -> bytes:
    """Minimal valid RGB PNG of the given dimensions, without Pillow."""
    raw = b''.join(b'\x00' + bytes(color) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw))
        + chunk(b'IEND', b'')
    )


PNG_A = make_png(1280, 800, (10, 20, 30))
PNG_B = make_png(1280, 800, (40, 50, 60))


class FakeDisplay:
    """In-memory `Display` implementing zoom and cursor position for desktop sandbox tests."""

    def __init__(self, *, png: bytes = PNG_A, size: tuple[int, int] = (1280, 800)) -> None:
        self.png = png
        self.size = size
        self.actions: list[DisplayAction] = []
        self.fail_screenshot: Exception | None = None

    async def display_size(self) -> tuple[int, int]:
        return self.size

    async def screenshot(self) -> bytes:
        if self.fail_screenshot is not None:
            raise self.fail_screenshot
        return self.png

    async def act(self, action: DisplayAction) -> None:
        self.actions.append(action)

    async def zoom(self, region: Region) -> bytes:
        del region
        return self.png

    async def cursor_position(self) -> Point:
        return Point(x=100, y=200)


class FakeDesktopBackend(LocalWorkspaceBackend):
    """Local folder workspace with an optional `Display`."""

    def __init__(self, ref: WorkspaceRef, directory: Path, *, env: Mapping[str, str], display: Display | None) -> None:
        super().__init__(directory, env=env)
        self._sandbox_ref = ref
        self._display = display

    @property
    def ref(self) -> WorkspaceRef:
        return self._sandbox_ref

    @property
    def display(self) -> Display | None:
        return self._display


@dataclass
class FakeDesktopProvider:
    """Fake sandbox provider supporting both plain and desktop sandboxes and optional native desktop URLs."""

    root: Path
    upstream: Path
    name: str = 'fake'
    title: str = 'Fake'
    working_dir: str = '/home/user/workspace'
    token_in_sandbox: bool = False
    has_desktop_url: bool = False
    desktop_url_is_bearer: bool = False
    desktop_at_create: bool = False
    requirement: str | None = None
    live_url_value: str | None = None
    live_url_error: Exception | None = None
    delete_error: Exception | None = None
    display: FakeDisplay = field(default_factory=FakeDisplay)
    created: list[tuple[GitHubCredential | None, bool]] = field(
        default_factory=list[tuple[GitHubCredential | None, bool]]
    )
    reopened: list[tuple[WorkspaceRef, bool]] = field(default_factory=list[tuple[WorkspaceRef, bool]])
    deleted: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])
    closed: list[WorkspaceRef] = field(default_factory=list[WorkspaceRef])

    def missing(self, *, desktop: bool = False) -> str | None:
        del desktop
        return self.requirement

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, FakeDesktopBackend]:
        self.created.append((github, desktop))
        env = {**SANDBOX_ENV, 'GH_TOKEN': 'stand-in'} if github is not None else SANDBOX_ENV
        ref = WorkspaceRef(provider=ref_provider(self.name, desktop=desktop), id=f'sbx-{len(self.created)}')
        folder = anyio.Path(self.root / ref.id)
        await folder.mkdir(exist_ok=True)
        await anyio.Path(self.root / f'{ref.id}.gitconfig').write_text(
            f'[url "{self.upstream.parent}/"]\n\tinsteadOf = https://github.com/owner/\n', encoding='utf-8'
        )
        return ref, self._backend(ref, env, desktop=desktop)

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> FakeDesktopBackend:
        self.reopened.append((ref, desktop))
        return self._backend(ref, {}, desktop=desktop)

    async def desktop_url(self, backend: FakeDesktopBackend) -> str | None:
        del backend
        if self.live_url_error is not None:
            raise self.live_url_error
        return self.live_url_value

    async def delete(self, ref: WorkspaceRef) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(ref)

    async def close(self, backend: FakeDesktopBackend) -> None:
        self.closed.append(backend.ref)

    def _backend(self, ref: WorkspaceRef, env: Mapping[str, str], *, desktop: bool) -> FakeDesktopBackend:
        git_env = {'GIT_CONFIG_GLOBAL': str(self.root / f'{ref.id}.gitconfig'), 'GIT_CONFIG_NOSYSTEM': '1'}
        return FakeDesktopBackend(
            ref, self.root / ref.id, env={**env, **git_env}, display=self.display if desktop else None
        )


def _not_installed(name: str) -> None:
    del name
    return None


def _gh_token(hostname: str) -> str:
    del hostname
    return 'gho_secret'


def _gh_user(hostname: str) -> str:
    del hostname
    return 'Ada'


@pytest.fixture(autouse=True)
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / 'global.gitconfig'
    config.touch()
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', str(config))
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.delenv('GIT_CONFIG_COUNT', raising=False)
    monkeypatch.delenv('GIT_CONFIG_PARAMETERS', raising=False)


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, Path]:
    seedling = tmp_path / 'seedling'
    seedling.mkdir()
    git(seedling, 'init', '--quiet', '--initial-branch=main')
    (seedling / 'README.md').write_text('hello\n', encoding='utf-8')
    git(seedling, 'add', '.')
    git(seedling, '-c', 'user.name=Up', '-c', 'user.email=up@example.com', 'commit', '--quiet', '-m', 'Initial')
    bare = tmp_path / 'github' / 'repo.git'
    git(tmp_path, 'clone', '--quiet', '--bare', str(seedling), str(bare))
    work = tmp_path / 'checkout'
    git(tmp_path, 'clone', '--quiet', str(bare), str(work))
    git(work, 'remote', 'set-url', 'origin', 'git@github.com:owner/repo.git')
    git(work, 'config', 'user.name', 'Ada')
    git(work, 'config', 'user.email', 'ada@example.com')
    return work, bare


@pytest.fixture
def desktop_provider(
    tmp_path: Path, checkout: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> FakeDesktopProvider:
    _, bare = checkout
    sandboxes = tmp_path / 'sandboxes'
    sandboxes.mkdir()
    fake = FakeDesktopProvider(sandboxes, bare)
    monkeypatch.setattr(sandbox, 'PROVIDERS', {fake.name: fake})
    monkeypatch.setattr(sandbox, 'gh_token', _gh_token)
    monkeypatch.setattr(sandbox, 'gh_user', _gh_user)
    return fake


def load(directory: Path, conversation: Transcript | None = None, *, terminal: bool = False) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='sandbox',
        console=Console(file=io.StringIO(), force_terminal=terminal, width=120),
        settings={},
        conversation=conversation or Transcript(),
        workspace=lambda: directory,
    )
    return load_plugin(SandboxPlugin, host)


def plugin_of(loaded: LoadedPlugin[None]) -> SandboxPlugin:
    assert isinstance(loaded.plugin, SandboxPlugin)
    return loaded.plugin


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
        show_hint=True,
    )
    assert ok_many is not None
    rendered_many = render_plain(ok_many)
    assert '3 actions · screenshot · watch live ↗ (/computer-use view)' in rendered_many
    ok_no_hint = render_computer_event(
        ActionsPerformedEvent(result=ComputerResult(performed=2)),
        live_url='http://127.0.0.1:9999/?token=t&sandbox=s',
    )
    assert ok_no_hint is not None
    assert '2 actions · screenshot · watch live ↗' in render_plain(ok_no_hint)
    assert '/computer-use view' not in render_plain(ok_no_hint)
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
    checkout: tuple[Path, Path], desktop_provider: FakeDesktopProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, _ = checkout
    loaded = load(work)
    plugin = plugin_of(loaded)
    monkeypatch.setattr(plugin._viewer, 'ensure_started', lambda: 43219)  # pyright: ignore[reportPrivateUsage]

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
    assert 'Computer use enabled' in armed
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use']

    started = await plugin.command(['fake'])
    assert 'Fake sandbox sbx-1' in started
    assert 'Computer use enabled (1280x800 desktop)' in started
    assert desktop_provider.created == [(GitHubCredential(login='Ada', token='gho_secret'), True)]
    assert [segment() for segment in plugin.get_status_segments()] == ['fake sandbox (computer-use)']
    ref = history_sandbox(session.messages)
    assert ref is not None
    assert ref == WorkspaceRef(provider='fake-desktop', id='sbx-1') and is_desktop_ref(ref)

    # Now a turn receives `computer`, starts the loopback viewer lazily, executes actions, and records the frame.
    result = await session.prompt('click and type')
    assert 'computer' in seen_tools[-1]
    assert result.response.workspace_ref == WorkspaceRef(provider='fake-desktop', id='sbx-1')
    assert desktop_provider.display.actions == [Click(x=120, y=240), TypeText(text='hi')]
    [first_req] = [m for m in result.new_messages() if isinstance(m, ModelRequest)][:1]
    assert 'graphical desktop' in (first_req.instructions or '')

    viewer_key = sandbox_key(WorkspaceRef(provider='fake', id='sbx-1'))
    viewer_session = plugin._viewer.get(viewer_key)  # pyright: ignore[reportPrivateUsage]
    assert viewer_session is not None
    assert len(viewer_session.frames) == 1
    assert viewer_session.frames[0].png == PNG_A
    assert viewer_session.frames[0].summary == 'click (120, 240) · type "hi"'
    assert viewer_session.frames[0].markers == (
        ActionMarker(kind='click', points=((120, 240),), label='click (120, 240)'),
    )

    # Render hook surfaces the inline `watch live ↗` link with a one-time `(/computer-use view)` hint.
    first_render = plugin.render(ActionsPerformedEvent(result=ComputerResult(performed=2)))
    assert first_render is not None
    assert 'watch live ↗ (/computer-use view)' in render_plain(first_render)
    second_render = plugin.render(ActionsPerformedEvent(result=ComputerResult(performed=1)))
    assert second_render is not None
    assert 'watch live ↗' in render_plain(second_render)
    assert '/computer-use view' not in render_plain(second_render)

    # Toggling `/computer-use` off disables the `computer` tool while keeping the sandbox running, and survives resume and compaction.
    disabled = await plugin.computer_use_command([])
    assert 'Computer use disabled' in disabled
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
    assert resumed_off.sandbox(ref).has_desktop is True
    await resumed_off.on_session_end(SessionEnd(reason='exit'))

    # Toggling `/computer-use` back on re-enables immediately without recreating the sandbox.
    re_enabled = await plugin.computer_use_command([])
    assert 'Computer use enabled' in re_enabled
    assert len(desktop_provider.created) == 1
    await session.prompt('back with desktop')
    assert 'computer' in seen_tools[-1]

    # Unowned and non-desktop refs are ignored by `_record_frame`, and _ObservedComputerUse handles missing actions/screenshots.
    unowned_ref = WorkspaceRef(provider='other', id='sbx-1')
    plugin._record_frame(unowned_ref, PNG_A, (), None)  # pyright: ignore[reportPrivateUsage]
    assert plugin._viewer.get(sandbox_key(unowned_ref)) is None  # pyright: ignore[reportPrivateUsage]
    plain_ref = WorkspaceRef(provider='fake', id='sbx-plain')
    plugin._record_frame(plain_ref, PNG_A, (), None)  # pyright: ignore[reportPrivateUsage]
    assert plugin._viewer.get(sandbox_key(plain_ref)) is None  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai import FunctionToolResultEvent, RunContext
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.computer_use import ComputerReturnPart
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage
    from pydantic_ai_pax.builtin_plugins.sandbox.computer_use import (
        _ObservedComputerUse,  # pyright: ignore[reportPrivateUsage]
    )

    captured_frames: list[bytes] = []
    observer = _ObservedComputerUse(
        AbstractCapability[None](), on_frame=lambda _r, data, _a, _res: captured_frames.append(data)
    )
    dummy_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    await observer.on_event(
        dummy_ctx, event=FunctionToolCallEvent(part=ComputerCallPart(tool_name='computer', args='{'))
    )
    await observer.on_event(
        dummy_ctx,
        event=FunctionToolResultEvent(
            part=ComputerReturnPart(tool_name='computer', content=ComputerResult(performed=0, error='boom'))
        ),
    )
    assert captured_frames == []

    # Resuming in a fresh plugin instance reopens with `desktop=True` and seeds the viewer from saved history.
    resumed_loaded = load(work, Transcript(messages=session.messages))
    resumed = plugin_of(resumed_loaded)
    resumed_session = resumed._viewer_session(resumed.sandbox(ref))  # pyright: ignore[reportPrivateUsage]
    assert len(resumed_session.frames) == 1
    assert resumed_session.frames[0].png == PNG_A
    assert await resumed.sandbox(ref).capture_screen() == PNG_A
    assert desktop_provider.reopened == [(ref, True), (ref, True)]

    # Viewer session for a sandbox not in the active transcript skips seeding from unrelated history.
    unrelated = plugin_of(load(work, Transcript(messages=[])))
    assert unrelated._viewer_session(unrelated.sandbox(ref)).frames == []  # pyright: ignore[reportPrivateUsage]

    # `/sandbox delete` is the single delete command; it removes the viewer session and closes cleanly.
    deleted = await plugin.command(['delete'])
    assert 'Deleted Fake sandbox sbx-1' in deleted
    assert desktop_provider.deleted == [ref]
    assert plugin._viewer.get(viewer_key) is None  # pyright: ignore[reportPrivateUsage]
    await plugin.on_session_end(SessionEnd(reason='exit'))
    await resumed.on_session_end(SessionEnd(reason='exit'))


async def test_computer_use_toggle_upgrade_and_errors(
    checkout: tuple[Path, Path], desktop_provider: FakeDesktopProvider
) -> None:
    work, _ = checkout
    plugin = plugin_of(load(work))
    assert list(plugin.computer_use_complete([])) == ['on', 'off', 'view']
    assert list(plugin.computer_use_complete(['on', ''])) == []

    with pytest.raises(ValueError, match=r'^Usage: /computer-use \[on \| off \| view\]$'):
        await plugin.computer_use_command(['unknown'])

    # Local conversation arm/disarm and automatic disarm when switching to another conversation.
    assert 'already off' in await plugin.computer_use_command(['off'])
    assert 'Computer use enabled' in await plugin.computer_use_command(['on'])
    assert 'Computer use disabled' in await plugin.computer_use_command(['off'])
    await plugin.computer_use_command(['on'])
    assert [segment() for segment in plugin.get_status_segments()] == ['computer-use']
    plugin.host.conversation = Transcript(conversation_id='c-2')
    assert [segment() for segment in plugin.get_status_segments()] == ['']

    # Retroactive attach (`desktop_at_create=False`): works immediately even after user turns have run, and survives resume.
    await plugin.command(['fake'])
    await plugin.host.conversation.commit_messages(
        [*plugin.host.conversation.messages, ModelRequest.user_text_prompt('ran a turn')]
    )
    assert 'already off' in await plugin.computer_use_command(['off'])
    enabled = await plugin.computer_use_command(['on'])
    assert 'Computer use enabled for Fake sandbox sbx-1' in enabled
    assert 'already enabled' in await plugin.computer_use_command(['on'])
    assert 'computer use on' in await plugin.command([])
    assert await plugin.sandbox(WorkspaceRef(provider='fake-desktop', id='sbx-1')).capture_screen() == PNG_A
    retro_resumed = plugin_of(load(work, Transcript(messages=plugin.host.conversation.messages)))
    assert [segment() for segment in retro_resumed.get_status_segments()] == ['fake sandbox (computer-use)']
    assert await retro_resumed.sandbox(WorkspaceRef(provider='fake', id='sbx-1')).capture_screen() == PNG_A
    assert desktop_provider.reopened[-1] == (WorkspaceRef(provider='fake', id='sbx-1'), True)

    # Deleted sandbox refuses `/computer-use` and `/computer-use view`.
    await plugin.command(['delete'])
    with pytest.raises(ValueError, match='was deleted'):
        await plugin.computer_use_command([])
    with pytest.raises(ValueError, match='was deleted'):
        await plugin.computer_use_command(['view'])

    # `desktop_at_create=True` on an UNUSED sandbox automatically restarts it with `desktop=True` (no `/new` needed).
    desktop_provider.desktop_at_create = True
    upgrade_plugin = plugin_of(load(work))
    await upgrade_plugin.command(['fake'])
    assert history_sandbox(upgrade_plugin.host.conversation.messages) == WorkspaceRef(provider='fake', id='sbx-2')
    upgraded = await upgrade_plugin.computer_use_command([])
    assert 'restarted unused Fake sandbox with a desktop (sbx-3)' in upgraded
    assert history_sandbox(upgrade_plugin.host.conversation.messages) == WorkspaceRef(
        provider='fake-desktop', id='sbx-3'
    )
    assert desktop_provider.deleted[-1] == WorkspaceRef(provider='fake', id='sbx-2')

    # If deleting the replaced unused sandbox fails, retain it and retry cleanup on `/sandbox delete`.
    orphan_plugin = plugin_of(load(work))
    await orphan_plugin.command(['fake'])
    old_orphan_ref = history_sandbox(orphan_plugin.host.conversation.messages)
    assert old_orphan_ref == WorkspaceRef(provider='fake', id='sbx-4')
    desktop_provider.delete_error = WorkspaceError('transient delete failure')
    orphan_msg = await orphan_plugin.computer_use_command(['on'])
    assert 'restarted unused Fake sandbox with a desktop (sbx-5)' in orphan_msg
    assert 'Could not delete replaced sandbox sbx-4; /sandbox delete will retry it.' in orphan_msg
    with pytest.raises(WorkspaceError, match='transient delete failure'):
        await orphan_plugin.command(['delete'])
    desktop_provider.delete_error = None
    assert 'Deleted Fake sandbox sbx-5' in await orphan_plugin.command(['delete'])
    assert desktop_provider.deleted[-2:] == [
        WorkspaceRef(provider='fake', id='sbx-4'),
        WorkspaceRef(provider='fake-desktop', id='sbx-5'),
    ]

    # `desktop_at_create=True` on an ALREADY USED sandbox refuses safely without losing work.
    used_plugin = plugin_of(load(work))
    await used_plugin.command(['fake'])
    await used_plugin.host.conversation.commit_messages(
        [*used_plugin.host.conversation.messages, ModelRequest.user_text_prompt('prior work')]
    )
    with pytest.raises(ValueError, match=r'already has conversation work.*Run /new, /computer-use, then /sandbox fake'):
        await used_plugin.computer_use_command(['on'])
    with pytest.raises(ValueError, match='Computer use is not active'):
        await used_plugin.computer_use_command(['view'])
    with pytest.raises(UserError, match='has no desktop display'):
        await used_plugin.sandbox(WorkspaceRef(provider='fake', id='sbx-6')).capture_screen()


async def test_view_native_url_fallback_and_side_panel(
    checkout: tuple[Path, Path], desktop_provider: FakeDesktopProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, _ = checkout
    opened: list[str] = []

    def fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(sandbox, 'open_browser', fake_open)

    # Side panel before enabling computer use returns instructional note.
    plugin = plugin_of(load(work, terminal=True))
    [panel] = plugin.get_side_panels()
    assert panel.id == 'desktop'
    note_view = await panel.view()
    assert note_view.url is None and '/computer-use' in note_view.note

    # Provider with native live desktop URL -> `/computer-use view` opens it via a private redirect file (no token in argv).
    desktop_provider.has_desktop_url = True
    desktop_provider.live_url_value = 'https://console.example/desktop/sbx-1'
    await plugin.computer_use_command(['on'])
    await plugin.command(['fake'])
    msg = await plugin.computer_use_command(['view'])
    assert msg == 'Opened live desktop in your browser: https://console.example/desktop/sbx-1'
    assert len(opened) == 1 and opened[0].startswith('file://')
    first_page = anyio.Path(opened[0].removeprefix('file://'))
    assert 'https://console.example/desktop/sbx-1' in await first_page.read_text(encoding='utf-8')

    # If native URL fails, falls back to local viewer (stubbing ensure_started so no socket binds inside sandbox).
    monkeypatch.setattr(plugin._viewer, 'ensure_started', lambda: 43210)  # pyright: ignore[reportPrivateUsage]
    desktop_provider.live_url_error = WorkspaceError('temporarily unavailable')
    fallback_msg = await plugin.computer_use_command(['view'])
    assert fallback_msg.startswith('Opened desktop viewer in your browser: http://127.0.0.1:43210/?token=')
    assert len(opened) == 2 and opened[1].startswith('file://') and 'token=' not in opened[1]
    second_page = anyio.Path(opened[1].removeprefix('file://'))
    assert 'http://127.0.0.1:43210/?token=' in await second_page.read_text(encoding='utf-8')

    # Session end unlinks the private redirect files.
    await plugin.on_session_end(SessionEnd(reason='exit'))
    assert not await first_page.exists() and not await second_page.exists()

    # Bearer native URL (e.g. Daytona) defaults to the loopback viewer instead of minting/printing a bearer link.
    desktop_provider.live_url_error = None
    desktop_provider.desktop_url_is_bearer = True
    bearer_plugin = plugin_of(load(work, terminal=True))
    monkeypatch.setattr(bearer_plugin._viewer, 'ensure_started', lambda: 43212)  # pyright: ignore[reportPrivateUsage]
    await bearer_plugin.computer_use_command(['on'])
    await bearer_plugin.command(['fake'])
    bearer_msg = await bearer_plugin.computer_use_command(['view'])
    assert bearer_msg.startswith('Opened desktop viewer in your browser: http://127.0.0.1:43212/?token=')
    await bearer_plugin.on_session_end(SessionEnd(reason='exit'))
    desktop_provider.desktop_url_is_bearer = False

    # Headless view does not call open_browser and tolerates failed initial capture.
    desktop_provider.has_desktop_url = False
    desktop_provider.display.fail_screenshot = WorkspaceError('display starting')
    headless = plugin_of(load(work, terminal=False))
    monkeypatch.setattr(headless._viewer, 'ensure_started', lambda: 43211)  # pyright: ignore[reportPrivateUsage]
    await headless.computer_use_command(['on'])
    await headless.command(['fake'])
    headless_msg = await headless.computer_use_command(['view'])
    assert headless_msg.startswith('Desktop viewer: http://127.0.0.1:43211/?token=')
    assert len(opened) == 3

    # Once a frame exists, `/computer-use view` reuses existing frames without triggering a live capture.
    desktop_provider.display.fail_screenshot = None
    headless_ref = history_sandbox(headless.host.conversation.messages)
    assert headless_ref is not None
    headless._record_frame(headless_ref, PNG_A, (), None)  # pyright: ignore[reportPrivateUsage]
    assert (await headless.computer_use_command(['view'])).startswith('Desktop viewer: http://127.0.0.1:43211/?token=')

    # Side panel now returns the viewer URL for the active desktop sandbox.
    [headless_panel] = headless.get_side_panels()
    active_view = await headless_panel.view()
    assert active_view.url is not None and active_view.url.startswith('http://127.0.0.1:43211/?token=')


# ---------------------------------------------------------------------------
# In-process DesktopViewer ASGI tests (security, frames, SSE, capture, live-url)
# ---------------------------------------------------------------------------


async def test_desktop_viewer_app_end_to_end() -> None:
    viewer = DesktopViewer()
    ref = WorkspaceRef(provider='fake-desktop', id='sbx-1')
    captured_png = PNG_B
    live_url_result: str | None = 'https://live.example/vnc'
    fail_capture = False
    fail_live = False

    async def capture() -> bytes:
        if fail_capture:
            raise WorkspaceError('screen offline')
        return captured_png

    async def live_url() -> str | None:
        if fail_live:
            raise WorkspaceError('preview failed')
        return live_url_result

    session = viewer.register(
        DesktopSession(ref=ref, title='Fake', capture=capture, live_url=live_url, has_live_url=True)
    )
    key = sandbox_key(ref)
    token = 'secret-token'
    app = create_viewer_app(viewer, token=token, hosts=frozenset({'127.0.0.1:8765'}))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:8765') as client:
        assert (await client.get(f'/?token={token}&sandbox={key}', headers={'host': 'evil.example'})).status_code == 403
        assert (await client.get(f'/?token=wrong&sandbox={key}')).status_code == 403
        assert (await client.post(f'/api/capture?sandbox={key}')).status_code == 403
        assert (
            await client.post(f'/api/capture?sandbox={key}', headers={HEADER: token, 'origin': 'https://evil.example'})
        ).status_code == 403

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
        assert 'Interactive desktop' in page.text

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

        # POST /api/live-url returns URL, 404 when absent, 502 on error.
        live = await client.post(f'/api/live-url?sandbox={key}', headers={HEADER: token})
        assert live.status_code == 200 and live.json() == {'url': 'https://live.example/vnc'}
        live_url_result = None
        assert (await client.post(f'/api/live-url?sandbox={key}', headers={HEADER: token})).status_code == 404
        fail_live = True
        assert (await client.post(f'/api/live-url?sandbox={key}', headers={HEADER: token})).status_code == 502

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


async def test_viewer_guard_and_server_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    async def inner(scope: Mapping[str, object], receive: object, send: object) -> None:
        del receive, send
        seen.append(scope['type'])

    async def noop_receive() -> dict[str, object]:
        return {}

    async def noop_send(message: Mapping[str, object]) -> None:
        del message

    guard = _ViewerGuard(inner, token='tok', hosts=frozenset({'127.0.0.1:1'}))
    await guard({'type': 'lifespan'}, noop_receive, noop_send)
    assert seen == ['lifespan']

    # Exercise DesktopViewer.ensure_started(), url(), and close() via a WebServer.serve stub (no bound socket needed).
    from pydantic_ai_pax.builtin_plugins.web import server as web_server

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


# ---------------------------------------------------------------------------
# Provider desktop=True wiring & native desktop_url tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(find_spec('boxd') is None, reason='boxd not installed')
async def test_boxd_desktop_url_and_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

    class _BoxdAccess:
        desktop_url = 'https://console.boxd.sh/machines/m1/desktop'

    class _BoxdMachine:
        id = 'm1'
        access = _BoxdAccess()

    boxd_p = boxd_provider.BoxdProvider()
    assert boxd_p.has_desktop_url is True and boxd_p.desktop_url_is_bearer is False
    monkeypatch.setenv(boxd_provider.KEY_NAME, 'boxd-key')
    boxd_backend = boxd_p.reopen(WorkspaceRef(provider='boxd-desktop', id='m1'), desktop=True)
    assert isinstance(boxd_backend, BoxdSandboxBackend) and boxd_backend.display is not None

    async def fake_boxd_get() -> _BoxdMachine:
        return _BoxdMachine()

    monkeypatch.setattr(boxd_backend, 'get_sandbox', fake_boxd_get)
    assert await boxd_p.desktop_url(boxd_backend) == 'https://console.boxd.sh/machines/m1/desktop'
    await boxd_p.close(boxd_backend)
    await boxd_p.close(boxd_backend)


@pytest.mark.skipif(find_spec('daytona') is None, reason='daytona not installed')
async def test_daytona_desktop_url_and_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    import daytona

    from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

    class _FakeDaytonaClient:
        closed = False

        def __init__(self, config: object = None) -> None:
            del config

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(daytona, 'AsyncDaytona', _FakeDaytonaClient)
    daytona_p = daytona_provider.DaytonaProvider()
    assert daytona_p.has_desktop_url is True and daytona_p.desktop_url_is_bearer is True
    monkeypatch.setenv(daytona_provider.KEY_NAME, 'dtn-key')
    dtn_backend = daytona_p.reopen(WorkspaceRef(provider='daytona-desktop', id='d1'), desktop=True)
    assert isinstance(dtn_backend, DaytonaSandboxBackend) and dtn_backend.display is not None

    signed_calls: list[tuple[int, int | None]] = []

    class _SignedUrl:
        url = 'https://6080-d1.proxy.daytona.works/?token=abc'

    class _FakeDaytonaSbx:
        async def create_signed_preview_url(self, port: int, *, expires_in_seconds: int | None = None) -> _SignedUrl:
            signed_calls.append((port, expires_in_seconds))
            return _SignedUrl()

    async def fake_dtn_size() -> tuple[int, int]:
        return (1280, 800)

    async def fake_dtn_get() -> _FakeDaytonaSbx:
        return _FakeDaytonaSbx()

    monkeypatch.setattr(dtn_backend.display, 'display_size', fake_dtn_size)
    monkeypatch.setattr(dtn_backend, 'get_sandbox', fake_dtn_get)
    assert await daytona_p.desktop_url(dtn_backend) == 'https://6080-d1.proxy.daytona.works/?token=abc'
    assert signed_calls == [(daytona_provider.VNC_PORT, daytona_provider.PREVIEW_EXPIRES_SECONDS)]

    dtn_plain = daytona_p.reopen(WorkspaceRef(provider='daytona', id='d1'), desktop=False)
    assert await daytona_p.desktop_url(dtn_plain) is None
    await daytona_p.close(dtn_backend)
    await daytona_p.close(dtn_plain)


@pytest.mark.skipif(find_spec('opensandbox') is None, reason='opensandbox not installed')
async def test_opensandbox_desktop_bootstraps_before_github_vault(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai_harness.opensandbox_sandbox import OpenSandboxBackend

    events: list[str] = []

    class _FakeVault:
        async def create(self, **kwargs: object) -> None:
            del kwargs
            events.append('vault_create')

    class _FakeSbx:
        id = 'osb-1'
        credential_vault = _FakeVault()

        async def patch_egress_rules(self, rules: list[object]) -> None:
            events.append(f'patch:{len(rules)}')

        async def delete_egress_rules(self, targets: list[str]) -> None:
            events.append(f'delete:{",".join(targets)}')

    class _FakeDisplay:
        async def display_size(self) -> tuple[int, int]:
            events.append('display_size')
            return (1280, 800)

    fake_sbx = _FakeSbx()
    fake_display = _FakeDisplay()

    async def fake_get_sandbox(self: OpenSandboxBackend) -> _FakeSbx:
        del self
        return fake_sbx

    async def fake_give_login(backend: object, github: GitHubCredential) -> None:
        del backend, github
        events.append('give_github_login')

    monkeypatch.setenv(opensandbox_provider.DOMAIN_NAME, 'localhost:8080')
    monkeypatch.setattr(OpenSandboxBackend, 'get_sandbox', fake_get_sandbox)
    monkeypatch.setattr(OpenSandboxBackend, 'display', property(lambda self: fake_display))
    monkeypatch.setattr(opensandbox_provider, 'give_github_login', fake_give_login)

    prov = opensandbox_provider.OpenSandboxProvider()
    ref, backend = await prov.create(github=GitHubCredential(login='Ada', token='gho_x'), desktop=True)
    assert ref == WorkspaceRef(provider='opensandbox-desktop', id='osb-1')
    assert events == [
        'patch:2',
        'display_size',
        'delete:deb.debian.org,security.debian.org',
        'vault_create',
        'give_github_login',
    ]
    await prov.close(backend)


async def test_smol_missing_and_reached_display(monkeypatch: pytest.MonkeyPatch) -> None:
    smol_p = local_provider.SmolProvider()
    monkeypatch.setattr(local_provider, 'find_spec', _not_installed)
    monkeypatch.setattr(local_provider.sys, 'platform', 'win32')
    assert 'no Windows build' in (smol_p.missing(desktop=True) or '')
    monkeypatch.setattr(local_provider.sys, 'platform', 'linux')
    assert 'needs smolmachines' in (smol_p.missing(desktop=True) or '')

    fake_display = FakeDisplay()
    inner_backend = FakeDesktopBackend(
        WorkspaceRef(provider='smol', id='vm-1'), Path('.'), env={}, display=fake_display
    )
    reached = local_provider._Reached(inner_backend)  # pyright: ignore[reportPrivateUsage]
    display = reached.display
    assert display is not None and display is reached.display
    assert isinstance(display, SupportsZoom) and isinstance(display, SupportsCursorPosition)
    assert reached.reached is False
    assert await display.display_size() == (1280, 800)
    assert reached.reached is True
    assert await display.screenshot() == PNG_A
    await display.act(Click(x=1, y=2))
    assert await display.zoom(Region(left=0, top=0, right=10, bottom=10)) == PNG_A
    assert await display.cursor_position() == Point(x=100, y=200)

    class _MinimalDisplay:
        async def display_size(self) -> tuple[int, int]:
            raise WorkspaceUnavailableError('gone')

        async def screenshot(self) -> bytes:
            return PNG_A

        async def act(self, action: DisplayAction) -> None:
            del action

    min_backend = FakeDesktopBackend(
        WorkspaceRef(provider='smol', id='vm-2'), Path('.'), env={}, display=_MinimalDisplay()
    )
    min_reached = local_provider._Reached(min_backend)  # pyright: ignore[reportPrivateUsage]
    min_display = min_reached.display
    assert min_display is not None
    assert isinstance(min_display, SupportsZoom) and isinstance(min_display, SupportsCursorPosition)
    with pytest.raises(WorkspaceUnavailableError):
        await min_display.display_size()
    assert min_reached.reached is False
    with pytest.raises(UnsupportedActionError):
        await min_display.zoom(Region(left=0, top=0, right=10, bottom=10))
    with pytest.raises(UnsupportedActionError):
        await min_display.cursor_position()


@pytest.mark.parametrize(
    ('prov', 'ref_prov', 'ref_id'),
    [
        pytest.param(
            e2b_provider.E2BProvider(),
            'e2b-desktop',
            'e1',
            id='e2b',
            marks=pytest.mark.skipif(find_spec('e2b') is None, reason='e2b not installed'),
        ),
        pytest.param(
            opensandbox_provider.OpenSandboxProvider(),
            'opensandbox-desktop',
            'o1',
            id='opensandbox',
            marks=pytest.mark.skipif(find_spec('opensandbox') is None, reason='opensandbox not installed'),
        ),
        pytest.param(
            modal_provider.ModalProvider(),
            'modal-desktop',
            'mod1',
            id='modal',
            marks=pytest.mark.skipif(find_spec('modal') is None, reason='modal not installed'),
        ),
        pytest.param(
            render_provider.RenderProvider(),
            'render-desktop',
            'r1',
            id='render',
            marks=pytest.mark.skipif(find_spec('render') is None, reason='render not installed'),
        ),
        pytest.param(
            sprites_provider.SpritesProvider(),
            'sprites-desktop',
            'sp1',
            id='sprites',
            marks=pytest.mark.skipif(find_spec('sprites') is None, reason='sprites not installed'),
        ),
        pytest.param(
            kubernetes_provider.KubernetesProvider(),
            'kubernetes-desktop',
            'default/k1',
            id='kubernetes',
            marks=pytest.mark.skipif(
                find_spec('k8s_agent_sandbox') is None or find_spec('grpc') is None,
                reason='k8s_agent_sandbox not installed',
            ),
        ),
        pytest.param(
            local_provider.SmolProvider(),
            'local-smol-desktop',
            f'pydantic-ai-{"0" * 32}',
            id='local-smol',
            marks=pytest.mark.skipif(find_spec('smol') is None, reason='smol not installed'),
        ),
        pytest.param(
            local_provider.BoxLiteProvider(),
            'local-boxlite-desktop',
            f'pydantic-ai-{"0" * 32}',
            id='local-boxlite',
            marks=pytest.mark.skipif(find_spec('boxlite') is None, reason='boxlite not installed'),
        ),
    ],
)
async def test_provider_reopen_desktop_wiring(
    prov: sandbox.SandboxProvider, ref_prov: str, ref_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(e2b_provider.KEY_NAME, 'e2b-key')
    monkeypatch.setenv(opensandbox_provider.DOMAIN_NAME, 'localhost:8080')
    monkeypatch.setenv(render_provider.KEY_NAME, 'r-key')
    monkeypatch.setenv(render_provider.WORKSPACE_KEY_NAME, 'tea-1')
    monkeypatch.setenv(sprites_provider.KEY_NAME, 's-key')
    backend = await anyio.to_thread.run_sync(
        lambda: prov.reopen(WorkspaceRef(provider=ref_prov, id=ref_id), desktop=True)
    )
    assert isinstance(backend, SupportsDisplay) and backend.display is not None, prov.name
    assert await prov.desktop_url(backend) is None
    await prov.close(backend)
