"""The AoE plugin reports through AoE's hook files, with tmux, `ps`, and `aoe` faked at the process boundary."""

import asyncio
import importlib.util
import io
import os
import stat
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest
from pydantic import JsonValue
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ask_user import AskUser, AskUserRequest, AskUserResponse
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2._app import DEFAULT_PLUGINS, create_shell
from pydantic_clai2.builtin_plugins import aoe
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import (
    ConversationChanged,
    LoadedPlugin,
    PluginHost,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
    load_plugin,
)
from pydantic_clai2.runtime._session import Session

INSTANCE = 'abcd1234ef567890'
PANE_PID = 4242


@dataclass
class System:
    """Answers the commands the plugin runs, as tmux, `ps`, and `aoe` would."""

    session_name: str = f'aoe_my-task_{INSTANCE[:8]}'
    hidden: str | None = f'AOE_INSTANCE_ID={INSTANCE}\n'
    pane_pid: str = str(PANE_PID)
    hidden_after: int = 0
    """How many lookups fail before the hidden variable appears, as when AoE has not set it yet."""
    display: bool = True
    processes: str | None = None
    current: str | None = f'{{"session": "my-task", "profile": "work", "id": "{INSTANCE}"}}'
    rename_error: str | None = None
    calls: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])
    aoe_calls: 'asyncio.Queue[tuple[str, ...]] | None' = None
    """Every `aoe` command, for a test to wait on; the plugin runs commands from worker threads."""
    loop: asyncio.AbstractEventLoop | None = None

    def __call__(self, *argv: str) -> subprocess.CompletedProcess[str] | None:
        self.calls.append(argv)
        if argv[0] == 'aoe' and self.loop is not None and self.aoe_calls is not None:
            self.loop.call_soon_threadsafe(self.aoe_calls.put_nowait, argv)
        match argv:
            case ('tmux', 'display-message', *_):
                return self.result(f'{self.session_name}\t{self.pane_pid}\n' if self.display else None)
            case ('tmux', 'show-environment', *_):
                if self.hidden_after:
                    self.hidden_after -= 1
                    return self.result(None)
                return self.result(self.hidden)
            case ('ps', *_):
                default = f'{os.getpid()} {PANE_PID} python\n{PANE_PID} 1 -zsh\n'
                return self.result(self.processes if self.processes is not None else default)
            case ('aoe', 'session', 'current', '--json'):
                return self.result(self.current)
            case _:
                return self.result('' if self.rename_error is None else None, stderr=self.rename_error or '')

    @staticmethod
    def result(stdout: str | None, stderr: str = '') -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess((), 0 if stdout is not None else 1, stdout or '', stderr)


@dataclass
class Fixture:
    system: System
    hooks: Path
    state: Path
    writes: 'asyncio.Queue[tuple[str, str]]'

    @property
    def directory(self) -> Path:
        return self.hooks / INSTANCE

    async def status(self, expected: str) -> None:
        """Wait until the plugin writes `expected` as the status."""
        while True:
            name, content = await asyncio.wait_for(self.writes.get(), timeout=10)
            if name == 'status' and content == expected:
                return

    async def write(self, expected: str, content: str | None = None) -> None:
        """Wait until the plugin writes the file `expected`, with `content` when given."""
        while True:
            name, written = await asyncio.wait_for(self.writes.get(), timeout=10)
            if name == expected and content in (None, written):
                return

    async def aoe(self, command: str) -> tuple[str, ...]:
        """Wait until the plugin runs `aoe ... <command> ...`."""
        assert self.system.aoe_calls is not None
        while command not in (call := await asyncio.wait_for(self.system.aoe_calls.get(), timeout=10)):
            pass
        return call

    def renames(self) -> list[str]:
        return [call[-1] for call in self.system.calls if call[1:4] == ('-p', 'work', 'session')]


def found(name: str) -> str:
    return name


def missing(name: str) -> None:
    return None


@pytest.fixture
async def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Fixture:
    system = System(aoe_calls=asyncio.Queue(), loop=asyncio.get_running_loop())
    monkeypatch.setattr(aoe, 'run_command', system)
    monkeypatch.setattr(aoe, 'hooks_base', lambda: tmp_path / 'hooks')
    monkeypatch.setattr(aoe, '_LOOKUP_INTERVAL', 0)
    monkeypatch.setattr('shutil.which', found)
    monkeypatch.setenv('TMUX', '/tmp/tmux-1/default,1,0')
    monkeypatch.setenv('TMUX_PANE', '%3')
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    monkeypatch.delenv('AOE_INSTANCE_ID', raising=False)
    monkeypatch.chdir(tmp_path)
    writes: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    write = aoe.HookFiles.write

    def recording(self: aoe.HookFiles, name: str, content: str) -> None:
        write(self, name, content)
        loop.call_soon_threadsafe(writes.put_nowait, (name, content))

    monkeypatch.setattr(aoe.HookFiles, 'write', recording)
    remember = aoe.Mappings.write

    def remembering(self: aoe.Mappings, instance_id: str, conversation_id: str) -> None:
        remember(self, instance_id, conversation_id)
        loop.call_soon_threadsafe(writes.put_nowait, ('mapping', conversation_id))

    monkeypatch.setattr(aoe.Mappings, 'write', remembering)
    return Fixture(
        system=system, hooks=tmp_path / 'hooks', state=tmp_path / 'state' / 'pydantic-clai2' / 'aoe', writes=writes
    )


def make_session(tmp_path: Path, model: FunctionModel | TestModel | None = None) -> Session[None, str]:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    return Session(Agent(model or TestModel(call_tools=[])), deps=None, conversations=store, workspace=tmp_path)


@pytest.fixture
async def started(env: Fixture, tmp_path: Path) -> AsyncIterator[tuple[Session[None, str], LoadedPlugin[None]]]:
    session = make_session(tmp_path)
    loaded = await load(session)
    await env.status('idle')
    await env.write('session_id')
    yield session, loaded
    await loaded.dispatch(SessionEnd(reason='exit'))


async def load(
    conversation: Session[None, str] | None = None,
    *,
    terminal: bool = True,
    output: io.StringIO | None = None,
    chosen: bool = False,
) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='aoe',
        console=Console(file=output or io.StringIO(), force_terminal=terminal),
        settings={},
        conversation=conversation,
    )
    loaded = load_plugin(aoe.AoePlugin, host)
    if conversation is not None:
        conversation.on_change = loaded.dispatch
        conversation.plugins = loaded.capabilities
    await loaded.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings(), conversation_chosen=chosen))
    return loaded


def reporting(loaded: LoadedPlugin[None]) -> bool:
    plugin = loaded.plugin
    assert isinstance(plugin, aoe.AoePlugin)
    return plugin.reporter is not None and plugin.reporter.hooks is not None


def test_default_is_opt_in() -> None:
    entry = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'aoe')
    assert not entry.enabled


@pytest.mark.parametrize('missing', ['TMUX', 'TMUX_PANE', 'windows', 'headless'])
async def test_inert_outside_an_interactive_tmux_pane(
    env: Fixture, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    if missing == 'windows':
        monkeypatch.setattr(aoe.sys, 'platform', 'win32')
    elif missing != 'headless':
        monkeypatch.delenv(missing)
    loaded = await load(terminal=missing != 'headless')
    assert loaded.capabilities == ()
    await loaded.dispatch(TurnStart(text='hello'))
    await loaded.dispatch(TurnEnd(text='hello', outcome='failed'))
    await loaded.dispatch(ConversationChanged(conversation_id='other', title=None))
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert env.system.calls == []
    assert not env.hooks.exists()


@pytest.mark.parametrize(
    'case',
    ['plain tmux', 'terminal session', 'never set', 'other instance', 'invalid', 'no tmux', 'bad pane', 'nested'],
)
async def test_inert_outside_an_aoe_agent_session(env: Fixture, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    system = env.system
    if case == 'plain tmux':
        system.session_name = 'main'
    elif case == 'terminal session':
        system.session_name = f'aoe_term_my-task_{INSTANCE[:8]}'
    elif case == 'never set':
        system.hidden = None
    elif case == 'other instance':
        system.hidden = 'AOE_INSTANCE_ID=ffff0000\n'
    elif case == 'invalid':
        monkeypatch.setenv('AOE_INSTANCE_ID', '../escape')
    elif case == 'no tmux':
        system.display = False
    elif case == 'bad pane':
        system.pane_pid = 'not-a-pid'
    else:
        system.processes = (
            f'{os.getpid()} 300 python\n300 200 /bin/sh -c pytest\n200 {PANE_PID} /usr/bin/python3 /usr/bin/clai2\n'
        )
    loaded = await load()
    assert not reporting(loaded)
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not env.hooks.exists()


async def test_reads_the_instance_once_aoe_sets_it(env: Fixture, tmp_path: Path) -> None:
    env.system.hidden_after = 3
    loaded = await load(make_session(tmp_path))
    await env.status('idle')
    assert reporting(loaded)
    await loaded.dispatch(SessionEnd(reason='exit'))


@pytest.mark.parametrize('case', ['launch wrapper', 'first process', 'no ps'])
async def test_not_nested(env: Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    system = env.system
    if case == 'launch wrapper':
        # AoE's launch wrapper names `clai2` but is not one.
        system.processes = f'  PID  PPID ARGS\n{os.getpid()} 300 python\n300 {PANE_PID} sh -c clai2\n{PANE_PID} 1\n'
    elif case == 'first process':
        system.pane_pid = str(os.getpid())
        system.processes = f'{os.getpid()} 1 /usr/bin/python3 /usr/bin/clai2\n'
    else:

        def no_ps(*argv: str) -> subprocess.CompletedProcess[str] | None:
            return None if argv[0] == 'ps' else system(*argv)

        monkeypatch.setattr(aoe, 'run_command', no_ps)
    loaded = await load(make_session(tmp_path))
    await env.status('idle')
    assert reporting(loaded)
    await loaded.dispatch(SessionEnd(reason='exit'))


async def test_the_instance_may_come_from_the_environment(
    env: Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('AOE_INSTANCE_ID', INSTANCE)
    env.system.session_name = 'renamed-by-hand'
    loaded = await load(make_session(tmp_path))
    await env.status('idle')
    assert not any(call[:2] == ('tmux', 'show-environment') for call in env.system.calls)
    await loaded.dispatch(SessionEnd(reason='exit'))


class QuestionModel(TestModel):
    def gen_tool_args(self, tool_def: ToolDefinition) -> JsonValue:
        return {'questions': [{'header': 'Pick', 'question': 'Which?', 'options': [{'label': 'A'}, {'label': 'B'}]}]}


async def test_states_session_and_title(env: Fixture, tmp_path: Path) -> None:
    async def answer(request: AskUserRequest) -> AskUserResponse:
        await env.status('waiting')
        return AskUserResponse(cancelled=True)

    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    agent = Agent(
        QuestionModel(call_tools=['ask_user_question']),
        deps_type=type(None),
        capabilities=[AskUser(answerer=answer)],
    )
    session = Session(agent, deps=None, conversations=store, workspace=tmp_path)
    loaded = await load(session)
    await env.status('idle')
    await env.write('session_id')
    base = env.hooks
    assert stat.S_IMODE(base.stat().st_mode) == 0o700
    assert stat.S_IMODE(env.directory.stat().st_mode) == 0o700
    first = session.conversation_id
    assert (env.directory / 'session_id').read_text() == first + '\n'
    mapping = env.state / INSTANCE
    assert mapping.read_text() == first + '\n'
    assert stat.S_IMODE(mapping.stat().st_mode) == 0o600

    await session.prompt('Fix the parser')
    await env.status('idle')
    assert (env.directory / 'status').read_text() == 'idle'
    assert await env.aoe('rename') == ('aoe', '-p', 'work', 'session', 'rename', INSTANCE, '--title=Fix the parser')

    await session.renamed(conversation_id=first, title='Fix the parser')
    await session.renamed(conversation_id=first, title='Parser fix')
    await env.aoe('rename')
    assert env.renames() == ['--title=Fix the parser', '--title=Parser fix']
    await session.clear()
    await env.write('session_id', session.conversation_id + '\n')
    assert (env.directory / 'session_id').read_text() == session.conversation_id + '\n'
    assert mapping.read_text() == session.conversation_id + '\n'
    assert not list(env.directory.glob('.*.tmp'))

    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not (env.directory / 'status').exists()
    assert (env.directory / 'session_id').exists()


async def test_a_failed_turn_is_an_error_until_the_next(env: Fixture, tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel(call_tools=[])),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), force_terminal=True),
        # The model cannot be resolved, so the turn fails before its agent run starts.
        settings=Settings(model='no-such-provider:model'),
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[PluginSettings(id='aoe', factory='pydantic_clai2.builtin_plugins.aoe')],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    try:
        await env.status('idle')
        ended = await shell.run_turn(TurnStart(text='hello'), headless=True)
        assert ended.outcome == 'failed'
        await shell.loader.fire(ended)
        await env.status('error')
        # A background fork that starts and completes meanwhile leaves the failure showing.
        await shell.loader.fire(TurnStart(text='forked'))
        await shell.loader.fire(TurnEnd(text='forked', outcome='completed'))
        [entry] = [entry for entry in shell.loader.entries() if entry.name == 'aoe']
        assert entry.loaded is not None and isinstance(plugin := entry.loaded.plugin, aoe.AoePlugin)
        assert plugin.reporter is not None and plugin.reporter.status == 'error'
        shell.session.model = None
        ended = await shell.run_turn(TurnStart(text='again'), headless=True)
        assert ended.outcome == 'completed'
        await env.status('idle')
        await shell.loader.fire(ended)
    finally:
        await shell.loader.close('exit')
    assert not (env.directory / 'status').exists()


async def test_nested_failure_and_background_runs(
    env: Fixture, started: tuple[Session[None, str], LoadedPlugin[None]]
) -> None:
    session, loaded = started
    plugin = loaded.plugin
    assert isinstance(plugin, aoe.AoePlugin) and plugin.reporter is not None

    async def broken_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ValueError('child')
        yield  # pragma: no cover

    child = Agent(FunctionModel(stream_function=broken_stream), deps_type=type(None))
    parent = Agent(TestModel(call_tools=['nested']), deps_type=type(None))

    @parent.tool_plain
    async def nested() -> str:
        with pytest.raises(ValueError, match='child'):
            await child.run('child', capabilities=loaded.capabilities, conversation_id=session.conversation_id)
        return 'recovered'

    await parent.run('parent', capabilities=loaded.capabilities, conversation_id=session.conversation_id)
    assert plugin.reporter.status == 'idle'

    # A background run, such as a `/fork`, has its own conversation and does not count.
    statuses: list[str] = []
    background = Agent(TestModel(call_tools=['hold']), deps_type=type(None))

    @background.tool_plain
    async def hold() -> str:
        assert plugin.reporter is not None
        statuses.append(plugin.reporter.status)
        return 'held'

    await background.run('fork', capabilities=loaded.capabilities, conversation_id='a-fork')
    assert statuses == ['idle']


async def test_running_is_rewritten_while_a_run_lasts(
    env: Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(aoe, '_RUNNING_REFRESH', 0.01)
    rewritten = anyio.Event()
    agent = Agent(TestModel(call_tools=['work']), deps_type=type(None))

    @agent.tool_plain
    async def work() -> str:
        await env.status('running')
        await env.status('running')
        rewritten.set()
        return 'done'

    session = Session(agent, deps=None, workspace=tmp_path)
    loaded = await load(session)
    await env.status('idle')
    await session.prompt('long task')
    assert rewritten.is_set()
    await loaded.dispatch(SessionEnd(reason='exit'))


async def test_restarted_pane_resumes_its_conversation(env: Fixture, tmp_path: Path) -> None:
    earlier = make_session(tmp_path)
    await earlier.prompt('earlier work')
    env.state.mkdir(parents=True)
    (env.state / INSTANCE).write_text(earlier.conversation_id + '\n')

    session = make_session(tmp_path)
    output = io.StringIO()
    loaded = await load(session, output=output)
    await env.status('idle')
    assert session.conversation_id == earlier.conversation_id
    assert session.messages == earlier.messages
    assert f'Resumed earlier work ({earlier.conversation_id}).' in Text.from_ansi(output.getvalue()).plain
    await loaded.dispatch(SessionEnd(reason='exit'))

    # `--session-id` chose an empty conversation, which wins.
    named = make_session(tmp_path)
    loaded = await load(named, chosen=True)
    await env.write('mapping', named.conversation_id)
    assert named.messages == []
    assert (env.state / INSTANCE).read_text() == named.conversation_id + '\n'
    await loaded.dispatch(SessionEnd(reason='exit'))

    # An explicit `--resume` already restored a conversation, which wins.
    other = make_session(tmp_path)
    await other.prompt('other work')
    explicit = make_session(tmp_path)
    await explicit.resume(other.conversation_id)
    loaded = await load(explicit)
    await env.write('mapping', other.conversation_id)
    assert explicit.conversation_id == other.conversation_id
    assert (env.state / INSTANCE).read_text() == other.conversation_id + '\n'
    await loaded.dispatch(SessionEnd(reason='exit'))


@pytest.mark.parametrize('saved', ['never-saved', 'not a valid id!'])
async def test_unusable_mapping_starts_fresh(env: Fixture, tmp_path: Path, saved: str) -> None:
    env.state.mkdir(parents=True)
    (env.state / INSTANCE).write_text(saved)
    session = make_session(tmp_path)
    fresh = session.conversation_id
    loaded = await load(session)
    await env.status('idle')
    assert session.conversation_id == fresh
    await loaded.dispatch(SessionEnd(reason='exit'))


@pytest.mark.parametrize(
    'failure',
    [
        'tied worktree',
        'linked worktree',
        'other profile',
        'unreadable',
        'not current',
        'no aoe',
        'timeout',
    ],
)
async def test_title_push_failures_are_benign(
    env: Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    if failure == 'tied worktree':
        env.system.rename_error = 'Stop the session before renaming its worktree directory or branch.'
    elif failure == 'linked worktree':
        # Not a failure: the rename is tried in a linked worktree too, and AoE decides what to do with it.
        (tmp_path / '.git').write_text('gitdir: /elsewhere/.git/worktrees/proj\n')
        (tmp_path / 'sub').mkdir()
        monkeypatch.chdir(tmp_path / 'sub')
    elif failure == 'timeout':
        system = env.system

        def timing_out(*argv: str) -> subprocess.CompletedProcess[str] | None:
            result = system(*argv)
            return None if 'rename' in argv else result

        monkeypatch.setattr(aoe, 'run_command', timing_out)
    elif failure == 'other profile':
        env.system.current = '{"session": "x", "profile": "work", "id": "someone-else"}'
    elif failure == 'unreadable':
        env.system.current = 'Update available!'
    elif failure == 'not current':
        env.system.current = None
    else:
        monkeypatch.setattr('shutil.which', missing)
    session = make_session(tmp_path)
    loaded = await load(session)
    await session.prompt('first title')
    calls = {'tied worktree': 1, 'linked worktree': 1, 'timeout': 1}.get(failure, 0)
    if calls or failure in ('other profile', 'unreadable', 'not current'):
        await env.aoe('rename' if calls else 'current')
    await session.renamed(conversation_id=session.conversation_id, title='-second title')
    if failure in ('linked worktree', 'timeout'):
        # Only a refusal stops renames; a timed-out one tries again with the next title.
        assert (await env.aoe('rename'))[-1] == '--title=-second title'
        calls += 1
    elif failure in ('other profile', 'unreadable', 'not current'):
        await env.aoe('current')
    await env.status('idle')
    await loaded.dispatch(SessionEnd(reason='exit'))
    attempts = [call for call in env.system.calls if call[0] == 'aoe' and 'rename' in call]
    assert len(attempts) == calls


@pytest.mark.parametrize('problem', ['group access', 'symlink', 'instance symlink', 'status is a directory'])
async def test_refuses_an_unsafe_hooks_directory(env: Fixture, tmp_path: Path, problem: str) -> None:
    target = tmp_path / 'elsewhere'
    target.mkdir(mode=0o700)
    if problem == 'group access':
        env.hooks.mkdir(mode=0o700)
        env.hooks.chmod(0o750)
    elif problem == 'symlink':
        env.hooks.symlink_to(target)
    elif problem == 'instance symlink':
        env.hooks.mkdir(mode=0o700)
        env.directory.symlink_to(target)
    else:
        env.hooks.mkdir(mode=0o700)
        env.directory.mkdir(mode=0o700)
        (env.directory / 'status').mkdir()
        (env.directory / 'status' / 'kept').touch()
    loaded = await load(make_session(tmp_path))
    await env.write('mapping')
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not list(target.iterdir())
    assert not (env.directory / 'status').is_file()
    assert not list(env.hooks.glob('**/.*.tmp'))


async def test_stop_tolerates_a_removed_directory(
    env: Fixture, started: tuple[Session[None, str], LoadedPlugin[None]]
) -> None:
    for path in env.directory.iterdir():
        path.unlink()
    env.directory.rmdir()
    _, loaded = started
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not env.directory.exists()
    env.hooks.rmdir()
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not env.hooks.exists()


def test_commands_that_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    assert aoe.run_command('/nonexistent/aoe-binary') is None
    monkeypatch.setattr(aoe, '_COMMAND_TIMEOUT', 0.01)
    assert aoe.run_command('sleep', '1') is None
    result = aoe.run_command('true')
    assert result is not None and result.returncode == 0


def test_state_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', 'relative')
    monkeypatch.setenv('HOME', str(tmp_path))
    assert aoe.state_directory() == tmp_path / '.local' / 'state' / 'pydantic-clai2' / 'aoe'
    assert aoe.hooks_base() == Path(f'/tmp/aoe-hooks-{os.geteuid()}')


async def test_unreadable_mapping_leaves_the_plugin_quiet(env: Fixture, tmp_path: Path) -> None:
    (env.state / INSTANCE).mkdir(parents=True)
    loaded = await load(make_session(tmp_path))
    assert not reporting(loaded)
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert not env.hooks.exists()


async def test_an_id_aoe_would_reject_is_only_mapped(env: Fixture, tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(
        Agent(TestModel()),
        deps=None,
        conversations=store,
        # Already titled, so the title push after the session step shows that step has finished.
        summary=ConversationSummary(id='-dash-first', workspace=str(tmp_path), revision=1, title='Saved'),
    )
    loaded = await load(session)
    await env.aoe('rename')
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert (env.state / INSTANCE).read_text() == '-dash-first\n'
    assert not (env.directory / 'session_id').exists()


async def test_a_failed_mapping_write_is_tried_again(
    env: Fixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write = aoe.Mappings.write
    failures = [OSError('disk full')]

    def flaky(self: aoe.Mappings, instance_id: str, conversation_id: str) -> None:
        if failures:
            raise failures.pop()
        write(self, instance_id, conversation_id)

    monkeypatch.setattr(aoe.Mappings, 'write', flaky)
    session = make_session(tmp_path)
    loaded = await load(session)
    await env.status('idle')
    assert not (env.state / INSTANCE).exists()
    # The next change publishes again, and this time the mapping lands.
    await session.prompt('any change')
    await env.write('session_id', session.conversation_id + '\n')
    assert (env.state / INSTANCE).read_text() == session.conversation_id + '\n'
    await loaded.dispatch(SessionEnd(reason='exit'))


def test_the_module_imports_without_posix_open_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    # Windows has neither, and the plugin must load there to stay inert.
    monkeypatch.delattr(os, 'O_NOFOLLOW')
    monkeypatch.delattr(os, 'O_DIRECTORY')
    spec = importlib.util.spec_from_file_location('aoe_without_posix_flags', aoe.__file__)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
