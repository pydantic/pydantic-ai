"""The `workspaces` plugin: kinds, the registry, Git worktrees from local and cloned repositories, and their CLAIs."""

import os
import signal
import socket
import stat
import subprocess
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path

import anyio
import pytest
from rich.console import Console
from starlette.datastructures import Headers

from pydantic_clai2._app import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.web.app import HEADER, Receive, Scope, Send
from pydantic_clai2.builtin_plugins.web.server import WebServer
from pydantic_clai2.builtin_plugins.workspaces import (
    USAGE,
    Workspaces,
    WorkspacesPlugin,
    WorkspacesSettings,
    WorktreeKind,
    describe,
    manager as manager_module,
)
from pydantic_clai2.builtin_plugins.workspaces.backend import is_url, repo_slug
from pydantic_clai2.builtin_plugins.workspaces.registry import Registry, Workspace, check_id, workspaces_dir
from pydantic_clai2.builtin_plugins.workspaces.sessions import Launcher, LocalProcesses, bound_port, probe
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, load_plugin
from pydantic_clai2.plugins.menus import Option
from pydantic_clai2.runtime import worktrees
from pydantic_clai2.runtime.worktrees import add_worktree, clone_or_fetch, commit_of, remove_worktree, repository_root
from tests.clai2.dialog_script import Back, Confirm, Pick, ScriptedDialogs, Type

pytestmark = pytest.mark.anyio


@dataclass
class CheckingDialogs:
    """`Dialogs` whose text dialog runs its `check` on each of `tries`, then backs out."""

    tries: list[str]
    checks: list[str | None] = field(default_factory=list[str | None])

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        raise AssertionError('not asked')  # pragma: no cover

    async def text(
        self,
        *,
        title: str,
        label: str = 'Value',
        placeholder: str = '',
        initial: str = '',
        secret: bool = False,
        check: Callable[[str], str | None] | None = None,
        description: str = '',
        multiline: bool = False,
    ) -> str | None:
        assert check is not None
        self.checks.extend(check(text) for text in self.tries)
        return None

    async def confirm(self, *, title: str, yes: str, no: str, description: str = '') -> bool:
        raise AssertionError('not asked')  # pragma: no cover

    async def view(self, *, title: str, text: str) -> None:
        raise AssertionError('not asked')  # pragma: no cover

    async def wait(
        self, *, title: str, description: str, task: Callable[[Callable[[str], None]], Awaitable[object]]
    ) -> bool:
        raise AssertionError('not asked')  # pragma: no cover


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    repo = tmp_path / 'my repo'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'CLAI Test')
    git(repo, 'config', 'user.email', 'clai@example.com')
    (repo / 'app.py').write_text('print("hi")\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'First')
    git(repo, 'branch', 'feature')
    return repo


@dataclass(frozen=True, kw_only=True)
class Spawned:
    argv: list[str]
    cwd: Path
    env: Mapping[str, str]
    log: Path


@dataclass
class FakeProcesses:
    """Processes that only exist in this object; SIGINT ends them unless `stubborn`."""

    spawned: list[Spawned] = field(default_factory=list[Spawned])
    running: set[int] = field(default_factory=set[int])
    signals: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    stubborn: bool = False
    """Ignore SIGINT."""
    lingering: int = 0
    """How many more times a process that was signalled to end reads as alive."""
    exits: bool = False
    """Exit as soon as started."""
    pid: int = 1000

    def spawn(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        self.pid += 1
        self.spawned.append(Spawned(argv=list(argv), cwd=cwd, env=env, log=log))
        # Appended, as `LocalProcesses` opens the log.
        with log.open('a', encoding='utf-8') as output:
            if self.exits:
                output.write('Traceback: it broke\n')
            else:
                # As `clai2 --web --port 0 --token-file` prints once it listens.
                output.write(f'Web UI running at http://127.0.0.1:{40000 + self.pid}/\n')
                self.running.add(self.pid)
        return self.pid

    def alive(self, pid: int) -> bool:
        if pid not in self.running and self.lingering:
            self.lingering -= 1
            return True
        return pid in self.running

    def signal(self, pid: int, signum: int) -> None:
        self.signals.append((pid, signum))
        if signum == signal.SIGKILL or not self.stubborn:
            self.running.discard(pid)


@dataclass
class FakeProbe:
    """Answers once `ready`, recording the tokens it was asked with."""

    ready: bool = True
    tokens: list[str] = field(default_factory=list[str])
    slow: int = 0
    """How many probes fail before it answers."""

    async def __call__(self, port: int, token: str) -> bool:
        self.tokens.append(token)
        if self.slow:
            self.slow -= 1
            return False
        return self.ready


@dataclass
class Rig:
    processes: FakeProcesses
    probe: FakeProbe
    launcher: Launcher
    directory: Path

    def workspaces(self, **kinds: WorktreeKind) -> Workspaces:
        return Workspaces(
            kinds=kinds or {'local-worktree': WorktreeKind()}, directory=self.directory, launcher=self.launcher
        )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    processes, answers = FakeProcesses(), FakeProbe()
    launcher = Launcher(
        processes=processes, probe=answers, python='python', interval=0, startup_timeout=5, stop_timeout=5
    )
    return Rig(processes=processes, probe=answers, launcher=launcher, directory=tmp_path / 'state')


def loaded(
    rig: Rig, settings: WorkspacesSettings | None = None, *, workspace: Callable[[], Path] = Path.cwd
) -> LoadedPlugin[None]:
    host = PluginHost(
        name='workspaces',
        console=Console(file=StringIO()),
        settings=(settings or WorkspacesSettings()).model_dump(mode='json'),
        workspace=workspace,
    )
    plugin = load_plugin(WorkspacesPlugin, host)
    assert isinstance(plugin.plugin, WorkspacesPlugin)
    plugin.plugin.workspaces = Workspaces(
        kinds=plugin.plugin.settings.kinds, directory=rig.directory, launcher=rig.launcher
    )
    return plugin


async def run(plugin: LoadedPlugin[None], text: str) -> str:
    return await plugin.commands.execute_async(text)


async def test_a_workspace_from_creation_to_destruction(
    rig: Rig, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/workspaces` makes a worktree, starts its CLAI with a token only you can read, stops it, and removes it."""
    opened: list[str] = []

    def open_browser(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.workspaces.open_browser', open_browser)
    # Whatever this process has, the workspace's CLAI is told not to load a `.env`.
    monkeypatch.delenv('PYTHON_DOTENV_DISABLED', raising=False)
    plugin = loaded(rig, WorkspacesSettings(kinds={'local-worktree': WorktreeKind(model='test')}))
    assert plugin.summary() == '1 command'
    assert await run(plugin, '/workspaces') == 'No workspaces. Make one with /workspaces create KIND [REPO [REF]].'

    created = await run(plugin, f'/workspaces create local-worktree "{repository}" feature')
    [workspace] = await rig.workspaces().listing()
    assert created == (
        f'Created workspace {workspace.id} at {workspace.path} on branch clai-{workspace.id}. '
        f'Start its CLAI with /workspaces open {workspace.id}.'
    )
    assert workspace.id.startswith('my-repo-') and workspace.status == 'ready'
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()
    assert git(workspace.path, 'branch', '--show-current') == f'clai-{workspace.id}'
    assert git(workspace.path, 'rev-parse', 'HEAD') == git(repository, 'rev-parse', 'feature')
    assert '/.worktrees/' in (repository / '.git/info/exclude').read_text()
    assert git(repository, 'status', '--porcelain') == '', 'the checkout is excluded from the repository'
    assert await run(plugin, '/workspaces list') == describe(workspace)

    shown = await run(plugin, f'/workspaces open {workspace.id}')
    [spawned] = rig.processes.spawned
    folder = rig.directory / 'registry' / workspace.id
    token = (folder / 'token').read_text().strip()
    assert spawned.argv == [
        *('python', '-P', '-m', 'pydantic_clai2', '--web', '--port', '0'),
        *('--token-file', str(folder / 'token'), '--model', 'test'),
    ]
    assert (spawned.cwd, spawned.log) == (workspace.path, folder / 'session.log')
    assert token not in ' '.join(spawned.argv), 'the token stays off the command line'
    assert spawned.env['PYTHON_DOTENV_DISABLED'] == '1', 'a `.env` in the checkout is not loaded'
    assert rig.probe.tokens == [token]
    for private in (folder / 'token', folder / 'workspace.json', folder / 'open.html'):
        assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    running = await rig.workspaces().get(workspace.id)
    assert running.status == 'running' and running.session is not None
    port = 40000 + rig.processes.pid
    assert running.session.port == port
    assert shown == f'Workspace {workspace.id} runs at http://127.0.0.1:{port}/; opened in your browser.'
    assert token not in shown and token not in describe(running)
    assert opened == [(folder / 'open.html').as_uri()]
    assert f'http://127.0.0.1:{port}/?token={token}' in (folder / 'open.html').read_text()
    assert await rig.workspaces().token(workspace.id) == token

    assert await run(plugin, f'/workspaces open {workspace.id}') == shown, 'a running CLAI is reused'
    assert len(rig.processes.spawned) == 1

    assert await run(plugin, f'/workspaces stop {workspace.id}') == f'Workspace {workspace.id}: stopped.'
    assert rig.processes.signals == [(rig.processes.pid, signal.SIGINT)]
    assert not (folder / 'token').exists()
    with pytest.raises(RuntimeError, match='has no CLAI running'):
        await rig.workspaces().token(workspace.id)

    destroyed = await run(plugin, f'/workspaces destroy {workspace.id}')
    assert destroyed == f'Destroyed workspace {workspace.id}; its checkout at {workspace.path} was removed.'
    assert not workspace.path.exists() and not folder.exists()
    assert git(repository, 'branch', '--list', f'clai-{workspace.id}'), 'the branch is kept'
    assert await rig.workspaces().listing() == []


async def test_a_workspace_from_a_clone(rig: Rig, repository: Path, tmp_path: Path) -> None:
    """A URL is cloned once into CLAI's state directory, fetched again for the next workspace, and checked out there."""
    url = repository.as_uri()
    workspaces = rig.workspaces(clones=WorktreeKind(repo=url, ref='feature', root=str(tmp_path / 'checkouts')))
    first = await workspaces.create(kind='clones')
    (repository / 'new.txt').write_text('new')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Second')
    second = await workspaces.create(kind='clones', ref='main')
    [clone] = (rig.directory / 'clones').iterdir()
    assert clone.name.startswith('my-repo-')
    assert first.repo == url and first.root == (tmp_path / 'checkouts').resolve()
    assert first.path.parent == second.path.parent == first.root
    assert (second.path / 'new.txt').exists() and not (first.path / 'new.txt').exists()
    assert git(clone, 'status', '--porcelain') == '', 'checkouts outside the clone leave it clean'
    default = await workspaces.create(kind='clones', ref='HEAD')
    assert (default.path / 'new.txt').exists()
    with pytest.raises(ValueError, match='nope is not a branch, tag, or commit'):
        await workspaces.create(kind='clones', ref='nope')


async def test_teardown_keep_leaves_the_checkout(rig: Rig, repository: Path) -> None:
    plugin = loaded(rig, WorkspacesSettings(kinds={'kept': WorktreeKind(repo=str(repository), teardown='keep')}))
    await run(plugin, '/workspaces create kept')
    [workspace] = await rig.workspaces().listing()
    assert workspace.ref == '' and git(workspace.path, 'rev-parse', 'HEAD') == git(repository, 'rev-parse', 'main')
    destroyed = await run(plugin, f'/workspaces destroy {workspace.id}')
    assert destroyed.endswith('was kept.')
    assert workspace.path.exists() and await rig.workspaces().listing() == []


async def test_a_relative_repository_is_in_the_sessions_directory(rig: Rig, repository: Path) -> None:
    """`/workspaces create KIND REPO` reads a relative `REPO` from the session's directory, not the process's."""
    plugin = loaded(rig, workspace=lambda: repository.parent)
    await run(plugin, f'/workspaces create local-worktree "{repository.name}"')
    [workspace] = await rig.workspaces().listing()
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()


async def test_a_kinds_relative_repository_is_in_the_sessions_directory(rig: Rig, repository: Path) -> None:
    """A kind's own relative `repo` is read from the session's directory too."""
    kinds = WorkspacesSettings(kinds={'here': WorktreeKind(repo=repository.name)})
    plugin = loaded(rig, kinds, workspace=lambda: repository.parent)
    await run(plugin, '/workspaces create here')
    [workspace] = await rig.workspaces().listing()
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()


async def test_usage_and_unknown_names(rig: Rig, repository: Path, tmp_path: Path) -> None:
    plugin = loaded(rig)
    for wrong in ('/workspaces open', '/workspaces create', '/workspaces create a b c d', '/workspaces nope x'):
        with pytest.raises(ValueError, match=USAGE.replace('[', r'\[').replace('|', r'\|')):
            await run(plugin, wrong)
    with pytest.raises(LookupError, match=r"No workspace kind 'docker'. Kinds: local-worktree."):
        await run(plugin, f'/workspaces create docker {repository}')
    with pytest.raises(LookupError, match='Kinds: none'):
        await Workspaces(kinds={}, directory=rig.directory).create(kind='x', repo='y')
    with pytest.raises(ValueError, match="kind 'local-worktree' has none of its own"):
        await run(plugin, '/workspaces create local-worktree')
    with pytest.raises(ValueError, match='is not in a Git repository'):
        await run(plugin, f'/workspaces create local-worktree {tmp_path}')
    for workspace_id in ('nope-123', '../escape', 'UPPER'):
        with pytest.raises(LookupError, match='No workspace'):
            await run(plugin, f'/workspaces stop {workspace_id}')


async def test_completion(rig: Rig, repository: Path) -> None:
    plugin = loaded(rig, WorkspacesSettings(kinds={'b': WorktreeKind(), 'a': WorktreeKind()}))
    workspace = await rig.workspaces().create(kind='local-worktree', repo=str(repository))
    [command] = plugin.commands
    assert list(command.complete([])) == ['list', 'create', 'open', 'stop', 'destroy']
    assert list(command.complete(['create', ''])) == ['a', 'b']
    assert list(command.complete(['destroy', ''])) == [workspace.id]
    assert list(command.complete(['create', 'a', ''])) == []


async def test_a_checkout_in_the_way(rig: Rig, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fixed(repo: str) -> str:
        return 'fixed-000000'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.workspaces.manager.new_id', fixed)
    workspaces = rig.workspaces()
    await workspaces.create(kind='local-worktree', repo=str(repository))
    with pytest.raises(ValueError, match='already exists'):
        await workspaces.create(kind='local-worktree', repo=str(repository))


async def test_destroy_refuses_a_folder_outside_the_workspaces_root(rig: Rig, repository: Path) -> None:
    """A record pointing anywhere else, by tampering or by accident, never gets that folder removed."""
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    for tampered in (replace(workspace, path=repository), replace(workspace, root=repository)):
        workspaces.registry.save(tampered)
        with pytest.raises(ValueError, match='Refusing to remove'):
            await workspaces.destroy(workspace.id)
        assert repository.exists() and workspace.path.exists()
    workspaces.registry.save(workspace)
    remove_worktree(path=workspace.path)
    await workspaces.destroy(workspace.id)
    assert await workspaces.listing() == [], 'a checkout already gone is only forgotten'


async def test_a_session_that_does_not_start(rig: Rig, repository: Path) -> None:
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    folder = workspaces.registry.folder(workspace.id)
    rig.probe.ready = False
    rig.processes.exits = True
    with pytest.raises(RuntimeError, match=f'exited while starting; its output is in {folder / "session.log"}'):
        await workspaces.start(workspace.id)
    assert not (folder / 'token').exists()
    assert (await workspaces.get(workspace.id)).status == 'ready'

    rig.processes.exits = False
    rig.launcher.startup_timeout = 0
    with pytest.raises(RuntimeError, match='did not answer in time'):
        await workspaces.start(workspace.id)
    assert rig.processes.signals == [(rig.processes.pid, signal.SIGKILL)]


async def test_stopping(rig: Rig, repository: Path) -> None:
    """Only the CLAI that answers with its token is signalled; one that ignores Ctrl-C is killed."""
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    assert await workspaces.stop(workspace.id) == workspace, 'nothing to stop'

    started = await workspaces.start(workspace.id)
    assert started.session is not None
    rig.processes.stubborn = True
    rig.launcher.stop_timeout = 0
    stopped = await workspaces.stop(workspace.id)
    assert stopped.status == 'stopped'
    assert rig.processes.signals == [(started.session.pid, signal.SIGINT), (started.session.pid, signal.SIGKILL)]

    rig.processes.signals.clear()
    started = await workspaces.start(workspace.id)
    assert started.session is not None
    rig.probe.ready = False
    await workspaces.stop(workspace.id)
    assert rig.processes.signals == [], 'a process that does not answer with the token may not be ours'

    rig.probe.ready = True
    started = await workspaces.start(workspace.id)
    assert started.session is not None
    started.session.token_file.unlink()
    await workspaces.stop(workspace.id)
    assert rig.processes.signals == [], 'without its token the process cannot be told apart'


async def test_a_session_takes_its_time(rig: Rig, repository: Path) -> None:
    """Starting waits for the first answer; stopping waits for the process to end after Ctrl-C."""
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    rig.probe.slow = 2
    started = await workspaces.start(workspace.id)
    assert started.status == 'running' and len(rig.probe.tokens) == 3
    rig.processes.lingering = 2
    await workspaces.stop(workspace.id)
    assert rig.processes.lingering == 0 and [signum for _, signum in rig.processes.signals] == [signal.SIGINT]


async def test_a_session_that_ended_reads_as_stopped(rig: Rig, repository: Path) -> None:
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    started = await workspaces.start(workspace.id)
    assert started.session is not None
    rig.processes.running.clear()
    [listed] = await workspaces.listing()
    assert listed.status == 'stopped' and describe(listed).startswith(f'{workspace.id}  stopped  ')
    workspaces.registry.save(replace(listed, status='running', session=None))
    assert (await workspaces.get(workspace.id)).status == 'stopped'


async def test_opening_without_a_browser(rig: Rig, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_browser(url: str) -> bool:
        return False

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.workspaces.open_browser', no_browser)
    plugin = loaded(rig, WorkspacesSettings(kinds={'local-worktree': WorktreeKind(repo=str(repository))}))
    await run(plugin, '/workspaces create local-worktree')
    [workspace] = await rig.workspaces().listing()
    shown = await run(plugin, f'/workspaces open {workspace.id}')
    assert shown.endswith(f'; open {rig.directory / "registry" / workspace.id / "open.html"} in your browser.')
    assert '--model' not in rig.processes.spawned[0].argv


async def test_the_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path))
    assert workspaces_dir() == tmp_path / 'pydantic-clai2' / 'workspaces'
    assert Workspaces(kinds={}).registry.directory == workspaces_dir() / 'registry'
    registry = Registry(tmp_path / 'missing')
    assert registry.all() == []
    (tmp_path / 'missing' / 'broken-1').mkdir(parents=True)
    (tmp_path / 'missing' / 'broken-1' / 'workspace.json').write_text('{}')
    (tmp_path / 'missing' / '.hidden').mkdir()
    assert registry.all() == []
    assert check_id('ok-1') == 'ok-1'


def test_write_private_leaves_nothing_behind_when_it_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_clai2.builtin_plugins.workspaces import registry

    def fail(source: object, target: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(registry.os, 'replace', fail)
    with pytest.raises(OSError, match='disk full'):
        registry.write_private(tmp_path / 'file', 'text')
    assert list(tmp_path.iterdir()) == []


def test_names() -> None:
    assert is_url('https://github.com/pydantic/pydantic-ai.git') and is_url('git@github.com:pydantic/pydantic-ai.git')
    assert not is_url('/home/me/pydantic-ai') and not is_url('C:\\code\\repo')
    assert repo_slug('git@github.com:pydantic/pydantic-ai.git') == 'pydantic-ai'
    assert repo_slug('file:///home/me/My%20Repo/') == 'my-repo'
    assert repo_slug('/') == 'repo'


def test_kind_settings_are_validated() -> None:
    with pytest.raises(ValueError, match='give an absolute path'):
        WorktreeKind(root='relative')
    with pytest.raises(ValueError):
        WorkspacesSettings(kinds={'Bad Name': WorktreeKind()})
    assert WorktreeKind(root='~/checkouts').root == '~/checkouts'
    [declared] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'workspaces']
    assert (declared.factory, declared.enabled) == ('pydantic_clai2.builtin_plugins.workspaces', False)


async def test_the_settings_menu(rig: Rig) -> None:
    plugin = loaded(rig)
    assert isinstance(plugin.plugin, WorkspacesPlugin)
    menu = plugin.plugin.settings_menu()
    source = menu.source
    rows = {row.key: row for row in source.rows()}
    assert list(rows) == [
        'local-worktree.repo',
        'local-worktree.ref',
        'local-worktree.root',
        'local-worktree.branch_prefix',
        'local-worktree.teardown',
        'local-worktree.model',
        'add',
        'remove',
    ]
    teardown = rows['local-worktree.teardown']
    assert (teardown.choices, teardown.allow_custom, teardown.display('keep')) == (
        ('remove', 'keep'),
        False,
        'Keep the checkout',
    )
    assert source.current(rows['local-worktree.branch_prefix']) == 'clai-'
    assert source.current(rows['add']) == 'local-worktree'
    assert (
        source.problem(rows['local-worktree.root'], 'relative')
        == 'Value error, give an absolute path, or leave it empty'
    )
    assert source.problem(rows['local-worktree.branch_prefix'], 'agent/') is None
    assert source.apply(rows['local-worktree.teardown'], 'keep') == 'local-worktree: On destroy: Keep the checkout.'
    assert plugin.host.settings(WorkspacesSettings).kinds['local-worktree'].teardown == 'keep'
    assert source.reset(rows['local-worktree.teardown']) == 'local-worktree: On destroy: Remove the checkout (default).'
    assert source.reset(rows['local-worktree.model']) == 'local-worktree: Model: (empty) (default).'

    dialogs = ScriptedDialogs([Type(text='docs')])
    assert await menu.actions['add'](dialogs) == ['Added workspace kind docs.']
    assert list(plugin.host.settings(WorkspacesSettings).kinds) == ['local-worktree', 'docs']
    [asked] = dialogs.asked
    assert asked.title == 'New workspace kind'
    assert await menu.actions['add'](ScriptedDialogs([Back()])) == []

    assert await menu.actions['remove'](ScriptedDialogs([Pick(value='docs'), Confirm(yes=False)])) == []
    assert await menu.actions['remove'](ScriptedDialogs([Back()])) == []
    assert await menu.actions['remove'](ScriptedDialogs([Pick(value='docs'), Confirm(yes=True)])) == [
        'Removed workspace kind docs.'
    ]
    assert list(plugin.host.settings(WorkspacesSettings).kinds) == ['local-worktree']


async def test_new_kind_names_are_checked(rig: Rig) -> None:
    plugin = loaded(rig)
    assert isinstance(plugin.plugin, WorkspacesPlugin)
    dialogs = CheckingDialogs(tries=['local-worktree', 'Bad', 'fine'])
    assert await plugin.plugin.settings_menu().actions['add'](dialogs) == []
    assert dialogs.checks == [
        "There is a kind 'local-worktree' already.",
        'Start with a lowercase letter; then lowercase letters, digits, and hyphens.',
        None,
    ]


async def test_local_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real child in its own process group: started, reaped when killed, and refused on Windows."""
    processes = LocalProcesses()
    log = tmp_path / 'log'

    async def reaped(pid: int) -> None:
        # The reaper thread may already have finished, having reaped the process.
        reapers = [thread for thread in threading.enumerate() if thread.name == f'clai2-workspace-{pid}']
        await anyio.to_thread.run_sync(lambda: [reaper.join(10) for reaper in reapers])

    pid = processes.spawn(['sleep', '30'], cwd=tmp_path, env=os.environ, log=log)
    assert processes.alive(pid)
    assert os.getpgid(pid) == pid
    processes.signal(pid, signal.SIGKILL)
    await reaped(pid)
    assert not processes.alive(pid)
    printed = processes.spawn(['echo', 'started'], cwd=tmp_path, env=os.environ, log=log)
    await reaped(printed)
    assert log.read_text() == 'started\n' and stat.S_IMODE(log.stat().st_mode) == 0o600
    assert not processes.alive(1) or os.geteuid() == 0, "another user's process is not ours"
    monkeypatch.setattr(sys, 'platform', 'win32')
    with pytest.raises(ValueError, match='macOS and Linux'):
        processes.spawn(['true'], cwd=tmp_path, env=os.environ, log=log)


async def test_probe() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        status = 200 if Headers(scope=scope).get(HEADER) == 'good' else 403
        await send({'type': 'http.response.start', 'status': status, 'headers': []})
        await send({'type': 'http.response.body', 'body': b''})

    server = WebServer.serve(lambda port: app, host='127.0.0.1', port=0)
    try:
        assert await probe(server.port, 'good')
        assert not await probe(server.port, 'bad')
    finally:
        await server.stop()
    with socket.socket() as unused:
        unused.bind(('127.0.0.1', 0))
        port = unused.getsockname()[1]
    assert not await probe(port, 'good'), 'nothing listens'


def test_the_port_comes_from_the_log(tmp_path: Path) -> None:
    assert bound_port(tmp_path / 'missing') is None
    (tmp_path / 'log').write_text('Loading plugins\nWeb UI running at http://127.0.0.1:4321/\n')
    assert bound_port(tmp_path / 'log') == 4321
    (tmp_path / 'log').write_text('Web UI running at http://192.168.1.2:4321/\n')
    assert bound_port(tmp_path / 'log') is None, 'only loopback'


def test_git_helpers_report_failures(repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match=r"(?s)Cannot create worktree: .*a branch named 'main' already exists"):
        add_worktree(repo=repository, path=tmp_path / 'a', branch='main', base='HEAD')
    with pytest.raises(ValueError, match='Cannot remove worktree'):
        remove_worktree(path=repository)
    with pytest.raises(ValueError, match='Cannot fetch'):
        clone_or_fetch(url=(tmp_path / 'nothing').as_uri(), path=tmp_path / 'clone')
    with pytest.raises(ValueError, match='is not a branch'):
        commit_of(repo=repository, ref='', remote=True)
    assert repository_root(repository / '.git' / '..') == repository.resolve()

    def broken(*args: str) -> str:
        raise OSError('no git')

    monkeypatch.setattr(worktrees, '_git', broken)
    with pytest.raises(ValueError, match='Cannot create worktree: no git'):
        add_worktree(repo=repository, path=tmp_path / 'b', branch='b', base='HEAD')
    with pytest.raises(ValueError, match=r'Cannot remove worktree .*: no git'):
        remove_worktree(path=repository)
    with pytest.raises(ValueError, match='Cannot fetch x: no git'):
        clone_or_fetch(url='x', path=tmp_path / 'clone')


async def test_a_restart_reads_its_own_port(rig: Rig, repository: Path) -> None:
    """Started again, a workspace's CLAI is found on its new port, not the one the last launch logged."""
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    first = await workspaces.start(workspace.id)
    await workspaces.stop(workspace.id)
    second = await workspaces.start(workspace.id)
    assert first.session is not None and second.session is not None
    assert (first.session.port, second.session.port) == (40000 + rig.processes.pid - 1, 40000 + rig.processes.pid)


async def test_a_token_gone_meanwhile(rig: Rig, repository: Path) -> None:
    """Another CLAI stopping the workspace between the status check and the read is a CLAI that does not run."""
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    started = await workspaces.start(workspace.id)
    assert started.session is not None
    started.session.token_file.unlink()
    with pytest.raises(RuntimeError, match='has no CLAI running'):
        await workspaces.token(workspace.id)


async def test_one_process_at_a_time(rig: Rig, repository: Path) -> None:
    """Two CLAIs starting one workspace at once start one child: the second waits for the first's lock."""
    first, second = rig.workspaces(), rig.workspaces()
    workspace = await first.create(kind='local-worktree', repo=str(repository))
    answer = anyio.Event()
    rig.probe.ready = False

    async def slowly(port: int, token: str) -> bool:
        await answer.wait()
        return True

    rig.launcher.probe = slowly
    started: list[Workspace] = []
    async with anyio.create_task_group() as group:

        async def start(workspaces: Workspaces) -> None:
            started.append(await workspaces.start(workspace.id))

        group.start_soon(start, first)
        await anyio.wait_all_tasks_blocked()
        group.start_soon(start, second)
        await anyio.wait_all_tasks_blocked()
        answer.set()
    assert len(rig.processes.spawned) == 1
    assert started[0].session == started[1].session


async def test_windows_needs_no_lock(rig: Rig, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspaces = rig.workspaces()
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    monkeypatch.setattr(manager_module.sys, 'platform', 'win32')
    assert (await workspaces.stop(workspace.id)).status == 'ready'


def test_branch_prefixes_git_refuses() -> None:
    for prefix in ('a..', 'a//', 'a/.', 'a.lock/'):
        with pytest.raises(ValueError, match='Git refuses branch names'):
            WorktreeKind(branch_prefix=prefix)
    assert WorktreeKind(branch_prefix='agent/').branch_prefix == 'agent/'
