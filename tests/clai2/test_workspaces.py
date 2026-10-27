"""The `workspaces` plugin: kinds, the registry, and Git worktrees from local and cloned repositories."""

import json
import os
import stat
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_clai2.app import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins.workspaces import (
    USAGE,
    Workspaces,
    WorkspacesPlugin,
    WorkspacesSettings,
    WorktreeKind,
    describe,
)
from pydantic_clai2.builtin_plugins.workspaces.backend import is_url, repo_slug
from pydantic_clai2.builtin_plugins.workspaces.registry import Registry, check_id, workspaces_dir
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


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """Where the registry and clones go, as CLAI's state directory holds them."""
    return tmp_path / 'state'


def workspaces_in(directory: Path, **kinds: WorktreeKind) -> Workspaces:
    return Workspaces(kinds=kinds or {'local-worktree': WorktreeKind()}, directory=directory)


def loaded(
    directory: Path, settings: WorkspacesSettings | None = None, *, workspace: Callable[[], Path] = Path.cwd
) -> LoadedPlugin[None]:
    host = PluginHost(
        name='workspaces',
        console=Console(file=StringIO()),
        settings=(settings or WorkspacesSettings()).model_dump(mode='json'),
        workspace=workspace,
    )
    plugin = load_plugin(WorkspacesPlugin, host)
    assert isinstance(plugin.plugin, WorkspacesPlugin)
    plugin.plugin.workspaces = Workspaces(kinds=plugin.plugin.settings.kinds, directory=directory)
    return plugin


async def run(plugin: LoadedPlugin[None], text: str) -> str:
    return await plugin.commands.execute_async(text)


async def test_a_workspace_from_creation_to_destruction(directory: Path, repository: Path) -> None:
    """`/workspaces` makes a worktree on its own branch, lists it, and removes it, keeping the branch."""
    plugin = loaded(directory)
    assert plugin.summary() == '1 command'
    assert await run(plugin, '/workspaces') == 'No workspaces. Make one with /workspaces create KIND [REPO [REF]].'

    created = await run(plugin, f'/workspaces create local-worktree "{repository}" feature')
    [workspace] = await workspaces_in(directory).listing()
    assert created == (
        f'Created workspace {workspace.id} at {workspace.path} on branch clai-{workspace.id}. '
        "Start a session there with the web UI's New session, or run CLAI in it."
    )
    assert workspace.id.startswith('my-repo-') and workspace.conversation is None
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()
    assert git(workspace.path, 'branch', '--show-current') == f'clai-{workspace.id}'
    assert git(workspace.path, 'rev-parse', 'HEAD') == git(repository, 'rev-parse', 'feature')
    assert '/.worktrees/' in (repository / '.git/info/exclude').read_text()
    assert git(repository, 'status', '--porcelain') == '', 'the checkout is excluded from the repository'
    listed = await run(plugin, '/workspaces list')
    assert listed == describe(workspace) == f'{workspace.id}  local-worktree  clai-{workspace.id}  {workspace.path}'
    folder = directory / 'registry' / workspace.id
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / 'workspace.json').stat().st_mode) == 0o600

    attached = await workspaces_in(directory).attach(workspace.id, 'conversation-1')
    assert attached == replace(workspace, conversation='conversation-1')
    assert await workspaces_in(directory).get(workspace.id) == attached

    destroyed = await run(plugin, f'/workspaces destroy {workspace.id}')
    assert destroyed == f'Destroyed workspace {workspace.id}; its checkout at {workspace.path} was removed.'
    assert not workspace.path.exists() and not folder.exists()
    assert git(repository, 'branch', '--list', f'clai-{workspace.id}'), 'the branch is kept'
    assert await workspaces_in(directory).listing() == []


async def test_what_an_older_clai_saved_still_reads(directory: Path, repository: Path) -> None:
    """A record written when each workspace ran a CLAI of its own, and a kind with its retired `model`, still load."""
    workspaces = workspaces_in(directory)
    workspace = await workspaces.create(kind='local-worktree', repo=str(repository))
    record = workspaces.registry.folder(workspace.id) / 'workspace.json'
    saved = json.loads(record.read_text())
    del saved['conversation']
    saved |= {
        'status': 'running',
        'model': 'openai:gpt-5',
        'session': {'pid': 1, 'port': 2, 'token_file': '/x/token', 'log': '/x/log', 'started': saved['created']},
    }
    record.write_text(json.dumps(saved))
    assert await workspaces.get(workspace.id) == workspace
    (record.parent / 'token').write_text('old')
    await workspaces.destroy(workspace.id)
    assert not record.parent.exists(), 'what the older CLAI kept there goes too'

    kind = WorktreeKind.model_validate({'repo': 'x', 'model': 'openai:gpt-5'})
    assert kind.model_dump() == WorktreeKind(repo='x').model_dump() | {'model': 'openai:gpt-5'}
    settings = WorkspacesSettings.model_validate({'kinds': {'local-worktree': {'model': ''}}})
    assert settings.model_dump(mode='json') == WorkspacesSettings().model_dump(mode='json')


async def test_a_workspace_from_a_clone(directory: Path, repository: Path, tmp_path: Path) -> None:
    """A URL is cloned once into CLAI's state directory, fetched again for the next workspace, and checked out there."""
    url = repository.as_uri()
    workspaces = workspaces_in(
        directory, clones=WorktreeKind(repo=url, ref='feature', root=str(tmp_path / 'checkouts'))
    )
    first = await workspaces.create(kind='clones')
    (repository / 'new.txt').write_text('new')
    git(repository, 'add', '.')
    git(repository, 'commit', '-m', 'Second')
    second = await workspaces.create(kind='clones', ref='main')
    [clone] = (directory / 'clones').iterdir()
    assert clone.name.startswith('my-repo-')
    assert first.repo == url and first.root == (tmp_path / 'checkouts').resolve()
    assert first.path.parent == second.path.parent == first.root
    assert (second.path / 'new.txt').exists() and not (first.path / 'new.txt').exists()
    assert git(clone, 'status', '--porcelain') == '', 'checkouts outside the clone leave it clean'
    default = await workspaces.create(kind='clones', ref='HEAD')
    assert (default.path / 'new.txt').exists()
    with pytest.raises(ValueError, match='nope is not a branch, tag, or commit'):
        await workspaces.create(kind='clones', ref='nope')


async def test_teardown_keep_leaves_the_checkout(directory: Path, repository: Path) -> None:
    plugin = loaded(directory, WorkspacesSettings(kinds={'kept': WorktreeKind(repo=str(repository), teardown='keep')}))
    await run(plugin, '/workspaces create kept')
    [workspace] = await workspaces_in(directory).listing()
    assert workspace.ref == '' and git(workspace.path, 'rev-parse', 'HEAD') == git(repository, 'rev-parse', 'main')
    destroyed = await run(plugin, f'/workspaces destroy {workspace.id}')
    assert destroyed.endswith('was kept.')
    assert workspace.path.exists() and await workspaces_in(directory).listing() == []


async def test_a_relative_repository_is_in_the_sessions_directory(directory: Path, repository: Path) -> None:
    """`/workspaces create KIND REPO` reads a relative `REPO` from the session's directory, not the process's."""
    plugin = loaded(directory, workspace=lambda: repository.parent)
    await run(plugin, f'/workspaces create local-worktree "{repository.name}"')
    [workspace] = await workspaces_in(directory).listing()
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()


async def test_a_kinds_relative_repository_is_in_the_sessions_directory(directory: Path, repository: Path) -> None:
    """A kind's own relative `repo` is read from the session's directory too."""
    kinds = WorkspacesSettings(kinds={'here': WorktreeKind(repo=repository.name)})
    plugin = loaded(directory, kinds, workspace=lambda: repository.parent)
    await run(plugin, '/workspaces create here')
    [workspace] = await workspaces_in(directory).listing()
    assert workspace.path == (repository / '.worktrees' / workspace.id).resolve()


async def test_usage_and_unknown_names(directory: Path, repository: Path, tmp_path: Path) -> None:
    plugin = loaded(directory)
    for wrong in ('/workspaces open', '/workspaces create', '/workspaces create a b c d', '/workspaces nope x'):
        with pytest.raises(ValueError, match=USAGE.replace('[', r'\[').replace('|', r'\|')):
            await run(plugin, wrong)
    with pytest.raises(LookupError, match=r"No workspace kind 'docker'. Kinds: local-worktree."):
        await run(plugin, f'/workspaces create docker {repository}')
    with pytest.raises(LookupError, match='Kinds: none'):
        await Workspaces(kinds={}, directory=directory).create(kind='x', repo='y')
    with pytest.raises(ValueError, match="kind 'local-worktree' has none of its own"):
        await run(plugin, '/workspaces create local-worktree')
    with pytest.raises(ValueError, match='is not in a Git repository'):
        await run(plugin, f'/workspaces create local-worktree {tmp_path}')
    for workspace_id in ('nope-123', '../escape', 'UPPER'):
        with pytest.raises(LookupError, match='No workspace'):
            await run(plugin, f'/workspaces destroy {workspace_id}')


async def test_completion(directory: Path, repository: Path) -> None:
    plugin = loaded(directory, WorkspacesSettings(kinds={'b': WorktreeKind(), 'a': WorktreeKind()}))
    workspace = await workspaces_in(directory).create(kind='local-worktree', repo=str(repository))
    [command] = plugin.commands
    assert list(command.complete([])) == ['list', 'create', 'destroy']
    assert list(command.complete(['create', ''])) == ['a', 'b']
    assert list(command.complete(['destroy', ''])) == [workspace.id]
    assert list(command.complete(['create', 'a', ''])) == []


async def test_a_checkout_in_the_way(directory: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fixed(repo: str) -> str:
        return 'fixed-000000'

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.workspaces.manager.new_id', fixed)
    workspaces = workspaces_in(directory)
    await workspaces.create(kind='local-worktree', repo=str(repository))
    with pytest.raises(ValueError, match='already exists'):
        await workspaces.create(kind='local-worktree', repo=str(repository))


async def test_destroy_refuses_a_folder_outside_the_workspaces_root(directory: Path, repository: Path) -> None:
    """A record pointing anywhere else, by tampering or by accident, never gets that folder removed."""
    workspaces = workspaces_in(directory)
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


async def test_the_settings_menu(directory: Path) -> None:
    plugin = loaded(directory)
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


async def test_new_kind_names_are_checked(directory: Path) -> None:
    plugin = loaded(directory)
    assert isinstance(plugin.plugin, WorkspacesPlugin)
    dialogs = CheckingDialogs(tries=['local-worktree', 'Bad', 'fine'])
    assert await plugin.plugin.settings_menu().actions['add'](dialogs) == []
    assert dialogs.checks == [
        "There is a kind 'local-worktree' already.",
        'Start with a lowercase letter; then lowercase letters, digits, and hyphens.',
        None,
    ]


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


def test_branch_prefixes_git_refuses() -> None:
    for prefix in ('a..', 'a//', 'a/.', 'a.lock/'):
        with pytest.raises(ValueError, match='Git refuses branch names'):
            WorktreeKind(branch_prefix=prefix)
    assert WorktreeKind(branch_prefix='agent/').branch_prefix == 'agent/'
