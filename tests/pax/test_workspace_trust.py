"""Folder trust: the first-run question, `/trust`, the footer, where the decision is kept, and what it sets."""

import io
import json
import os
import subprocess
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.input import create_pipe_input
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import Authority
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax.app import _decide_trust  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_pax.cli import cli, headless, web
from pydantic_ai_pax.commands import config_command
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings, load_project_settings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.runtime.process import ClaiProcess
from pydantic_ai_pax.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_ai_pax.runtime.session_workspace import SessionWorkspace
from pydantic_ai_pax.runtime.workspace_trust import (
    WorkspaceTrust,
    repository_files,
    trust_command,
    trust_scope,
    undecided_notice,
)
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.status import Status


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '--quiet', '--initial-branch=main')
    git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '--quiet', '--allow-empty', '-m', 'first')
    return repo.resolve()


class AuthorityProbe(AbstractCapability[None]):
    """Records the authority each run's workspace gives the folder's instruction file."""

    def __init__(self) -> None:
        self.seen: list[Authority] = []

    async def before_run(self, ctx: RunContext[None]) -> None:
        self.seen.append(await ctx.workspace.authority_of('AGENTS.md'))


def shell_in(directory: Path, store: SettingsStore, *plugins: AbstractCapability[None]) -> SessionRuntime[None, str]:
    return create_shell(
        Agent(TestModel(call_tools=[]), deps_type=type(None), output_type=str),
        deps=None,
        plugins=plugins,
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=directory,
    )


async def test_a_folder_is_untrusted_until_you_trust_it(tmp_path: Path) -> None:
    """Nothing decided: the run's files are the repository's (`'none'`); `/trust yes` makes them yours."""
    probe = AuthorityProbe()
    store = SettingsStore(tmp_path / 'config.db')
    shell = shell_in(tmp_path, store, probe)
    try:
        assert shell.session.trust is not None and not shell.session.trust.decided
        assert 'untrusted' in shell.status.text()
        await shell.session.prompt('one')
        assert await shell.commands.execute_async('/trust yes') == (
            f'Trusted {tmp_path.resolve()}. Applies from the next prompt.'
        )
        assert 'untrusted' not in shell.status.text() and 'trusted' in shell.status.text()
        await shell.session.prompt('two')
        assert (await shell.commands.execute_async('/trust no')).startswith(f'Untrusted {tmp_path.resolve()}.')
        await shell.session.prompt('three')
        assert (await shell.commands.execute_async('/trust')).startswith('This folder is untrusted.\n')
        assert probe.seen == ['none', 'operator', 'none']
        forked = await shell.fork_session(None, ())
        await forked.prompt('four')
        assert probe.seen == ['none', 'operator', 'none', 'none']
        assert store.workspace_trust(tmp_path.resolve()) is False
        with pytest.raises(ValueError, match=r'Usage: /trust \[yes\|no\]'):
            await shell.commands.execute_async('/trust maybe')
    finally:
        await shell.close('exit')


async def test_trust_shows_what_it_covers(repository: Path) -> None:
    (repository / 'AGENTS.md').write_text('Be careful.')
    (repository / '.claude' / 'skills').mkdir(parents=True)
    (repository / '.clai').mkdir()
    (repository / '.clai' / 'settings.json').write_text('{}')
    store = SettingsStore(repository.parent / 'config.db')
    shell = shell_in(repository / '.claude', store)
    try:
        assert await shell.commands.execute_async('/trust') == (
            f'This folder is untrusted, as every folder is until you decide.\n'
            f'{repository}: this repository and its worktrees.\n'
            'Found here: ../AGENTS.md, skills (agent folders still reach the model as yours either way, for now).\n'
            'Untrusted, the agent can still read, edit and run code here, '
            "and instruction files and skills reach the model as the repository's, not yours.\n"
            'Either way, these still wait for their own approval: ../.clai/settings.json (/project approve, /plugins enable).\n'
            '/trust yes trusts it; /trust no keeps it untrusted.'
        )
    finally:
        await shell.close('exit')


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Console:
    """A terminal console with a terminal to answer in, as the interactive shell has."""
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    return Console(file=io.StringIO(), force_terminal=True, width=300, height=30)


async def test_first_run_question_is_asked_once_and_remembered(tmp_path: Path, terminal: Console) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    folder = tmp_path / 'project'
    folder.mkdir()
    asked: list[str] = []

    async def answer(details: str) -> bool:
        asked.append(details)
        return True

    console = terminal
    shell = shell_in(folder, store)
    try:
        await _decide_trust(shell, console=console, ask=answer)
        assert shell.session.trust is not None and shell.session.trust.trusted
    finally:
        await shell.close('exit')
    assert len(asked) == 1 and asked[0].startswith(f'{folder.resolve()}: this folder and its subfolders.')
    assert store.workspace_trust(folder.resolve()) is True

    again = shell_in(folder, store)
    try:
        await _decide_trust(again, console=console, ask=answer)
        assert again.session.trust is not None and again.session.trust.trusted
    finally:
        await again.close('exit')
    assert len(asked) == 1, 'a decided folder is not asked about again'


async def test_no_keeps_the_folder_untrusted_and_is_remembered(tmp_path: Path, terminal: Console) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    console = terminal

    async def decline(details: str) -> bool:
        return False

    shell = shell_in(tmp_path, store)
    try:
        await _decide_trust(shell, console=console, ask=decline)
        assert shell.session.trust is not None and shell.session.trust.authority == 'none'
    finally:
        await shell.close('exit')
    assert store.workspace_trust(tmp_path.resolve()) is False
    assert 'Untrusted' in console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]
    assert WorkspaceTrust.saved(store, tmp_path).decided


async def test_nobody_is_asked_without_a_terminal(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')

    async def never(details: str) -> bool:  # pragma: no cover
        raise AssertionError('asked without a terminal')

    shell = shell_in(tmp_path, store)
    try:
        await _decide_trust(shell, console=Console(file=io.StringIO()), ask=never)
        assert shell.session.trust is not None and not shell.session.trust.decided
    finally:
        await shell.close('exit')


@pytest.mark.parametrize(
    ('keys', 'trusted', 'saved'),
    [('1', True, True), ('\r', False, False), ('2', False, False), ('\x1b', False, None)],
)
async def test_the_question_defaults_to_untrusted(
    tmp_path: Path,
    terminal: Console,
    monkeypatch: pytest.MonkeyPatch,
    keys: str,
    trusted: bool,
    saved: bool | None,
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    shell = shell_in(tmp_path, store)
    try:
        with create_pipe_input() as pipe:
            monkeypatch.setattr('pydantic_ai_pax.ui.prompt.question_input.create_input', lambda: pipe)
            pipe.send_text(keys)
            await _decide_trust(shell, console=terminal)
        assert shell.session.trust is not None
        assert shell.session.trust.trusted is trusted
        assert shell.session.trust.decided is (saved is not None)
    finally:
        await shell.close('exit')
    shown = Text.from_ansi(terminal.file.getvalue()).plain  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownArgumentType]
    assert f'{tmp_path.resolve()}: this folder and its subfolders.' in ' '.join(shown.split())
    assert store.workspace_trust(tmp_path.resolve()) is saved


def test_a_repository_is_trusted_as_a_whole(repository: Path, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    (repository / 'src').mkdir()
    assert trust_scope(repository / 'src') == trust_scope(repository)
    assert trust_scope(repository).key == repository and trust_scope(repository).repository
    WorkspaceTrust.saved(store, repository / 'src').decide(store, trusted=True)
    assert store.workspace_trust(repository) is True
    assert WorkspaceTrust.saved(store, repository).trusted


def test_a_worktree_shares_its_repository_decision(repository: Path, tmp_path: Path) -> None:
    worktree = tmp_path / 'linked'
    git(repository, 'worktree', 'add', '--quiet', '-b', 'side', str(worktree))
    assert trust_scope(worktree).key == repository


def test_a_nested_repository_is_asked_about_on_its_own(repository: Path, tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    nested = repository / 'vendor' / 'clone'
    nested.mkdir(parents=True)
    git(nested, 'init', '--quiet')
    WorkspaceTrust.saved(store, repository).decide(store, trusted=True)
    assert trust_scope(nested).key == nested.resolve()
    assert not WorkspaceTrust.saved(store, nested).decided


def test_a_git_file_cannot_borrow_a_trusted_repository(repository: Path, tmp_path: Path) -> None:
    """A downloaded folder cannot inherit a trusted repository's decision via a forged `.git` file or symlink."""
    store = SettingsStore(tmp_path / 'config.db')
    WorkspaceTrust.saved(store, repository).decide(store, trusted=True)

    # 1. `.git` file pointing directly at a trusted repository's `.git` directory.
    spoof = tmp_path / 'download'
    spoof.mkdir()
    (spoof / '.git').write_text(f'gitdir: {repository / ".git"}\n')
    assert trust_scope(spoof).key == spoof.resolve()
    assert not WorkspaceTrust.saved(store, spoof).trusted

    # 2. `.git` file pointing at a real linked worktree's gitdir belonging to another path.
    real_wt = tmp_path / 'real-wt'
    git(repository, 'worktree', 'add', '--quiet', '-b', 'real', str(real_wt))
    wt_gitdir = (real_wt / '.git').read_text().strip()
    other_wt = tmp_path / 'other-wt'
    other_wt.mkdir()
    (other_wt / '.git').write_text(f'{wt_gitdir}\n')
    assert trust_scope(other_wt).key == other_wt.resolve()

    # 3. Symlink `.git` pointing at a real linked worktree's `.git` file.
    symlinked = tmp_path / 'symlinked'
    symlinked.mkdir()
    (symlinked / '.git').symlink_to(real_wt / '.git')
    assert trust_scope(symlinked).key == symlinked.resolve()

    # 4. Forged local worktree gitdir whose `commondir` points at the trusted repo's `.git`.
    forged = tmp_path / 'forged'
    fake_gitdir = forged / 'fake_wt'
    fake_gitdir.mkdir(parents=True)
    (forged / '.git').write_text('gitdir: ./fake_wt\n')
    (fake_gitdir / 'gitdir').write_text('../.git\n')
    (fake_gitdir / 'commondir').write_text(f'{repository / ".git"}\n')
    assert trust_scope(forged).key == forged.resolve()

    # 5. Corrupt or unreadable `.git` file.
    corrupt = tmp_path / 'corrupt'
    corrupt.mkdir()
    (corrupt / '.git').write_text('not a gitdir pointer\n')
    assert trust_scope(corrupt).key == corrupt.resolve()
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / '.git').write_bytes(b'\xff\xfe')
    assert trust_scope(empty).key == empty.resolve()

    # 6. Non-regular metadata file (directory or FIFO) is rejected without blocking.
    non_regular = tmp_path / 'non-regular'
    non_regular_gitdir = non_regular / 'wt'
    non_regular_gitdir.mkdir(parents=True)
    (non_regular / '.git').write_text('gitdir: ./wt\n')
    if hasattr(os, 'mkfifo'):
        os.mkfifo(non_regular_gitdir / 'gitdir')
    else:  # pragma: no cover -- Windows has no mkfifo; a directory exercises the same `not is_file()` guard.
        (non_regular_gitdir / 'gitdir').mkdir()
    (non_regular_gitdir / 'commondir').write_text(f'{repository / ".git"}\n')
    assert trust_scope(non_regular).key == non_regular.resolve()


def test_bare_repository_worktree_gets_its_own_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    bare = tmp_path / 'bare.git'
    git(tmp_path, 'init', '--quiet', '--bare', str(bare))
    wt = tmp_path / 'bare-wt'
    git(bare, 'worktree', 'add', '--quiet', '--orphan', '-b', 'main', str(wt))
    assert trust_scope(wt).key == wt.resolve()


def test_home_inside_a_repository_or_worktree_stays_session_only(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    WorkspaceTrust.saved(store, repository).decide(store, trusted=True)

    # Home as a linked worktree of a trusted repository stays session-only.
    home_wt = tmp_path / 'home-wt'
    git(repository, 'worktree', 'add', '--quiet', '-b', 'home-wt', str(home_wt))
    monkeypatch.setattr(Path, 'home', staticmethod(lambda: home_wt))
    scope = trust_scope(home_wt)
    assert scope.key == home_wt.resolve() and not scope.remembered and not scope.repository
    assert not WorkspaceTrust.saved(store, home_wt).decided

    # A dotfiles `.git` in `$HOME` does not capture subfolders of `$HOME`.
    sub = home_wt / 'notes'
    sub.mkdir()
    sub_scope = trust_scope(sub)
    assert sub_scope.key == sub.resolve() and sub_scope.remembered and not sub_scope.repository


def test_outside_a_repository_a_decision_covers_subfolders(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    parent = tmp_path / 'notes'
    child = parent / 'drafts' / 'old'
    child.mkdir(parents=True)
    WorkspaceTrust.saved(store, parent).decide(store, trusted=True)
    inherited = WorkspaceTrust.saved(store, child)
    assert trust_command(inherited, store, child, []).startswith(
        f'This folder is trusted, as you decided for {parent.resolve()}.\n'
    )
    assert (inherited.trusted, inherited.decided_for, inherited.scope.key) == (True, parent.resolve(), child.resolve())
    inherited.decide(store, trusted=False)
    assert not WorkspaceTrust.saved(store, child).trusted, 'the nearest decision wins'
    assert WorkspaceTrust.saved(store, parent / 'drafts').trusted


async def test_trust_in_the_home_folder_holds_for_the_session_only(tmp_path: Path) -> None:
    home = Path.home().resolve()
    home.mkdir(parents=True, exist_ok=True)
    store = SettingsStore(tmp_path / 'config.db')
    shell = shell_in(home, store)
    try:
        assert shell.session.trust is not None and not shell.session.trust.scope.remembered
        assert await shell.commands.execute_async('/trust yes') == (
            f'Trusted {home} for this session. Applies from the next prompt.'
        )
        assert shell.session.trust.trusted
        assert 'holds for this session only' in await shell.commands.execute_async('/trust')
    finally:
        await shell.close('exit')
    assert store.workspace_trust(home) is None
    assert not WorkspaceTrust.saved(store, home).decided, 'asked again next launch'
    (home / 'scratch').mkdir()
    store.set_workspace_trust(home, trusted=True)  # As an older build or a hand edit could have left it.
    assert not WorkspaceTrust.saved(store, home / 'scratch').decided, 'a decision above is never read for home'


async def test_a_project_file_cannot_trust_its_repository(repository: Path, tmp_path: Path) -> None:
    (repository / '.clai').mkdir()
    (repository / '.clai' / 'settings.json').write_text(
        json.dumps({'trust': True, 'trusted': True, 'workspace_trust': 'trusted', 'hasTrustDialogAccepted': True})
    )
    store = SettingsStore(tmp_path / 'config.db')
    project = load_project_settings(repository, store=store)
    assert project.unknown == ('hasTrustDialogAccepted', 'trust', 'trusted', 'workspace_trust')
    assert not WorkspaceTrust.saved(store, repository).decided
    with pytest.raises(ValueError, match='Unknown settings'):
        config_command(store, ['set', 'workspace_trust', 'true'])


def test_footer_shows_the_trust_level() -> None:
    level: list[str] = ['untrusted']
    status = Status(
        workspace='/a/very/long/project/folder', trust=lambda: 'untrusted' if level[0] == 'untrusted' else 'trusted'
    )
    assert (theme.WARNING, 'untrusted') in status.toolbar()
    assert '/a/very/long/project/folder | untrusted' in status.text()
    short = status.toolbar(width=72)
    assert (theme.MUTED, '…/folder') in short and (theme.WARNING, 'untrusted') in short
    level[0] = 'trusted'
    assert (theme.MUTED, 'trusted') in status.toolbar()
    assert (theme.WARNING, 'untrusted') in Status(trust=lambda: 'untrusted').toolbar()
    assert 'trusted' not in Status().text(), "a host's own session shows no trust level"
    assert Status(trust=lambda: 'trusted', fields=lambda: ['model']).text() == 'agent default'


async def test_headless_run_uses_the_saved_decision_without_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    probe = AuthorityProbe()
    store = SettingsStore(tmp_path / 'config.db')

    async def run() -> int:
        agent = Agent(TestModel(call_tools=[], custom_output_text='done'), deps_type=type(None), capabilities=[probe])
        return await headless.run_headless(
            text='go', settings=Settings(model=None), store=store, project=ProjectSettings(), agent=agent
        )

    assert await run() == 0
    assert capsys.readouterr().err == (
        f'{tmp_path.resolve()} is untrusted, as you have not decided whether to trust it. '
        'Start Pax there and answer its question, or enter /trust yes, to trust it.\n'
    )
    await anyio.to_thread.run_sync(lambda: store.set_workspace_trust(tmp_path.resolve(), trusted=True))
    assert await run() == 0
    assert capsys.readouterr().err == ''
    assert probe.seen == ['none', 'operator']


def test_repository_files_lists_names_only(repository: Path) -> None:
    (repository / 'CLAUDE.md').write_text('x')
    (repository / '.agents' / 'agents').mkdir(parents=True)
    (repository / '.mcp.json').write_text('{}')
    files = repository_files(repository)
    assert (files.trusted, files.approved_separately) == (
        ('CLAUDE.md', '.agents/agents'),
        (('.mcp.json', '/mcp trust'),),
    )


def test_without_a_home_folder_every_folder_is_remembered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_home() -> Path:
        raise RuntimeError('Could not determine home directory.')

    monkeypatch.setattr(Path, 'home', staticmethod(no_home))
    assert trust_scope(tmp_path).remembered


async def test_web_launch_names_an_undecided_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = SettingsStore(tmp_path / 'config.db')
    levels: list[bool] = []

    async def serve(shell: SessionRuntime[None, str], **_: object) -> None:
        assert shell.session.trust is not None
        levels.append(shell.session.trust.trusted)

    monkeypatch.setattr(web, '_serve_web', serve)
    settings_for = cli._settings_for(store, model=None, request_limit=None)  # pyright: ignore[reportPrivateUsage]

    async def launch() -> str:
        console = Console(file=io.StringIO(), width=1000)
        await web.run_web(settings_for=settings_for, store=store, project=ProjectSettings(), console=console)
        return console.file.getvalue()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]

    assert 'is untrusted, as' in await launch()
    await anyio.to_thread.run_sync(lambda: store.set_workspace_trust(tmp_path.resolve(), trusted=True))
    assert 'untrusted' not in await launch()
    assert levels == [False, True]
    assert undecided_notice(None) is None


async def test_trust_changes_reach_other_live_sessions_in_the_same_process(repository: Path, tmp_path: Path) -> None:
    """`/trust` in one session updates every other live session sharing its scope, including repo subfolders and `$HOME`."""
    store = SettingsStore(tmp_path / 'config.db')
    process: ClaiProcess[None] = ClaiProcess.open(store)
    sub = repository / 'sub'
    sub.mkdir()
    probe_root, probe_sub = AuthorityProbe(), AuthorityProbe()

    def open_in(directory: Path, probe: AuthorityProbe) -> SessionRuntime[None, str]:
        return create_shell(
            Agent(TestModel(call_tools=[]), deps_type=type(None), output_type=str),
            deps=None,
            plugins=(probe,),
            usage_limits=None,
            console=Console(file=io.StringIO()),
            settings=Settings(model=None),
            store=None,
            process=process,
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            workspace=directory,
        )

    first = open_in(repository, probe_root)
    second = open_in(sub, probe_sub)
    try:
        assert first.session.trust is second.session.trust
        await first.commands.execute_async('/trust yes')
        await second.session.prompt('trusted-run')
        assert 'untrusted' not in second.status.text() and 'trusted' in second.status.text()

        await first.commands.execute_async('/trust no')
        await second.session.prompt('revoked-run')
        assert 'untrusted' in second.status.text()
        assert (await second.commands.execute_async('/trust')).startswith('This folder is untrusted.\n')
        assert probe_sub.seen == ['operator', 'none']
    finally:
        await second.close('exit')
        await first.close('exit')

    # Moving a session into the repository through `SessionWorkspace` joins the same shared trust.
    other = tmp_path / 'other'
    other.mkdir()
    mover = open_in(other, probe_root)
    try:
        workspace = SessionWorkspace(
            session=mover.session,
            context=mover.context,
            loader=mover.loader,
            status=mover.status,
            settings=lambda _: Settings(model=None),
        )
        await workspace.move(str(sub))
        assert mover.session.trust is process.trusts[repository]
    finally:
        await mover.close('exit')

    # Non-repo parent changes refresh live descendant scopes unless a child has its own decision.
    parent = tmp_path / 'notes'
    child = parent / 'drafts'
    child.mkdir(parents=True)
    probe_child = AuthorityProbe()
    parent_shell = open_in(parent, AuthorityProbe())
    child_shell = open_in(child, probe_child)
    try:
        await parent_shell.commands.execute_async('/trust yes')
        await child_shell.session.prompt('inherited-yes')
        await parent_shell.commands.execute_async('/trust no')
        await child_shell.session.prompt('inherited-no')
        await child_shell.commands.execute_async('/trust yes')
        await parent_shell.commands.execute_async('/trust no')
        await child_shell.session.prompt('own-yes-wins')
        assert probe_child.seen == ['operator', 'none', 'operator']
    finally:
        await child_shell.close('exit')
        await parent_shell.close('exit')

    # Session-only `$HOME` trust is shared across live sessions in the same process and seeded on reload.
    home = Path.home().resolve()
    home.mkdir(parents=True, exist_ok=True)
    home_a = open_in(home, AuthorityProbe())
    home_b = open_in(home, AuthorityProbe())
    try:
        await home_a.commands.execute_async('/trust yes')
        assert home_b.session.trust is not None and home_b.session.trust.trusted
        reloaded: ClaiProcess[None] = ClaiProcess.open(store)
        assert reloaded.trust_for(home, initial=home_a.session.trust) is home_a.session.trust
        assert reloaded.trust_for(home) is home_a.session.trust
    finally:
        await home_b.close('exit')
        await home_a.close('exit')


async def test_concurrent_trust_lookups_and_updates_share_one_instance(tmp_path: Path) -> None:
    """Concurrent `resolve_trust` misses return the same `WorkspaceTrust`, and concurrent updates stay consistent."""
    store = SettingsStore(tmp_path / 'config.db')
    process: ClaiProcess[None] = ClaiProcess.open(store)
    parent = tmp_path / 'shared'
    child = parent / 'nested'
    child.mkdir(parents=True)
    found: list[WorkspaceTrust] = []

    async def lookup(directory: Path) -> None:
        found.append(await process.resolve_trust(directory))

    async with anyio.create_task_group() as group:
        for _ in range(4):
            group.start_soon(lookup, parent)

    assert len(found) == 4 and all(item is found[0] for item in found)
    # A subsequent fast-path lookup returns the cached instance.
    assert await process.resolve_trust(parent) is found[0]

    # Updating `parent` while another coroutine looks up `child` and inserts new scopes never raises or loses updates.
    child_resolved: list[WorkspaceTrust] = []

    async def lookup_child() -> None:
        child_resolved.append(await process.resolve_trust(child))

    async def trust_parent() -> None:
        await process.decide_trust(found[0], trusted=True)

    async with anyio.create_task_group() as group:
        group.start_soon(trust_parent)
        group.start_soon(lookup_child)
        for i in range(4):
            sibling = tmp_path / f'sibling-{i}'
            sibling.mkdir()
            group.start_soon(lookup, sibling)

    assert child_resolved[0].trusted is True
    await process.decide_trust(found[0], trusted=False)
    assert child_resolved[0].trusted is False
