"""Headless project/session navigation and safe, responsive frame rendering."""

import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from itertools import chain, count, repeat
from pathlib import Path, PurePath
from types import SimpleNamespace
from uuid import UUID, uuid4, uuid5

import pytest
from termflow.ansi.utils import visible_length
from termflow.tui.keys import Key

import pydantic_clai2.ui.menus.session_browser as module
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, GitCheckout
from pydantic_clai2.runtime.project_identity import (
    MISSING,
    ProjectIdentity,
    git_checkout,
    placements,
    project_identity,
)
from pydantic_clai2.ui.menus.session_browser import SessionBrowser, date_label, plain


@pytest.fixture(autouse=True)
def fictional_checkouts_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared `/a` and `/b` workspaces stand for checkouts that still exist."""

    def identify(workspace: str) -> ProjectIdentity:
        identity = project_identity(workspace)
        return replace(identity, missing=False) if workspace.startswith(('/a/', '/b/')) else identity

    monkeypatch.setattr(module, 'project_identity', identify)


def git(*args: str) -> None:
    subprocess.run(['git', *args], check=True, capture_output=True)


def repository(repo: Path, *, linked: Path, branch: str) -> None:
    """A repository on `main` with a linked worktree checked out on `branch`."""
    git('init', '-b', 'main', str(repo))
    git(
        '-C',
        str(repo),
        '-c',
        'user.name=Test',
        '-c',
        'user.email=test@example.com',
        'commit',
        '--allow-empty',
        '-m',
        'init',
    )
    git('-C', str(repo), 'worktree', 'add', '-b', branch, str(linked))


def browser(
    *, keys: list[str] | None = None, now: datetime | None = None
) -> tuple[SessionBrowser, list[ConversationSummary]]:
    entries = [
        ConversationSummary(id='one', workspace='/a/project', title='Fix renderer', subtitle='Race', tags=('tests',)),
        ConversationSummary(id='two', workspace='/b/project', title='Add resume', message_count=24, total_tokens=1200),
        ConversationSummary(id='three', workspace='/a/project', title='Other', outcome='cancelled'),
    ]
    source = iter(keys or [])

    def refresh(query: str, limit: int, archived: bool) -> list[ConversationSummary]:
        found = [e for e in entries if query.lower() in e.title.lower() and (archived or e.archived_at is None)]
        return found[:limit]

    def rename(entry: ConversationSummary, title: str) -> None:
        entries[entries.index(entry)] = replace(entry, title=title)

    def archive(entry: ConversationSummary, archived: bool) -> None:
        index = next(index for index, e in enumerate(entries) if e.id == entry.id)
        entries[index] = replace(entry, archived_at=datetime.now(UTC) if archived else None)

    widget = SessionBrowser(
        entries=entries,
        workspace='/a/project',
        active_id='one',
        refresh=refresh,
        preview=lambda session_id: f'Transcript {session_id}\nuser: hello',
        delete=entries.remove,
        rename=rename,
        archive=archive,
        output=StringIO(),
        key_source=lambda: next(source),
        size=lambda: (110, 24),
        now=lambda: now or datetime.now(UTC),
    )
    return widget, entries


@pytest.mark.parametrize('width,height', [(120, 30), (80, 24), (40, 12), (12, 5), (1, 3)])
def test_frame_fits_and_handles_narrow_terminals(width: int, height: int) -> None:
    widget, _ = browser()
    for mode in ('projects', 'sessions', 'preview'):
        widget.mode = mode
        frame = widget.frame(width=width, height=height)
        assert len(frame) <= height
        assert all(visible_length(line) <= max(1, width - 1) for line in frame)
    assert '\x1b' not in plain('unsafe\x1b[2J')


@pytest.mark.parametrize('color', ['', 'truecolor'])
@pytest.mark.parametrize('width', [40, 120])
def test_focus_labels_and_persistent_project_highlight(monkeypatch: pytest.MonkeyPatch, color: str, width: int) -> None:
    monkeypatch.setenv('COLORTERM', color)
    widget, _ = browser()
    highlight = module.theme.sgr(module.theme.INFO, bold=True)
    project = f'{highlight}> /a/project (2)\x1b[0m'
    assert project in '\n'.join(widget.frame(width=120, height=24))
    assert 'SELECT PROJECT' in '\n'.join(widget.frame(width=width, height=24))
    assert 'Enter/Right open project' in widget.footer()
    widget.handle_key(Key.ENTER)
    assert project in '\n'.join(widget.frame(width=120, height=24))
    frame = '\n'.join(widget.frame(width=width, height=24))
    assert 'SELECT SESSION' in frame
    assert 'SELECT PROJECT' not in frame
    assert f'{highlight}> ' in frame
    assert 'Left/Esc projects' in widget.footer()
    widget.handle_key(Key.DOWN)
    cards = widget.frame(width=40, height=24)
    assert any(f'{highlight}> ' in line and 'Other' in line for line in cards)
    assert not any(f'{highlight}> ' in line and 'Fix renderer' in line for line in cards)
    widget.handle_key(Key.LEFT)
    assert 'SELECT PROJECT' in '\n'.join(widget.frame(width=width, height=24))
    widget.handle_key(Key.DOWN)
    assert f'{highlight}> /b/project (1)\x1b[0m' in '\n'.join(widget.frame(width=120, height=24))
    assert project not in '\n'.join(widget.frame(width=120, height=24))


def test_project_selection_preview_and_back() -> None:
    widget, _ = browser()
    assert widget.project == '/a/project'
    widget.handle_key(Key.ENTER)
    assert widget.mode == 'sessions'
    widget.handle_key(Key.DOWN)
    assert widget.selected is not None and widget.selected.id == 'three'
    widget.handle_key(Key.RIGHT)
    assert widget.mode == 'preview'
    assert 'three' in widget.preview_text
    widget.handle_key(Key.DOWN)
    assert widget.preview_offset == 1
    widget.handle_key(Key.UP)
    widget.handle_key(Key.ESCAPE)
    assert widget.mode == 'sessions'
    assert widget.handle_key(Key.ENTER) == 'three'
    widget.handle_key(Key.LEFT)
    widget.handle_key(Key.DOWN)
    widget.handle_key(Key.ENTER)
    assert widget.selected is not None and widget.selected.id == 'two'
    assert widget.handle_key(Key.ENTER) == 'two'
    assert widget.handle_key('ctrl-c') == ''


def test_search_sort_rename_delete_and_stable_refresh() -> None:
    widget, entries = browser()
    widget.handle_key('/')
    for key in 'resume':
        widget.handle_key(key)
    widget.handle_key(Key.ENTER)
    assert [e.id for e in widget.sessions] == ['two']
    assert 'Search all projects' in '\n'.join(widget.frame(width=120, height=24))
    widget.handle_key('s')
    widget.handle_key('s')
    widget.handle_key('s')
    widget.handle_key('r')
    widget.buffer = 'My title'
    widget.handle_key(Key.ENTER)
    assert entries[1].title == 'My title'
    widget.handle_key(Key.ESCAPE)
    widget.handle_key(Key.ENTER)
    widget.handle_key('d')
    assert 'active session' in widget.notice
    widget.handle_key(Key.DOWN)
    widget.handle_key('d')
    assert widget.confirm is not None
    assert widget.footer() == 'Delete saved session: Other? y confirm - any other key cancel'
    widget.handle_key('n')
    assert len(entries) == 3
    widget.handle_key('d')
    widget.handle_key('y')
    assert len(entries) == 2
    widget.selected_id = 'one'
    entries[0] = replace(entries[0], title='Renamed in background')
    widget.reload()
    assert widget.selected is not None and widget.selected.id == 'one'
    assert widget.selected.title == 'Renamed in background'
    widget.handle_key('m')
    assert widget.limit == 400
    widget.handle_key('/')
    widget.handle_key('x')
    widget.handle_key(Key.BACKSPACE)
    assert widget.buffer == ''
    widget.handle_key(Key.ESCAPE)


def test_empty_search_and_scripted_loop() -> None:
    widget, entries = browser(keys=['', Key.ENTER, Key.ENTER])
    assert widget.run() == 'one'
    entries.clear()
    widget.reload()
    assert widget.selected is None
    widget.mode = 'sessions'
    assert 'No saved sessions' in '\n'.join(widget.frame(width=120, height=24))
    assert widget.handle_key(Key.ENTER) is None
    widget.handle_key(Key.ESCAPE)
    assert widget.handle_key(Key.ESCAPE) == ''


@pytest.mark.parametrize('workspace', ['/a/project', '/b/project', '/missing/clai2/worktree'])
def test_resume_without_directory_confirmation(workspace: str) -> None:
    widget, entries = browser(keys=[Key.ENTER, '', Key.ENTER, '', 'ctrl-c'])
    entries[:] = [replace(entries[1], workspace=workspace)]
    widget.reload()
    assert widget.loop() == 'two'
    assert widget.confirm is None
    assert widget.workspace == '/a/project'
    assert entries[0].workspace == workspace


def test_date_buckets() -> None:
    now = datetime(2026, 9, 18, 12, tzinfo=UTC)
    assert date_label(now, now=now) == 'TODAY'
    assert date_label(now - timedelta(days=1), now=now) == 'YESTERDAY'
    assert date_label(now - timedelta(days=6), now=now) == 'LAST 7 DAYS'
    assert date_label(now - timedelta(days=29), now=now) == 'LAST 30 DAYS'
    assert date_label(now - timedelta(days=365), now=now) == 'SEPTEMBER 2025'


def test_idle_refresh_and_storage_errors_stay_in_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, entries = browser(keys=['', 'ctrl-c'])
    ticks = iter([0.0, 1.0, 1.0])
    # Replace the browser's `time`, not `module.time.monotonic`: that is the stdlib's clock, which other
    # threads' event loops read too.
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    entries[0] = replace(entries[0], title='Named in background')
    assert widget.loop() == ''
    assert isinstance(widget.output, StringIO)
    assert 'Named in background' in widget.output.getvalue()

    def fail_preview(session_id: str) -> str:
        raise ValueError('Cannot read this session')

    widget, _ = browser(keys=[Key.ENTER, Key.RIGHT, 'ctrl-c'])
    widget.preview = fail_preview
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 0.0))
    assert widget.loop() == ''
    assert 'Cannot read this session' in widget.notice


def test_moving_the_selection_never_waits_on_a_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refreshes wait for a poll without a key, so a held arrow key never waits on storage."""
    widget, _ = browser(keys=[Key.ENTER, Key.DOWN, Key.UP, Key.DOWN, '', 'ctrl-c'])
    refresh = widget.refresh
    refreshed: list[str | None] = []

    def counted(query: str, limit: int, archived: bool) -> list[ConversationSummary]:
        refreshed.append(widget.selected_id)
        return refresh(query, limit, archived)

    widget.refresh = counted
    # Every key arrives long after the last refresh.
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=count(step=1.0).__next__))
    assert widget.loop() == ''
    assert refreshed == ['three']


def test_render_long_cards_all_modes_and_empty_projects() -> None:
    widget, entries = browser()
    entries[0] = replace(entries[0], title='界' * 200, subtitle='detail' * 50, tags=('verylongtag' * 4,))
    widget.reload()
    widget.mode = 'sessions'
    for sort in range(3):
        widget.sort = sort
        assert all(visible_length(line) <= 119 for line in widget.frame(width=120, height=12))
    widget.handle_key('r')
    assert 'rename:' in widget.footer()
    widget.handle_key(Key.ESCAPE)
    widget.handle_key('/')
    widget.buffer = 'not present'
    widget.handle_key(Key.ENTER)
    assert widget.selected is None
    widget.handle_key('d')
    widget.handle_key('s')
    widget.handle_key(Key.ESCAPE)
    widget.handle_key('ctrl-p')
    widget.mode = 'preview'
    widget.handle_key('x')
    widget.handle_key('q')
    widget.handle_key('q')
    assert widget.handle_key('q') == ''


def test_preview_work_is_bounded_and_truncation_visible() -> None:
    widget, _ = browser()
    widget.mode = 'preview'
    widget.preview_text = 'x' * 1_000_000
    widget.preview_offset = 1_000_000
    assert 'Preview truncated' in '\n'.join(widget.frame(width=120, height=24))
    bottom = widget.preview_offset
    assert 0 < bottom < 1_000_000
    widget.handle_key(Key.UP)
    widget.frame(width=120, height=24)
    assert widget.preview_offset == bottom - 1
    widget.preview_text = 'short'
    widget.frame(width=120, height=24)
    assert widget.preview_offset == 0


def test_ignored_keys_and_empty_rename() -> None:
    widget, entries = browser()
    widget.handle_key(Key.ENTER)
    widget.handle_key('unknown')
    widget.handle_key('r')
    widget.buffer = ''
    widget.handle_key(Key.ENTER)
    assert entries[0].title == 'Fix renderer'
    widget.mode = 'rename'
    widget.handle_key('ctrl-p')
    entries.clear()
    widget.reload()
    widget.handle_key(Key.DOWN)
    widget.handle_key(Key.ENTER)
    assert not widget.entries
    widget.mode = 'sessions'
    widget.handle_key(Key.DOWN)


def test_multiline_metadata_cannot_inject_terminal_rows() -> None:
    widget, entries = browser()
    entries[0] = replace(
        entries[0], workspace='/a/line\nbreak', title='title\nnext', subtitle='sub\nnext', tags=('tag\nnext',)
    )
    widget.reload()
    widget.project = entries[0].workspace
    widget.query = 'global'
    for mode in ('projects', 'sessions', 'rename'):
        widget.mode = mode
        widget.buffer = 'rename\nnext'
        assert all('\n' not in line and '\r' not in line for line in widget.frame(width=120, height=24))
    widget.confirm = entries[0]
    assert '\n' not in widget.footer()
    assert plain('first\nsecond', multiline=True) == 'first\nsecond'
    assert plain('first\nsecond') == 'first second'


def test_idle_loop_redraws_only_for_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, _ = browser(keys=['', '', '', 'ctrl-c'])
    sizes = iter([(110, 24), (110, 24), (100, 24), (100, 24)])
    widget.size = lambda: next(sizes)
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: 0.0))
    assert widget.loop() == ''
    assert isinstance(widget.output, StringIO)
    assert widget.output.getvalue().count('CLAI > Resume session') == 2


def test_failed_idle_refresh_waits_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, _ = browser(keys=['', '', '', 'ctrl-c'])
    ticks = chain([0.0, 0.6, 0.6, 0.7], repeat(0.8))
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    attempts: list[str] = []

    def fail(query: str, limit: int, archived: bool) -> list[ConversationSummary]:
        attempts.append(query)
        raise OSError('database unavailable')

    widget.refresh = fail
    assert widget.loop() == ''
    assert attempts == ['']
    assert 'database unavailable' in widget.notice


@pytest.mark.parametrize('repo_name', ['repo', 'repo\nname'])
def test_repository_grouping_and_checkout_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_name: str
) -> None:
    """Real Git metadata, not directory naming conventions, unifies linked checkouts."""
    repo = tmp_path / repo_name
    linked = tmp_path / '.herdr' / 'worktrees' / f'unrelated-long-directory-{repo_name}'
    repository(repo, linked=linked, branch='feature/resume')
    subdir = linked / 'src'
    subdir.mkdir()
    widget, entries = browser()
    entries[:] = [
        replace(entries[0], workspace=str(repo)),
        replace(entries[1], workspace=str(linked)),
        replace(entries[2], workspace=str(subdir)),
    ]
    widget.workspace = str(subdir)
    widget.reload()
    assert len(widget.projects) == 1
    assert widget.project_label(widget.project) == repo_name
    assert [e.id for e in widget.sessions] == ['one', 'two', 'three']
    frame = '\n'.join(widget.frame(width=120, height=24))
    assert f'{plain(repo_name)} (3)' in frame
    assert '[main]' in frame and '[feature/resume]' in frame
    assert str(linked) not in frame
    widget.handle_key(Key.ENTER)
    widget.selected_id = 'three'
    assert widget.handle_key(Key.ENTER) == 'three'
    widget.selected_id = 'two'
    assert widget.handle_key(Key.ENTER) == 'two'
    widget.reload()
    assert widget.selected is not None and widget.selected.id == 'two'

    resolved: list[str] = []

    def observe_resolve(workspace: str) -> ProjectIdentity:
        resolved.append(workspace)
        return project_identity(workspace)

    monkeypatch.setattr(module, 'project_identity', observe_resolve)
    assert module.project_identity(str(linked)).checkout == 'feature/resume'
    widget.reload()
    widget.query = 'global'
    assert f'[{plain(repo_name)}: feature/resume]' in '\n'.join(widget.frame(width=120, height=24))
    assert resolved == [str(linked)]
    git('-C', str(linked), 'checkout', '--detach')
    assert project_identity(str(linked)).checkout == f'{linked.name} (detached)'


@pytest.mark.parametrize('workspace', ['relative/project', '/missing/clai2/worktree'])
def test_identity_fallback(workspace: str) -> None:
    identity = project_identity(workspace)
    assert identity.key == workspace
    assert identity.name == Path(workspace).name
    assert identity.checkout == ''
    assert identity.missing == workspace.startswith('/')


def test_checkout_tree_and_missing_folders(tmp_path: Path) -> None:
    """Worktrees nest, dimmed, under their repository; deleted folders gather in one group last.

    Rows are keyed by folder, so deleted folders sharing a name stay apart, labelled by path.
    """
    repo, linked, gone = tmp_path / 'repo', tmp_path / 'linked', tmp_path / 'gone'
    old, twin = gone / 'old-worktree', gone / 'elsewhere' / 'old-worktree'
    repository(repo, linked=linked, branch='feature')
    widget, entries = browser()
    entries[:] = [
        replace(entries[0], workspace=str(repo)),
        replace(entries[1], workspace=str(old)),
        replace(entries[2], workspace=str(linked)),
        ConversationSummary(id='four', workspace=str(gone / 'other'), title='Older'),
        ConversationSummary(id='five', workspace=str(twin), title='Oldest'),
    ]
    widget.reload()
    key = widget.project
    assert widget.rows == [
        (key, ''),
        (key, widget.identity(str(repo)).root),
        (key, widget.identity(str(linked)).root),
        (MISSING, ''),
        (MISSING, str(old)),
        (MISSING, str(gone / 'other')),
        (MISSING, str(twin)),
    ]
    assert widget.checkout_label(MISSING, str(twin)) == str(twin)
    muted = module.theme.sgr(module.theme.MUTED)
    highlight = module.theme.sgr(module.theme.INFO, bold=True)
    frame = '\n'.join(widget.frame(width=120, height=24))
    assert f'{highlight}> repo (2)\x1b[0m' in frame
    assert f'{muted}  ├─ main (1)\x1b[0m' in frame
    assert f'{muted}  └─ feature (1)\x1b[0m' in frame
    assert '  missing folders (3)' in frame
    assert f'{muted}  ├─ other (1)\x1b[0m' in frame
    widget.handle_key(Key.DOWN)
    assert [e.id for e in widget.sessions] == ['one']
    assert 'SESSIONS: repo / main' in '\n'.join(widget.frame(width=120, height=24))
    widget.handle_key(Key.DOWN)
    assert [e.id for e in widget.sessions] == ['three']
    assert f'{highlight}> └─ feature (1)\x1b[0m' in '\n'.join(widget.frame(width=120, height=24))
    for _ in range(4):
        widget.handle_key(Key.DOWN)
    assert (widget.project, widget.checkout) == (MISSING, str(twin))
    assert [e.id for e in widget.sessions] == ['five']
    assert f'SESSIONS: missing folders / {twin}'[:60] in '\n'.join(widget.frame(width=300, height=24))
    del entries[3:]
    widget.reload()
    assert (widget.project, widget.checkout) == (MISSING, '')
    assert [e.id for e in widget.sessions] == ['two']
    widget.query = 'global'
    assert '[missing folders: old-worktree]' in '\n'.join(widget.frame(width=120, height=24))


def test_removed_worktree_stays_in_its_repository(tmp_path: Path) -> None:
    """A session saved with its checkout is listed under its repository after the worktree is removed."""
    repo, gone = tmp_path / 'repo', tmp_path / 'gone'
    repository(repo, linked=gone, branch='fix')
    checkout = git_checkout(project_identity(str(gone)))
    git('-C', str(repo), 'worktree', 'remove', str(gone))
    widget, entries = browser()
    entries[:] = [
        replace(entries[0], workspace=str(repo)),
        replace(entries[1], workspace=str(gone / 'src'), checkout=checkout),
    ]
    widget.reload()
    assert widget.projects == [widget.project]
    assert widget.project_label(widget.project) == 'repo'
    frame = '\n'.join(widget.frame(width=120, height=24))
    # Gathered with the other removed and idle worktrees, its sessions still saying where they ran.
    assert '├─ main (1)' in frame and '└─ 1 older worktree (1)' in frame and '[fix (removed)]' in frame


def test_placements_of_removed_folders() -> None:
    """A removed folder takes its saved checkout, else a known checkout holding it, else a worktree layout's."""
    key = '/code/repo/.git'
    main = ProjectIdentity(key=key, name='repo', root='/code/repo', checkout='main')
    linked = ProjectIdentity(key=key, name='repo', root='/wt/repo/feature', checkout='feature')
    other = ProjectIdentity(key='/code/other/.git', name='other', root='/code/other', checkout='main')
    plain = ProjectIdentity(key='/plain', name='plain', root='/plain')
    fix = GitCheckout(repository=key, name='repo', root='/wt/repo/fix', branch='fix')
    entries = [
        ConversationSummary(workspace='/wt/repo/fix/src', checkout=fix),
        ConversationSummary(workspace='/wt/repo/fix/src', checkout=replace(fix, branch='older')),
        ConversationSummary(workspace='/code/repo/src', checkout=replace(fix, root='/code/repo')),
        ConversationSummary(workspace='/plain'),
    ]
    removed = [
        '/wt/repo/fix/src',
        '/wt/repo/fix',
        '/wt/repo/fix/docs',
        '/code/repo/.worktrees/task/src',
        '/nowhere/.worktrees/task',
        '/code/repo/src',
        '/wt/repo/old/src',
        '/home/worktrees/other/wip',
        '/home/worktrees/unknown/wip',
        '/tmp/repo/x',
        '/',
    ]
    identities = {i.root: i for i in (main, linked, other, plain)} | {
        workspace: ProjectIdentity(key=workspace, name='', root=workspace, missing=True) for workspace in removed
    }
    placed = placements(entries, identities=identities)
    recorded = ProjectIdentity(key=key, name='repo', root='/wt/repo/fix', checkout='fix', missing=True)

    def guessed(identity: ProjectIdentity, root: str) -> ProjectIdentity:
        return replace(identity, root=str(PurePath(root)), checkout=PurePath(root).name, missing=True)

    def unclaimed(workspace: str, label: str) -> ProjectIdentity:
        return ProjectIdentity(key=MISSING, name='Removed folders', root=workspace, checkout=label, missing=True)

    assert placed == {
        '/code/repo': main,
        '/wt/repo/feature': linked,
        '/code/other': other,
        '/plain': plain,
        '/wt/repo/fix/src': recorded,
        '/wt/repo/fix': recorded,
        '/wt/repo/fix/docs': recorded,
        '/code/repo/.worktrees/task/src': guessed(main, '/code/repo/.worktrees/task'),
        '/nowhere/.worktrees/task': unclaimed('/nowhere/.worktrees/task', 'task'),
        '/code/repo/src': main,
        '/wt/repo/old/src': guessed(linked, '/wt/repo/old'),
        '/home/worktrees/other/wip': guessed(other, '/home/worktrees/other/wip'),
        '/home/worktrees/unknown/wip': unclaimed('/home/worktrees/unknown/wip', 'wip'),
        '/tmp/repo/x': unclaimed('/tmp/repo/x', 'x'),
        '/': unclaimed('/', '/'),
    }
    assert [git_checkout(identity) for identity in (plain, recorded)] == [None, None]
    assert git_checkout(main) == GitCheckout(repository=key, name='repo', root='/code/repo', branch='main')


def test_independent_same_named_repositories(tmp_path: Path) -> None:
    widget, entries = browser()
    for index, parent in enumerate(('a', 'b')):
        repo = tmp_path / parent / 'repo'
        repo.mkdir(parents=True)
        subprocess.run(['git', 'init', '-b', 'main', str(repo)], check=True, capture_output=True)
        entries[index] = replace(entries[index], workspace=str(repo))
    entries.pop()
    widget.reload()
    assert len(widget.projects) == 2
    assert widget.project_label(widget.projects[0]) != widget.project_label(widget.projects[1])
    assert len(widget.sessions) == 1


@pytest.mark.parametrize('error', [FileNotFoundError('git'), subprocess.TimeoutExpired('git', 2)])
def test_git_unavailable_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, 'run', fail)
    assert project_identity(str(tmp_path)) == ProjectIdentity(key=str(tmp_path), name=tmp_path.name, root=str(tmp_path))


def test_archiving_hides_a_session_until_archived_ones_are_listed() -> None:
    widget, entries = browser()
    assert 'h archived' in widget.footer()
    widget.handle_key(Key.ENTER)
    assert 'a archive - h archived' in widget.footer()
    widget.handle_key('a')
    assert widget.notice == 'Cannot archive the active session. Use /new first.'
    widget.handle_key(Key.DOWN)
    widget.handle_key('a')
    assert widget.notice == 'Archived Other. h lists archived sessions.'
    assert [entry.id for entry in widget.entries] == ['one', 'two']
    widget.handle_key('h')
    assert widget.notice == 'Archived sessions are listed.'
    frame = widget.frame(width=110, height=24)
    assert 'archived included' in frame[0]
    assert any('Other ! (archived)' in line for line in frame)
    widget.selected_id = 'three'
    widget.handle_key('a')
    assert widget.notice == 'Unarchived Other.' and entries[2].archived_at is None
    widget.handle_key('h')
    assert widget.notice == 'Archived sessions are hidden.'
    assert [entry.id for entry in widget.entries] == ['one', 'two', 'three']


def test_sources_filter_and_idle_worktrees_fold(tmp_path: Path) -> None:
    """`f` lists one source's sessions; removed and idle worktrees gather in one row under active ones."""
    now = datetime(2026, 9, 18, 12, tzinfo=UTC)
    repo, feature, stale = tmp_path / 'repo', tmp_path / 'feature', tmp_path / 'stale'
    repository(repo, linked=feature, branch='feature')
    git('-C', str(repo), 'worktree', 'add', '-b', 'stale', str(stale))
    widget, entries = browser(now=now)
    entries[:] = [
        ConversationSummary(id=str(uuid4()), workspace=str(repo), title='Native', updated_at=now),
        ConversationSummary(
            id=str(uuid5(UUID(int=0), 'claude')),
            workspace=str(feature),
            title='From Claude',
            tags=('claude',),
            updated_at=now - timedelta(days=1),
        ),
        ConversationSummary(
            id=str(uuid5(UUID(int=0), 'codex')),
            workspace=str(stale),
            title='From Codex',
            tags=('codex',),
            updated_at=now - timedelta(days=35),
        ),
    ]
    widget.reload()
    key = widget.project
    assert widget.rows == [
        (key, ''),
        (key, widget.identity(str(repo)).root),
        (key, widget.identity(str(feature)).root),
        (key, module.OLDER),
    ]
    frame = '\n'.join(widget.frame(width=120, height=30))
    assert '└─ 1 older worktree (1)' in frame and 'stale (1)' not in frame
    # Headings widen with age; an older card shows its date.
    older = f'{(now - timedelta(days=35)).astimezone():%b %d} From Codex'
    assert all(heading in frame for heading in ('TODAY', 'YESTERDAY', 'AUGUST 2026', older))
    for _ in range(3):
        widget.handle_key(Key.DOWN)
    assert [e.title for e in widget.sessions] == ['From Codex']
    assert 'SESSIONS: repo / 1 older worktree' in '\n'.join(widget.frame(width=120, height=30))
    titles: list[list[str]] = []
    for _ in range(4):
        widget.handle_key('f')
        titles.append([e.title for e in widget.sessions])
    assert titles == [['Native'], ['From Claude'], ['From Codex'], ['Native', 'From Claude', 'From Codex']]
    widget.handle_key('f')
    assert widget.notice == 'Listing CLAI sessions.' and 'Source: CLAI' in widget.frame(width=120, height=30)[0]
    # The older row is gone with the worktrees it gathered, so the project is selected instead.
    assert widget.rows == [(key, '')] and widget.checkout == ''
    for _ in range(3):
        widget.handle_key('f')
    assert widget.notice == 'Listing all sessions.' and 'f source' in widget.footer()
    # The repository's own checkout keeps its row however idle; only worktrees are gathered.
    entries[0] = replace(entries[0], updated_at=now - timedelta(days=60))
    widget.reload()
    assert widget.rows == [
        (key, ''),
        (key, widget.identity(str(repo)).root),
        (key, widget.identity(str(feature)).root),
        (key, module.OLDER),
    ]
