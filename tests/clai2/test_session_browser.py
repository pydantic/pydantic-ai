"""Headless project/session navigation and safe, responsive frame rendering."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO

import pytest
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.keys import Key  # pyright: ignore[reportMissingTypeStubs]

import pydantic_clai2.session_browser as module
from pydantic_clai2.session_browser import SessionBrowser, date_label, plain


def browser(*, keys: list[str] | None = None) -> tuple[SessionBrowser, list[ConversationSummary]]:
    entries = [
        ConversationSummary(id='one', workspace='/a/project', title='Fix renderer', subtitle='Race', tags=('tests',)),
        ConversationSummary(id='two', workspace='/b/project', title='Add resume', message_count=24, total_tokens=1200),
        ConversationSummary(id='three', workspace='/a/project', title='Other', outcome='cancelled'),
    ]
    source = iter(keys or [])

    def refresh(query: str, limit: int) -> list[ConversationSummary]:
        return [e for e in entries if query.lower() in e.title.lower()][:limit]

    def rename(entry: ConversationSummary, title: str) -> None:
        entries[entries.index(entry)] = replace(entry, title=title)

    widget = SessionBrowser(
        entries=entries,
        workspace='/a/project',
        active_id='one',
        refresh=refresh,
        preview=lambda session_id: f'Transcript {session_id}\nuser: hello',
        delete=entries.remove,
        rename=rename,
        output=StringIO(),
        key_source=lambda: next(source),
        size=lambda: (110, 24),
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
    assert widget.handle_key(Key.ENTER) is None
    assert 'saved in /b/project' in widget.footer()
    assert widget.handle_key('y') == 'two'
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


def test_date_buckets() -> None:
    now = datetime(2026, 9, 18, 12, tzinfo=UTC)
    assert date_label(now, now=now) == 'TODAY'
    assert date_label(now - timedelta(days=1), now=now) == 'YESTERDAY'
    assert date_label(now - timedelta(days=365), now=now).startswith(str((now - timedelta(days=365)).year))


def test_idle_refresh_and_storage_errors_stay_in_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, entries = browser(keys=['', 'ctrl-c'])
    ticks = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    entries[0] = replace(entries[0], title='Named in background')
    assert widget.loop() == ''
    assert isinstance(widget.output, StringIO)
    assert 'Named in background' in widget.output.getvalue()

    def fail_preview(session_id: str) -> str:
        raise ValueError('Cannot read this session')

    widget, _ = browser(keys=[Key.ENTER, Key.RIGHT, 'ctrl-c'])
    widget.preview = fail_preview
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.0)
    assert widget.loop() == ''
    assert 'Cannot read this session' in widget.notice


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
    widget.confirm_action = f'Resume in {entries[0].workspace}'
    assert '\n' not in widget.footer()
    assert plain('first\nsecond', multiline=True) == 'first\nsecond'
    assert plain('first\nsecond') == 'first second'


def test_idle_loop_redraws_only_for_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, _ = browser(keys=['', '', '', 'ctrl-c'])
    sizes = iter([(110, 24), (110, 24), (100, 24), (100, 24)])
    widget.size = lambda: next(sizes)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.0)
    assert widget.loop() == ''
    assert isinstance(widget.output, StringIO)
    assert widget.output.getvalue().count('CLAI > Resume session') == 2


def test_failed_idle_refresh_waits_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    widget, _ = browser(keys=['', '', '', 'ctrl-c'])
    ticks = iter([0.0, 0.6, 0.6, 0.7, 0.8])
    monkeypatch.setattr(module.time, 'monotonic', lambda: next(ticks))
    attempts: list[str] = []

    def fail(query: str, limit: int) -> list[ConversationSummary]:
        attempts.append(query)
        raise OSError('database unavailable')

    widget.refresh = fail
    assert widget.loop() == ''
    assert attempts == ['']
    assert 'database unavailable' in widget.notice
