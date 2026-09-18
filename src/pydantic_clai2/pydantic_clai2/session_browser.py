"""Project/session browser inspired by Code Puppy, using Termflow's terminal and layout primitives.

MenuBuilder's single selectable pane cannot express independently navigable projects
and two-line session cards. This widget keeps the same pure frame/scripted-key seam.
"""

from __future__ import annotations

import sys
import textwrap
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.keys import Key  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.layout import collapsed, split_frame, truncate  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.terminal import terminal_session, terminal_size  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from .menu_worker import menu_key


def plain(text: str, *, multiline: bool = False) -> str:
    """Saved and model-generated text is untrusted terminal content."""
    if not multiline:
        text = ' '.join(text.splitlines())
    return ''.join(c for c in text if c.isprintable() or c == '\n')


def date_label(moment: datetime, *, now: datetime | None = None) -> str:
    """Local calendar buckets, with a year on older sessions."""
    day = moment.astimezone().date()
    today = (now or datetime.now(UTC)).astimezone().date()
    if day == today:
        return 'TODAY'
    if day == today - timedelta(days=1):
        return 'YESTERDAY'
    return day.isoformat()


def colored(text: str, *, role: str = theme.INFO) -> str:
    """Use the shared brand palette, including the 16-color fallback."""
    return f'{theme.sgr(role)}{plain(text)}\x1b[0m'


class SessionBrowser:
    """Pure browsing state plus a thin, injectable terminal loop.

    Callbacks run on the menu thread; the async runner bridges storage operations
    to the application loop. Background updates retain selection by stable ID.
    """

    def __init__(
        self,
        *,
        entries: Sequence[ConversationSummary],
        workspace: str,
        active_id: str,
        refresh: Callable[[str, int], list[ConversationSummary]],
        preview: Callable[[str], str],
        delete: Callable[[ConversationSummary], None],
        rename: Callable[[ConversationSummary, str], None],
        output: TextIO | None = None,
        key_source: Callable[[], str] = menu_key,
        size: Callable[[], tuple[int, int]] = terminal_size,
    ) -> None:
        """Bind an initial catalog and explicit IO operations."""
        self.entries = list(entries)
        self.workspace = workspace
        self.active_id = active_id
        self.refresh = refresh
        self.preview = preview
        self.delete = delete
        self.rename = rename
        self.output = output or sys.stdout
        self.key_source = key_source
        self.size = size
        self.project = workspace if workspace in self.projects else next(iter(self.projects), '')
        self.mode = 'projects'
        self.query = ''
        self.buffer = ''
        self.sort = 0
        self.limit = 200
        self.selected_id: str | None = None
        self.notice = ''
        self.preview_text = ''
        self.preview_offset = 0
        self.confirm: ConversationSummary | None = None
        self.confirm_action = ''

    @property
    def projects(self) -> list[str]:
        """Projects in most-recent activity order, without basename identity collisions."""
        return list(dict.fromkeys(entry.workspace for entry in self.entries))

    @property
    def sessions(self) -> list[ConversationSummary]:
        """Search results are global; otherwise show the selected project."""
        entries = [e for e in self.entries if self.query or e.workspace == self.project]
        if self.sort == 1:
            entries.sort(key=lambda e: e.message_count, reverse=True)
        elif self.sort == 2:
            entries.sort(key=lambda e: e.total_tokens, reverse=True)
        return entries

    @property
    def selected(self) -> ConversationSummary | None:
        """Stable selection survives naming edits and resorting."""
        entries = self.sessions
        return next((e for e in entries if e.id == self.selected_id), entries[0] if entries else None)

    def reload(self) -> None:
        """Refresh cached summaries, including names generated while the browser is idle."""
        selected = self.selected
        self.entries = self.refresh(self.query, self.limit)
        if selected is not None:
            self.selected_id = selected.id
        if self.project not in self.projects:
            self.project = next(iter(self.projects), '')

    def frame(self, *, width: int, height: int) -> list[str]:
        """Produce a bounded responsive frame without reading storage or a terminal."""
        width, height = max(1, width - 1), max(3, height)
        budget = max(1, height - 5)
        header = colored(f'CLAI > Resume session    {len(self.entries)} sessions / {len(self.projects)} projects')
        if self.mode == 'preview':
            preview = self.preview_text[:24_000]
            if len(self.preview_text) > len(preview):
                preview += '\n[Preview truncated to 24,000 characters.]'
            body = [
                line
                for paragraph in plain(preview, multiline=True).splitlines()
                for line in (textwrap.wrap(paragraph, width=width) or [''])
            ] or ['No messages.']
            self.preview_offset = min(self.preview_offset, max(0, len(body) - budget))
            body = body[self.preview_offset : self.preview_offset + budget]
        else:
            left = self._projects_frame(budget=budget)
            pane_width = width if collapsed(width) else width - min(30, max(12, width // 3)) - 3
            right = self._sessions_frame(budget=budget, width=pane_width)
            body = split_frame(
                left,
                right,
                width=width,
                list_width=min(30, max(12, width // 3)),
                focus='left' if self.mode == 'projects' else 'right',
            )[:budget]
        return [truncate(line, width) for line in [header, '', *body, self.notice, self.footer()]][:height]

    def _projects_frame(self, *, budget: int) -> list[str]:
        projects = self.projects
        cursor = projects.index(self.project) if self.project in projects else 0
        start = max(0, cursor - budget + 2)
        lines = [colored('PROJECTS')]
        labels = [Path(p).name for p in projects]
        for project in projects[start : start + budget - 1]:
            label = Path(project).name or project
            if labels.count(label) > 1:
                label = project
            count = sum(e.workspace == project for e in self.entries)
            marker = '> ' if project == self.project else '  '
            lines.append(plain(f'{marker}{label} ({count})'))
        return lines

    def _sessions_frame(self, *, budget: int, width: int) -> list[str]:
        selected = self.selected
        entries = self.sessions
        cursor = entries.index(selected) if selected in entries else 0
        # Three lines per card leaves room for a date heading without splitting cards.
        capacity = max(1, (budget - 1) // 3)
        start = max(0, cursor - capacity + 1)
        lines = [
            colored(
                f'{"Search all projects: " + self.query if self.query else self.project} | '
                f'Sort: {("recent", "messages", "tokens")[self.sort]}'
            )
        ]
        last_day = ''
        for entry in entries[start : start + capacity]:
            day = date_label(entry.updated_at)
            if self.sort == 0 and day != last_day:
                lines.append(colored(day, role=theme.MUTED))
                last_day = day
            marker = '> ' if entry == selected else '  '
            risk = ' !' if entry.outcome in ('running', 'failed', 'cancelled') else ''
            title = f'{marker}{entry.updated_at.astimezone():%H:%M} {entry.title}{risk}'
            counts = f'{entry.message_count} msgs / {entry.total_tokens:,} tok'
            if len(counts) > width // 2:
                counts = f'{entry.message_count} msgs'
            title = truncate(plain(title), max(1, width - len(counts) - 1))
            lines.append(title + ' ' * max(1, width - visible_length(title) - len(counts)) + counts)
            tags = list(entry.tags)
            while tags and len(' '.join(f'#{t}' for t in tags)) > width // 2:
                tags.pop()
            chips = ' '.join(f'#{t}' for t in tags)
            detail = (
                truncate(plain(f'  {entry.subtitle}'), max(1, width - len(chips) - 1)).replace('\x1b[0m', '')
                + ' '
                + chips
            )
            if self.query:
                detail += f'  [{entry.workspace}]'
            lines.append(colored(detail, role=theme.MUTED))
        if not entries:
            lines.append('No saved sessions match. Esc goes back.')
        return lines

    def footer(self) -> str:
        """Mode-specific hints, including explicit confirmation of destructive actions."""
        if self.confirm is not None:
            return plain(f'{self.confirm_action}: {self.confirm.title}? y confirm / any other key cancel')
        if self.mode in ('search', 'rename'):
            return plain(f'{self.mode}: {self.buffer} | Enter apply / Esc cancel')
        if self.mode == 'preview':
            return 'Up/Down scroll - Esc back - Ctrl-C close'
        return 'Esc back / Ctrl-C close - Up/Down move - Enter resume - Right preview - / search - s sort - r rename - d delete - m more'

    def handle_key(self, key: str) -> str | None:
        """Return a session ID on selection, an empty string on close, otherwise continue."""
        if key == 'ctrl-c':
            return ''
        if self.confirm is not None:
            return self._confirm_key(key)
        if self.mode in ('search', 'rename'):
            self._text_key(key)
            return None
        if key in (Key.ESCAPE, 'q'):
            if self.mode == 'projects':
                return ''
            if self.query:
                self.query = ''
                self.reload()
            self.mode = 'sessions' if self.mode == 'preview' else 'projects'
        elif key in (Key.UP, Key.DOWN, 'ctrl-p', 'ctrl-n'):
            self._move(-1 if key in (Key.UP, 'ctrl-p') else 1)
        elif self.mode == 'preview':
            return None
        elif key == '/':
            self.mode, self.buffer = 'search', self.query
        elif key == 'm':
            self.limit += 200
            self.reload()
        elif self.mode == 'projects':
            if key in (Key.ENTER, Key.RIGHT, Key.TAB):
                self.mode = 'sessions'
        else:
            return self._session_key(key)
        return None

    def _move(self, delta: int) -> None:
        if self.mode == 'preview':
            self.preview_offset = max(0, self.preview_offset + delta)
        elif self.mode == 'projects' and self.projects:
            index = self.projects.index(self.project)
            self.project = self.projects[max(0, min(len(self.projects) - 1, index + delta))]
            self.selected_id = None
        elif self.selected is not None:
            entries = self.sessions
            index = entries.index(self.selected)
            self.selected_id = entries[max(0, min(len(entries) - 1, index + delta))].id

    def _session_key(self, key: str) -> str | None:
        entry = self.selected
        if key in (Key.LEFT, Key.TAB):
            self.mode = 'projects'
        elif key == 's':
            self.sort = (self.sort + 1) % 3
        elif entry is not None:
            if key == Key.ENTER:
                if entry.workspace != self.workspace:
                    self.confirm, self.confirm_action = (
                        entry,
                        f'Resume in current directory (saved in {entry.workspace})',
                    )
                else:
                    return entry.id
            elif key in (Key.RIGHT, 'e'):
                self.preview_text = self.preview(entry.id)
                self.preview_offset, self.mode = 0, 'preview'
            elif key == 'd':
                if entry.id == self.active_id:
                    self.notice = 'Cannot delete the active session. Use /new first.'
                else:
                    self.confirm, self.confirm_action = entry, 'Delete saved session'
            elif key == 'r':
                self.mode, self.buffer = 'rename', entry.title
        return None

    def _confirm_key(self, key: str) -> str | None:
        entry, self.confirm = self.confirm, None
        if key.lower() != 'y' or entry is None:
            return None
        if self.confirm_action.startswith('Resume'):
            return entry.id
        self.delete(entry)
        self.reload()
        return None

    def _text_key(self, key: str) -> None:
        if key == Key.ESCAPE:
            self.mode = 'sessions'
        elif key == Key.ENTER:
            if self.mode == 'search':
                self.query = self.buffer.strip()
            elif self.selected is not None and self.buffer.strip():
                self.rename(self.selected, self.buffer.strip()[:100])
            self.mode = 'sessions'
            self.reload()
        elif key == Key.BACKSPACE:
            self.buffer = self.buffer[:-1]
        elif len(key) == 1 and key.isprintable() and len(self.buffer) < 200:
            self.buffer += key

    def run(self) -> str:
        """Own the alternate screen until exit, including when the async runner cancels."""
        with terminal_session(self.output):
            return self.loop()

    def loop(self) -> str:
        """Scriptable loop; poll metadata changes even when no keys arrive."""
        refreshed = time.monotonic()
        dirty = True
        previous_size: tuple[int, int] | None = None
        while True:
            size = self.size()
            if dirty or size != previous_size:
                frame = self.frame(width=size[0], height=size[1])
                self.output.write('\x1b[H' + '\r\n'.join(f'{line}\x1b[K' for line in frame) + '\x1b[J')
                self.output.flush()
                previous_size = size
                dirty = False
            try:
                key = self.key_source()
                dirty = bool(key)
                result = self.handle_key(key)
                if result is not None:
                    return result
                if time.monotonic() - refreshed >= 0.5:
                    try:
                        self.reload()
                    finally:
                        refreshed = time.monotonic()
                    dirty = True
            except Exception as exc:  # noqa: BLE001 -- storage errors stay inside the alternate screen.
                self.notice = plain(str(exc))
                dirty = True
