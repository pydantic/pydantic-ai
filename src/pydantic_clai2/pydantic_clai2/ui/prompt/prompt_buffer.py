"""Pure draft editing, undo, and history navigation for the pinned prompt."""

from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field

from termflow.ansi.utils import visible_length

UNDO_KEYS = ('ctrl-z', 'super-z')
"""Ctrl+Z on every platform, since the editor's raw mode receives it instead of suspending; Cmd+Z when reported."""
REDO_KEYS = ('ctrl-y', 'ctrl-shift-z', 'super-shift-z')
UNDO_LIMIT = 100
"""Undo steps kept per draft; the oldest is forgotten first."""
_MOVES = ('left', 'right', 'home', 'ctrl-a', 'end', 'ctrl-e', 'alt-b', 'ctrl-left', 'alt-f', 'ctrl-right')


@dataclass(frozen=True, kw_only=True)
class _Snapshot:
    """A draft as one undo or redo step restores it."""

    text: str
    cursor: int
    pastes: tuple[tuple[int, int], ...]


def _steps() -> deque[_Snapshot]:
    return deque(maxlen=UNDO_LIMIT)


@dataclass(kw_only=True)
class PromptBuffer:
    """Text state independent of terminal input and painting."""

    text: str = ''
    cursor: int = 0
    history: list[str] = field(default_factory=list[str])
    history_index: int | None = None
    saved_draft: str = ''
    search: str | None = None
    search_original: str = ''
    _pastes: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]], init=False, repr=False)
    _entries: list[str] = field(default_factory=list[str], init=False, repr=False)
    _undo: deque[_Snapshot] = field(default_factory=_steps, init=False, repr=False)
    _redo: deque[_Snapshot] = field(default_factory=_steps, init=False, repr=False)
    _group: tuple[str, int] | None = field(default=None, init=False, repr=False)
    """The kind of the last change and the cursor after it, while further changes extend its undo step."""

    def replace(self, text: str, *, group: str | None = None) -> None:
        """Set a draft and put its cursor at the end.

        A change is one undo step; consecutive changes in the same `group`, such as a history walk, share one.
        """
        if text != self.text:
            self._checkpoint(group)
        self.text, self.cursor = text, len(text)
        self._pastes.clear()
        self._group = None if group is None else (group, self.cursor)

    def reset(self) -> None:
        """Empty the draft after it was submitted; its edits can no longer be undone."""
        self.replace('')
        self._undo.clear()
        self._redo.clear()
        self._group = None

    def replace_range(self, start: int, end: int, text: str, *, group: str | None = None) -> None:
        """Replace an editing range, revealing overlapping pastes.

        An edit that changes text ends a recall walk, so the edited text becomes the draft the
        next walk restores. A no-op deletion at either end keeps the walk going. Consecutive
        edits in the same `group` that continue where the last one left the cursor share an
        undo step; any other change is a step of its own.
        """
        if text or start != end:
            self.history_index = None
            self._checkpoint(group)
        shift = len(text) - (end - start)
        self._pastes = [
            (left, right) if right <= start else (left + shift, right + shift)
            for left, right in self._pastes
            if right <= start or left >= end
        ]
        self.text = self.text[:start] + text + self.text[end:]
        self.cursor = start + len(text)
        if text or start != end:
            self._group = None if group is None else (group, self.cursor)

    def _checkpoint(self, group: str | None) -> None:
        """Save the draft before a change, unless the change extends the last one's undo step."""
        if group is None or self._group != (group, self.cursor):
            self._undo.append(self._snapshot())
        self._redo.clear()

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(text=self.text, cursor=self.cursor, pastes=tuple(self._pastes))

    def undo(self) -> None:
        """Restore the draft before the latest change, keeping the current one for redo."""
        self._step(self._undo, self._redo)

    def redo(self) -> None:
        """Reapply the change the latest undo reverted, until a new change discards it."""
        self._step(self._redo, self._undo)

    def _step(self, source: deque[_Snapshot], target: deque[_Snapshot]) -> None:
        # Even an undo with nothing to restore ends the current step, so the next change saves the draft.
        self._group = None
        # A history walk that came back to the draft leaves a step with nothing to change.
        while source and source[-1].text == self.text:
            source.pop()
        if not source:
            return
        target.append(self._snapshot())
        snapshot = source.pop()
        self.text, self.cursor, self._pastes = snapshot.text, snapshot.cursor, list(snapshot.pastes)
        # Restored text is an edit: it ends a recall walk, and the next change starts its own step.
        self.history_index = None

    def insert(self, text: str, *, paste: bool = False) -> None:
        """Insert literal text; terminal control bytes do not become escape output.

        Typed characters share an undo step per word, so undo removes the last word typed.
        Newlines, pastes, and inserted strings are steps of their own.
        """
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = ''.join(char for char in text if char.isprintable() or char in ('\n', '\t'))
        start = self.cursor
        typed = not paste and len(text) == 1 and text != '\n'
        if typed and not text.isspace() and self.text[:start][-1:].isspace():
            self._group = None
        self.replace_range(start, start, text, group='type' if typed else None)
        if paste and (len(text.splitlines()) >= 5 or len(text) >= 1000):
            self._pastes.append((start, self.cursor))
            self._pastes.sort()

    @property
    def recall_offset(self) -> int | None:
        """Steps back from the draft during a recall walk: 0 is the draft, -1 the newest entry."""
        return None if self.history_index is None else self.history_index - len(self._entries)

    def recall(self, *, backwards: bool, queued: tuple[str, ...] = (), recorded: tuple[str, ...] = ()) -> None:
        """Walk chronological history, preserving the draft beyond its newest entry.

        `queued` prompts are newer than any history, so a walk that starts here visits them
        between the draft and history. `recorded` holds the raw history copies made when they
        were queued; the newest match of each is skipped, so older identical history stays.
        """
        if self.history_index is None:
            self.saved_draft = self.text
            newest_first = self.history[::-1]
            for text in recorded:
                with suppress(ValueError):
                    newest_first.remove(text)
            self._entries = newest_first[::-1] + list(queued)
            self.history_index = len(self._entries)
        self.history_index = min(len(self._entries), max(0, self.history_index + (-1 if backwards else 1)))
        self.replace(
            self.saved_draft if self.history_index == len(self._entries) else self._entries[self.history_index],
            group='recall',
        )

    def vertical(self, *, backwards: bool, queued: tuple[str, ...] = (), recorded: tuple[str, ...] = ()) -> None:
        """Move within multiline text before falling back to history recall."""
        lines = self.text.split('\n')
        before = self.text[: self.cursor]
        row, column = before.count('\n'), len(before.rsplit('\n', 1)[-1])
        target = row + (-1 if backwards else 1)
        if len(lines) > 1 and 0 <= target < len(lines):
            self._group = None
            self.cursor = sum(len(line) + 1 for line in lines[:target]) + min(column, len(lines[target]))
        else:
            self.recall(backwards=backwards, queued=queued, recorded=recorded)

    def search_key(self, key: str) -> None:
        """Search backwards without submitting the selected history entry."""
        assert self.search is not None
        if key in ('enter', 'escape'):
            self.search = None
        elif key == 'ctrl-g':
            self.replace(self.search_original, group='search')
            self.search = None
        else:
            if key == 'backspace':
                self.search = self.search[:-1]
            elif len(key) == 1 and key.isprintable():
                self.search += key
            matches = [entry for entry in reversed(self.history) if self.search in entry]
            if matches:
                index = (matches.index(self.text) + 1) % len(matches) if key == 'ctrl-r' and self.text in matches else 0
                self.replace(matches[index], group='search')
                self.history_index = None

    def edit(self, key: str) -> bool:
        """Apply an editing key; return false when the owner should handle it."""
        if self.search is not None:
            self.search_key(key)
            return True
        before = self.text[: self.cursor]
        if key == 'ctrl-r':
            # Each search is its own undo step, even when it picks the same entry as the last.
            self._group = None
            self.search_original, self.search = self.text, ''
        elif key in _MOVES:
            self._move(key)
        elif key in UNDO_KEYS:
            self.undo()
        elif key in REDO_KEYS:
            self.redo()
        elif key == 'backspace':
            self.replace_range(len(before[:-1]), self.cursor, '', group='backspace')
        elif key == 'delete':
            self.replace_range(self.cursor, min(len(self.text), self.cursor + 1), '', group='delete')
        elif key in ('ctrl-u', 'ctrl-k', 'ctrl-w', 'alt-backspace'):
            self._kill(key)
        elif key in ('up', 'down'):
            self.vertical(backwards=key == 'up')
        elif len(key) == 1 and key.isprintable():
            self.insert(key)
        else:
            return False
        return True

    def _move(self, key: str) -> None:
        """Move the cursor; the next change starts a new undo step, even back where the last one ended."""
        self._group = None
        before, after = self.text[: self.cursor], self.text[self.cursor :]
        if key in ('left', 'right'):
            self.cursor = max(0, min(len(self.text), self.cursor + (-1 if key == 'left' else 1)))
        elif key in ('home', 'ctrl-a'):
            self.cursor = before.rfind('\n') + 1
        elif key in ('end', 'ctrl-e'):
            self.cursor += after.find('\n') if '\n' in after else len(after)
        elif key in ('alt-b', 'ctrl-left'):
            self.cursor = len(before.rstrip().rsplit(' ', 1)[0]) + 1 if ' ' in before.rstrip() else 0
        else:
            self.cursor += len(after) - len(after.lstrip())
            tail = self.text[self.cursor :]
            self.cursor += tail.find(' ') if ' ' in tail else len(tail)

    def _kill(self, key: str) -> None:
        """Delete everything before the cursor (`ctrl-u`), after it (`ctrl-k`), or the word before it."""
        if key == 'ctrl-k':
            self.replace_range(self.cursor, len(self.text), '')
            return
        start = 0
        if key in ('ctrl-w', 'alt-backspace'):
            stripped = self.text[: self.cursor].rstrip()
            words = stripped.rsplit(maxsplit=1)
            start = len(stripped) - len(words[-1]) if words else 0
        self.replace_range(start, self.cursor, '')

    def display(self) -> tuple[str, int]:
        """Fold pasted ranges unless the cursor has entered them to edit."""
        self._pastes = [(start, end) for start, end in self._pastes if not start < self.cursor < end]
        parts: list[str] = []
        previous, cursor = 0, self.cursor
        for start, end in self._pastes:
            lines = max(1, len(self.text[start:end].splitlines()))
            label = f'[paste {lines} lines]'
            parts.extend((self.text[previous:start], label))
            previous = end
            if self.cursor >= end:
                cursor += len(label) - (end - start)
        parts.append(self.text[previous:])
        return ''.join(parts), cursor

    def rows(self, *, width: int, limit: int) -> list[str]:
        """Wrap into terminal cells and keep the nonblinking cursor in view."""
        width = max(1, width)
        rows = ['']
        cells = 0
        cursor_row = 0
        text, cursor = self.display()
        for index, char in enumerate(text + ' '):
            is_cursor = index == cursor
            if index == len(text) and not is_cursor:
                break
            if char == '\n' and not is_cursor:
                rows.append('')
                cells = 0
                continue
            display = ' ' if char == '\n' else '    ' if char == '\t' else char if char.isprintable() else '?'
            size = visible_length(display)
            if cells + size > width:
                rows.append('')
                cells = 0
            if is_cursor:
                cursor_row = len(rows) - 1
            rows[-1] += f'\x1b[7m{display}\x1b[27m' if is_cursor else display
            cells += size
            if char == '\n':
                rows.append('')
                cells = 0
        start = max(0, cursor_row - limit + 1)
        return rows[start : start + limit]
