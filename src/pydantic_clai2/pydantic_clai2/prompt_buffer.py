"""Pure draft editing and history navigation for the pinned prompt."""

from dataclasses import dataclass, field

from termflow.ansi.utils import visible_length  # pyright: ignore[reportMissingTypeStubs]


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

    def replace(self, text: str) -> None:
        """Set a draft and put its cursor at the end."""
        self.text, self.cursor = text, len(text)

    def insert(self, text: str) -> None:
        """Insert literal text; terminal control bytes do not become escape output."""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = ''.join(char for char in text if char.isprintable() or char in ('\n', '\t'))
        self.text = self.text[: self.cursor] + text + self.text[self.cursor :]
        self.cursor += len(text)
        self.history_index = None

    def recall(self, *, backwards: bool) -> None:
        """Walk chronological history, preserving the draft beyond its newest entry."""
        if self.history_index is None:
            self.saved_draft = self.text
            self.history_index = len(self.history)
        self.history_index = min(len(self.history), max(0, self.history_index + (-1 if backwards else 1)))
        self.replace(self.saved_draft if self.history_index == len(self.history) else self.history[self.history_index])

    def vertical(self, *, backwards: bool) -> None:
        """Move within multiline text before falling back to history recall."""
        lines = self.text.split('\n')
        before = self.text[: self.cursor]
        row, column = before.count('\n'), len(before.rsplit('\n', 1)[-1])
        target = row + (-1 if backwards else 1)
        if len(lines) > 1 and 0 <= target < len(lines):
            self.cursor = sum(len(line) + 1 for line in lines[:target]) + min(column, len(lines[target]))
        else:
            self.recall(backwards=backwards)

    def search_key(self, key: str) -> None:
        """Search backwards without submitting the selected history entry."""
        assert self.search is not None
        if key in ('enter', 'escape'):
            self.search = None
        elif key == 'ctrl-g':
            self.replace(self.search_original)
            self.search = None
        else:
            if key == 'backspace':
                self.search = self.search[:-1]
            elif len(key) == 1 and key.isprintable():
                self.search += key
            matches = [entry for entry in reversed(self.history) if self.search in entry]
            if matches:
                index = (matches.index(self.text) + 1) % len(matches) if key == 'ctrl-r' and self.text in matches else 0
                self.replace(matches[index])

    def edit(self, key: str) -> bool:
        """Apply an editing key; return false when the owner should handle it."""
        if self.search is not None:
            self.search_key(key)
            return True
        before, after = self.text[: self.cursor], self.text[self.cursor :]
        if key == 'ctrl-r':
            self.search_original, self.search = self.text, ''
        elif key in ('left', 'right'):
            self.cursor = max(0, min(len(self.text), self.cursor + (-1 if key == 'left' else 1)))
        elif key in ('home', 'ctrl-a'):
            self.cursor = before.rfind('\n') + 1
        elif key in ('end', 'ctrl-e'):
            self.cursor += after.find('\n') if '\n' in after else len(after)
        elif key == 'backspace':
            self.text = before[:-1] + after
            self.cursor = len(before[:-1])
        elif key == 'delete':
            self.text = before + after[1:]
        elif key in ('ctrl-u', 'ctrl-k', 'ctrl-w', 'alt-backspace'):
            start = 0
            if key in ('ctrl-w', 'alt-backspace'):
                stripped = before.rstrip()
                words = stripped.rsplit(maxsplit=1)
                start = len(stripped) - len(words[-1]) if words else 0
            self.text = before if key == 'ctrl-k' else before[:start] + after
            self.cursor = len(before) if key == 'ctrl-k' else min(start, len(self.text))
        elif key in ('alt-b', 'ctrl-left', 'alt-f', 'ctrl-right'):
            if key in ('alt-b', 'ctrl-left'):
                self.cursor = len(before.rstrip().rsplit(' ', 1)[0]) + 1 if ' ' in before.rstrip() else 0
            else:
                self.cursor += len(after) - len(after.lstrip())
                tail = self.text[self.cursor :]
                self.cursor += tail.find(' ') if ' ' in tail else len(tail)
        elif key in ('up', 'down'):
            self.vertical(backwards=key == 'up')
        elif len(key) == 1 and key.isprintable():
            self.insert(key)
        else:
            return False
        return True

    def rows(self, *, width: int, limit: int) -> list[str]:
        """Wrap into terminal cells and keep the nonblinking cursor in view."""
        width = max(1, width)
        rows = ['']
        cells = 0
        cursor_row = 0
        for index, char in enumerate(self.text + ' '):
            is_cursor = index == self.cursor
            if index == len(self.text) and not is_cursor:
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
