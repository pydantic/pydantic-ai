"""A proposed change to the workspace: the diff a listener sees, and the request that announces it."""

from __future__ import annotations

import difflib
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.filesystem._events import (
    MAX_DIFF_SOURCE_CHARS,
    MAX_EVENT_DIFF_CHARS,
    FileChangeRequestEvent,
    FileEditedEvent,
    FileOperation,
)

_REFUSALS: dict[FileOperation, str] = {
    'write': 'was not written',
    'edit': 'was not edited',
    'create_directory': 'was not created',
}

_NO_NEWLINE = '\\ No newline at end of file'
"""The marker `git diff` prints after a final line that lacks a newline."""

_MAX_DIFF_LINE_PAIRS = 65536
"""Bound the quadratic line-matching work before entering `difflib`."""


def _lines(text: str) -> list[str]:
    """Split on newlines only, keeping each one, so a final line without one stays distinguishable."""
    lines = text.split('\n')
    last = lines.pop()
    return [f'{line}\n' for line in lines] + ([last] if last else [])


def _quoted(name: str) -> str:
    """`name` as a diff header label, quoted as `git diff` quotes a name with a control character in it.

    A model-supplied name with a newline in it would otherwise forge a
    header or a hunk in the diff a listener is shown.
    """
    if name.isprintable():
        return name
    return '"' + name.encode('unicode_escape').decode('ascii').replace('"', '\\"') + '"'


def _diff_lines(old: str, new: str, *, path: str) -> Iterator[str]:
    """The diff lines without terminators, marking a final line that has none the way `git diff` does."""
    fromfile, tofile = _quoted(f'a/{path}'), _quoted(f'b/{path}')
    lines = difflib.unified_diff(_lines(old), _lines(new), fromfile=fromfile, tofile=tofile, lineterm='')
    for index, line in enumerate(lines):
        if line.endswith('\n'):
            yield line[:-1]
            continue
        yield line
        # The first two lines name the files and a hunk header starts with
        # `@@`; only a content line can be missing its newline.
        if index >= 2 and not line.startswith('@@'):
            yield _NO_NEWLINE


def unified_diff(old: str | None, new: str, *, path: str) -> tuple[str, bool]:
    """Unified diff from `old` to `new`, cut at `MAX_EVENT_DIFF_CHARS`.

    Returns the diff and whether it was cut. Two equal texts diff to an
    empty string, so a `create_directory` proposes no diff at all. A text
    longer than `MAX_DIFF_SOURCE_CHARS` on either side is not diffed: the
    result is the two file headers, marked as cut, so a large write does
    not pay for a diff that would be cut anyway. `old` is `None` when the
    caller found the current content that large and did not decode it. The
    headers-only fallback also applies when the product of conservative
    line counts exceeds `_MAX_DIFF_LINE_PAIRS`, before matching starts.
    """
    if (
        old is None
        or len(old) > MAX_DIFF_SOURCE_CHARS
        or len(new) > MAX_DIFF_SOURCE_CHARS
        or (old.count('\n') + 1) * (new.count('\n') + 1) > _MAX_DIFF_LINE_PAIRS
    ):
        # The quoted name of a control-character-heavy path can pass the cap
        # on its own, so the headers are cut like every other return.
        header = f'--- {_quoted(f"a/{path}")}\n+++ {_quoted(f"b/{path}")}'
        return header[:MAX_EVENT_DIFF_CHARS], True
    diff = '\n'.join(_diff_lines(old, new, path=path))
    if len(diff) <= MAX_EVENT_DIFF_CHARS:
        return diff, False
    # Cut on a line boundary so the kept part is still whole diff lines. A
    # path so long that the first header alone passes the cap (possible only
    # where the OS allows such paths) leaves no newline to cut at; the bound
    # holds regardless.
    cut = diff.rfind('\n', 0, MAX_EVENT_DIFF_CHARS + 1)
    return diff[: cut if cut >= 0 else MAX_EVENT_DIFF_CHARS], True


@dataclass(kw_only=True)
class Change:
    """A change to announce before it is applied and to report after."""

    path: str
    root_dir: str
    operation: FileOperation
    diff: str
    truncated: bool

    @classmethod
    def propose(
        cls, *, path: str, root_dir: str, operation: FileOperation, old: str | None = '', new: str = ''
    ) -> Change:
        diff, truncated = unified_diff(old, new, path=path)
        return cls(path=path, root_dir=root_dir, operation=operation, diff=diff, truncated=truncated)

    async def request(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Announce the change to the run's listeners.

        Returns the tool result for the model when a listener cancelled the
        change, or `None` when it may proceed.
        """
        event = FileChangeRequestEvent(
            path=self.path, root_dir=self.root_dir, operation=self.operation, diff=self.diff, truncated=self.truncated
        )
        await ctx.emit(event)
        if not event.cancelled:
            return None
        reason = event.cancel_reason or 'cancelled by a listener'
        return f'[{self.path!r} {_REFUSALS[self.operation]}: {reason}]'

    def edited(self, *, content_hash: str) -> FileEditedEvent:
        """The notification for this change once `edit_file` has applied it."""
        return FileEditedEvent(
            path=self.path, root_dir=self.root_dir, content_hash=content_hash, diff=self.diff, truncated=self.truncated
        )
