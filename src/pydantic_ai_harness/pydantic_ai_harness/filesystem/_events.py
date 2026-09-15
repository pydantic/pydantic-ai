"""Events emitted by the filesystem capability.

Each event carries two path fields:

- `path`: normalized and relative to the emitting filesystem's root, never an
  absolute host path, so it is safe to echo to the model or a UI.
- `root_dir`: the absolute, symlink-resolved root the `path` is relative to,
  so a subscriber can locate the file (`Path(root_dir) / path`) without
  assuming it shares the emitter's root.

A diff in an event is a unified diff cut at `MAX_EVENT_DIFF_CHARS` with a
`truncated` flag, so an event stream that is persisted or forwarded to a UI
cannot be flooded by one large write. A change whose text is longer than
`MAX_DIFF_SOURCE_CHARS` on either side is not diffed at all.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic_ai import CapabilityEvent

FILE_SYSTEM_EVENTS = 'file_system'

MAX_EVENT_DIFF_CHARS = 8192
"""Characters kept in an event's `diff` before it is cut."""

MAX_DIFF_SOURCE_CHARS = 4 * MAX_EVENT_DIFF_CHARS
"""Longest text on either side of a change that is still diffed; past it the `diff` is the headers alone."""

FileOperation = Literal['write', 'edit', 'create_directory']
"""The change a `FileChangeRequestEvent` announces."""

SearchKind = Literal['find', 'grep']
"""Whether a `FilesSearchedEvent` matched names (`find_files`) or contents (`search_files`)."""


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='file_read'):
    """A text file was read successfully."""

    path: str
    root_dir: str
    content_hash: str


@dataclass(kw_only=True)
class DirectoryListedEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='directory_listed'):
    """A directory was listed successfully."""

    path: str
    root_dir: str
    entry_count: int


@dataclass(kw_only=True)
class FileWrittenEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='file_written'):
    """A text file was written or edited successfully."""

    path: str
    root_dir: str
    content_hash: str


@dataclass(kw_only=True)
class FileEditedEvent(FileWrittenEvent, name='file_edited'):
    """An existing text file was changed in place by `edit_file`.

    A `FileWrittenEvent` with the change itself: `diff` is the unified diff
    from the content before the edit to the content after it. Listeners for
    `FileWrittenEvent` receive edits too, since this is a subclass.
    """

    diff: str
    truncated: bool


@dataclass(kw_only=True)
class DirectoryCreatedEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='directory_created'):
    """A directory was created, along with any missing parents."""

    path: str
    root_dir: str


@dataclass(kw_only=True)
class FilesSearchedEvent(CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='files_searched'):
    """A search under `path` finished.

    `match_count` is the number of matches the model received; `truncated`
    says the search stopped at the capability's result cap.
    """

    path: str
    root_dir: str
    pattern: str
    search: SearchKind
    match_count: int
    truncated: bool


@dataclass(kw_only=True)
class FileChangeRequestEvent(
    CapabilityEvent, namespace=FILE_SYSTEM_EVENTS, name='file_change_request', dispatch='immediate'
):
    """A file or directory is about to change; listeners may cancel it first.

    Fires after the path has passed the access checks and, for a write or
    edit, after the conflict check, so a listener only sees changes that
    would otherwise go ahead; a write re-checks under its open descriptor,
    so a file replaced in the meantime can still fail after it was announced.
    A cancelled change is not applied and the model gets `cancel_reason` as
    the tool result.

    `diff` is the unified diff from the current content to the proposed
    content; a new file diffs from empty, and a `create_directory` has none.
    """

    path: str
    root_dir: str
    operation: FileOperation
    diff: str
    truncated: bool
    cancelled: bool = False
    cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Stop the change from being applied; a `reason` given by any listener reaches the model."""
        self.cancelled = True
        if reason is not None or self.cancel_reason is None:
            self.cancel_reason = reason
