"""Filesystem capability: gives agents configurable, sandboxed file system access."""

from pydantic_ai_harness.filesystem._capability import FileSystem
from pydantic_ai_harness.filesystem._events import (
    FILE_SYSTEM_EVENTS,
    MAX_DIFF_SOURCE_CHARS,
    MAX_EVENT_DIFF_CHARS,
    DirectoryCreatedEvent,
    DirectoryListedEvent,
    FileChangeRequestEvent,
    FileEditedEvent,
    FileOperation,
    FileReadEvent,
    FilesSearchedEvent,
    FileWrittenEvent,
    SearchKind,
)
from pydantic_ai_harness.filesystem._toolset import READ_ONLY_TOOL_NAMES, FileSystemToolset

__all__ = [
    'FILE_SYSTEM_EVENTS',
    'MAX_DIFF_SOURCE_CHARS',
    'MAX_EVENT_DIFF_CHARS',
    'READ_ONLY_TOOL_NAMES',
    'DirectoryCreatedEvent',
    'DirectoryListedEvent',
    'FileChangeRequestEvent',
    'FileEditedEvent',
    'FileOperation',
    'FileReadEvent',
    'FileSystem',
    'FileSystemToolset',
    'FilesSearchedEvent',
    'FileWrittenEvent',
    'SearchKind',
]
