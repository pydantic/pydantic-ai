"""Tool kinds for the filesystem tools, so other code can recognize them whatever they are named.

Each tool's kind is in the `file_system` namespace, the one `FileSystem`'s capability events use,
and is named after the tool's default name: `read_file` has the kind `file_system.read_file`. The
kind id is stored in every message history that records a call, so it stays the same when the tool
is renamed or prefixed, and cannot change later.

Each kind declares the shape of its tool's arguments, and its call parts are promoted to a typed
`ToolCallPart` subclass. Only `read_file` has a typed return part: its result is the file, which
other capabilities look for in a history. The other tools return a formatted confirmation or listing
for the model, not a format to parse; the capability events carry their outcome as structured data.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NotRequired

from typing_extensions import TypedDict

from pydantic_ai import ToolKind
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai_harness.filesystem._events import FILE_SYSTEM_EVENTS


class ReadFileArgs(TypedDict):
    """Arguments of a `read_file` call."""

    path: str
    """File path relative to the working directory."""
    offset: NotRequired[int]
    """Zero-based line offset to start reading from."""
    limit: NotRequired[int | None]
    """Maximum number of lines to return."""


class WriteFileArgs(TypedDict):
    """Arguments of a `write_file` call."""

    path: str
    """File path relative to the working directory."""
    content: str
    """The text content to write."""
    expected_hash: NotRequired[str | None]
    """The hash the file must have for the write to go ahead; only offered with `content_hashes=True`."""


class ReplacementArgs(TypedDict):
    """One replacement in an `edit_file` call's `replacements`."""

    old_text: str
    """The exact text to find, which must appear exactly once."""
    new_text: str
    """The replacement text."""


class EditFileArgs(TypedDict):
    """Arguments of an `edit_file` call: one `old_text`/`new_text` pair, or several as `replacements`."""

    path: str
    """File path relative to the working directory."""
    old_text: NotRequired[str | None]
    """The exact text to find, which must appear exactly once."""
    new_text: NotRequired[str | None]
    """The replacement text."""
    replacements: NotRequired[list[ReplacementArgs] | None]
    """Replacements to apply in order, instead of a single pair."""
    expected_hash: NotRequired[str | None]
    """The hash the file must have for the edit to go ahead; only offered with `content_hashes=True`."""


class ListDirectoryArgs(TypedDict):
    """Arguments of a `list_directory` call."""

    path: NotRequired[str]
    """Directory path relative to the working directory."""


class SearchFilesArgs(TypedDict):
    """Arguments of a `search_files` call."""

    pattern: str
    """Regex pattern to search for."""
    path: NotRequired[str]
    """Directory to search in, relative to the working directory."""
    include_glob: NotRequired[str | None]
    """Glob matched against root-relative paths to narrow the search."""


class FindFilesArgs(TypedDict):
    """Arguments of a `find_files` call."""

    pattern: str
    """Glob pattern to match, relative to `path`."""
    path: NotRequired[str]
    """Directory to search in, relative to the working directory."""


class CreateDirectoryArgs(TypedDict):
    """Arguments of a `create_directory` call."""

    path: str
    """Directory path relative to the working directory."""


class FileInfoArgs(TypedDict):
    """Arguments of a `file_info` call."""

    path: str
    """File or directory path relative to the working directory."""


class ListFilesArgs(TypedDict):
    """Arguments of a `list_files` call."""

    path: NotRequired[str]
    """Directory to list, relative to the working directory."""
    glob: NotRequired[str | None]
    """Glob the listed files must match."""


class GrepArgs(TypedDict):
    """Arguments of a `grep` call."""

    pattern: str
    """Regular expression (ripgrep syntax), or exact text when `literal` is set."""
    path: NotRequired[str]
    """Directory or file to search, relative to the working directory."""
    glob: NotRequired[str | None]
    """Glob the searched files must match."""
    file_type: NotRequired[str | None]
    """Ripgrep file type the searched files must have."""
    ignore_case: NotRequired[bool]
    """Whether to match case-insensitively."""
    literal: NotRequired[bool]
    """Whether `pattern` is exact text rather than a regular expression."""
    context: NotRequired[int]
    """Lines of context to show around each match."""


# `read_file` declares no content shape: its result is `str | list[str | BinaryContent]`, which
# `ToolKind(content=...)` cannot take, as it accepts a class and not a union.
READ_FILE_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'read_file', args=ReadFileArgs)
"""The kind of `FileSystem`'s `read_file` tool, whatever it is named."""

WRITE_FILE_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'write_file', args=WriteFileArgs)
"""The kind of `FileSystem`'s `write_file` tool, whatever it is named."""

EDIT_FILE_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'edit_file', args=EditFileArgs)
"""The kind of `FileSystem`'s `edit_file` tool, whatever it is named."""

LIST_DIRECTORY_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'list_directory', args=ListDirectoryArgs)
"""The kind of `FileSystem`'s `list_directory` tool, whatever it is named."""

SEARCH_FILES_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'search_files', args=SearchFilesArgs)
"""The kind of `FileSystem`'s `search_files` tool, whatever it is named."""

FIND_FILES_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'find_files', args=FindFilesArgs)
"""The kind of `FileSystem`'s `find_files` tool, whatever it is named."""

CREATE_DIRECTORY_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'create_directory', args=CreateDirectoryArgs)
"""The kind of `FileSystem`'s `create_directory` tool, whatever it is named."""

FILE_INFO_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'file_info', args=FileInfoArgs)
"""The kind of `FileSystem`'s `file_info` tool, whatever it is named."""

LIST_FILES_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'list_files', args=ListFilesArgs)
"""The kind of `FileSystem`'s ripgrep-backed `list_files` tool, whatever it is named."""

GREP_KIND = ToolKind(FILE_SYSTEM_EVENTS, 'grep', args=GrepArgs)
"""The kind of `FileSystem`'s ripgrep-backed `grep` tool, whatever it is named."""


class ReadFileCallPart(ToolCallPart[ReadFileArgs], kind=READ_FILE_KIND):
    """A call to `FileSystem`'s `read_file` tool, under whatever name it was offered."""


class ReadFileReturnPart(ToolReturnPart, kind=READ_FILE_KIND):
    """A successful return of `FileSystem`'s `read_file` tool, under whatever name it was offered.

    `content` is the numbered text with its header as a `str`, or `[header, BinaryContent]` for an
    image or PDF. A return that failed or was denied stays a plain `ToolReturnPart` with this kind's
    `tool_kind`, as its content is an error.
    """


class WriteFileCallPart(ToolCallPart[WriteFileArgs], kind=WRITE_FILE_KIND):
    """A call to `FileSystem`'s `write_file` tool, under whatever name it was offered."""


class EditFileCallPart(ToolCallPart[EditFileArgs], kind=EDIT_FILE_KIND):
    """A call to `FileSystem`'s `edit_file` tool, under whatever name it was offered."""


class ListDirectoryCallPart(ToolCallPart[ListDirectoryArgs], kind=LIST_DIRECTORY_KIND):
    """A call to `FileSystem`'s `list_directory` tool, under whatever name it was offered."""


class SearchFilesCallPart(ToolCallPart[SearchFilesArgs], kind=SEARCH_FILES_KIND):
    """A call to `FileSystem`'s `search_files` tool, under whatever name it was offered."""


class FindFilesCallPart(ToolCallPart[FindFilesArgs], kind=FIND_FILES_KIND):
    """A call to `FileSystem`'s `find_files` tool, under whatever name it was offered."""


class CreateDirectoryCallPart(ToolCallPart[CreateDirectoryArgs], kind=CREATE_DIRECTORY_KIND):
    """A call to `FileSystem`'s `create_directory` tool, under whatever name it was offered."""


class FileInfoCallPart(ToolCallPart[FileInfoArgs], kind=FILE_INFO_KIND):
    """A call to `FileSystem`'s `file_info` tool, under whatever name it was offered."""


class ListFilesCallPart(ToolCallPart[ListFilesArgs], kind=LIST_FILES_KIND):
    """A call to `FileSystem`'s ripgrep-backed `list_files` tool, under whatever name it was offered."""


class GrepCallPart(ToolCallPart[GrepArgs], kind=GREP_KIND):
    """A call to `FileSystem`'s ripgrep-backed `grep` tool, under whatever name it was offered."""


FILE_SYSTEM_TOOL_KINDS: dict[str, ToolKind] = {
    'read_file': READ_FILE_KIND,
    'write_file': WRITE_FILE_KIND,
    'edit_file': EDIT_FILE_KIND,
    'list_directory': LIST_DIRECTORY_KIND,
    'search_files': SEARCH_FILES_KIND,
    'find_files': FIND_FILES_KIND,
    'create_directory': CREATE_DIRECTORY_KIND,
    'file_info': FILE_INFO_KIND,
    'list_files': LIST_FILES_KIND,
    'grep': GREP_KIND,
}
"""The kind of each tool in `FILE_SYSTEM_TOOL_NAMES`, keyed by the name `FileSystem` registers it under."""


def step_aside_for_read_file(ctx: RunContext[Any], tool_def: ToolDefinition) -> ToolDefinition:
    """A tool `prepare` that has the tool step aside for `read_file` when `FileSystem` is on the agent.

    For a capability whose own reader reads files `FileSystem` can reach too: the tool is left out of
    any run step with a tool of `READ_FILE_KIND`, whatever that tool is named.
    """
    return replace(tool_def, unless_tool_kind=READ_FILE_KIND)
