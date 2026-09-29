"""Filesystem capability that provides bounded file system access to the run's workspace."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored, warn_argument_renamed
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.filesystem._toolset import (
    DEFAULT_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    FileSystemToolset,
    root_spelling,
)

_DEFAULT_READ_CHARS = 50_000

_DEFAULT_READ_ONLY: tuple[str, ...] = (
    '**/.git/*',
    '**/.env',
    '**/.env.*',
    '*.pem',
    '*.key',
    '**/secrets*',
    '**/.pydantic-ai-harness/**',
)


@dataclass
class FileSystem(AbstractCapability[AgentDepsT]):
    """File system access to the run's workspace, scoped to a root directory.

    Every operation goes through `ctx.workspace`, so attach a workspace to the
    run: `LocalWorkspace(...)` for a local checkout, or a sandbox provider's
    capability. A run without one fails at its start. Relative paths resolve
    from the workspace's working directory.

    `root_dir` bounds the model's file tools: before each operation, the target
    must be inside it both as written and once the workspace has resolved its
    symlinks, and `read_only_patterns` guard what may be written. This is a
    guardrail checked before each operation, not isolation: a symlink swapped in
    between the check and the use is not caught, and `Shell` commands are not
    bounded at all. The workspace is the isolation boundary. Walks do not descend
    into a directory whose real path is outside `root_dir`.
    """

    root_dir: str | Path | None = None
    """The containment boundary for all file operations, as a workspace path.

    `None` (the default) is the workspace's working directory; a relative path
    resolves against it. Set it higher to let the model reach beyond the
    working directory, e.g. a parent holding sibling projects. The working
    directory must be inside it: a relative path below it (e.g. `'src'`) is
    rejected here, and an absolute one that does not contain it fails the run on
    its first file operation. `'/'` turns the containment checks off, while any
    access patterns still match a symlink's target.
    """

    cwd: str | Path | None = None
    """Deprecated and ignored: relative paths resolve from the workspace's working directory.

    Set the working directory on the workspace instead, e.g. `LocalWorkspace('./repo')`.
    """

    allowed_patterns: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only paths matching at least one glob pattern are accessible."""

    denied_patterns: Sequence[str] = field(default_factory=list[str])
    """Paths matching any of these glob patterns are rejected, even if `allowed_patterns` matches them."""

    read_only_patterns: Sequence[str] = _DEFAULT_READ_ONLY
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to `.git/`, `.env`, key files, secrets, and `.pydantic-ai-harness/`, where
    capabilities keep files such as spilled tool output and background job status.
    Set to an empty list to make every path writable.
    """

    max_read_lines: int = 2000
    """Number of lines a `read_file` call returns when it passes no `limit`."""

    max_read_chars: int | None = _DEFAULT_READ_CHARS
    """Maximum characters in a single `read_file` result, header and hint included.

    The window ends on the last complete line that fits, and the continuation
    hint names the first line not shown, so a caller paging by `offset` cannot
    skip content. Defaults to 50,000 characters. Set this at or below any
    downstream tool-output cap; a cap applied after the fact cuts mid-line and
    drops or strands the hint. `None` leaves only `max_read_lines` in force. A cap
    smaller than the header and hint themselves cannot be met; the result still carries them.
    """

    max_list_results: int = 1000
    """Maximum number of entries returned by `list_directory`."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files`."""

    read_only: bool = False
    """Whether to expose only the tools in `READ_ONLY_TOOL_NAMES`.

    A read-only workspace (`Workspace.read_only`) narrows the tools the same way for that run.
    """

    content_hashes: bool = True
    """Whether tool results report content hashes and `write_file`/`edit_file` accept `expected_hash`.

    The hashes give a model optimistic concurrency control over a workspace
    that something else may also be editing. Turn them off for a single-writer
    coding agent, where they only add tokens to every read and write.
    """

    tools: Sequence[str] = DEFAULT_TOOL_NAMES
    """Which tools to register, from `FILE_SYSTEM_TOOL_NAMES`.

    The default is every tool that needs only the workspace's filesystem. Name
    `list_files` and `grep` to add the ripgrep-backed listing and search tools,
    which run the `rg` executable inside the workspace (the `coder` extra
    installs it for a local workspace) and respect `.gitignore`.
    `read_only` further narrows the selection to `READ_ONLY_TOOL_NAMES`.
    """

    max_retries: int | None = field(default=None, kw_only=True)
    """How many consecutive retries each file tool allows before the run fails.

    A denied path, a missing parent, or an edit whose `old_text` no longer matches asks the model
    to retry; a success resets the count. `None` (the default) uses the agent's tool retry budget.
    """

    protected_patterns: Sequence[str] | None = field(default=None, kw_only=True)
    """Deprecated: renamed to `read_only_patterns`."""

    _toolset: FileSystemToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The one toolset, built on first use; it resolves the boundary per workspace on its first operation there."""

    def __post_init__(self) -> None:
        if self.cwd is not None:
            warn_argument_ignored('FileSystem', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE)
        if self.protected_patterns is not None:
            if self.read_only_patterns is not _DEFAULT_READ_ONLY:
                raise TypeError('Pass `read_only_patterns` only: `protected_patterns` is its deprecated name.')
            warn_argument_renamed('FileSystem', 'protected_patterns', 'read_only_patterns', stacklevel=4)
            self.read_only_patterns, self.protected_patterns = self.protected_patterns, None
        root_spelling(None if self.root_dir is None else Path(self.root_dir))
        # Runtime validation: dataclass field annotations are advisory, not enforced.
        # A config-driven caller could pass a string that would otherwise propagate.
        values: dict[str, Any] = {
            'max_read_lines': self.max_read_lines,
            'max_list_results': self.max_list_results,
            'max_search_results': self.max_search_results,
            'max_find_results': self.max_find_results,
        }
        if self.max_read_chars is not None:
            values['max_read_chars'] = self.max_read_chars
        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        if self.max_retries is not None and (not isinstance(self.max_retries, int) or self.max_retries < 0):
            raise ValueError(f'max_retries must be a non-negative integer, got {self.max_retries!r}')

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail without a workspace, without touching it: the boundary waits for the first file operation."""
        require_workspace(ctx.workspace, 'FileSystem', ctx.messages)

    def get_toolset(self) -> FileSystemToolset[AgentDepsT] | FilteredToolset[AgentDepsT]:
        """The filesystem toolset, the same one for every run, so durable execution sees the leaf it registered."""
        toolset = self._file_system_toolset()
        if self.read_only:
            return FilteredToolset(toolset, lambda ctx, tool: tool.name in READ_ONLY_TOOL_NAMES)
        return toolset

    def _file_system_toolset(self) -> FileSystemToolset[AgentDepsT]:
        if self._toolset is None:
            self._toolset = FileSystemToolset[AgentDepsT](
                root_dir=None if self.root_dir is None else Path(self.root_dir),
                allowed_patterns=self.allowed_patterns,
                denied_patterns=self.denied_patterns,
                read_only_patterns=self.read_only_patterns,
                max_read_lines=self.max_read_lines,
                max_read_chars=self.max_read_chars,
                max_list_results=self.max_list_results,
                max_search_results=self.max_search_results,
                max_find_results=self.max_find_results,
                id=self.id or 'file_system',
                content_hashes=self.content_hashes,
                tools=self.tools,
                max_retries=self.max_retries,
            )
        return self._toolset
