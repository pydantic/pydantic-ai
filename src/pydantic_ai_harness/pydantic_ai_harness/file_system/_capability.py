"""Filesystem capability that provides bounded file system access to the run's workspace."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PureWindowsPath

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FilteredToolset
from pydantic_ai_harness._conjoin import RecordsStated, agreed, intersection, union
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored, warn_argument_renamed
from pydantic_ai_harness._workspace import require_workspace, workspace_path
from pydantic_ai_harness.announce_working_directory import AnnounceWorkingDirectory
from pydantic_ai_harness.file_system._content import require_pillow
from pydantic_ai_harness.file_system._toolset import (
    DEFAULT_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    FileSystemToolset,
    root_spelling,
)

_DEFAULT_READ_CHARS = 50_000

_DEFAULT_READ_ONLY: tuple[str, ...] = (
    '**/.git',
    '**/.env',
    '**/.env.*',
    '*.pem',
    '*.key',
    '**/secrets*',
    '**/.pydantic-ai-harness/**',
)


@dataclass
class FileSystem(RecordsStated, AbstractCapability[AgentDepsT]):
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
    bounded at all. Isolation depends on the configured workspace backend;
    `LocalWorkspace` provides none. Walks do not descend into a directory whose
    real path is outside `root_dir`.

    An agent has one `FileSystem` per `id`, `'file_system'` by default. Two that share it, such as a
    `FileSystem` added beside `Coder()`, are combined so that every restriction either states applies
    (see `combine`), so adding one only ever narrows. Give each its own `id` to keep two apart.
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
    """Paths matching any of these glob patterns are rejected, even if `allowed_patterns` matches them.

    Matching ignores case, so a differently cased spelling cannot reach a denied file on a
    case-insensitive filesystem.
    """

    read_only_patterns: Sequence[str] = _DEFAULT_READ_ONLY
    """Paths matching these patterns are read-only (writes are rejected).

    Defaults to protecting `.git` (the directory, or the file a worktree or
    submodule has), `.env` files, key files, secrets, and `.pydantic-ai-harness/`
    at any depth. Protected paths stay readable; use `denied_patterns` to block
    reads too. Set to an empty list to make every path writable. Like
    `denied_patterns`, these match regardless of case.
    """

    max_read_lines: int = 2000
    """Most lines one `read_file` call returns, and the number it returns when it passes no `limit`."""

    max_read_chars: int | None = _DEFAULT_READ_CHARS
    """Maximum characters in a single `read_file` result, header and hint included.

    The window ends on the last complete line that fits, and the continuation
    hint names the first line not shown, so a caller paging by `offset` cannot
    skip content. Defaults to 50,000 characters. Set this at or below any
    downstream tool-output cap; a cap applied after the fact cuts mid-line and
    drops or strands the hint. `None` leaves only `max_read_lines` in force. A cap
    smaller than the header and hint themselves cannot be met; the result still carries them.
    """

    max_image_side: int | None = field(default=None, kw_only=True)
    """Longest side, in pixels, of an image `read_file` returns; a larger one is downscaled to fit.

    The copy keeps the aspect ratio and the file's format, and is never upscaled.
    `None` (the default) sends images at their full size; `Coder` sets 1024.
    Downscaling needs Pillow, which the `coder` extra installs: setting this
    without it raises an `ImportError`.
    """

    max_list_results: int = 1000
    """Maximum number of entries returned by `list_directory`."""

    max_search_results: int = 1000
    """Maximum number of matches returned by `search_files` and `grep`."""

    max_find_results: int = 1000
    """Maximum number of matches returned by `find_files` and `list_files`."""

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

    id: str | None = field(default='file_system', kw_only=True)
    """The capability id, `'file_system'` by default. Two `FileSystem`s with one id are combined into one."""

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
            self._stated |= {'read_only_patterns'}
        root_spelling(None if self.root_dir is None else Path(self.root_dir))
        # Runtime validation: dataclass field annotations are advisory, not enforced.
        # A config-driven caller could pass a string that would otherwise propagate.
        values: dict[str, object] = {
            'max_read_lines': self.max_read_lines,
            'max_list_results': self.max_list_results,
            'max_search_results': self.max_search_results,
            'max_find_results': self.max_find_results,
        }
        if self.max_read_chars is not None:
            values['max_read_chars'] = self.max_read_chars
        if self.max_image_side is not None:
            values['max_image_side'] = self.max_image_side
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        if self.max_image_side is not None:
            require_pillow()
        retries: object = self.max_retries
        if retries is not None and (not isinstance(retries, int) or isinstance(retries, bool) or retries < 0):
            raise ValueError(f'max_retries must be a non-negative integer, got {self.max_retries!r}')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Combine `FileSystem`s sharing an id so that every restriction any of them states applies.

        The innermost `root_dir` wins, `None` (the working directory) included; denied and read-only
        patterns add up; the smaller limits win; `read_only` from any holds, `content_hashes` needs all;
        only tools every instance registers are kept; `max_retries` must agree. Two different
        `allowed_patterns` lists raise, since one `FileSystem` holds one list, and so do roots that can't
        be compared before the workspace resolves them.

        Only the fields an instance was constructed with take part, so a `FileSystem(allowed_patterns=[...])`
        added beside `Coder()` keeps `Coder`'s limits and tools. A copy made with `dataclasses.replace()`
        counts every field as passed.
        """
        file_systems = [capability for capability in capabilities if isinstance(capability, FileSystem)]
        assert len(file_systems) == len(capabilities)
        allowed = [list(fs.allowed_patterns) for fs in file_systems if fs.allowed_patterns]
        if len({tuple(patterns) for patterns in allowed}) > 1:
            # Glob lists can't be intersected as text, and `FileSystem` holds one list.
            raise UserError(
                'These `FileSystem` capabilities share an id, so both of their `allowed_patterns` lists would '
                f'have to apply, but one `FileSystem` holds one list: {", ".join(map(repr, allowed))}. State one '
                'list, or give each `FileSystem` its own `id` to keep them apart.'
            )
        read_only_lists = cls._stated_values(file_systems, 'read_only_patterns')
        tools = intersection(
            cls._stated_values(file_systems, 'tools'), owner='FileSystem', field='tools', empty_means_all=False
        )
        return replace(
            file_systems[-1],
            cwd=None,
            root_dir=_innermost_root([fs.root_dir for fs in file_systems]),
            allowed_patterns=allowed[0] if allowed else [],
            denied_patterns=union(fs.denied_patterns for fs in file_systems),
            read_only_patterns=union(read_only_lists) if read_only_lists else _DEFAULT_READ_ONLY,
            max_read_lines=cls._smallest(file_systems, 'max_read_lines'),
            max_read_chars=cls._smallest(file_systems, 'max_read_chars'),
            max_image_side=cls._smallest(file_systems, 'max_image_side'),
            max_list_results=cls._smallest(file_systems, 'max_list_results'),
            max_search_results=cls._smallest(file_systems, 'max_search_results'),
            max_find_results=cls._smallest(file_systems, 'max_find_results'),
            read_only=any(fs.read_only for fs in file_systems),
            content_hashes=all(fs.content_hashes for fs in file_systems),
            tools=DEFAULT_TOOL_NAMES if tools is None else tools,
            max_retries=agreed([fs.max_retries for fs in file_systems], owner='FileSystem', field='max_retries'),
        )

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail without a workspace, without touching it: the boundary waits for the first file operation."""
        require_workspace(ctx.workspace, 'FileSystem', ctx.messages)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Bring `AnnounceWorkingDirectory` along; copies from other capabilities merge into one line.

        A deferred one doesn't: naming the directory starts a sandbox the model may never load this capability for.
        """
        if self.defer_loading:
            return self
        return CombinedCapability([self, AnnounceWorkingDirectory()])

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
                max_image_side=self.max_image_side,
            )
        return self._toolset


def _innermost_root(roots: Sequence[str | Path | None]) -> str | Path | None:
    """The root inside every one of `roots`, `None` being the working directory.

    Every root must contain the working directory, so the roots lie on one line of its ancestors and the
    innermost is inside all of them. A root spelled relative (`..`) and one spelled absolute can't be
    compared before the workspace resolves them, so they are refused.
    """
    if any(root is None for root in roots):
        return None
    spellings = {posixpath.normpath(workspace_path(Path(root))): root for root in roots if root is not None}
    if '.' in spellings:
        return spellings['.']
    # A Windows path such as `C:/repo` is absolute too, as `root_spelling` reads it.
    absolute = [s for s in spellings if posixpath.isabs(s) or PureWindowsPath(s).drive]
    if absolute and len(absolute) < len(spellings):
        raise UserError(
            'These `FileSystem` capabilities share an id, so every `root_dir` applies, but some are relative and '
            f'some absolute ({", ".join(map(repr, spellings))}), which can only be compared in the workspace. '
            'Spell them the same way, or give each `FileSystem` its own `id` to keep them apart.'
        )
    # The longest spelling is innermost: the most `..` components is outermost, and so is the shortest
    # absolute path.
    if not absolute:
        return spellings[min(spellings, key=lambda spelling: spelling.count('..'))]
    innermost = max(absolute, key=len)
    for spelling in absolute:
        if posixpath.commonpath([spelling, innermost]) != spelling:
            raise UserError(
                'These `FileSystem` capabilities share an id, so every `root_dir` applies, but '
                f'{spelling!r} and {innermost!r} are not inside one another, so no working directory is inside both. '
                'Give each `FileSystem` its own `id` to keep them apart.'
            )
    return spellings[innermost]
