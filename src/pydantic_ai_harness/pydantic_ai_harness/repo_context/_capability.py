"""RepoContext: discover and load a repo's accumulated context engineering."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic_ai._instructions import SourcedInstruction, append_on_change_by_default
from pydantic_ai._utils import is_str_dict, replace_no_init
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextContent,
    ToolCallPart,
    UserPromptPart,
    post_compaction_window,
)
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.workspaces import SupportsCommands, SupportsRealpath, Workspace
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, HarnessDeprecationWarning, warn_argument_ignored
from pydantic_ai_harness._workspace import in_working_dir, innermost_backend, require_workspace, workspace_path
from pydantic_ai_harness.filesystem import DirectoryListedEvent, FileReadEvent
from pydantic_ai_harness.filesystem._tool_kinds import step_aside_for_read_file
from pydantic_ai_harness.repo_context._loader import (
    ContextFile,
    context_file_label,
    dir_context_file_path,
    directories_up_to,
    discover_instruction_files,
    find_dir_context_file,
    render_context_file,
    render_context_file_path,
    render_context_files,
)
from pydantic_ai_harness.repo_context._toolset import RepoContextToolset

_INVENTORY_HINT = (
    'Call `{tool_name}` to map where this repo keeps its coding-assistant setup '
    '(instruction dirs, skills, sub-agents, and hooks) so you can inspect it.'
)
_ASSET_ROOTS_HINT = (
    'This repo may keep its coding-assistant setup in {roots}: `skills/*/SKILL.md`, '
    '`agents/*.md` (sub-agents), and `settings.json` (hooks). Inspect it with your file tools, '
    'or call `{tool_name}` to map it when that tool is offered.'
)
_POINTER_NOTE = (
    'A directory you just accessed has an instruction file, not loaded here. Its path is in the '
    '`<context-file>` tag that follows. Read the file if it is relevant to your task.'
)
_NESTED_PRECEDENCE = (
    'Instruction files ({names}) apply to the directory they are in and everything below it. Where instruction '
    "files conflict, the more deeply nested one takes precedence, and the user's direct instructions take "
    'precedence over all of them.'
)
_NESTED_NOTE_ORIGIN = 'pydantic-ai-harness.repo_context.nested_note.v1'
"""The `origin` in the `metadata` of every part a nested note enqueues, next to the surfaced `directory`."""
_DEFAULT_TRAVERSAL_TOOL_NAMES = frozenset({'list_directory', 'read_file'})
_DEFAULT_TRAVERSAL_PATH_ARG = 'path'
_TRAVERSAL_DEPRECATION = (
    '`RepoContext.traversal_tool_names` and `RepoContext.traversal_path_arg` are deprecated. '
    'Traversal detection now reacts to `FileReadEvent` and `DirectoryListedEvent`. Hosts can emit these events by '
    'importing them from `pydantic_ai_harness.filesystem`. The customized tool-sniffing fallback remains active for '
    'this configuration.'
)


@dataclass
class RepoContext(AbstractCapability[AgentDepsT]):
    """Discover and load a repo's accumulated coding-assistant context engineering.

    Three strategies, each independently toggleable:

    Everything is read through the run's workspace, anchored at its working
    directory; a run without a workspace fails at its start.

    1. Walk-up instruction autoload (`autoload_instructions`, on by default):
       load `CLAUDE.md`/`AGENTS.md` from the working directory and every ancestor
       up to `home_dir`, deduped, ancestor-first. These are read once at run start
       and injected as system instructions via `get_instructions`, so they stay in
       the cached prefix, stable within a run; a change between runs of a
       conversation is appended rather than rewritten.

    2. Asset inventory (`expose_inventory_tool`, off by default): a tool that
       reports where the repo's CE assets live (`.claude`/`.agents`/`.codex`/
       `.grok` and their `skills/`, `agents/`, `settings.json`). It locates
       assets; it does not parse them. It steps aside for `FileSystem`'s
       `read_file` when the asset roots are inside the working directory, and
       the instructions then point at the roots instead.

    3. Nested-on-traversal (`nested_traversal`, off by default): when the model
       lists a directory or reads a file through a filesystem capability event,
       surface the `CLAUDE.md`/`AGENTS.md` of that directory and of every
       directory between it and the working directory, shallowest first. The
       working directory's own files are not repeated when Strategy 1 loaded
       them. The note is enqueued in
       the message tail, not added to system instructions, so it does not
       invalidate the cached prefix. `nested_inject='pointer'` (default)
       enqueues a one-line pointer; `'contents'` inlines the file body. With it
       on, the instructions also gain one static sentence on how instruction
       files relate: each applies to its subtree, the deeper one wins, and the
       user's direct instructions win over all of them.

    Cache note: injecting file contents into the system prompt costs prompt-cache
    stability. Strategy 1 is safe because its files are stable within a run, and a
    change between runs is appended rather than rewritten; the volatile Strategy 3
    content rides in the message tail instead.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness.repo_context import RepoContext

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[LocalWorkspace('.'), RepoContext(home_dir=Path.home())],
    )
    ```
    """

    workspace_dir: Path | None = None
    """Deprecated and ignored: the walk-up and asset scan are anchored at the workspace's working directory.

    Set the working directory on the workspace instead, e.g. `LocalWorkspace('./repo')`.
    """

    home_dir: str | Path | None = None
    """The shallowest workspace directory to stop the walk-up at, inclusive, as a workspace path.

    A relative path resolves against the working directory. `None` (the default)
    scans only the working directory -- no walk-up.
    """

    filenames: Sequence[str] = ('CLAUDE.md', 'AGENTS.md')
    """Instruction filenames to look for, in within-directory precedence order."""

    autoload_instructions: bool = True
    """Strategy 1: load instruction files into the system prompt."""

    autoload_symlink_roots: Sequence[str | Path] = field(default=(), kw_only=True)
    """Authorize external autoload targets beneath these workspace paths, empty by default.

    Roots and targets are resolved through the workspace. Requires a backend implementing
    `SupportsRealpath` or command execution for workspace symlink resolution. External files
    retain their scanned entry's label. Nested traversal and inventory are unaffected.
    """

    expose_inventory_tool: bool = False
    """Strategy 2: expose the asset-inventory tool."""

    inventory_tool_name: str = 'inventory_agent_context'
    """Name of the inventory tool exposed to the model."""

    nested_traversal: bool = False
    """Strategy 3: surface the instruction files between the working directory and
    a directory the model lists or reads a file in. Off by default -- it couples to the list/read tools.
    With it on, the instructions also state how instruction files relate."""

    nested_inject: Literal['pointer', 'contents'] = 'pointer'
    """For Strategy 3: append a one-line `pointer`, or inline the file `contents`."""

    traversal_tool_names: frozenset[str] = _DEFAULT_TRAVERSAL_TOOL_NAMES
    """Deprecated tool names used by the compatibility traversal detector."""

    traversal_path_arg: str = _DEFAULT_TRAVERSAL_PATH_ARG
    """Deprecated path argument used by the compatibility traversal detector."""

    asset_roots: Sequence[str] = ('.claude', '.agents', '.codex', '.grok')
    """Root directories the inventory tool scans, relative to the working directory."""

    _context_files: list[ContextFile] | None = field(default=None, init=False, repr=False, compare=False)
    """Walk-up result for this run, loaded once in `before_run` via `ctx.workspace`."""

    _pending_dirs: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Directories whose Strategy 3 note is enqueued but not yet in history."""

    _loaded_dirs: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Directories whose instruction files Strategy 1 put in the instructions, which Strategy 3 never repeats."""

    _latest_compaction: CompactionPart | None = field(default=None, init=False, repr=False, compare=False)
    """The latest compaction in history when `_pending_dirs` was last checked; a new one clears them."""

    _cached_working_dir: Path | None = field(default=None, init=False, repr=False, compare=False)
    """The workspace's working directory for this run."""

    _toolset: RepoContextToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The one inventory toolset, shared with every per-run copy so durable execution sees the leaf it registered."""

    _sniff_traversal_tools: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.workspace_dir is not None:
            warn_argument_ignored('RepoContext', 'workspace_dir', SET_WORKING_DIR_ON_THE_WORKSPACE)
        self._sniff_traversal_tools = (
            self.traversal_tool_names != _DEFAULT_TRAVERSAL_TOOL_NAMES
            or self.traversal_path_arg != _DEFAULT_TRAVERSAL_PATH_ARG
        )
        if self._sniff_traversal_tools:
            warnings.warn(_TRAVERSAL_DEPRECATION, HarnessDeprecationWarning, stacklevel=2)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RepoContext[AgentDepsT]:
        """Return a fresh per-run instance with isolated traversal/cache state.

        The copy skips `__post_init__`, so deprecated arguments warned about at construction are not warned again.
        """
        run = replace_no_init(self)
        run._context_files = None
        run._pending_dirs = set()
        run._loaded_dirs = set()
        run._latest_compaction = None
        run._cached_working_dir = None
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail without a workspace, and load walk-up instruction files so `get_instructions` is sync."""
        require_workspace(ctx.workspace, 'RepoContext', ctx.messages)
        if not self.autoload_instructions:
            return
        workspace = ctx.workspace
        working_dir = await self._working_dir(workspace)
        home = None
        if self.home_dir is not None:
            home = Path(await workspace.resolve(workspace_path(Path(self.home_dir))))
        symlink_roots: list[Path] = []
        if self.autoload_symlink_roots:
            if not isinstance(innermost_backend(workspace), (SupportsRealpath, SupportsCommands)):
                raise UserError(
                    '`RepoContext.autoload_symlink_roots` requires a workspace backend implementing '
                    '`SupportsRealpath` or `SupportsCommands` to resolve symlinks safely.'
                )
            for root in self.autoload_symlink_roots:
                symlink_roots.append(Path(await workspace.realpath(workspace_path(Path(root)))))
        self._context_files = await discover_instruction_files(
            workspace, working_dir, home, self.filenames, autoload_symlink_roots=symlink_roots
        )
        # The walk-up loaded the working directory's instruction files (or an identical ancestor copy), so nested
        # traversal has nothing to add there. A file that appears there later in the run is still surfaced.
        if self.nested_traversal and await dir_context_file_path(workspace, working_dir, self.filenames) is not None:
            self._loaded_dirs.add(str(working_dir))

    def get_instructions(self) -> str | Callable[[RunContext[AgentDepsT]], str | None] | None:
        """Cache-stable instructions resolved after `before_run` loads workspace files."""
        if not self.autoload_instructions:
            # Neither part depends on the run, so they stay static and inside the cached prefix.
            return '\n\n'.join(self._static_instructions()) or None

        def instructions(_ctx: RunContext[AgentDepsT]) -> str | None:
            return self._render_instructions()

        return instructions

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        # The context files are read each run, so an edit to one between runs is appended as a change (a
        # diff, when that's shorter) rather than rewriting the cached prompt.
        return append_on_change_by_default(super()._collect_instructions(), key='repo_context')

    def _render_instructions(self) -> str | None:
        parts: list[str] = []
        if self._context_files:
            assert self._cached_working_dir is not None, '`before_run` resolves it before loading files'
            parts.append(render_context_files(self._context_files, relative_to=self._cached_working_dir))
        parts.extend(self._static_instructions())
        return '\n\n'.join(parts) or None

    def _static_instructions(self) -> list[str]:
        parts: list[str] = []
        if self.expose_inventory_tool:
            parts.append(self._inventory_hint())
        if self.nested_traversal:
            parts.append(_NESTED_PRECEDENCE.format(names=', '.join(f'`{name}`' for name in self.filenames)))
        return parts

    def _inventory_hint(self) -> str:
        """Point at the inventory tool, and at the asset roots when a file reader can replace it.

        It names no reader, so it is the same whether or not `FileSystem` is on the agent.
        """
        if self._asset_roots_in_working_dir:
            roots = ', '.join(f'`{root}/`' for root in self.asset_roots)
            return _ASSET_ROOTS_HINT.format(roots=roots, tool_name=self.inventory_tool_name)
        return _INVENTORY_HINT.format(tool_name=self.inventory_tool_name)

    @property
    def _asset_roots_in_working_dir(self) -> bool:
        """Whether every asset root is inside the working directory, which `FileSystem`'s root contains.

        The inventory reads the run's workspace, so this is all it takes for `read_file` to reach the same files.
        """
        return bool(self.asset_roots) and all(in_working_dir(root) for root in self.asset_roots)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """The asset-inventory toolset, or `None` when the tool is disabled."""
        if not self.expose_inventory_tool:
            return None
        if self._toolset is None:
            toolset = RepoContextToolset[AgentDepsT](
                self.asset_roots, self.inventory_tool_name, id=self.id or 'repo_context'
            )
            if self._asset_roots_in_working_dir:
                toolset.tools[self.inventory_tool_name].prepare = step_aside_for_read_file
            self._toolset = toolset
        return self._toolset

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Support customized legacy traversal tool and argument names."""
        if (
            not self.nested_traversal
            or not self._sniff_traversal_tools
            or call.tool_name not in self.traversal_tool_names
        ):
            return result
        raw_path = args.get(self.traversal_path_arg)
        if not isinstance(raw_path, str):
            return result
        await self._enqueue_context(ctx, await self._resolve_directory(ctx.workspace, raw_path))
        return result

    @on_event(FileReadEvent, DirectoryListedEvent)
    async def _on_file_traversal(
        self, ctx: RunContext[AgentDepsT], event: FileReadEvent | DirectoryListedEvent
    ) -> None:
        """Enqueue nested context after an authorized filesystem traversal."""
        if not self.nested_traversal:
            return
        working_dir = await self._working_dir(ctx.workspace)
        path = Path(await ctx.workspace.resolve(event.path, base=event.root_dir))
        directory = path.parent if isinstance(event, FileReadEvent) else path
        try:
            directory.relative_to(working_dir)
        except ValueError:
            return
        await self._enqueue_context(ctx, directory)

    async def _enqueue_context(self, ctx: RunContext[AgentDepsT], directory: Path) -> None:
        """Surface the instruction file of `directory` and of each directory between it and the working directory.

        Files are enqueued shallowest first, so the most specific file is the latest one the model reads.
        """
        working_dir = await self._working_dir(ctx.workspace)
        in_history = self._sync_surfaced_dirs(ctx.messages)
        for current in directories_up_to(directory, working_dir):
            await self._enqueue_directory(ctx, current, in_history)

    async def _enqueue_directory(self, ctx: RunContext[AgentDepsT], directory: Path, in_history: set[str]) -> None:
        key = str(directory)
        if self._surfaced(key, in_history):
            return
        context_file = await find_dir_context_file(ctx.workspace, directory, self.filenames)
        # Parallel traversals can probe the same directory concurrently; re-check after the await.
        if context_file is None or self._surfaced(key, in_history):
            return
        self._pending_dirs.add(key)
        ctx.enqueue(*_stamp(self._render_note(context_file), key))

    def _surfaced(self, key: str, in_history: set[str]) -> bool:
        return key in self._loaded_dirs or key in self._pending_dirs or key in in_history

    def _sync_surfaced_dirs(self, messages: Sequence[ModelMessage]) -> set[str]:
        """Update the pending notes against history, and return the directories whose note is in the window.

        Call it once per traversal, before any `_surfaced` check.

        A note survives in history until compaction summarizes it away, so this is derived from the window
        rather than remembered, and it covers notes from earlier runs of the conversation. Notes enqueued but
        not yet in history are tracked until they appear there, or until a new compaction replaces the history
        they would have been in.
        """
        latest = _latest_compaction(messages)
        if latest is not self._latest_compaction:
            self._latest_compaction = latest
            self._pending_dirs.clear()
        in_history = {
            directory
            for message in post_compaction_window(messages)
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
            if isinstance(item, TextContent) and (directory := _note_directory(item.metadata)) is not None
        }
        self._pending_dirs -= in_history
        return in_history

    async def _resolve_directory(self, workspace: Workspace, raw_path: str) -> Path:
        working_dir = await self._working_dir(workspace)
        text = await workspace.resolve(raw_path, base=working_dir.as_posix())
        candidate = Path(text)
        try:
            entry = await workspace.stat(text)
        except (FileNotFoundError, NotADirectoryError):
            return candidate
        return candidate.parent if not entry.is_dir else candidate

    def _render_note(self, context_file: ContextFile) -> Sequence[str | SystemPromptPart]:
        """The file's contents as repository data, or a framework pointer followed by the file's path as data.

        The contents and the path are the repository's, not the operator's or the framework's, so they go on
        the user channel in a `<context-file>` tag. Only the pointer's own wording is in system voice.
        """
        assert self._cached_working_dir is not None, 'a note is only rendered once the directory is resolved'
        label = context_file_label(context_file.path, self._cached_working_dir)
        if self.nested_inject == 'contents':
            return [render_context_file(context_file, label=label)]
        return [SystemPromptPart(_POINTER_NOTE), render_context_file_path(label)]

    async def _working_dir(self, workspace: Workspace) -> Path:
        if self._cached_working_dir is None:
            self._cached_working_dir = Path(await workspace.working_dir())
        return self._cached_working_dir

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'RepoContext'


def _stamp(parts: Sequence[str | SystemPromptPart], directory: str) -> Iterator[TextContent | SystemPromptPart]:
    """Mark a note's parts as RepoContext's, so history scans and compaction recognize them by metadata."""
    for part in parts:
        # A dict per part, so mutating one part's metadata doesn't change the other's.
        metadata = {'origin': _NESTED_NOTE_ORIGIN, 'directory': directory}
        yield (
            replace(part, metadata=metadata)
            if isinstance(part, SystemPromptPart)
            else TextContent(part, metadata=metadata)
        )


def _latest_compaction(messages: Sequence[ModelMessage]) -> CompactionPart | None:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, CompactionPart):
                return part
    return None


def _note_directory(metadata: object) -> str | None:
    if not is_str_dict(metadata) or metadata.get('origin') != _NESTED_NOTE_ORIGIN:
        return None
    directory = metadata.get('directory')
    return directory if isinstance(directory, str) else None
