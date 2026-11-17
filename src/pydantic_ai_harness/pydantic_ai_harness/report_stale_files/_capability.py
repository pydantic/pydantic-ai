"""`ReportStaleFiles`: tell the model when files it read have changed in the workspace."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai._framing import resembles_frame
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionCapabilityPart, ModelRequest, OperatorText, WorkspaceSource
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import Workspace, WorkspaceError, WorkspacePaths
from pydantic_ai_harness._durable import in_durable_context
from pydantic_ai_harness.file_system import FileReadEvent, FileWrittenEvent
from pydantic_ai_harness.file_system._content import bytes_hash

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

_CHANGED = (
    'These files changed in the workspace after you last read them, so what you saw of them is out of date. '
    'Read them again before relying on them or editing them:'
)
_DELETED = 'These files you read were deleted from the workspace:'

_AUTHORITIES = ('none', 'user', 'operator')
"""Authorities from least to most."""

_CONVERSATIONS_KEPT = 1024
"""How many conversations' observations an instance keeps, most recently used first."""


@dataclass(kw_only=True)
class StaleFilesPart(InstructionCapabilityPart, namespace='report_stale_files', name='stale_files'):
    """The request-scoped notice `ReportStaleFiles` enqueues, naming files that changed or were deleted.

    Its source is the workspace the files are in, so a workspace with less than `'operator'` authority gets the
    notice as data rather than in the application's voice. Find notices in history by this type.
    """

    changed: tuple[str, ...] = ()
    """The changed files, as the notice names them. Never sent to the model as a field."""

    deleted: tuple[str, ...] = ()
    """The deleted files, as the notice names them. Never sent to the model as a field."""


@dataclass(frozen=True)
class _Observation:
    content_hash: str
    root: str
    """The root of the file tool that read or wrote the file, which a check doesn't follow a symlink out of."""


_Observations = OrderedDict[str, _Observation]
"""Observations by absolute workspace path, least recently observed first."""


@dataclass
class ReportStaleFiles(AbstractCapability[AgentDepsT]):
    """Tell the model when files it read or wrote have since changed or been deleted in the workspace.

    It records the content hash of every file a `FileSystem` tool reads or writes, from `FileReadEvent` and
    `FileWrittenEvent`. Before each model request it reads the recorded files through the run's workspace,
    and when one no longer matches, it enqueues a request-scoped `StaleFilesPart` naming it. A changed file is
    named on every request until the model reads it again; a deleted file is named once. A file that can't
    be read for another reason, such as a permission error or a workspace failure, or that doesn't fit in
    `max_checked_bytes`, is skipped and checked again on the next request.

    Observations are kept per conversation, in memory on the instance the agent was built with, so a run
    continued from an earlier run's `message_history` hears about changes made between the runs.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness import FileSystem, ReportStaleFiles

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[LocalWorkspace('.'), FileSystem(), ReportStaleFiles()],
    )
    ```
    """

    max_tracked_files: int = 100
    """How many files to check before each request; past it, the least recently read or written is dropped."""

    max_reported_files: int = 20
    """How many paths one notice names; the rest are counted."""

    max_checked_bytes: int = 8 << 20
    """How many bytes of recorded files to read before each request, most recently read or written first.

    A file that doesn't fit in what is left is not checked on that request.
    """

    _conversations: OrderedDict[str, _Observations] = field(
        default_factory=OrderedDict[str, _Observations], init=False, repr=False, compare=False
    )
    _observations: _Observations = field(default_factory=_Observations, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_tracked_files < 1:
            raise ValueError('`max_tracked_files` must be at least 1.')
        if self.max_reported_files < 1:
            raise ValueError('`max_reported_files` must be at least 1.')
        if self.max_checked_bytes < 1:
            raise ValueError('`max_checked_bytes` must be at least 1.')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ReportStaleFiles[AgentDepsT]:
        """Bind the run to its conversation's observations; a run without a conversation id keeps its own."""
        run = replace(self)
        run._conversations = self._conversations
        if ctx.conversation_id is not None:
            run._observations = self._conversations.setdefault(ctx.conversation_id, _Observations())
            self._conversations.move_to_end(ctx.conversation_id)
            while len(self._conversations) > _CONVERSATIONS_KEPT:
                self._conversations.popitem(last=False)
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Refuse durable execution, whose replays and workers don't share this process's records."""
        if in_durable_context(ctx):
            raise UserError(
                '`ReportStaleFiles` cannot run under durable execution: its records of the files the model read are '
                'in process memory, which a replay or another worker does not have.'
            )

    @on_event(FileReadEvent, FileWrittenEvent)
    async def _observe(self, ctx: RunContext[AgentDepsT], event: FileReadEvent | FileWrittenEvent) -> None:
        path = await ctx.workspace.resolve(event.path, base=event.root_dir)
        self._observations[path] = _Observation(event.content_hash, event.root_dir)
        self._observations.move_to_end(path)
        while len(self._observations) > self.max_tracked_files:
            self._observations.popitem(last=False)

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Send a request-scoped notice naming the recorded files that changed or were deleted."""
        messages = request_context.messages
        # A provider-resume turn ends in a suspended `ModelResponse` that is echoed to the provider verbatim.
        if not messages or not isinstance(messages[-1], ModelRequest) or not self._observations:
            return request_context
        changed: list[str] = []
        deleted: list[str] = []
        budget = self.max_checked_bytes
        for path, observation in reversed(list(self._observations.items())):
            state, read = await _state(ctx.workspace, path, observation, budget)
            budget -= read
            if state == 'deleted':
                self._observations.pop(path, None)
                deleted.append(path)
            elif state == 'changed':
                changed.append(path)
        if changed or deleted:
            ctx.enqueue(await self._notice(ctx.workspace, changed, deleted))
        return request_context

    async def _notice(self, workspace: Workspace, changed: list[str], deleted: list[str]) -> StaleFilesPart:
        """A notice naming the files, whose source is the least trusted of their workspace sources."""
        sources = [await WorkspaceSource.from_workspace(workspace, path) for path in (*changed, *deleted)]
        working_dir = await workspace.working_dir()
        shown_changed = tuple(_shown(path, working_dir, workspace.paths) for path in changed)
        shown_deleted = tuple(_shown(path, working_dir, workspace.paths) for path in deleted)
        return StaleFilesPart(
            content=OperatorText(_render(shown_changed, shown_deleted, self.max_reported_files)),
            source=min(sources, key=lambda source: _AUTHORITIES.index(source.authority)),
            scope='request',
            changed=shown_changed,
            deleted=shown_deleted,
        )


async def _state(
    workspace: Workspace, path: str, observation: _Observation, budget: int
) -> tuple[Literal['fresh', 'changed', 'deleted', 'unknown'], int]:
    """Whether the file at `path` still has the observed content, and how many bytes were read to tell.

    `'unknown'` when it can't be read, is larger than `budget`, or now resolves outside the observing tool's root.
    """
    try:
        if not _contains(workspace.paths, observation.root, await workspace.realpath(path)):
            return 'unknown', 0
        entry = await workspace.stat(path)
        if entry.is_dir or (entry.size is not None and entry.size > budget):
            return 'unknown', 0
        content = await workspace.read_bytes(path)
    except (FileNotFoundError, NotADirectoryError):
        return 'deleted', 0
    except (OSError, WorkspaceError):
        return 'unknown', 0
    if len(content) > budget:
        return 'unknown', len(content)
    return ('fresh' if bytes_hash(content) == observation.content_hash else 'changed'), len(content)


def _contains(paths: WorkspacePaths, root: str, path: str) -> bool:
    return paths.normcase(path).startswith(paths.normcase(paths.join(root, '')))


def _shown(path: str, working_dir: str, paths: WorkspacePaths) -> str:
    """`path` as the file tools take it: relative to the working directory when inside it, else absolute."""
    if _contains(paths, working_dir, path):
        path = paths.relpath(path, working_dir)
    return path.replace(paths.sep, '/')


def _render(changed: tuple[str, ...], deleted: tuple[str, ...], limit: int) -> str:
    sections: list[str] = []
    for heading, paths in ((_CHANGED, changed), (_DELETED, deleted)):
        if not paths:
            continue
        # A path that looks like a framework frame can't go in the application's voice, so it's only counted.
        shown = [path for path in paths if not resembles_frame(path)][:limit]
        limit -= len(shown)
        lines = [heading, *(f'- `{path}`' for path in shown)]
        if hidden := len(paths) - len(shown):
            lines.append(f'- and {hidden} more')
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections)
