"""`ReportStaleFiles`: tell the model when files it read have changed in the workspace."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness.file_system import FileReadEvent, FileWrittenEvent
from pydantic_ai_harness.file_system._content import bytes_hash

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

STALE_FILES_ORIGIN = 'pydantic-ai-harness.report_stale_files.notice.v1'
"""The `origin` in the `metadata` of the request-scoped `SystemPromptPart` a notice is sent as."""

_CHANGED = (
    'These files changed in the workspace after you last read them, so what you saw of them is out of date. '
    'Read them again before relying on them or editing them:'
)
_DELETED = 'These files you read were deleted from the workspace:'

_SHOWN_PATH = re.compile(r'[A-Za-z0-9._/@+-]{1,200}')
"""Paths a notice names. The notice is in system voice, so a path with other characters is only counted."""

_MAX_CHECKED_BYTES = 64 << 20
"""Files larger than this, which `read_file` won't return either, are not read to be checked."""

_CONVERSATIONS_KEPT = 1024
"""How many conversations' observations an instance keeps, most recently used first."""

_Observations = OrderedDict[str, str]
"""Content hashes by absolute workspace path, least recently observed first."""


@dataclass
class ReportStaleFiles(AbstractCapability[AgentDepsT]):
    """Tell the model when files it read or wrote have since changed or been deleted in the workspace.

    It records the content hash of every file a `FileSystem` tool reads or writes, from `FileReadEvent` and
    `FileWrittenEvent`. Before each model request it reads the recorded files through the run's workspace,
    and when one no longer matches, it sends a request-scoped system prompt naming it. A changed file is
    named on every request until the model reads it again; a deleted file is named once. A file that can't
    be read for another reason, such as a permission error or a workspace failure, or that is over 64 MiB, is
    skipped and checked again on the next request.

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

    _conversations: OrderedDict[str, _Observations] = field(
        default_factory=OrderedDict[str, _Observations], init=False, repr=False, compare=False
    )
    _observations: _Observations = field(default_factory=_Observations, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_tracked_files < 1:
            raise ValueError('`max_tracked_files` must be at least 1.')
        if self.max_reported_files < 1:
            raise ValueError('`max_reported_files` must be at least 1.')

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

    @on_event(FileReadEvent, FileWrittenEvent)
    async def _observe(self, ctx: RunContext[AgentDepsT], event: FileReadEvent | FileWrittenEvent) -> None:
        path = await ctx.workspace.resolve(event.path, base=event.root_dir)
        self._observations[path] = event.content_hash
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
        for path, content_hash in list(self._observations.items()):
            try:
                entry = await ctx.workspace.stat(path)
                if entry.is_dir or (entry.size is not None and entry.size > _MAX_CHECKED_BYTES):
                    continue
                content = await ctx.workspace.read_bytes(path)
            except (FileNotFoundError, NotADirectoryError):
                self._observations.pop(path, None)
                deleted.append(path)
                continue
            except (OSError, WorkspaceError):
                continue
            if bytes_hash(content) != content_hash:
                changed.append(path)
        if changed or deleted:
            working_dir = Path(await ctx.workspace.working_dir())
            notice = _render(_shown(changed, working_dir), _shown(deleted, working_dir), self.max_reported_files)
            ctx.enqueue(SystemPromptPart(notice, scope='request', metadata={'origin': STALE_FILES_ORIGIN}))
        return request_context


def _shown(paths: list[str], working_dir: Path) -> list[str]:
    """Each path as the file tools take it: relative to the working directory when inside it, else absolute."""
    return [
        (Path(path).relative_to(working_dir) if Path(path).is_relative_to(working_dir) else Path(path)).as_posix()
        for path in paths
    ]


def _render(changed: list[str], deleted: list[str], limit: int) -> str:
    sections: list[str] = []
    for heading, paths in ((_CHANGED, changed), (_DELETED, deleted)):
        if not paths:
            continue
        shown = [path for path in paths if _SHOWN_PATH.fullmatch(path)][:limit]
        limit -= len(shown)
        lines = [heading, *(f'- `{path}`' for path in shown)]
        if hidden := len(paths) - len(shown):
            lines.append(f'- and {hidden} more')
        sections.append('\n'.join(lines))
    return '\n\n'.join(sections)
