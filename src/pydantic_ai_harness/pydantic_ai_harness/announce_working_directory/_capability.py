"""`AnnounceWorkingDirectory`: tell the model the workspace's working directory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceUnavailableError

_ID = 'announce_working_directory'


@dataclass(kw_only=True)
class AnnounceWorkingDirectory(AbstractCapability[AgentDepsT]):
    """Tell the model the workspace's working directory, so it uses relative paths instead of guessing absolute ones.

    `Shell` and `FileSystem` add it themselves. Add it to an agent whose own tools work in `ctx.workspace`.
    """

    id: str | None = _ID
    """One-off: two of them would state the same line, so they share a fixed id and merge into one.

    `None` or `''` means the default, `announce_working_directory`.
    """

    _line: str | None = field(default=None, init=False, repr=False, compare=False)
    """This run's line, once the workspace has been asked."""

    _asked: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.id = self.id or _ID

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """A copy for this run, so the line is worked out once and stays the same on every request of the run."""
        return replace(self)

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]:
        """Name the working directory, asking the workspace, which starts a sandbox that isn't running."""
        return self._working_directory

    async def _working_directory(self, ctx: RunContext[AgentDepsT]) -> str | None:
        # Asked once per run: a workspace that goes away partway through mustn't rewrite the prompt's prefix.
        if not self._asked:
            self._asked = True
            try:
                self._line = f'The working directory is `{await ctx.workspace.working_dir()}`.'
            except WorkspaceUnavailableError:
                # No workspace, or one that is gone: the first tool call reports why.
                self._line = None
        return self._line
