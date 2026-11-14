"""A workspace backend whose every operation raises `WorkspaceUnavailableError` with a reason."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never

from .protocol import SupportsCommands, WorkspaceBackend, WorkspaceCommand, WorkspaceUnavailableError

__all__ = ('UnavailableWorkspace',)


class UnavailableWorkspace(WorkspaceBackend, SupportsCommands):
    """A `WorkspaceBackend` whose operations raise `WorkspaceUnavailableError` with a configured reason."""

    def __init__(self, reason: str):
        self.reason = reason

    @property
    def ref(self) -> None:
        """Always `None`: there is no environment to name, so nothing can be reconnected to later."""
        return None

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise WorkspaceUnavailableError(self.reason)

    async def working_dir(self) -> Never:
        raise WorkspaceUnavailableError(self.reason)


NO_WORKSPACE = UnavailableWorkspace(
    "No workspace is attached to this run. Attach `capabilities=[LocalWorkspace('.')]` to the agent, or pass "
    "`workspace=LocalWorkspaceBackend('.')` to the run method, to use the local machine (unsafe: commands and "
    'file operations run with the full permissions of this process); attach another capability that supplies a '
    'workspace through its `get_workspace` hook; or, with a capability that can reconnect to an existing '
    'environment, pass its `WorkspaceRef`. '
    'See https://pydantic.dev/docs/ai/workspace/ for details.'
)
"""The backend of a run with no workspace. Unlike an `UnavailableWorkspace` a caller passes, it is not a choice:
a child run handed it still selects its own workspace."""
