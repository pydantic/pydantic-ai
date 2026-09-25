"""A workspace backend whose every operation explains why no workspace is available."""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SupportsCommands, WorkspaceBackend, WorkspaceCommand

__all__ = ('UnavailableWorkspace',)


class UnavailableWorkspace(WorkspaceBackend, SupportsCommands):
    """A `WorkspaceBackend` whose execution operations raise `UserError` with a configured reason."""

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
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise UserError(self.reason)

    async def working_dir(self) -> Never:
        raise UserError(self.reason)
