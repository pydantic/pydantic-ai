"""A workspace backend that reports why execution is unavailable.

[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] gives workspace operations
the same explicit failure mode. Pydantic AI uses it where a live execution
environment cannot safely exist, and applications can pass one deliberately to disable
execution with a policy-specific explanation.

It implements the required backend operations so every operation surfaces the configured reason.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import WorkspaceBackend, WorkspaceCommand

__all__ = ('UnavailableWorkspace',)


class UnavailableWorkspace(WorkspaceBackend):
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
