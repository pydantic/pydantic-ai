"""A workspace backend that reports why execution is unavailable.

[`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] gives workspace operations
the same explicit failure mode. Pydantic AI uses it where a live execution
environment cannot safely exist, and applications can pass one deliberately to disable
execution with a policy-specific explanation.

It implements the required backend operations so every operation raises
[`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] with the configured reason,
which tools catch like any other `WorkspaceError`.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SupportsCommands, WorkspaceBackend, WorkspaceCommand, WorkspaceUnavailableError

__all__ = ('UnavailableWorkspace',)


class UnavailableWorkspace(WorkspaceBackend, SupportsCommands):
    """A `WorkspaceBackend` whose operations raise `WorkspaceUnavailableError` with a configured reason."""

    def __init__(self, reason: str):
        self.reason = reason

    def _error(self) -> Exception:
        return WorkspaceUnavailableError(self.reason)

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
        raise self._error()

    async def working_dir(self) -> Never:
        raise self._error()


class _UnattachedWorkspace(UnavailableWorkspace):  # pyright: ignore[reportUnusedClass]
    """The workspace of a run that has none attached: using it is a configuration mistake, so it raises `UserError`."""

    def _error(self) -> Exception:
        return UserError(self.reason)
