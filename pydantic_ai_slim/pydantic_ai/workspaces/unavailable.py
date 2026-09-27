"""A workspace backend whose every operation raises `WorkspaceUnavailableError` with a reason."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SupportsCommands, WorkspaceBackend, WorkspaceCommand, WorkspaceUnavailableError

__all__ = ('UnavailableWorkspace',)


class _UnavailableBackend(WorkspaceBackend, SupportsCommands):
    def __init__(self, reason: str):
        self.reason = reason

    @abstractmethod
    def _error(self) -> Exception: ...

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


class UnavailableWorkspace(_UnavailableBackend):
    """A `WorkspaceBackend` whose operations raise `WorkspaceUnavailableError` with a configured reason."""

    def _error(self) -> Exception:
        return WorkspaceUnavailableError(self.reason)


class _UnattachedWorkspace(_UnavailableBackend):  # pyright: ignore[reportUnusedClass]
    """The workspace of a run that has none attached: using it is a configuration mistake, so it raises `UserError`."""

    def _error(self) -> Exception:
        return UserError(self.reason)
