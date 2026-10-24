"""A policy wrapper that makes a workspace read-only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Never

from .computer import ComputerAction
from .protocol import WorkspaceCommand, WorkspaceReadOnlyError
from .workspace import WrapperWorkspace

__all__ = ('ReadOnlyWorkspace',)


_READ_ONLY_REASON = (
    'This workspace is read-only: running commands, modifying files, and acting on the display are disabled. '
    'Reading files, listing directories, checking that paths exist, and taking screenshots are allowed.'
)


class ReadOnlyWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] that allows reads and refuses commands, file changes, and display actions.

    Commands are refused too, since they could change files. This restricts the workspace API; it is not isolation.
    """

    @property
    def read_only(self) -> bool:
        return True

    async def write_bytes(self, path: str, data: bytes) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY_REASON)

    async def make_dir(self, path: str) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY_REASON)

    async def remove(self, path: str) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY_REASON)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY_REASON)

    async def act(self, action: ComputerAction) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY_REASON)
