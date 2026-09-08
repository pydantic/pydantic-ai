"""A policy wrapper that makes an existing workspace backend read-only.

[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] wraps any
[`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend]: file reads pass through unchanged,
while command execution and file mutation raise `UserError`. Commands are blocked along with
writes because they execute against the same filesystem (the one-environment contract): a
workspace that refused `write_bytes` but ran `rm` would not be read-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SupportsFilesystem, WorkspaceBackend, WorkspaceCommand, WorkspaceFileEntry, WorkspaceRef
from .workspace import Workspace

__all__ = ('ReadOnlyWorkspace',)


_READ_ONLY_REASON = (
    'This workspace is read-only: running commands and modifying files are disabled. '
    'Reading files, listing directories, and checking that paths exist are allowed.'
)


class ReadOnlyWorkspace(WorkspaceBackend, SupportsFilesystem):
    """A [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] that forwards reads to a wrapped backend and refuses everything else.

    Reads (`working_dir`, `read_bytes`, `stat`, `list_dir`, `exists`) forward to the wrapped
    backend, using its native filesystem methods or the shell fallback; `run` and file mutations raise
    [`UserError`][pydantic_ai.exceptions.UserError] explaining the restriction. `ref`
    is the wrapped backend's own: a
    [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] names the environment, never the policy,
    so whoever supplies the workspace re-applies the wrapper on every (re)connection.

    The wrapper is a policy boundary for access through the workspace API, not an isolation
    mechanism. Read-only *with* command execution is only possible when the environment itself
    enforces it (e.g. a read-only mount). Over a backend without native file methods, reads run
    standard utilities such as `base64` inside the environment, so an environment someone has
    tampered with is not protected by this wrapper.
    """

    def __init__(self, wrapped: WorkspaceBackend):
        self._wrapped = wrapped
        # Use the ordinary wrapper internally so a run-only backend retains the same shell read
        # fallback without exposing command execution through this policy wrapper.
        self._workspace = Workspace(wrapped)

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._wrapped.ref

    async def working_dir(self) -> str:
        return await self._wrapped.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self._workspace.read_bytes(path)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        return await self._workspace.stat(path)

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        return await self._workspace.list_dir(path)

    async def exists(self, path: str) -> bool:
        return await self._workspace.exists(path)

    async def write_bytes(self, path: str, data: bytes) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def make_dir(self, path: str) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def remove(self, path: str) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise UserError(_READ_ONLY_REASON)
