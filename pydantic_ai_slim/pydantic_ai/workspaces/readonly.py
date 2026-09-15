"""A policy wrapper that makes an existing workspace backend read-only.

[`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] wraps any
[`Workspace`][pydantic_ai.workspaces.Workspace] facade: file reads pass through unchanged,
while command execution and file mutation raise `UserError`. Commands are blocked along with
writes because they execute against the same filesystem (the one-environment contract): a
workspace that refused `write_bytes` but ran `rm` would not be read-only.
"""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import WorkspaceCommand
from .workspace import WrapperWorkspace

__all__ = ('ReadOnlyWorkspace',)


_READ_ONLY_REASON = (
    'This workspace is read-only: running commands and modifying files are disabled. '
    'Reading files, listing directories, and checking that paths exist are allowed.'
)


class ReadOnlyWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] facade that forwards reads and refuses mutations.

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
