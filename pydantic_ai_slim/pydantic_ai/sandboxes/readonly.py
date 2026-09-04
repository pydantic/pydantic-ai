"""A policy wrapper that makes an existing sandbox backend read-only.

[`ReadOnlySandbox`][pydantic_ai.sandboxes.ReadOnlySandbox] wraps any
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend]: file reads pass through unchanged,
while command execution and file mutation raise `UserError`. Commands are blocked along with
writes because they execute against the same filesystem (the one-environment contract): a
sandbox that refused `write_bytes` but ran `rm` would not be read-only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from typing_extensions import Never

from pydantic_ai.exceptions import UserError

from .protocol import SandboxCommand, SandboxRef
from .sandbox import Sandbox

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SandboxFileEntry, SupportsFilesystem
    from .unavailable import UnavailableSandbox

__all__ = ('ReadOnlySandbox',)


_READ_ONLY_REASON = (
    'This sandbox is read-only: running commands and modifying files are disabled. '
    'Reading files, listing directories, and checking that paths exist are allowed.'
)


class ReadOnlySandbox:
    """A [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] that forwards reads to a wrapped backend and refuses everything else.

    Reads (`working_dir`, `read_bytes`, `stat`, `list_dir`, `exists`) forward to the wrapped
    backend, using its native filesystem methods or the shell fallback; `run` and file mutations raise
    [`UserError`][pydantic_ai.exceptions.UserError] explaining the restriction. `ref`
    is the wrapped backend's own: a
    [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] names the environment, never the policy,
    so whoever supplies the sandbox re-applies the wrapper on every (re)connection.

    The wrapper is a policy boundary for access through the sandbox API, not an isolation
    mechanism. Read-only *with* command execution is only possible when the environment itself
    enforces it (e.g. a read-only mount).
    """

    def __init__(self, wrapped: SandboxBackend):
        self._wrapped = wrapped
        # Use the ordinary wrapper internally so a run-only backend retains the same shell read
        # fallback without exposing command execution through this policy wrapper.
        self._sandbox = Sandbox(wrapped)

    @property
    def ref(self) -> SandboxRef | None:
        return self._wrapped.ref

    async def working_dir(self) -> str:
        return await self._wrapped.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self._sandbox.read_bytes(path)

    async def stat(self, path: str) -> SandboxFileEntry:
        return await self._sandbox.stat(path)

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        return await self._sandbox.list_dir(path)

    async def exists(self, path: str) -> bool:
        return await self._sandbox.exists(path)

    async def write_bytes(self, path: str, data: bytes) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def make_dir(self, path: str) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def remove(self, path: str) -> Never:
        raise UserError(_READ_ONLY_REASON)

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Never:
        raise UserError(_READ_ONLY_REASON)


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    _conforms: SandboxBackend = ReadOnlySandbox(UnavailableSandbox(''))
    _filesystem_conforms: SupportsFilesystem = ReadOnlySandbox(UnavailableSandbox(''))
