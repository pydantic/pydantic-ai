"""Recover per-run workspace policy lazily inside a Temporal activity."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import anyio

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities._workspace import get_run_workspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    LazyWorkspace,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceTimeoutError,
)


class ActivityWorkspaceBackend(LazyWorkspace[Workspace], WorkspaceBackend, SupportsFilesystem):
    """Keep synchronous context decoding from bypassing an async `for_run` replacement.

    The inner facade preserves native filesystem access and shell fallbacks. Direct backend
    calls take the same recovery path as facade calls, so neither can bypass per-run policy.
    """

    def __init__(self, capability: AbstractCapability[Any], ctx: RunContext[Any], ref: WorkspaceRef | None) -> None:
        super().__init__()
        self._capability = capability
        self._ctx = ctx
        self._ref = ref

    async def create_or_attach(self) -> Workspace:
        capability = await self._capability.for_run(self._ctx)
        selection = get_run_workspace(capability, self._ctx, self._ref)
        if selection is None:
            raise UserError('The per-run workspace capability declined the serialized workspace reference.')
        return Workspace(selection.backend)

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._live.ref if self._live is not None else self._ref

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> WorkspaceResult:
        started = time.monotonic()
        workspace: Workspace | None = None
        with anyio.move_on_after(timeout):
            workspace = await self.workspace
        if workspace is None:
            raise WorkspaceTimeoutError('Workspace recovery exceeded the command deadline.', timeout=timeout)
        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
        return await workspace.run(command, shell=shell, cwd=cwd, env=env, timeout=remaining)

    async def working_dir(self) -> str:
        workspace = await self.workspace
        return await workspace.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        workspace = await self.workspace
        return await workspace.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        workspace = await self.workspace
        await workspace.write_bytes(path, data)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        workspace = await self.workspace
        return await workspace.stat(path)

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        workspace = await self.workspace
        return await workspace.list_dir(path)

    async def make_dir(self, path: str) -> None:
        workspace = await self.workspace
        await workspace.make_dir(path)

    async def remove(self, path: str) -> None:
        workspace = await self.workspace
        await workspace.remove(path)

    async def exists(self, path: str) -> bool:
        workspace = await self.workspace
        return await workspace.exists(path)
