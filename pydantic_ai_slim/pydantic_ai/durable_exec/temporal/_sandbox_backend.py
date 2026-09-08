"""Recover per-run sandbox policy lazily inside a Temporal activity."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import anyio

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities._sandbox import get_run_sandbox
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import (
    LazySandbox,
    Sandbox,
    SandboxBackend,
    SandboxCommand,
    SandboxFileEntry,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SupportsFilesystem,
)
from pydantic_ai.tools import RunContext


class ActivitySandboxBackend(LazySandbox[Sandbox], SandboxBackend, SupportsFilesystem):
    """Keep synchronous context decoding from bypassing an async `for_run` replacement.

    The inner facade preserves native filesystem access and shell fallbacks. Direct backend
    calls take the same recovery path as facade calls, so neither can bypass per-run policy.
    """

    def __init__(self, capability: AbstractCapability[Any], ctx: RunContext[Any], ref: SandboxRef | None) -> None:
        super().__init__()
        self._capability = capability
        self._ctx = ctx
        self._ref = ref

    async def create_or_attach(self) -> Sandbox:
        capability = await self._capability.for_run(self._ctx)
        selection = get_run_sandbox(capability, self._ctx, self._ref)
        if selection is None:
            raise UserError('The per-run sandbox capability declined the serialized sandbox reference.')
        return Sandbox(selection.backend)

    @property
    def ref(self) -> SandboxRef | None:
        return self._live.ref if self._live is not None else self._ref

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        started = time.monotonic()
        sandbox: Sandbox | None = None
        with anyio.move_on_after(timeout):
            sandbox = await self.sandbox
        if sandbox is None:
            raise SandboxTimeoutError('Sandbox recovery exceeded the command deadline.', timeout=timeout)
        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
        return await sandbox.run(command, shell=shell, cwd=cwd, env=env, timeout=remaining)

    async def working_dir(self) -> str:
        sandbox = await self.sandbox
        return await sandbox.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        sandbox = await self.sandbox
        return await sandbox.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        sandbox = await self.sandbox
        await sandbox.write_bytes(path, data)

    async def stat(self, path: str) -> SandboxFileEntry:
        sandbox = await self.sandbox
        return await sandbox.stat(path)

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        sandbox = await self.sandbox
        return await sandbox.list_dir(path)

    async def make_dir(self, path: str) -> None:
        sandbox = await self.sandbox
        await sandbox.make_dir(path)

    async def remove(self, path: str) -> None:
        sandbox = await self.sandbox
        await sandbox.remove(path)

    async def exists(self, path: str) -> bool:
        sandbox = await self.sandbox
        return await sandbox.exists(path)
