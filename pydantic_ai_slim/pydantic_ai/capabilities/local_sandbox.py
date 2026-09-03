"""A capability for running commands on the local machine."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai._utils import run_in_executor
from pydantic_ai.sandboxes import LocalSandboxBackend, SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability
from .durable_operation import durable_operation

__all__ = ('LocalSandbox',)

_LOCAL_PREFIX = 'local:'


@dataclass
class LocalSandbox(AbstractCapability[AgentDepsT]):
    """Attach the local machine as a sandbox for trusted development and tests.

    !!! warning "`LocalSandbox` does not isolate code"
        Commands run as host subprocesses with the host process's privileges. Use this capability
        only for trusted development and tests, never for untrusted model-generated commands.

    `root` is the host directory exposed to the run. When omitted, each run gets the deterministic
    system-temp path `pydantic-ai-sandbox-<run_id>`, which is removed on release. Under durable
    execution, it assumes every durable unit runs on one host because local filesystems are not
    shared between workers.
    """

    id: str | None = 'local_sandbox'
    root: Path | None = None

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = self.root.resolve()

    def _path(self, ctx: RunContext[AgentDepsT]) -> Path:
        if self.root is not None:
            return self.root
        return Path(tempfile.gettempdir()) / f'pydantic-ai-sandbox-{ctx.run_id}'

    async def acquire_sandbox(self, ctx: RunContext[AgentDepsT]) -> SandboxRef:
        return SandboxRef(sandbox_id=f'{_LOCAL_PREFIX}{self._path(ctx).as_posix()}')

    def resolve_sandbox(self, ctx: RunContext[AgentDepsT], ref: SandboxRef) -> SandboxBackend | None:
        if not ref.sandbox_id.startswith(_LOCAL_PREFIX):
            return None
        return LocalSandboxBackend(root=Path(ref.sandbox_id.removeprefix(_LOCAL_PREFIX)))

    @durable_operation('release_sandbox')
    async def release_sandbox(self, ctx: RunContext[AgentDepsT], ref: SandboxRef) -> None:
        if self.root is None:
            path = Path(ref.sandbox_id.removeprefix(_LOCAL_PREFIX))
            await run_in_executor(lambda: shutil.rmtree(path, ignore_errors=True))
