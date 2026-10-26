"""Capability that supplies a local smolvm microVM to an agent run."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

import smol

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir, running_on_asyncio
from pydantic_ai_harness.smol_sandbox import _backend
from pydantic_ai_harness.smol_sandbox._backend import SmolSandboxBackend


@dataclass(kw_only=True)
class SmolSandbox(AbstractCapability[AgentDepsT]):
    """Supply a [smolvm](https://github.com/smol-machines/smolvm) microVM on this machine as the run's workspace.

    A run with no reference boots a fresh VM on first use. Pass a `WorkspaceRef` supplied by the
    application to attach to a VM from an earlier run; only ids in the form `SmolSandbox` names its VMs are accepted.

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`.
    """

    image: str | None = None
    """OCI image a new VM runs, such as `'python:3.12-alpine'`; `None` boots smolvm's own Alpine Linux without pulling anything."""

    cpus: int | None = None
    """vCPUs for a new VM; smolvm's default when `None`."""

    memory_mb: int | None = None
    """Memory for a new VM in MiB; smolvm's default when `None`."""

    allow_internet_access: bool = False
    """Whether a new VM gets unrestricted outbound network access, made from this machine's network: the internet, but also its local network and services."""

    working_dir: str | None = '/workspace'
    """Absolute directory commands start in and relative paths resolve against, created if missing; `None` uses the VM's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached VM; nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`SmolSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        check_integer('cpus', self.cpus, optional=True)
        check_integer('memory_mb', self.memory_mb, optional=True)
        check_working_dir(self.working_dir)

    def backend(self, ref: WorkspaceRef) -> SmolSandboxBackend:
        """Attach lazily to an existing VM, one `SmolSandbox` created."""
        _backend.check_ref(ref)
        return SmolSandboxBackend(ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a VM by ref; one that is already gone returns quietly.

        A boot of it still in progress in this process finishes first, and the deletion isn't abandoned
        when the caller is cancelled. The SDK has no delete by ID for a local VM, so this attaches first,
        which boots a stopped VM briefly.
        """
        _backend.check_ref(ref)
        if not running_on_asyncio():
            raise UserError(_backend.ASYNCIO_ONLY)
        await _backend.settled(_backend.start_lifecycle(ref.id, lambda: _delete(ref.id)))

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it boots or attaches on first use."""
        del ctx
        if ref is not None and ref.provider != 'smol':
            return None
        if ref is not None:
            # A ref from message history only reaches VMs `SmolSandbox` created.
            _backend.check_ref(ref)
        return SmolSandboxBackend(
            ref=ref,
            image=self.image,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            allow_internet_access=self.allow_internet_access,
            working_dir=self.working_dir,
            env=self.env,
        )


async def _delete(machine_id: str) -> None:
    try:
        machine = await asyncio.to_thread(smol.Machine.connect, machine_id, _backend.local_target())
    except smol.SmolError as error:
        if error.code == 'NOT_FOUND':
            return
        raise
    await asyncio.to_thread(machine.delete)
