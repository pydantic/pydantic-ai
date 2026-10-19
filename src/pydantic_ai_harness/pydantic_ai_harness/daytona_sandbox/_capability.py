"""Capability that supplies a Daytona workspace to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir
from pydantic_ai_harness.daytona_sandbox._backend import (
    DaytonaSandboxBackend,
    _close_sdk_client,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from daytona import AsyncDaytona


class _RunBackend(DaytonaSandboxBackend):
    """A backend `DaytonaSandbox` built for one run; that run closes its API client when it ends."""

    run_id: str | None = None
    """The run that built this backend. A child run handed it through `workspace=` has another ID."""


@dataclass(kw_only=True)
class DaytonaSandbox(AbstractCapability[AgentDepsT]):
    """Run the agent's commands and file operations in an isolated [Daytona](https://www.daytona.io) sandbox.

    A run with no workspace ref creates a fresh sandbox; pass a `WorkspaceRef` to attach to an
    existing one. Shell commands run under `/bin/sh -c`.

    This capability supplies the sandbox only. Compose it with capabilities that use the run's
    workspace, such as `Coder`, `Shell`, or `FileSystem`.

    Without `client=`, each run opens its own `AsyncDaytona` API client on first use and closes it
    when the run ends; the sandbox keeps running, and `result.workspace` opens a new client if it
    is used afterwards.
    """

    client: AsyncDaytona | None = None
    """A caller-owned `daytona.AsyncDaytona` client, never closed by the capability. Supply one to
    share it across runs and own its lifecycle with `async with AsyncDaytona() as client:`."""

    snapshot: str | None = None
    """Daytona snapshot used for a newly created workspace; Daytona's default when `None`."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; the sandbox's
    default when `None`. It applies to attached sandboxes too."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command in the sandbox gets; a command's own `env` is layered on
    top. Nothing is read from the agent process's environment."""

    auto_stop_interval: int | None = None
    """Idle minutes before Daytona stops a newly created workspace; `0` disables it, and `None`
    keeps Daytona's default (15 minutes)."""

    network_block_all: bool = False
    """Whether to block outbound traffic from a newly created workspace."""

    def __post_init__(self) -> None:
        # Checked here rather than when the backend first creates a sandbox, so a bad value fails
        # where it is written instead of at the first workspace operation of some later run.
        check_working_dir(self.working_dir)
        check_integer('auto_stop_interval', self.auto_stop_interval, minimum=0, optional=True)
        if self.defer_loading:
            # Core skips deferred capabilities when it selects the run's workspace.
            raise UserError('DaytonaSandbox cannot be deferred: a deferred capability never supplies the workspace.')

    def backend(self, ref: WorkspaceRef) -> DaytonaSandboxBackend:
        """Build a lazy backend for an existing sandbox."""
        return DaytonaSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox without starting it, including one Daytona has stopped."""
        if ref.provider != 'daytona' or not ref.id.strip():
            raise ValueError('expected a Daytona workspace ref with a nonempty id')
        if self.client is not None:
            await (await self.client.get(ref.id)).delete()
        else:
            import daytona  # optional SDK is imported only when needed

            client = daytona.AsyncDaytona()
            try:
                await (await client.get(ref.id)).delete()
            finally:
                # Deleting a sandbox must release the API client even if deletion is cancelled.
                with anyio.move_on_after(30, shield=True):
                    await _close_sdk_client(client)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'daytona':
            return None
        backend = _RunBackend(
            client=self.client,
            ref=ref,
            snapshot=self.snapshot,
            auto_stop_interval=self.auto_stop_interval,
            working_dir=self.working_dir,
            env=self.env,
            network_block_all=self.network_block_all,
        )
        backend.run_id = ctx.run_id
        return backend

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        try:
            return await handler()
        finally:
            # Only the backend this run built: one passed through `workspace=`, such as a parent
            # run's handed to a subagent, may be shared with runs still using its client.
            backend = innermost_backend(ctx.workspace)
            if isinstance(backend, _RunBackend) and backend.run_id == ctx.run_id:
                await backend.aclose()
