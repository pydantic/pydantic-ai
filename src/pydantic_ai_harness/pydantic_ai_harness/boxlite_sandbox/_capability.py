"""Capability that supplies a local BoxLite micro-VM to an agent run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace_provider import check_integer, check_working_dir, running_on_asyncio
from pydantic_ai_harness.boxlite_sandbox import _backend
from pydantic_ai_harness.boxlite_sandbox._backend import BoxLiteSandboxBackend


@dataclass(kw_only=True)
class BoxLiteSandbox(AbstractCapability[AgentDepsT]):
    """Supply a [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VM on this machine as the run's workspace.

    A run with no reference creates a fresh box on first use. Pass a `WorkspaceRef` supplied by the
    application to attach to a box from an earlier run; only ids in the form `BoxLiteSandbox` names its boxes are accepted.

    This capability supplies execution only. Compose it with tools or capabilities that use the
    workspace, such as `Coder`, `Shell`, or `FileSystem`.
    """

    image: str | None = None
    """OCI image a new box runs, such as `'python:3.12-slim'`; BoxLite's default, Alpine Linux, when `None`."""

    cpus: int | None = None
    """vCPUs for a new box; BoxLite's default when `None`."""

    memory_mb: int | None = None
    """Memory for a new box in MiB; BoxLite's default when `None`."""

    allow_internet_access: bool = False
    """Whether a new box gets outbound network access, made from this machine's network: the internet, but also its local network and services."""

    allowed_hosts: Sequence[str] | None = None
    """With `allow_internet_access`, the only hosts, IPs or CIDRs a new box can connect to, such as `['pypi.org', 'files.pythonhosted.org']`; `None` allows any."""

    working_dir: str | None = '/workspace'
    """Absolute directory commands start in and relative paths resolve against, created if missing; `None` uses the box's own."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached box; nothing is read from the host."""

    box_options: Mapping[str, Any] | None = field(default=None, repr=False)
    """Other keyword arguments for a new box's `boxlite.BoxOptions`, such as `volumes`, `ports`, `secrets` or `disk_size_gb`.

    The settings above own `image`, `cpus`, `memory_mib`, `network`, `working_dir`, `env` and `auto_delete` (and
    its deprecated `auto_remove`), so passing one of those here raises `UserError`.
    """

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError(
                '`BoxLiteSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        check_integer('cpus', self.cpus, optional=True)
        check_integer('memory_mb', self.memory_mb, optional=True)
        check_working_dir(self.working_dir)
        try:
            # Built once here so a bad option fails when the agent is defined, not on its first run.
            self._new_backend(None)
        except (TypeError, ValueError) as error:
            raise UserError(str(error)) from error

    def backend(self, ref: WorkspaceRef) -> BoxLiteSandboxBackend:
        """Attach lazily to an existing box, one `BoxLiteSandbox` created."""
        _backend.check_ref(ref)
        return BoxLiteSandboxBackend(ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a box by ref; one that is already gone returns quietly.

        It waits for a creation of the box still in progress in this process. Once the deletion has started,
        a cancelled caller waits for it to end.
        """
        _backend.check_ref(ref)
        if not running_on_asyncio():
            raise UserError(_backend.ASYNCIO_ONLY)
        await _backend.delete(ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it creates or attaches on first use."""
        del ctx
        if ref is not None:
            if ref.provider != 'boxlite':
                return None
            # A ref from message history only reaches boxes `BoxLiteSandbox` created.
            _backend.check_ref(ref)
        return self._new_backend(ref)

    def _new_backend(self, ref: WorkspaceRef | None) -> BoxLiteSandboxBackend:
        return BoxLiteSandboxBackend(
            ref=ref,
            image=self.image,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            allow_internet_access=self.allow_internet_access,
            allowed_hosts=self.allowed_hosts,
            working_dir=self.working_dir,
            env=self.env,
            box_options=self.box_options,
        )
