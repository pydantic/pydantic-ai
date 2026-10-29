"""Capability that supplies a boxd machine to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.boxd_sandbox._backend import BoxdSandboxBackend, destroy_machine

if TYPE_CHECKING:
    from boxd import AsyncBoxd

    from pydantic_ai.agent import AgentRunResult

_VCPUS = (1, 2, 4, 8)


class _SuppliedBackend(BoxdSandboxBackend):
    """The backend `BoxdSandbox` built for one run; the capability closes its client when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id
        # Until a run claims it, nothing closes this backend: Temporal builds one per activity and
        # drops it, so its own client closes after each operation instead.
        self._close_after_operation = True

    def claim(self, owned: bool) -> None:
        """Keep the client open across operations while the run that closes it is in progress."""
        self._close_after_operation = not owned


@dataclass(kw_only=True)
class BoxdSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [boxd](https://boxd.sh) machine as the run's workspace.

    A run with no reference creates a fresh machine on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the machine: boxd suspends it when idle and
    keeps it, disk included, until you delete it, unless your organization's defaults set a delete timer.
    Commands run under `sh -c` as the machine's non-root `boxd` user.

    The machine is isolated: it holds no boxd credentials and has no peers on your organization's
    private network, and attaching to one that is not isolated raises `WorkspaceUnavailableError`.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: AsyncBoxd | None = None
    """A `boxd.AsyncBoxd` to share across runs on one event loop; you close it. When `None`,
    each run opens one from the environment (`BOXD_TOKEN` or `BOXD_API_KEY`) and closes it when the run ends."""

    org: str | None = None
    """boxd organization a newly created machine belongs to; the API key's organization, or your
    default one, when `None`."""

    snapshot: str | None = None
    """boxd snapshot a newly created machine starts from, by name or id; boxd's default image when
    `None`. An unknown snapshot raises `WorkspaceUnavailableError` on first use."""

    vcpu: Literal[1, 2, 4, 8] | None = None
    """Size of a newly created machine: 1, 2, 4, or 8 vCPUs, with 4 GiB of memory each; the
    organization's default size when `None`. It cannot be combined with `snapshot`: a machine
    created from a snapshot keeps the size it was saved at."""

    auto_suspend_timeout: int | None = None
    """Idle network seconds before boxd suspends a newly created machine; `0` disables it, and
    `None` keeps the organization's default. A command wakes a suspended machine."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the
    machine's own (`/home/boxd`). Created on a new machine; an attached machine must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached machine; a command's `env` is
    layered on top. Nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`BoxdSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('auto_suspend_timeout', self.auto_suspend_timeout, minimum=0, optional=True)
        if self.vcpu is not None and (type(self.vcpu) is not int or self.vcpu not in _VCPUS):
            raise UserError(f'vcpu must be 1, 2, 4, 8 or None, got {self.vcpu!r}.')
        if self.snapshot is not None and self.vcpu is not None:
            raise UserError('`snapshot` keeps the size it was saved at, so it cannot be combined with `vcpu`.')
        check_env(self.env)

    def backend(self, ref: WorkspaceRef) -> BoxdSandboxBackend:
        """Construct a lazy backend for an existing machine."""
        return BoxdSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a machine by id, whatever state it is in.

        A machine that no longer exists is already gone, so destroying it returns quietly. A machine
        that is not isolated raises `ValueError` and is left alone: `BoxdSandbox` never created it.
        """
        if ref.provider != 'boxd':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boxd'")
        await destroy_machine(self.client, ref.id)

    async def fork(self, ref: WorkspaceRef) -> WorkspaceRef:
        """Copy a machine into a new isolated machine and return the copy's ref.

        The copy has the machine's disk and memory as they are now, running processes included, and the
        two change independently after: pass its ref as `workspace=` to run an agent in the copy, and
        `destroy` it when you are done. A machine that is not isolated, or that is on a named boxd network, raises
        `WorkspaceUnavailableError`, as attaching to it does.
        """
        backend = self.backend(ref)
        try:
            return await backend.fork()
        finally:
            await backend.aclose()

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'boxd':
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            org=self.org,
            snapshot=self.snapshot,
            vcpu=self.vcpu,
            auto_suspend_timeout=self.auto_suspend_timeout,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        # Only the backend this run built: one passed in as `workspace=`, such as a parent run's
        # handed to a subagent, belongs to the run that built it and may still be in use.
        backend = innermost_backend(ctx.workspace)
        owned = backend if isinstance(backend, _SuppliedBackend) and backend.run_id == ctx.run_id else None
        if owned is not None:
            owned.claim(True)
        try:
            return await handler()
        finally:
            if owned is not None:
                # `result.workspace` outlives the run, with no one left to close it.
                owned.claim(False)
                await owned.aclose()
