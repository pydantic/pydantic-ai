"""Capability that supplies a Boat sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_env, check_integer, check_working_dir
from pydantic_ai_harness.boat_sandbox._backend import BoatSandboxBackend, destroy_sandbox

if TYPE_CHECKING:
    import httpx

    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(BoatSandboxBackend):
    """The backend `BoatSandbox` built for one run; the capability closes its client when that run ends."""

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
class BoatSandbox(AbstractCapability[AgentDepsT]):
    """Supply an isolated [Boat](https://boat.dev) sandbox as the run's workspace.

    A run with no reference creates a fresh sandbox on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the sandbox: Boat stops it after `auto_stop`
    and keeps it until you delete it. Commands run under `sh -c` as the sandbox's `user` account,
    and a new sandbox gets none of your Boat account's credentials.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: httpx.AsyncClient | None = None
    """An `httpx.AsyncClient` for Boat's API, with its `base_url` and an `Authorization: Bearer`
    header set, to share across runs on one event loop; you close it. When `None`, each run opens
    one from `BOAT_API_KEY` and closes it when the run ends."""

    machine_type: str | None = None
    """Boat machine size for a newly created sandbox: `'small'`, `'default'`, `'large'` or
    `'xlarge'`; Boat's default when `None`. A size Boat refuses raises `WorkspaceUnavailableError`
    on first use."""

    snapshot: str | None = None
    """Named Boat snapshot a newly created sandbox starts from; Boat's base image when `None`.
    An unknown snapshot raises `WorkspaceUnavailableError` on first use."""

    auto_stop: int | None = None
    """Seconds after it starts, and after each resume, before Boat stops a newly created sandbox,
    keeping its disk; `0` disables it, and `None` keeps Boat's default of an hour. Command activity
    does not postpone it. Attaching resumes a stopped sandbox."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the sandbox's own.
    Created on a new sandbox; an attached sandbox must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, also on an attached sandbox; a command's `env` is
    layered on top. Nothing is read from the host."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                '`BoatSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)
        check_integer('auto_stop', self.auto_stop, minimum=0, optional=True)
        check_env(self.env)

    def backend(self, ref: WorkspaceRef) -> BoatSandboxBackend:
        """Construct a lazy backend for an existing sandbox."""
        return BoatSandboxBackend(client=self.client, ref=ref, working_dir=self.working_dir, env=self.env)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a sandbox by ID without resuming it, including one Boat has stopped.

        A sandbox that no longer exists is already gone, so destroying it returns quietly.
        """
        if ref.provider != 'boat':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'boat'")
        await destroy_sandbox(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'boat':
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            machine_type=self.machine_type,
            snapshot=self.snapshot,
            auto_stop=self.auto_stop,
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
