"""Capability that supplies a Fly.io Sprite sandbox to an agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sprites import AsyncSpritesClient

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness._mounts import S3Mount, mounted
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_provider, check_working_dir, reject_defer_loading
from pydantic_ai_harness.sprites_sandbox._backend import SpritesSandboxBackend, destroy_sprite

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(SpritesSandboxBackend):
    """The backend `SpritesSandbox` built for one run; the capability closes its client when that run ends."""

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
class SpritesSandbox(AbstractCapability[AgentDepsT]):
    """Run the agent's workspace in a persistent [Fly.io Sprite](https://sprites.dev).

    A run with no reference creates a fresh Sprite on first use; pass a `WorkspaceRef` to attach
    to an existing one. Ending a run does not delete the Sprite: it sleeps when idle and persists
    until you delete it. Commands run under `sh -c` in the Sprite's own shell environment.

    This capability supplies the workspace only. Compose it with capabilities that use it, such
    as `Coder`, `Shell`, and `FileSystem`. See
    [Workspaces](https://pydantic.dev/docs/ai/core-concepts/workspace/) for more.
    """

    client: AsyncSpritesClient | None = None
    """A caller-owned `sprites.AsyncSpritesClient`, which is never closed for you. When omitted,
    each run's backend creates one on first use from `SPRITE_TOKEN` and closes it when the run
    ends; a backend this capability supplies outside a run (under Temporal, one per activity)
    closes its client after each operation. A backend from `backend(ref)` needs `aclose()`. Supply one to share its connections across runs, on one event loop."""

    runtime: str | None = None
    """Runtime for a newly created Sprite; an unknown runtime fails on first use."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the Sprite's default.
    Created on a new Sprite; an attached Sprite must already have it."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, on top of the Sprite's own; a command's `env` is layered on top."""

    mounts: (
        Sequence[S3Mount]
        | Callable[[RunContext[AgentDepsT]], Sequence[S3Mount]]
        | Callable[[RunContext[AgentDepsT]], Awaitable[Sequence[S3Mount]]]
    ) = ()
    """Buckets to show as folders in the Sprite, or a function of the run, sync or async, returning them with fresh keys."""

    def __post_init__(self) -> None:
        reject_defer_loading('SpritesSandbox', self.defer_loading)
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
        runtime: str | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> SpritesSandbox[Any]:
        """Build from an agent spec, naming the fields a spec can express.

        Core reads this signature to generate the spec's JSON schema. Inherited, core reads `__init__`, and
        `client`'s `AsyncSpritesClient` hint has no JSON schema, so only the bare `'SpritesSandbox'` is published.
        `mounts` is left out: a spec can't hold keys.
        """
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            runtime=runtime,
            working_dir=working_dir,
            env=env,
        )

    def backend(self, ref: WorkspaceRef) -> SpritesSandboxBackend:
        """Construct a lazy backend for an existing Sprite."""
        return SpritesSandboxBackend(
            client=self.client, ref=ref, runtime=self.runtime, working_dir=self.working_dir, env=self.env
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a Sprite by id without attaching or waking it.

        A Sprite that no longer exists is already gone, so destroying it returns quietly.
        """
        check_provider(ref, 'sprites')
        await destroy_sprite(self.client, ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'sprites':
            return None
        backend = _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            runtime=self.runtime,
            working_dir=self.working_dir,
            env=self.env,
        )
        return mounted(backend, backend._run_as_root, self.mounts, ctx)  # pyright: ignore[reportPrivateUsage]

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
