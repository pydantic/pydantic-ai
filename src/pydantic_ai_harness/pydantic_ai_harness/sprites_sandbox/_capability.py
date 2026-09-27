"""Capability that supplies a Fly.io Sprite sandbox to an agent run."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from sprites import AsyncSpritesClient

from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import check_working_dir
from pydantic_ai_harness.sprites_sandbox._backend import SpritesSandboxBackend

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult


class _SuppliedBackend(SpritesSandboxBackend):
    """The backend `SpritesSandbox` built for one run; the capability closes its client when that run ends."""

    def __init__(self, *, run_id: str | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = run_id


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
    ends. Supply one to share its connections across runs, on one event loop."""

    runtime: str | None = None
    """Runtime for a newly created Sprite; an unknown runtime fails on first use."""

    working_dir: str | None = None
    """Absolute directory commands start in and relative paths resolve against; `None` uses the Sprite's default."""

    env: Mapping[str, str] | None = field(default=None, repr=False)
    """Environment variables every command gets, on top of the Sprite's own; a command's `env` is layered on top."""

    def __post_init__(self) -> None:
        if self.defer_loading:
            # Core picks the run's workspace from the always-on capabilities only.
            raise UserError(
                'defer_loading must be False for SpritesSandbox: a deferred capability never supplies the workspace.'
            )
        # Checked here rather than at the first workspace operation, so a bad value fails where it is written.
        check_working_dir(self.working_dir)

    def backend(self, ref: WorkspaceRef) -> SpritesSandboxBackend:
        """Construct a lazy backend for an existing Sprite."""
        return SpritesSandboxBackend(
            client=self.client, ref=ref, runtime=self.runtime, working_dir=self.working_dir, env=self.env
        )

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Delete a Sprite by id without attaching or waking it."""
        if ref.provider != 'sprites':
            raise ValueError(f"unsupported workspace provider {ref.provider!r}; expected 'sprites'")
        if self.client is not None:
            await self.client.destroy_sprite(ref.id)
        else:
            token = os.getenv('SPRITE_TOKEN')
            if not token:
                raise ValueError('SPRITE_TOKEN is required to delete a Sprite')
            async with AsyncSpritesClient(token=token) as client:
                await client.destroy_sprite(ref.id)

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        """Build the backend for this run. No I/O here: it attaches or creates on first use."""
        if ref is not None and ref.provider != 'sprites':
            return None
        return _SuppliedBackend(
            run_id=ctx.run_id,
            client=self.client,
            ref=ref,
            runtime=self.runtime,
            working_dir=self.working_dir,
            env=self.env,
        )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        try:
            return await handler()
        finally:
            # Only the backend this run built: one passed in as `workspace=`, such as a parent run's
            # handed to a subagent, belongs to the run that built it and may still be in use.
            backend = innermost_backend(ctx.workspace)
            if isinstance(backend, _SuppliedBackend) and backend.run_id == ctx.run_id:
                await backend.aclose()
