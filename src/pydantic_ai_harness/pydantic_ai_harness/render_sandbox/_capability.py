"""Provide Render as an ordinary Pydantic AI workspace for Harness tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, TracebackType
from typing import Self

from render import RenderAsync

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceRef

from ._backend import RenderSandboxBackend, RenderSandboxConfig
from ._types import RenderSandboxEvent


class _RunBackend(RenderSandboxBackend):
    """Identify the run that supplied a backend without state on the shared capability."""

    def __init__(
        self,
        client: RenderAsync,
        *,
        config: RenderSandboxConfig,
        ref: WorkspaceRef | None,
        on_sandbox_event: Callable[[RenderSandboxEvent], None] | None,
        run_id: str | None,
        owner: object,
    ) -> None:
        super().__init__(client, config=config, ref=ref, on_sandbox_event=on_sandbox_event)
        self.run_id = run_id
        self.owner = owner


@dataclass(kw_only=True)
class RenderSandbox(AbstractCapability[AgentDepsT]):
    """Supply a Render workspace to standard Shell, FileSystem, or Coder capabilities.

    A fresh environment is created on first use. With auto_cleanup=True, this capability
    destroys only the fresh backend it supplied for this run. Attached references and
    caller-supplied workspaces always retain their caller's ownership. Set auto_cleanup=False
    to retain a fresh workspace, then explicitly destroy it after retaining its ref.

    on_finish receives the public Workspace after successful model completion and before
    owned cleanup. It can verify results and capture files using ordinary workspace methods.
    The client, environment values, and seed file contents are excluded from repr.
    """

    client: RenderAsync = field(default_factory=RenderAsync, repr=False)
    config: RenderSandboxConfig = field(default_factory=RenderSandboxConfig, repr=False)
    auto_cleanup: bool = True
    files: Mapping[str, str | bytes] | None = field(default=None, repr=False)
    on_finish: Callable[[Workspace], Awaitable[None]] | None = field(default=None, repr=False)
    on_sandbox_event: Callable[[RenderSandboxEvent], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.defer_loading:
            raise UserError('RenderSandbox cannot be deferred: the workspace is selected before a run')
        if self.files is not None:
            self.files = MappingProxyType(dict(self.files))

    async def __aenter__(self) -> Self:
        """Use this capability's SDK client within one application-owned async scope."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the SDK's HTTP pool after all users of this client have finished.

        This also closes an explicitly supplied client. It does not terminate retained
        sandboxes. Use one context around all concurrent runs, not one per shared run.
        """
        # Render 1.2 exposes transport cleanup on SandboxClient's public API client,
        # but has no close method on RenderAsync itself.
        await self.client.experimental.sandboxes.client.get_async_httpx_client().aclose()

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_workspace(self, ctx: RunContext[AgentDepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if ctx.in_durable_context:
            raise UserError('RenderSandbox does not support durable execution; use it outside a durable container.')
        if ref is not None and ref.provider != 'render':
            return None
        return _RunBackend(
            self.client,
            config=self.config,
            ref=ref,
            on_sandbox_event=self.on_sandbox_event,
            run_id=ctx.run_id,
            owner=self,
        )

    def get_instructions(self) -> str:
        return (
            'The workspace is an isolated Render sandbox. Use the standard workspace tools '
            'to run commands and read or write files. Each command starts in the workspace '
            'working directory. A nonzero exit code is a command result. Command deadlines '
            "stop that command's foreground process group; the workspace remains available."
        )

    def backend(self, ref: WorkspaceRef) -> RenderSandboxBackend:
        """Build a lazy backend for an existing sandbox without acquiring or owning it."""
        return RenderSandboxBackend(self.client, config=self.config, ref=ref, on_sandbox_event=self.on_sandbox_event)

    async def destroy(self, ref: WorkspaceRef) -> None:
        """Explicitly terminate a Render sandbox; no agent run or attachment is required."""
        await self.backend(ref).destroy()

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        if ctx.in_durable_context:
            raise UserError('RenderSandbox does not support durable execution; use it outside a durable container.')
        backend = ctx.workspace.backend
        cleanup: RenderSandboxBackend | None = None
        if (
            isinstance(backend, _RunBackend)
            and backend.owner is self
            and backend.run_id == ctx.run_id
            and backend.owned
            and self.auto_cleanup
        ):
            cleanup = backend
        run_error: BaseException | None = None
        try:
            # Standard Pydantic tools use our workspace's native content transfers and
            # public metadata fallback. The provider registers no competing tools.
            for path, content in (self.files or {}).items():
                await ctx.workspace.write_bytes(path, content.encode('utf-8') if isinstance(content, str) else content)
            result = await handler()
            if self.on_finish is not None:
                await self.on_finish(ctx.workspace)
            return result
        except BaseException as error:
            run_error = error
            raise
        finally:
            if cleanup is not None:
                try:
                    await cleanup.destroy()
                except BaseException as cleanup_error:
                    if run_error is not None:
                        # An explicit cause survives Pydantic graph teardown, which can
                        # replace implicit exception context while closing its task group.
                        run_error.add_note(f'Workspace cleanup also failed; retain {cleanup.ref!r} for recovery.')
                        raise run_error from cleanup_error
                    raise
