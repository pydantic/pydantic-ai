"""Runtime capability creation for agent-authored Pydantic AI capabilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio.to_thread

from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._durable import RetryRequest
from pydantic_ai_harness.capability_creation._store import AuthoredCapability, CapabilityStore
from pydantic_ai_harness.capability_creation._toolset import (
    AUTHORING_TOOL_METADATA_KEY,
    CapabilityCreationOperations,
    CapabilityCreationToolset,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import ValidatedToolArgs
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

_DEFAULT_GUIDANCE = (
    'You can author new pydantic-ai capabilities at runtime with `author_capability(name, code)`. '
    'A capability is a subclass of `pydantic_ai.capabilities.AbstractCapability` that constructs with '
    'no arguments and overrides one or more lifecycle hooks (a single overridden hook is a valid '
    'capability). Authored capabilities are validated and saved immediately, but they do not take effect '
    'in this run; whether a later run loads them depends on how this agent is set up. Use '
    '`list_authored_capabilities` and `disable_authored_capability` to manage them.'
)


@dataclass
class CapabilityCreation(AbstractCapability[AgentDepsT]):
    """Create Pydantic AI capabilities during one run for activation on the next.

    Exposes `author_capability(name, code)`, `list_authored_capabilities`, and
    `disable_authored_capability`. Authoring validates the Python source (exactly
    one `AbstractCapability` subclass that constructs with no arguments and whose
    static getters run) before it replaces `<name>.py` in `directory`; code that
    fails is kept for inspection and, while a working version exists, does not
    replace it. Authored capabilities hold live code, so they are persisted as source
    rather than as a spec.

    Pydantic AI resolves the capability set once at the start of each run, so an
    authored capability becomes usable on the next run: pass the result of
    `load_active(workspace)` as `capabilities=`.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.workspaces import LocalWorkspaceBackend
    from pydantic_ai_harness.capability_creation import CapabilityCreation

    workspace = LocalWorkspaceBackend('.')
    creation = CapabilityCreation(directory=Path('.authored'))
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[creation])

    # Inject whatever the agent has authored so far.
    extra = await creation.load_active(workspace)
    result = await agent.run('build a logging capability', capabilities=extra, workspace=workspace)
    ```

    Authoring imports model-written Python into the agent's own process, on this
    machine. So a run refuses to start unless its workspace is this machine
    (`LocalWorkspace`) and writable: next to a sandbox, the model's code would
    escape the sandbox, and a read-only workspace promises the model changes
    nothing. The run is also refused unless `directory` has `'operator'`
    [authority][pydantic_ai.messages.Authority] in that workspace, because the
    code there runs as the operator's. With `defer_loading`, which skips
    `before_run`, the same checks refuse each authoring tool call instead, and
    `load_active` makes them against the workspace it is given before it loads anything.

    Each store read and write runs as a durable operation, so under durable
    execution a recovered run reuses the recorded result instead of writing
    the authored module and manifest again.
    """

    directory: Path
    """Directory holding the authored `<name>.py` files and the `manifest.json` index.

    A path on the machine running the agent, relative to the process's current directory, not a workspace path.
    It must hold only code the operator wrote or this capability authored; a run is refused unless the workspace's
    [`authority_of`][pydantic_ai.workspaces.Workspace.authority_of] gives it `'operator'`, which a directory outside
    the workspace's working directory only gets when `path_authority` names it.
    """

    guidance: str | None = None
    """Static system-prompt guidance on authoring. Cache-stable. Leave `None` for the
    default, or set `''` to omit guidance entirely."""

    _: KW_ONLY
    id: str | None = 'capability_creation'
    """Stable identity for durable execution, which records each store read and write under it."""

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one configuration stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    @property
    def store(self) -> CapabilityStore:
        """The disk-backed store. Its `load_active()` runs the authored code without checking `directory`'s authority."""
        return CapabilityStore(self.directory)

    async def load_active(self, workspace: WorkspaceBackend) -> list[AbstractCapability[object]]:
        """Construct every active authored capability for the next run in `workspace`, as `store.load_active()` does.

        Raises `UserError`, before running any authored code, when `workspace` would refuse a run. Only `workspace` is
        checked, so pass the one the run uses, as `agent.run(..., workspace=...)`.
        """
        await self._check_workspace(workspace if isinstance(workspace, Workspace) else Workspace(workspace))
        return await anyio.to_thread.run_sync(self.store.load_active, abandon_on_cancel=True)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Refuse the run up front, rather than on the first tool call, when the workspace doesn't allow authoring."""
        await self._check_workspace(ctx.workspace)

    async def before_tool_execute(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        """Check again before an authoring tool runs: a capability loaded with `defer_loading` skips `before_run`."""
        if tool_def.metadata and tool_def.metadata.get(AUTHORING_TOOL_METADATA_KEY):
            await self._check_workspace(ctx.workspace)
        return args

    async def _check_workspace(self, workspace: Workspace) -> None:
        """Raise unless the workspace is this machine and writable, and holds `directory` at `'operator'`."""
        if workspace.ref is None or workspace.ref.provider != 'local' or workspace.read_only:
            raise UserError(
                '`CapabilityCreation` imports model-written Python into the agent process, so it only runs when '
                "the workspace is this machine and writable. Use a writable `LocalWorkspaceBackend('.')` as the run's "
                "`workspace=` and in `load_active`, or a `LocalWorkspace('.')` capability, not a sandbox or a "
                'read-only workspace.'
            )
        written = await self._absolute_directory()
        # `authority_of` collapses `..` as text, while the store's file operations climb from a link's target.
        for directory in dict.fromkeys((written, await workspace.realpath(written))):
            authority = await workspace.authority_of(directory)
            if authority != 'operator':
                raise UserError(
                    f"`CapabilityCreation` runs the code in its `directory` as the operator's, but {directory!r} has "
                    f"`'{authority}'` authority in the workspace. Point `directory` at a location that holds "
                    "only operator-written code and give it `'operator'` authority with `path_authority`, which a "
                    "directory outside the workspace's working directory needs too."
                )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on the authoring tools."""
        guidance = _DEFAULT_GUIDANCE if self.guidance is None else self.guidance
        return guidance or None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the authoring tools over this capability's store."""
        return CapabilityCreationToolset[AgentDepsT](
            self.store,
            id=self.id,
            operations=CapabilityCreationOperations(write=self._write, list_all=self._list_all, disable=self._disable),
        )

    @durable_operation('absolute_directory')
    async def _absolute_directory(self) -> str:
        # `directory` is relative to the process's current directory, not the workspace's working directory.
        return str(await anyio.to_thread.run_sync(self.directory.absolute))

    @durable_operation('write')
    async def _write(self, name: str, code: str) -> AuthoredCapability | RetryRequest:
        try:
            return await anyio.to_thread.run_sync(self.store.write, name, code, abandon_on_cancel=True)
        except ValueError as exc:
            return RetryRequest(str(exc))

    @durable_operation('list_all')
    async def _list_all(self) -> list[AuthoredCapability]:
        return await anyio.to_thread.run_sync(self.store.list_all, abandon_on_cancel=True)

    @durable_operation('disable')
    async def _disable(self, name: str) -> bool:
        return await anyio.to_thread.run_sync(self.store.disable, name, abandon_on_cancel=True)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability holds a live, disk-backed store."""
        return None
