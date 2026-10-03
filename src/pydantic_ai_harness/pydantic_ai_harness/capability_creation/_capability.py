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
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._durable import RetryRequest
from pydantic_ai_harness.capability_creation._store import AuthoredCapability, CapabilityStore
from pydantic_ai_harness.capability_creation._toolset import CapabilityCreationOperations, CapabilityCreationToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

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
    `disable_authored_capability`. Authoring writes Python source to `directory`,
    imports it, and validates it (exactly one `AbstractCapability` subclass that
    constructs with no arguments and whose static getters run). Authored
    capabilities hold live code, so they are not spec-serializable and are
    persisted as source rather than as a spec.

    Activation boundary: a capability cannot be added to a live, already-executing
    run -- pydantic-ai resolves the capability set once at the start of each run.
    The authored capability becomes usable on the next `agent.run(...)`. The
    integration contract is one line on the orchestrator side: thread the store's
    active capabilities into the next run.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness.capability_creation import CapabilityCreation

    creation = CapabilityCreation(directory=Path('.authored'))
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[LocalWorkspace('.'), creation])

    # Loop: each iteration injects whatever the agent has authored so far.
    result = await agent.run('build a logging capability', capabilities=creation.store.load_active())
    ```

    Authoring imports model-written Python into the agent's own process, on this
    machine. So a run refuses to start unless its workspace is this machine
    (`LocalWorkspace`) and writable: next to a sandbox, the model's code would
    escape the sandbox, and a read-only workspace promises the model changes
    nothing.

    Each store read and write runs as a durable operation, so under durable
    execution a recovered run reuses the recorded result instead of writing
    the authored module and manifest again.
    """

    directory: Path
    """Directory holding the authored `<name>.py` files and the `manifest.json` index.

    A path on the machine running the agent, relative to the process's current directory, not a workspace path.
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
        """The disk-backed store. Call `store.load_active()` to inject authored capabilities into the next run."""
        return CapabilityStore(self.directory)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Refuse the run unless its workspace is this machine and writable."""
        workspace = ctx.workspace
        if workspace.ref is None or workspace.ref.provider != 'local' or workspace.read_only:
            raise UserError(
                '`CapabilityCreation` imports model-written Python into the agent process, so it only runs when '
                "the run's workspace is this machine and writable. Attach a writable `LocalWorkspace('.')`, not "
                'a sandbox or a read-only workspace.'
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

    @durable_operation('write')
    async def _write(self, name: str, code: str) -> AuthoredCapability | RetryRequest:
        try:
            return await anyio.to_thread.run_sync(self.store.write, name, code)
        except ValueError as exc:
            return RetryRequest(str(exc))

    @durable_operation('list_all')
    async def _list_all(self) -> list[AuthoredCapability]:
        return await anyio.to_thread.run_sync(self.store.list_all)

    @durable_operation('disable')
    async def _disable(self, name: str) -> bool:
        return await anyio.to_thread.run_sync(self.store.disable, name)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability holds a live, disk-backed store."""
        return None
