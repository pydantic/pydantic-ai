from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability, CapabilityOrdering
from .wrapper import WrapperCapability

CapabilityFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractCapability[AgentDepsT] | None | Awaitable[AbstractCapability[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a capability."""


@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    """A capability that builds another capability dynamically using a function that takes the run context.

    The factory is called once per agent run from
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]. The returned
    capability's instructions, model settings, native tools, and hooks flow through
    normally; its toolset is exposed through a stable dynamic toolset contributed at
    agent construction time, which reuses the run's resolved capability instance.

    Under durable execution, a stable `id` is required on `DynamicCapability`: it
    names the durable units (activities/steps/tasks) that list and call the
    contributed tools. The factory itself runs in workflow/flow code, which durable
    engines re-execute on replay, recovery, or flow retry, so it must be
    deterministic given the run's dependencies; leave I/O to the toolset it
    returns, whose use is checkpointed inside the durable units. In-process
    engines (DBOS, Prefect) reuse the run's resolved capability inside those
    units; Temporal re-runs the factory inside its activities (the activity
    boundary can't carry the resolved instance).

    Pass a [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] directly
    to `Agent(capabilities=[...])` or `agent.run(capabilities=[...])` and it
    will be wrapped in a `DynamicCapability` automatically.

    `defer_loading` on the wrapper itself is rejected because `for_run` replaces
    the wrapper with the factory's return value. Set it on the returned
    capability instead.
    For history replay, set a stable `id` on the capability the factory returns
    rather than on the wrapper.
    """

    capability_func: CapabilityFunc[AgentDepsT]
    """The function that takes the run context and returns a capability or `None`."""

    def __post_init__(self) -> None:
        # Forwarding this to the returned capability would be ambiguous: the factory
        # may return None, or a capability that deliberately chose its own loading state/id.
        if self.defer_loading is True:
            raise UserError(
                '`defer_loading` is not supported on `DynamicCapability` — '
                'set it on the capability the factory returns instead.'
            )
        # Built eagerly: this single instance's identity is what durability engines register
        # against at agent construction and match at run time, so it must never fork.
        self._toolset = DynamicToolset(toolset_func=self._resolve_toolset, per_run_step=False, id=self.id)

    def get_toolset(self) -> DynamicToolset[AgentDepsT]:
        return self._toolset

    async def _resolve_toolset(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        # Inside a run, `for_run` has already resolved this capability once: reuse that
        # instance so hooks, instructions, and tools all observe the same per-run state,
        # and so the factory keeps its once-per-run contract. Read the registry via
        # `__dict__` because contexts rehydrated across a durable boundary (e.g.
        # `TemporalRunContext` inside an activity) deliberately don't carry it and raise
        # on regular attribute access.
        registry: dict[str, AbstractCapability[AgentDepsT]] = ctx.__dict__.get('capabilities') or {}
        for capability in registry.values():
            if isinstance(capability, ResolvedDynamicCapability) and capability.dynamic_toolset is self._toolset:
                return await _evaluate_agent_toolset(capability.wrapped.get_toolset(), ctx)
            if capability is self:
                # `for_run` kept this capability as-is because the factory returned `None`.
                return None
        # No resolved instance to reuse: the toolset is being used standalone, or inside a
        # durable unit (activity) whose deserialized context carries no capability registry —
        # re-resolve the factory there, where its I/O is allowed.
        capability = await self._resolve_capability(ctx)
        if capability is None:
            return None
        return await _evaluate_agent_toolset(capability.get_toolset(), ctx)

    async def _resolve_capability(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT] | None:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        if capability is None:
            return None
        assert ctx.agent is not None, 'CapabilityFunc requires an agent run context'
        capability = capability.for_agent(ctx.agent)
        return await capability.for_run(ctx)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        capability = await self._resolve_capability(ctx)
        if capability is None:
            return self
        return ResolvedDynamicCapability(wrapped=capability, dynamic_toolset=self.get_toolset())


@dataclass
class ResolvedDynamicCapability(WrapperCapability[AgentDepsT]):
    """The per-run replacement for a [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability].

    Delegates to the factory's resolved capability, except that the resolved capability's own
    toolset contribution is replaced by the `DynamicCapability`'s stable dynamic toolset — the
    one registered with any durable execution engine at agent construction time.
    """

    dynamic_toolset: DynamicToolset[AgentDepsT]

    def get_toolset(self) -> DynamicToolset[AgentDepsT]:
        return self.dynamic_toolset

    def get_ordering(self) -> CapabilityOrdering | None:
        # `CombinedCapability.for_run` re-sorts its (replaced) capabilities, so the resolved
        # capability's ordering constraints must survive the wrapper.
        return self.wrapped.get_ordering()


async def _evaluate_agent_toolset(
    toolset: AgentToolset[AgentDepsT] | None, ctx: RunContext[AgentDepsT]
) -> AbstractToolset[AgentDepsT] | None:
    """Normalize a capability's toolset contribution: evaluate the toolset-*function* arm with the run context."""
    if toolset is None or isinstance(toolset, AbstractToolset):
        # Pyright can't narrow Callable type aliases out of unions after an isinstance check
        return toolset  # pyright: ignore[reportUnknownVariableType]
    resolved: AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None] = toolset(ctx)
    if inspect.isawaitable(resolved):
        resolved = await resolved
    return resolved


def wrap_capability_funcs(
    capabilities: Sequence[AbstractCapability[AgentDepsT] | CapabilityFunc[AgentDepsT]] | None,
) -> list[AbstractCapability[AgentDepsT]]:
    """Wrap any [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] entries in a `DynamicCapability`."""
    if not capabilities:
        return []
    return [cap if isinstance(cap, AbstractCapability) else DynamicCapability(cap) for cap in capabilities]
