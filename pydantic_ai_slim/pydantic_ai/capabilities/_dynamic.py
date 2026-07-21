from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from .abstract import AbstractCapability
from .wrapper import WrapperCapability

CapabilityFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    AbstractCapability[AgentDepsT] | None | Awaitable[AbstractCapability[AgentDepsT] | None],
]
"""A sync/async function which takes a run context and returns a capability."""


@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    """A capability that builds another capability dynamically using a function that takes the run context.

    The factory is called once per agent run for capability resolution from
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run], and once per
    run for toolset resolution through a contributed dynamic toolset. The factory
    must therefore be deterministic given `ctx.deps`. The returned capability's
    instructions, model settings, native tools, and hooks flow through normally;
    its toolset is exposed through the dynamic toolset.

    Under durable execution, the factory is additionally called inside the durable
    unit whenever tools are listed or called there. A stable `id` is required on
    `DynamicCapability` because it names those durable units.

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
        self._toolset: DynamicToolset[AgentDepsT] | None = None
        # Forwarding this to the returned capability would be ambiguous: the factory
        # may return None, or a capability that deliberately chose its own loading state/id.
        if self.defer_loading is True:
            raise UserError(
                '`defer_loading` is not supported on `DynamicCapability` — '
                'set it on the capability the factory returns instead.'
            )

    def get_toolset(self) -> DynamicToolset[AgentDepsT]:
        if self._toolset is None:
            self._toolset = DynamicToolset(toolset_func=self._resolve_toolset, per_run_step=False, id=self.id)
        return self._toolset

    async def _resolve_toolset(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        if capability is None:
            return None
        assert ctx.agent is not None, 'CapabilityFunc requires an agent run context'
        capability = capability.for_agent(ctx.agent)
        capability = await capability.for_run(ctx)
        toolset = capability.get_toolset()
        if toolset is None or isinstance(toolset, AbstractToolset):
            # Pyright can't narrow Callable type aliases out of unions after an isinstance check
            return toolset  # pyright: ignore[reportUnknownVariableType]
        # A capability may contribute a toolset *function*; evaluate it here, where the
        # run context is available, instead of nesting another dynamic toolset.
        resolved: AbstractToolset[AgentDepsT] | None | Awaitable[AbstractToolset[AgentDepsT] | None] = toolset(ctx)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        return resolved

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        if capability is None:
            return self
        assert ctx.agent is not None, 'CapabilityFunc requires an agent run context'
        capability = capability.for_agent(ctx.agent)
        capability = await capability.for_run(ctx)
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


def wrap_capability_funcs(
    capabilities: Sequence[AbstractCapability[AgentDepsT] | CapabilityFunc[AgentDepsT]] | None,
) -> list[AbstractCapability[AgentDepsT]]:
    """Wrap any [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] entries in a `DynamicCapability`."""
    if not capabilities:
        return []
    return [
        cap if isinstance(cap, AbstractCapability) else DynamicCapability(capability_func=cap) for cap in capabilities
    ]
