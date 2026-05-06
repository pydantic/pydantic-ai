from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from pydantic_ai._run_context import AgentDepsT, RunContext

from .abstract import AbstractCapability

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
    capability replaces this wrapper for the rest of the run, so its
    instructions, model settings, toolset, builtin tools, and hooks all flow
    through normally.

    Pass a [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] directly
    to `Agent(capabilities=[...])` or `agent.run(capabilities=[...])` and it
    will be wrapped in a `DynamicCapability` automatically.
    """

    capability_func: CapabilityFunc[AgentDepsT]
    """The function that takes the run context and returns a capability or `None`."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        capability = self.capability_func(ctx)
        if inspect.isawaitable(capability):
            capability = await capability
        if capability is None:
            return self
        return await capability.for_run(ctx)


def wrap_capability_funcs(
    capabilities: Sequence[AbstractCapability[AgentDepsT] | CapabilityFunc[AgentDepsT]] | None,
) -> list[AbstractCapability[AgentDepsT]]:
    """Wrap any [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] entries in a `DynamicCapability`."""
    if not capabilities:
        return []
    return [
        cap if isinstance(cap, AbstractCapability) else DynamicCapability(capability_func=cap) for cap in capabilities
    ]
