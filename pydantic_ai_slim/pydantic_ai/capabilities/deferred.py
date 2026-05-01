from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._deferred import DeferredLoadingRegistry
from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability import DeferredCapabilityToolset

from .abstract import AbstractCapability, CapabilityOrdering


@dataclass
class DeferredLoadingCapability(AbstractCapability[AgentDepsT]):
    """Framework capability that provides the ``load_capability`` tool for deferred capabilities.

    Added by the agent for a run when any capability has loadable deferred outputs.
    Provides a :class:`~pydantic_ai.toolsets._deferred_capability.DeferredCapabilityToolset`
    that exposes a single ``load_capability(id)`` tool. The tool description includes a
    catalog of all unloaded capabilities so the model knows what's available.
    """

    registry: DeferredLoadingRegistry[AgentDepsT]
    """Run-local catalog and loadable outputs for deferred capabilities."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return None

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityToolset(wrapped=toolset, registry=self.registry)
