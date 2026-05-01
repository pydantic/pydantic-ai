from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import AgentToolset
from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability import DeferredCapabilityToolset

from .abstract import AbstractCapability, CapabilityOrdering


@dataclass
class DeferredLoadingCapability(AbstractCapability[AgentDepsT]):
    """Framework capability that provides the ``load_capability`` tool for deferred capabilities.

    Auto-injected by the agent when any capability has ``defer_loading=True``.
    Provides a :class:`~pydantic_ai.toolsets._deferred_capability.DeferredCapabilityToolset`
    that exposes a single ``load_capability(id)`` tool. The tool description includes a
    catalog of all unloaded capabilities so the model knows what's available.
    """

    deferred_capabilities: dict[str, AbstractCapability[AgentDepsT]]
    """The deferred capabilities to expose in the `load_capability` catalog."""

    # I am going to need the id to load the instructions back from get_capability_instructions I believe

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        # There are no instructions to use this capability, it it an internal capability to reveal the deferred capabilities
        return None

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost')

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        # Find the tools which are not deferred and return a toolset with them
        # But the problem is that the ones which are deferred, how would they get revealed anyway?

        # I should bundle all the tools of deferred capabilities together and return a toolset with them
        # I should honour their defer_loading flag along with it
        return None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityToolset(wrapped=toolset, deferred_capabilities=self.deferred_capabilities)
