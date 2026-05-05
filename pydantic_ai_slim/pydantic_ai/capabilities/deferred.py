from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
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

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        def create_catalog(ctx: RunContext[AgentDepsT]) -> str:
            catalog: list[tuple[str, str]] = []
            for cap in ctx.capabilities.values():
                if not cap.defer_loading:
                    continue

                if cap.id is None:
                    continue

                catalog.append((cap.id, cap.get_description(ctx) or ''))

            return f'The following capabilities are deferred and can be loaded via load_capability: {", ".join(f"{id}: {description}" for id, description in catalog)}'

        return create_catalog

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost')

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityToolset(wrapped=toolset)
