from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability_loader import DeferredCapabilityLoaderToolset

from .abstract import (
    AbstractCapability,
    CapabilityOrdering,
    resolve_capability_description,
)
from .instrumentation import Instrumentation


@dataclass
class DeferredCapabilityLoader(AbstractCapability[AgentDepsT]):
    """Internal capability that installs deferred capability catalog and loading support."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        async def create_catalog(ctx: RunContext[AgentDepsT]) -> str:
            catalog: list[tuple[str, str | None]] = []
            for cap_id, cap in ctx.capabilities.items():
                if cap.defer_loading is not True:
                    continue

                description = await resolve_capability_description(cap.get_description(), ctx)
                catalog.append((cap_id, description))

            entries = '\n'.join(
                f'- {cap_id}: {description}' if description else f'- {cap_id}' for cap_id, description in catalog
            )
            return f'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n{entries}'

        return create_catalog

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityLoaderToolset(wrapped=toolset)
