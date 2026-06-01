from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai._run_context import RunContext
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability_loader import DeferredCapabilityLoaderToolset

from .abstract import (
    AbstractCapability,
    CapabilityDescription,
    CapabilityOrdering,
)
from .instrumentation import Instrumentation

DEFERRED_CAPABILITY_CATALOG_PREFIX = (
    'The following capabilities are deferred and can be loaded using the `load_capability` tool:'
)


async def _resolve_capability_description(
    description: CapabilityDescription[AgentDepsT] | None,
    ctx: RunContext[AgentDepsT],
) -> str | None:
    if description is None:
        return None
    if isinstance(description, str):
        return description
    return await SystemPromptRunner[AgentDepsT](description).run(ctx)


async def _render_deferred_capability_catalog(ctx: RunContext[AgentDepsT]) -> str:
    # Deliberately lists EVERY deferred capability on every turn, including ones the model
    # has already loaded — do not filter by load state here.
    #
    # This catalog is a dynamic instruction, so it renders into the request *prefix* (ahead
    # of the message history). With static descriptions it renders byte-identical on every
    # request, which keeps the provider's prompt-cache prefix warm across loads — the entire
    # reason the native tool-search path exists. Dropping (or annotating) already-loaded
    # capabilities would mutate that prefix the moment any capability loads, and because
    # instructions sit at the very front, it would invalidate essentially the whole cached
    # prefix on every single load.
    #
    # The cost of keeping the list stable is that a loaded capability still appears as
    # "loadable". That is intentional and cheap: the model rarely re-loads something whose
    # instructions and tools it can already see, and if it does, the loader tool bounces the
    # redundant call with an "already available" ModelRetry. One occasional wasted retry is
    # far cheaper than busting the prefix cache on every load.
    catalog = {
        cap_id: await _resolve_capability_description(cap.get_description(), ctx)
        for cap_id, cap in ctx.capabilities.items()
        if cap.defer_loading is True
    }
    entries = '\n'.join(
        f'- {cap_id}: {description}' if description else f'- {cap_id}' for cap_id, description in catalog.items()
    )
    return f'{DEFERRED_CAPABILITY_CATALOG_PREFIX}\n{entries}'


@dataclass
class DeferredCapabilityLoader(AbstractCapability[AgentDepsT]):
    """Internal capability that installs deferred capability catalog and loading support."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _render_deferred_capability_catalog

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityLoaderToolset(wrapped=toolset)
