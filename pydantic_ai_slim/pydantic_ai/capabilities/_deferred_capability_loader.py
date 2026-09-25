from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic_ai._instructions import AgentInstructions, SourcedInstruction
from pydantic_ai._run_context import RunContext
from pydantic_ai.native_tools._tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability_loader import (
    DEFERRED_CAPABILITY_CATALOG_INSTRUCTION_NAME,
    DeferredCapabilityLoaderToolset,
    deferred_capability_catalog,
)

from .abstract import (
    AbstractCapability,
    CapabilityOrdering,
)
from .instrumentation import Instrumentation

DEFERRED_CAPABILITY_CATALOG_PREFIX = (
    'The following capabilities are deferred and can be loaded using the `load_capability` tool. '
    "A capability's tools stay hidden until it is loaded:"
)
DEFERRED_CAPABILITY_CATALOG_PREFIX_WITH_SEARCH = (
    'The following capabilities are deferred and can be loaded using the `load_capability` tool. '
    "A capability's tools stay hidden until it is loaded — load the capability first rather than searching for its tools:"
)


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
    # redundant call with an "already active" ModelRetry. One occasional wasted retry is
    # far cheaper than busting the prefix cache on every load.
    catalog = await deferred_capability_catalog(ctx)
    entries = '\n'.join(
        f'- {cap_id}: {description}' if description else f'- {cap_id}' for cap_id, description in catalog.items()
    )
    # Steer the model away from tool search only when a search surface actually exists in the
    # run — mentioning searching in a run that has none invites hallucinated search calls. Two
    # signals cover the two ways a surface arises: the `search_tools` definition is registered
    # when `ToolSearch` runs with its local fallback enabled and a non-empty corpus, and a
    # searchable (non-capability) deferred tool marks the corpus itself for the named-native
    # strategies (`'bm25'`/`'regex'`), which register no local fallback but either get a native
    # surface or fail the run before this catalog matters. Both signals are authored, never
    # mutated mid-run, so the chosen variant is as byte-stable across the run as the rest of
    # this catalog.
    has_search_surface = TOOL_SEARCH_FUNCTION_TOOL_NAME in ctx.tools or any(
        tool_def.defer_loading and tool_def.capability_id is None for tool_def in ctx.tools.values()
    )
    prefix = (
        DEFERRED_CAPABILITY_CATALOG_PREFIX_WITH_SEARCH if has_search_surface else DEFERRED_CAPABILITY_CATALOG_PREFIX
    )
    return f'{prefix}\n{entries}'


@dataclass
class DeferredCapabilityLoader(AbstractCapability[AgentDepsT]):
    """Internal capability that installs deferred capability catalog and loading support."""

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return _render_deferred_capability_catalog

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        # Named so a model can tell the catalog from the rest of the instructions: a decision model offers
        # the capabilities as options of their own instead. With no `id`, the part stays unaddressable.
        return [
            replace(sourced, name=DEFERRED_CAPABILITY_CATALOG_INSTRUCTION_NAME)
            for sourced in super()._collect_instructions()
        ]

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityLoaderToolset(wrapped=toolset)
