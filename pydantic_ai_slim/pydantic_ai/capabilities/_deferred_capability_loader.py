from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai._deferred_capabilities import DEFERRED_CAPABILITY_CATALOG_ID
from pydantic_ai._instructions import AgentInstructions, SourcedInstruction, append_on_change_by_default
from pydantic_ai._run_context import RunContext
from pydantic_ai._system_prompt import SystemPromptRunner
from pydantic_ai.models import DeferredCapabilityEntry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability_loader import DeferredCapabilityLoaderToolset

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


async def _render_deferred_capability_catalog(
    ctx: RunContext[AgentDepsT], catalog: list[DeferredCapabilityEntry]
) -> str:
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
    # instructions and tools it can already see, and if it does, the loader tool answers the
    # redundant call with an "already active" no-op return. One occasional wasted call is
    # far cheaper than busting the prefix cache on every load.
    entries = '\n'.join(
        f'- {entry.id}: {entry.description}' if entry.description else f'- {entry.id}' for entry in catalog
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
    has_search_surface = any(
        tool_def.tool_kind == 'tool-search' or (tool_def.defer_loading and tool_def.capability_id is None)
        for tool_def in ctx.tools.values()
    )
    prefix = (
        DEFERRED_CAPABILITY_CATALOG_PREFIX_WITH_SEARCH if has_search_surface else DEFERRED_CAPABILITY_CATALOG_PREFIX
    )
    return f'{prefix}\n{entries}'


@dataclass
class DeferredCapabilityLoader(AbstractCapability[AgentDepsT]):
    """Internal capability that installs deferred capability catalog and loading support."""

    _step: tuple[str | None, int] | None = field(default=None, init=False, repr=False)
    _catalog: list[DeferredCapabilityEntry] = field(
        default_factory=list[DeferredCapabilityEntry], init=False, repr=False
    )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return self._render_catalog

    async def _render_catalog(self, ctx: RunContext[AgentDepsT]) -> str:
        return await _render_deferred_capability_catalog(ctx, await self.catalog(ctx))

    async def catalog(self, ctx: RunContext[AgentDepsT]) -> list[DeferredCapabilityEntry]:
        """Every deferred capability, with its description and whether it's active, for the current run step.

        The one source for the catalog: the instructions are rendered from it, and the step's
        [`ModelRequestParameters.deferred_capabilities`][pydantic_ai.models.ModelRequestParameters.deferred_capabilities]
        is it. Resolved once per step, the first time it's asked for. That's when the instructions are rendered,
        after the step's tools are prepared, so a callable `description` sees them in `ctx.tools`; the request
        parameters are built after that. A request resumed without rendering instructions resolves it then.
        """
        step = (ctx.run_id, ctx.run_step)
        if step == self._step:
            return self._catalog
        active = ctx.active_capability_ids
        catalog: list[DeferredCapabilityEntry] = []
        for capability_id, capability in ctx.capabilities.items():
            if capability.defer_loading is not True:
                continue
            description = capability.get_description()
            if description is not None and not isinstance(description, str):
                description = await SystemPromptRunner[AgentDepsT](description).run(ctx)
            catalog.append(
                DeferredCapabilityEntry(id=capability_id, description=description, active=capability_id in active)
            )
        self._step, self._catalog = step, catalog
        return catalog

    def _collect_instructions(self) -> list[SourcedInstruction[AgentDepsT]]:
        # The catalog gets a key of its own, though this capability has no `id`, so a catalog that changes
        # between runs (a skill added to the workspace) is appended as a change instead of rewriting the
        # cached prompt.
        return append_on_change_by_default(super()._collect_instructions(), key=DEFERRED_CAPABILITY_CATALOG_ID)

    def get_ordering(self) -> CapabilityOrdering | None:
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        return DeferredCapabilityLoaderToolset(wrapped=toolset)
