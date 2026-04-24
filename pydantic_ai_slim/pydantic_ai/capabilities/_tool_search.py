"""Tool search capability: provider-adaptive discovery of deferred tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .._run_context import AgentDepsT
from ..builtin_tools import ToolSearchNativeStrategy, ToolSearchTool
from ..tools import AgentBuiltinTool
from ..toolsets import AbstractToolset
from ..toolsets._tool_search import ToolSearchFunc, ToolSearchStrategy, ToolSearchToolset
from .abstract import AbstractCapability, CapabilityOrdering


@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    """Capability that provides tool discovery for large toolsets.

    Tools marked with ``defer_loading=True`` are hidden from the model until discovered.
    Auto-injected into every agent — zero overhead when no deferred tools exist.

    When the model supports native tool search (Anthropic BM25/regex, OpenAI Responses),
    discovery is handled by the provider: the deferred tools are sent with ``defer_loading``
    on the wire and the provider exposes them once they've been discovered. Otherwise,
    discovery happens locally via a ``search_tools`` function that the model can call.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ToolSearch

    # Default: native search on supporting providers, local token matching elsewhere.
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch()])

    # Force a specific Anthropic native strategy; errors on providers that can't honor it.
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy='regex')])

    # Always run the local token-overlap algorithm, regardless of provider.
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy='substring')])

    # Custom search function — used locally, and by provider-native "client-executed" modes when supported.
    def my_search(query, tools):
        return [t.name for t in tools if query.lower() in (t.description or '').lower()]

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy=my_search)])
    ```
    """

    strategy: ToolSearchStrategy | None = None
    """The search strategy to use.

    * ``None`` (default): let Pydantic AI pick the best strategy for the current provider
      — native on supporting models (Anthropic BM25, OpenAI server-executed tool search),
      local token matching elsewhere. The choice may change in future versions.
    * ``'substring'``: always use the local token-overlap algorithm.
    * ``'bm25'`` / ``'regex'``: force a specific Anthropic native strategy. Raises on
      providers that can't honor the choice (including OpenAI, which has no named
      native strategies).
    * Callable ``(query, tools) -> names``: custom search function. Used locally, and by
      the native "client-executed" surface on providers that support it (Anthropic custom
      tool-reference blocks, OpenAI ``execution='client'``).
    """

    max_results: int = 10
    """Maximum number of matches returned by the local search algorithm."""

    tool_description: str | None = None
    """Custom description for the local ``search_tools`` function shown to the model."""

    search_guidance: str | None = None
    """Custom description for the local ``search_tools`` ``keywords`` parameter."""

    _search_fn: ToolSearchFunc | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # A callable strategy is used as the local search function. Named strategies
        # (``'substring'``, ``'bm25'``, ``'regex'``) don't plug into the local callable:
        # ``'substring'`` means the default token-matching algorithm runs locally, while
        # ``'bm25'``/``'regex'`` only take effect on the native provider path.
        self._search_fn = self.strategy if callable(self.strategy) else None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ToolSearch'

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        # ``'substring'`` is explicitly-local: no builtin, the local ``search_tools``
        # function tool carries the full search workload.
        if self.strategy == 'substring':
            return []

        # A callable strategy surfaces as the provider's "client-executed" native mode
        # (Anthropic: regular function tool with tool_reference result formatting;
        # OpenAI: ``execution='client'``). Models without support drop the builtin as
        # optional and fall back to the local ``search_tools`` function — same callable,
        # just executed locally instead of routed through the provider.
        if callable(self.strategy):
            return [ToolSearchTool(custom=True, optional=True)]

        # ``None`` means "pick the best native option available, otherwise fall back
        # locally" — ``optional=True`` so the swap silently falls back on unsupported
        # models.
        if self.strategy is None:
            return [ToolSearchTool(optional=True)]

        # Explicit named native strategy (``'bm25'`` / ``'regex'``). The user committed
        # to a specific algorithm, so ``optional=False``: if the model can't honor it,
        # the request must error rather than silently substitute a different algorithm.
        #
        # Assumes no local implementation of bm25/regex exists — if we ever port either
        # to Python, ``optional`` should flip to ``True`` for that strategy and
        # ``ToolSearchToolset`` should gain a matching branch, so models without native
        # support can still honor the choice via the local path.
        named: ToolSearchNativeStrategy = self.strategy
        return [ToolSearchTool(strategy=named, optional=False)]

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # For explicit named native strategies, the local ``search_tools`` function
        # tool is not a valid fallback — the user's ``strategy='bm25'``/``'regex'``
        # choice must be honored by the provider or error out. Suppress the local
        # tool so ``prepare_request`` can raise cleanly when the builtin is unsupported.
        local_fallback = not (isinstance(self.strategy, str) and self.strategy in ('bm25', 'regex'))
        return ToolSearchToolset(
            wrapped=toolset,
            search_fn=self._search_fn,
            max_results=self.max_results,
            tool_description=self.tool_description,
            search_guidance=self.search_guidance,
            local_fallback=local_fallback,
        )
