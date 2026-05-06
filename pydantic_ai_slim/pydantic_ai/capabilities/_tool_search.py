"""Tool search capability: provider-adaptive discovery of deferred tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .._run_context import AgentDepsT
from ..builtin_tools.tool_search import (
    ToolSearchFunc,
    ToolSearchNativeStrategy,
    ToolSearchStrategy,
    ToolSearchTool,
)

# `ToolDefinition` is referenced via forward-string from `ToolSearchFunc`
# (defined in `builtin_tools/tool_search.py`, where it can't be eagerly imported because
# of the `tools.py` ↔ `builtin_tools` circular). Import it eagerly here so dataclass-spec
# generation (`get_type_hints` on `ToolSearch.__init__`) can resolve the forward reference
# against this module's globals.
from ..tools import (
    AgentBuiltinTool,
    ToolDefinition,  # pyright: ignore[reportUnusedImport]  # noqa: F401  (resolves forward ref)
)
from ..toolsets import AbstractToolset
from ..toolsets._tool_search import ToolSearchToolset, keywords_search_fn
from .abstract import AbstractCapability, CapabilityOrdering


@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    """Capability that provides tool discovery for large toolsets.

    Tools marked with `defer_loading=True` are hidden from the model until discovered.
    Auto-injected into every agent — zero overhead when no deferred tools exist.

    When the model supports native tool search (Anthropic BM25/regex, OpenAI Responses),
    discovery is handled by the provider: the deferred tools are sent with `defer_loading`
    on the wire and the provider exposes them once they've been discovered. Otherwise,
    discovery happens locally via a `search_tools` function that the model can call.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ToolSearch

    # Default: native search on supporting providers, local token matching elsewhere.
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch()])

    # Force a specific Anthropic native strategy; errors on providers that can't honor it.
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy='regex')])

    # Always run the local keyword-overlap algorithm, regardless of provider.
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy='keywords')])

    # Custom search function — used locally, and by provider-native "client-executed" modes when supported.
    def my_search(ctx, query, tools):
        return [t.name for t in tools if query.lower() in (t.description or '').lower()]

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy=my_search)])
    ```
    """

    strategy: ToolSearchStrategy[AgentDepsT] | None = None
    """The search strategy to use.

    * `None` (default): let Pydantic AI pick the best strategy for the current provider
      — native on supporting models (Anthropic BM25, OpenAI server-executed tool search),
      local keyword matching elsewhere. The choice may change in future versions.
    * `'keywords'`: always use the local keyword-overlap algorithm.
    * `'bm25'` / `'regex'`: force a specific Anthropic native strategy. Raises on
      providers that can't honor the choice (including OpenAI, which has no named
      native strategies).
    * Callable `(ctx, query, tools) -> names`: custom search function (sync or async).
      Used locally, and by the native "client-executed" surface on providers that support
      it (Anthropic custom tool-reference blocks, OpenAI `execution='client'`).
    """

    max_results: int = 10
    """Maximum number of matches returned by the local search algorithm."""

    tool_description: str | None = None
    """Custom description for the local `search_tools` function shown to the model."""

    parameter_description: str | None = None
    """Custom description for the `keywords` parameter on the local `search_tools` function."""

    _search_fn: ToolSearchFunc[AgentDepsT] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # `'keywords'` and a callable strategy both run their algorithm on our side and
        # both engage the provider's "client-executed" native mode where supported, so
        # they share a `_search_fn` that the toolset routes through `_run_search_fn`.
        # The named strategies `'bm25'` / `'regex'` only take effect server-side
        # (Anthropic) — no local implementation today — and `None` falls through to
        # the toolset's default keyword-overlap algorithm.
        if self.strategy == 'keywords':
            self._search_fn = keywords_search_fn
        elif callable(self.strategy):
            self._search_fn = self.strategy
        else:
            self._search_fn = None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ToolSearch'

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        # `'keywords'` and a callable strategy both register the `'custom'` builtin so
        # the provider's "client-executed" native mode engages where supported (cache
        # benefit on Anthropic and OpenAI), and silently fall back to the local
        # `search_tools` function tool elsewhere via `optional=True`. Same dispatch
        # path differs only in *which* algorithm runs as `_search_fn`.
        if self.strategy == 'keywords' or callable(self.strategy):
            return [ToolSearchTool(strategy='custom', optional=True)]

        # `None` means "pick the best native option available, otherwise fall back
        # locally" — `optional=True` so the swap silently falls back on unsupported
        # models.
        if self.strategy is None:
            return [ToolSearchTool(optional=True)]

        # Explicit named native strategy (`'bm25'` / `'regex'`). The user committed
        # to a specific algorithm, so `optional=False`: if the model can't honor it,
        # the request must error rather than silently substitute a different algorithm.
        #
        # Assumes no local implementation of bm25/regex exists — if we ever port either
        # to Python, the strategy should join the `'keywords'` branch above so models
        # without native support can still honor the choice via the local path.
        named: ToolSearchNativeStrategy = self.strategy
        return [ToolSearchTool(strategy=named, optional=False)]

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # For explicit named native strategies (`'bm25'` / `'regex'`) the
        # `ToolSearchTool` builtin is registered with `optional=False` (see
        # `get_builtin_tools` above), so `prepare_request` will raise on a model
        # without native support — there's no risk of the local `search_tools`
        # function silently substituting a different algorithm. Always wrap with
        # `ToolSearchToolset` so the corpus is exposed and `search_tools` keeps
        # the prompt prefix stable across discovery steps; the function tool gets
        # dropped on the wire whenever the builtin is supported via its
        # `unless_builtin='tool_search'` flag.
        return ToolSearchToolset(
            wrapped=toolset,
            search_fn=self._search_fn,
            max_results=self.max_results,
            tool_description=self.tool_description,
            parameter_description=self.parameter_description,
        )
