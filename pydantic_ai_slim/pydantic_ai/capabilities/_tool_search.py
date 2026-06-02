"""Tool search capability: provider-adaptive discovery of deferred tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .._run_context import AgentDepsT
from ..native_tools._tool_search import (
    ToolSearchFunc,
    ToolSearchNativeStrategy,
    ToolSearchStrategy,
    ToolSearchTool,
)

# `ToolDefinition` is referenced via forward-string from `ToolSearchFunc`
# (defined in `native_tools/_tool_search.py`, where it can't be eagerly imported because
# of the `tools.py` ↔ `native_tools` circular). Import it eagerly here so dataclass-spec
# generation (`get_type_hints` on `ToolSearch.__init__`) can resolve the forward reference
# against this module's globals.
from ..tools import (
    AgentNativeTool,
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

    On providers that support a native "client-executed" surface (Anthropic, OpenAI),
    the discovery message is delivered append-only — prompt cache is preserved across
    discovery turns, so growing the message history with discovered-tool results does
    not invalidate the cached prefix.

    ```python
    from collections.abc import Sequence

    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.tools import ToolDefinition


    # Tools become deferred via `defer_loading=True`. They stay hidden from the model
    # until tool search discovers them.
    def get_weather(city: str) -> str:
        ...


    weather_tool = Tool(get_weather, defer_loading=True)

    # Default: native search on supporting providers, local keyword matching elsewhere.
    agent = Agent('anthropic:claude-sonnet-4-6', tools=[weather_tool], capabilities=[ToolSearch()])

    # Force a specific Anthropic native strategy; errors on providers that can't honor it.
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        tools=[weather_tool],
        capabilities=[ToolSearch(strategy='regex')],
    )

    # Always run the local keyword-overlap algorithm, regardless of provider.
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        tools=[weather_tool],
        capabilities=[ToolSearch(strategy='keywords')],
    )

    # Custom search function — used locally, and by provider-native "client-executed"
    # modes when supported.
    def my_search(
        ctx: RunContext[None], queries: Sequence[str], tools: Sequence[ToolDefinition]
    ) -> list[str]:
        return [
            t.name
            for t in tools
            if any(q.lower() in (t.description or '').lower() for q in queries)
        ]

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        tools=[weather_tool],
        capabilities=[ToolSearch(strategy=my_search)],
    )
    ```
    """

    strategy: ToolSearchStrategy[AgentDepsT] | None = None
    """The search strategy to use.

    * `None` (default): let Pydantic AI pick the best strategy for the current provider
      — native on supporting models (Anthropic BM25, OpenAI server-executed tool search),
      local keyword matching elsewhere. The choice may change in future versions.
    * `'keywords'`: always use the local keyword-overlap algorithm. Still prompt-cache
      compatible on providers that expose a "client-executed" native surface (Anthropic,
      OpenAI): the algorithm rides the same `defer_loading` wire as a custom callable,
      so the tool list stays stable across discovery rounds and the cached prefix is
      preserved.
    * `'bm25'` / `'regex'`: force a specific Anthropic native strategy. Raises on
      providers that can't honor the choice (including OpenAI, which has no named
      native strategies).
    * Callable `(ctx, queries, tools) -> names`: custom search function (sync or async).
      Used locally, and by the native "client-executed" surface on providers that support
      it (Anthropic custom tool-reference blocks, OpenAI `execution='client'`).
    """

    max_results: int = 10
    """Maximum number of matches returned by the local search algorithm."""

    tool_description: str | None = None
    """Custom description for the local `search_tools` function shown to the model."""

    parameter_description: str | None = None
    """Custom description for the `queries` parameter on the local `search_tools` function."""

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

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
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
        elif self.strategy is None:
            return [ToolSearchTool(optional=True)]
        # Explicit named native strategy (`'bm25'` / `'regex'`). The user committed
        # to a specific algorithm, so `optional=False`: if the model can't honor it,
        # the request must error rather than silently substitute a different algorithm.
        #
        # Assumes no local implementation of bm25/regex exists — if we ever port either
        # to Python, the strategy should join the `'keywords'` branch above so models
        # without native support can still honor the choice via the local path.
        else:
            named: ToolSearchNativeStrategy = self.strategy
            return [ToolSearchTool(strategy=named, optional=False)]

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # For explicit named native strategies (`'bm25'` / `'regex'`) the
        # `ToolSearchTool` builtin is registered with `optional=False` (see
        # `get_native_tools` above), so `prepare_request` will raise on a model
        # without native support. To make that raise actually fire — and to avoid
        # emitting a redundant `search_tools` function tool alongside the native
        # builtin on supported providers — the toolset must NOT emit the local
        # `search_tools` function at all in this mode. We signal that via
        # `enable_fallback=False` for the named-native strategies; `None`,
        # `'keywords'`, and callable strategies all have a real local
        # implementation and keep `search_tools` wired up.
        #
        # Always wrap with `ToolSearchToolset` so the deferred corpus is exposed
        # via the per-tool `with_native='tool_search'` flag — the wrapper toolset
        # is what teaches `_resolve_builtin_tool_swap` which function tools belong
        # to the tool-search corpus, regardless of whether `search_tools` itself is
        # emitted.
        return ToolSearchToolset(
            wrapped=toolset,
            search_fn=self._search_fn,
            max_results=self.max_results,
            tool_description=self.tool_description,
            parameter_description=self.parameter_description,
            enable_fallback=self.strategy not in ('bm25', 'regex'),
        )
