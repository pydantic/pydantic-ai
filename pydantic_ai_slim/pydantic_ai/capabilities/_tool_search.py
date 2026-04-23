"""Tool search capability: provider-adaptive discovery of deferred tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from .._run_context import AgentDepsT
from ..builtin_tools import ToolSearchFunc, ToolSearchNamedStrategy, ToolSearchStrategy, ToolSearchTool
from ..tools import AgentBuiltinTool
from ..toolsets import AbstractToolset
from ..toolsets._tool_search import ToolSearchToolset
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

    # Force a specific Anthropic strategy.
    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy='regex')])

    # Custom local search function.
    def my_search(query, tools):
        return [t.name for t in tools if query.lower() in (t.description or '').lower()]

    agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[ToolSearch(strategy=my_search)])
    ```
    """

    strategy: ToolSearchStrategy | None = None
    """The search strategy to use.

    * ``None`` (default): use the provider's default native search when supported (Anthropic
      BM25, OpenAI server-executed tool search) with a local token-matching fallback.
    * ``'bm25'`` / ``'regex'``: force a specific Anthropic native strategy. Falls back to
      local token matching on other providers.
    * Callable ``(query, tools) -> names``: custom local search function. Used as the local
      search algorithm on all providers.
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
        # (``'bm25'``, ``'regex'``) are provider-specific and only take effect on the
        # native path — the local fallback uses the default token-matching algorithm.
        self._search_fn = self.strategy if callable(self.strategy) else None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ToolSearch'

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        # Register `ToolSearchTool` for every strategy. Named strategies surface as the
        # provider's server-side native search (bm25/regex on Anthropic, server-executed
        # tool_search on OpenAI). A custom callable surfaces as the provider's
        # "client-executed" native mode (Anthropic: regular function tool with
        # tool_reference result formatting; OpenAI: `execution='client'`), with the
        # model's search calls routed back to our local `search_tools` function. On
        # models that don't support `ToolSearchTool`, the builtin is dropped as optional
        # and the capability's local path takes over.
        custom = callable(self.strategy)
        named_strategy = self.strategy if not custom else None
        return [ToolSearchTool(strategy=cast('ToolSearchNamedStrategy | None', named_strategy), custom=custom)]

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ToolSearchToolset(
            wrapped=toolset,
            search_fn=self._search_fn,
            max_results=self.max_results,
            tool_description=self.tool_description,
            search_guidance=self.search_guidance,
        )
