from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import WebSearchTool, WebSearchUserLocation
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

WebSearchLocalStrategy = Literal['duckduckgo']
"""Named local strategies accepted by `WebSearch.local`. `local=True` resolves to `'duckduckgo'`."""


@dataclass(init=False)
class WebSearch(BuiltinOrLocalTool[AgentDepsT]):
    """Web search capability.

    Uses the model's builtin web search when available, falling back to a local
    function tool (DuckDuckGo by default) when it isn't.
    """

    search_context_size: Literal['low', 'medium', 'high'] | None
    """Controls how much context is retrieved from the web. Builtin-only; ignored by local tools."""

    user_location: WebSearchUserLocation | None
    """Localize search results based on user location. Builtin-only; ignored by local tools."""

    blocked_domains: list[str] | None
    """Domains to exclude from results. Requires builtin support."""

    allowed_domains: list[str] | None
    """Only include results from these domains. Requires builtin support."""

    max_uses: int | None
    """Maximum number of web searches per run. Requires builtin support."""

    def __init__(
        self,
        *,
        builtin: WebSearchTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[WebSearchTool | None] | WebSearchTool | None]
        | bool = True,
        local: WebSearchLocalStrategy | Tool[AgentDepsT] | Callable[..., Any] | bool | None = None,
        search_context_size: Literal['low', 'medium', 'high'] | None = None,
        user_location: WebSearchUserLocation | None = None,
        blocked_domains: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_uses: int | None = None,
    ) -> None:
        self.builtin = builtin
        self.local = local
        self.search_context_size = search_context_size
        self.user_location = user_location
        self.blocked_domains = blocked_domains
        self.allowed_domains = allowed_domains
        self.max_uses = max_uses
        self.__post_init__()

    def _default_builtin(self) -> WebSearchTool:
        kwargs: dict[str, Any] = {}
        if self.search_context_size is not None:
            kwargs['search_context_size'] = self.search_context_size
        if self.user_location is not None:
            kwargs['user_location'] = self.user_location
        if self.blocked_domains is not None:
            kwargs['blocked_domains'] = self.blocked_domains
        if self.allowed_domains is not None:
            kwargs['allowed_domains'] = self.allowed_domains
        if self.max_uses is not None:
            kwargs['max_uses'] = self.max_uses
        return WebSearchTool(**kwargs)

    def _builtin_unique_id(self) -> str:
        return WebSearchTool.kind

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        try:
            from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
        except ImportError:
            # No DDG installed → the auto-fallback path can't run, so there's no deprecated
            # behavior to warn about. If the model also doesn't support the builtin, the user
            # will hit a clear UserError at request time with `local=…` migration hints.
            return None

        import warnings

        warnings.warn(
            'WebSearch will stop auto-selecting DuckDuckGo based on package availability in v2. '
            "To keep this fallback, pass `local='duckduckgo'` (or `local=True`). "
            'To disable the fallback, pass `local=False`.',
            DeprecationWarning,
            stacklevel=4,
        )
        return duckduckgo_search_tool()

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        # True → the default strategy (DuckDuckGo)
        strategy = 'duckduckgo' if name is True else name
        if strategy == 'duckduckgo':
            try:
                from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
            except ImportError as e:
                raise UserError(
                    "WebSearch(local='duckduckgo') requires the `duckduckgo` optional group — "
                    '`pip install "pydantic-ai-slim[duckduckgo]"`.'
                ) from e
            return duckduckgo_search_tool()
        raise UserError(
            f'WebSearch(local={name!r}) is not a known strategy. '
            "Supported: 'duckduckgo' (or `local=True`). Or pass a Tool/callable directly."
        )

    def _requires_builtin(self) -> bool:
        return self.blocked_domains is not None or self.allowed_domains is not None or self.max_uses is not None
