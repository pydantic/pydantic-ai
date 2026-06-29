from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import WebSearchTool, WebSearchUserLocation
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

WebSearchLocalStrategy = Literal['duckduckgo']
"""Named local strategies accepted by `WebSearch.local`. `local=True` resolves to `'duckduckgo'`."""


@dataclass(init=False)
class WebSearch(NativeOrLocalTool[AgentDepsT]):
    """Web search capability.

    Uses the model's native web search and raises `UserError` on models that
    don't support it natively. Pass `local='duckduckgo'` (or `local=True`) to opt into a
    local DuckDuckGo fallback — requires the `duckduckgo` optional group:

    ```bash
    pip install "pydantic-ai-slim[duckduckgo]"
    ```

    `local=` also accepts any callable, `Tool`, or `AbstractToolset` for a custom fallback.
    """

    search_context_size: Literal['low', 'medium', 'high'] | None
    """Controls how much context is retrieved from the web. Native-only; ignored by local tools."""

    user_location: WebSearchUserLocation | None
    """Localize search results based on user location. Native-only; ignored by local tools."""

    blocked_domains: list[str] | None
    """Domains to exclude from results. Requires native support."""

    allowed_domains: list[str] | None
    """Only include results from these domains. Requires native support."""

    max_uses: int | None
    """Maximum number of web searches per run. Requires native support."""

    def __init__(
        self,
        *,
        native: WebSearchTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[WebSearchTool | None] | WebSearchTool | None]
        | bool = True,
        local: WebSearchLocalStrategy | Tool[AgentDepsT] | Callable[..., Any] | bool | None = None,
        search_context_size: Literal['low', 'medium', 'high'] | None = None,
        user_location: WebSearchUserLocation | None = None,
        blocked_domains: list[str] | None = None,
        allowed_domains: list[str] | None = None,
        max_uses: int | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
        self.local = local
        self.search_context_size = search_context_size
        self.user_location = user_location
        self.blocked_domains = blocked_domains
        self.allowed_domains = allowed_domains
        self.max_uses = max_uses
        self.__post_init__()

    def _default_native(self) -> WebSearchTool:
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

    def _native_unique_id(self) -> str:
        return WebSearchTool.kind

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

    def _requires_native(self) -> bool:
        return self.blocked_domains is not None or self.allowed_domains is not None or self.max_uses is not None
