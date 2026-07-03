from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import WebFetchTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool


@dataclass(init=False)
class WebFetch(NativeOrLocalTool[AgentDepsT]):
    """URL fetching capability.

    Uses the model's native URL fetching and raises `UserError` on models that
    don't support it natively. Pass `local=True` to opt into a local fallback
    (requires the `web-fetch` optional group):

    ```bash
    pip install "pydantic-ai-slim[web-fetch]"
    ```
    """

    allowed_domains: list[str] | None
    """Only fetch from these domains. Enforced locally when native is unavailable."""

    blocked_domains: list[str] | None
    """Never fetch from these domains. Enforced locally when native is unavailable."""

    max_uses: int | None
    """Maximum number of fetches per run. Requires native support."""

    enable_citations: bool | None
    """Enable citations for fetched content. Native-only; ignored by local tools."""

    max_content_tokens: int | None
    """Maximum content length in tokens. Native-only; ignored by local tools."""

    def __init__(
        self,
        *,
        native: WebFetchTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[WebFetchTool | None] | WebFetchTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | bool | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_uses: int | None = None,
        enable_citations: bool | None = None,
        max_content_tokens: int | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
        self.local = local
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.max_uses = max_uses
        self.enable_citations = enable_citations
        self.max_content_tokens = max_content_tokens
        self.__post_init__()

    def _default_native(self) -> WebFetchTool:
        kwargs: dict[str, Any] = {}
        if self.allowed_domains is not None:
            kwargs['allowed_domains'] = self.allowed_domains
        if self.blocked_domains is not None:
            kwargs['blocked_domains'] = self.blocked_domains
        if self.max_uses is not None:
            kwargs['max_uses'] = self.max_uses
        if self.enable_citations is not None:
            kwargs['enable_citations'] = self.enable_citations
        if self.max_content_tokens is not None:
            kwargs['max_content_tokens'] = self.max_content_tokens
        return WebFetchTool(**kwargs)

    def _native_unique_id(self) -> str:
        return WebFetchTool.kind

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        if name is True:
            try:
                from pydantic_ai.common_tools.web_fetch import web_fetch_tool
            except ImportError as e:
                raise UserError(
                    'WebFetch(local=True) requires the `web-fetch` optional group — '
                    '`pip install "pydantic-ai-slim[web-fetch]"`.'
                ) from e
            return web_fetch_tool(
                allowed_domains=self.allowed_domains,
                blocked_domains=self.blocked_domains,
            )
        raise UserError(
            f'WebFetch(local={name!r}) is not a known strategy. '
            'Pass `local=True` for the default markdownify-based tool, or a Tool/callable directly.'
        )

    def _requires_native(self) -> bool:
        return self.max_uses is not None
