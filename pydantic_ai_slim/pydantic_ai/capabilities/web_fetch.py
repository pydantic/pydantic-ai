from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import WebFetchTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool


@dataclass(init=False)
class WebFetch(BuiltinOrLocalTool[AgentDepsT]):
    """URL fetching capability.

    Uses the model's builtin URL fetching when available, falling back to a local
    function tool (markdownify-based fetch by default) when it isn't.

    The local fallback requires the `web-fetch` optional group::

        pip install "pydantic-ai-slim[web-fetch]"
    """

    allowed_domains: list[str] | None
    """Only fetch from these domains. Enforced locally when builtin is unavailable."""

    blocked_domains: list[str] | None
    """Never fetch from these domains. Enforced locally when builtin is unavailable."""

    max_uses: int | None
    """Maximum number of fetches per run. Requires builtin support."""

    enable_citations: bool | None
    """Enable citations for fetched content. Builtin-only; ignored by local tools."""

    max_content_tokens: int | None
    """Maximum content length in tokens. Builtin-only; ignored by local tools."""

    def __init__(
        self,
        *,
        builtin: WebFetchTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[WebFetchTool | None] | WebFetchTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        max_uses: int | None = None,
        enable_citations: bool | None = None,
        max_content_tokens: int | None = None,
    ) -> None:
        self.builtin = builtin
        self.local = local
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.max_uses = max_uses
        self.enable_citations = enable_citations
        self.max_content_tokens = max_content_tokens
        self.__post_init__()

    def _default_builtin(self) -> WebFetchTool:
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

    def _builtin_unique_id(self) -> str:
        return WebFetchTool.kind

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        try:
            from pydantic_ai.common_tools.web_fetch import web_fetch_tool

            return web_fetch_tool(
                allowed_domains=self.allowed_domains,
                blocked_domains=self.blocked_domains,
            )
        except ImportError:
            import warnings

            warnings.warn(
                'WebFetch local fallback requires the `web-fetch` optional group — '
                '`pip install "pydantic-ai-slim[web-fetch]"`. '
                'Without it, WebFetch only works with models that support it natively.',
                UserWarning,
                stacklevel=2,
            )
            return None

    def _requires_builtin(self) -> bool:
        return self.max_uses is not None
