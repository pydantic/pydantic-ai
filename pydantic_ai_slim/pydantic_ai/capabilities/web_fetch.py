from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import WebFetchTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_tool import BuiltinToolCapability


async def _web_fetch_impl(url: str) -> str:
    """Fetch the text content of a URL with SSRF protection."""
    # TODO: Convert HTML to markdown for better LLM consumption (needs html-to-markdown package)
    from pydantic_ai._ssrf import safe_download

    response = await safe_download(url)
    return response.text


@dataclass(init=False)
class WebFetch(BuiltinToolCapability[AgentDepsT]):
    """URL fetching capability.

    Uses the model's builtin URL fetching when available, falling back to a local
    httpx-based fetcher with SSRF protection otherwise.
    """

    allowed_domains: list[str] | None
    """Only fetch from these domains. Requires builtin support."""

    blocked_domains: list[str] | None
    """Never fetch from these domains. Requires builtin support."""

    max_uses: int | None
    """Maximum number of fetches per run. Requires builtin support."""

    enable_citations: bool | None
    """Enable citations for fetched content. Builtin-only; ignored by local tools."""

    max_content_tokens: int | None
    """Maximum content length in tokens. Builtin-only; ignored by local tools."""

    def __init__(
        self,
        builtin: WebFetchTool | AgentBuiltinTool[AgentDepsT] | bool = True,
        local: Tool[Any] | Callable[..., Any] | Literal[False] | None = None,
        *,
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

    def _default_local(self) -> Tool[Any] | AbstractToolset[Any] | None:
        return Tool(_web_fetch_impl, name='web_fetch', description='Fetch the text content of a URL.')

    def _requires_builtin(self) -> bool:
        return bool(self.allowed_domains or self.blocked_domains or self.max_uses)
