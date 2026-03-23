from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import WebFetchTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, Tool

from .builtin_or_local import BuiltinOrLocalTool


@dataclass(init=False)
class WebFetch(BuiltinOrLocalTool[AgentDepsT]):
    """URL fetching capability.

    Uses the model's builtin URL fetching when available. No default local
    fallback — provide a custom `local` tool if needed.
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
        *,
        builtin: WebFetchTool | AgentBuiltinTool[AgentDepsT] | bool = True,
        local: Tool[Any] | Callable[..., Any] | Literal[False] | None = None,
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

    def _requires_builtin(self) -> bool:
        return self.allowed_domains is not None or self.blocked_domains is not None or self.max_uses is not None
