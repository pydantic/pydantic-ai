from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal

import httpx
from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

__all__ = ('BochaSearchTool', 'bocha_search_tool')

_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from None in factory kwargs."""

DEFAULT_BOCHA_SEARCH_API_URL = 'https://api.bochaai.com/v1/web-search'


class BochaSearchResult(TypedDict):
    """A Bocha web search result."""

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    content: str
    """A short description or summary of the search result."""
    site_name: str | None
    """The source site name, if available."""
    published_date: str | None
    """The published or crawled date, if available."""


bocha_search_ta = TypeAdapter(list[BochaSearchResult])


@dataclass
class BochaSearchTool:
    """The Bocha web search tool."""

    client: httpx.AsyncClient
    """The async HTTP client used to call Bocha."""

    _: KW_ONLY

    api_key: str
    """The Bocha API key."""

    api_url: str = DEFAULT_BOCHA_SEARCH_API_URL
    """The Bocha web search API URL."""

    count: int = 5
    """The maximum number of results to return."""

    async def __call__(
        self,
        query: str,
        freshness: Literal['noLimit', 'oneDay', 'oneWeek', 'oneMonth', 'oneYear'] = 'noLimit',
        summary: bool = True,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[BochaSearchResult]:
        """Searches Bocha for the given query and returns source-linked results.

        Args:
            query: The search query to execute with Bocha.
            freshness: The freshness window for search results.
            summary: Whether to request summarized snippets when supported.
            include_domains: List of domains to specifically include in the search results.
            exclude_domains: List of domains to specifically exclude from the search results.

        Returns:
            A list of search results from Bocha.
        """
        request_body: dict[str, Any] = {
            'query': query,
            'freshness': freshness,
            'summary': summary,
            'count': self.count,
        }
        if include_domains:
            request_body['include'] = '|'.join(include_domains)
        if exclude_domains:
            request_body['exclude'] = '|'.join(exclude_domains)

        response = await self.client.post(
            self.api_url,
            headers={'Authorization': f'Bearer {self.api_key}'},
            json=request_body,
        )
        response.raise_for_status()
        payload = response.json()
        web_pages = payload.get('data', {}).get('webPages', payload.get('webPages', {}))
        results = web_pages.get('value', [])
        return bocha_search_ta.validate_python([_map_result(result) for result in results])


def _map_result(result: dict[str, Any]) -> BochaSearchResult:
    return BochaSearchResult(
        title=result.get('name') or result.get('title') or '',
        url=result.get('url') or result.get('displayUrl') or '',
        content=result.get('summary') or result.get('snippet') or result.get('description') or '',
        site_name=result.get('siteName'),
        published_date=result.get('datePublished') or result.get('dateLastCrawled'),
    )


def bocha_search_tool(
    api_key: str | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    api_url: str = DEFAULT_BOCHA_SEARCH_API_URL,
    count: int = 5,
    freshness: Literal['noLimit', 'oneDay', 'oneWeek', 'oneMonth', 'oneYear'] = _UNSET,
    summary: bool = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]:
    """Creates a Bocha web search tool.

    `count` is developer-controlled and does not appear in the LLM tool schema.
    Other parameters, when provided, are fixed for all searches and hidden from the LLM's
    tool schema. Parameters left unset remain available for the LLM to set per-call.

    Args:
        api_key: The Bocha API key. Required.
        client: An existing HTTPX async client. This is useful for sharing a client across multiple tool instances.
        api_url: The Bocha web search API URL.
        count: The maximum number of results to return.
        freshness: The freshness window for search results.
        summary: Whether to request summarized snippets when supported.
        include_domains: List of domains to specifically include in the search results.
        exclude_domains: List of domains to specifically exclude from the search results.
    """
    if api_key is None:
        raise ValueError('api_key must be provided')
    if client is None:
        client = httpx.AsyncClient()
    func = BochaSearchTool(client=client, api_key=api_key, api_url=api_url, count=count).__call__

    kwargs: dict[str, Any] = {}
    if freshness is not _UNSET:
        kwargs['freshness'] = freshness
    if summary is not _UNSET:
        kwargs['summary'] = summary
    if include_domains is not _UNSET:
        kwargs['include_domains'] = include_domains
    if exclude_domains is not _UNSET:
        kwargs['exclude_domains'] = exclude_domains

    if kwargs:
        original = func
        func = partial(func, **kwargs)
        func.__name__ = original.__name__  # type: ignore[union-attr]
        func.__qualname__ = original.__qualname__
        # partial with keyword args only updates defaults, not removes params.
        # Set __signature__ explicitly to exclude bound params from the tool schema.
        orig_sig = signature(original)
        func.__signature__ = orig_sig.replace(  # type: ignore[attr-defined]
            parameters=[p for name, p in orig_sig.parameters.items() if name not in kwargs]
        )

    return Tool[Any](
        func,  # pyright: ignore[reportArgumentType]
        name='bocha_search',
        description='Searches Bocha for the given query and returns source-linked web results.',
    )
