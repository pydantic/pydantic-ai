from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal, overload

from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from tavily import AsyncTavilyClient
except ImportError as _import_error:
    raise ImportError(
        'Please install `tavily-python` to use the Tavily search tool, '
        'you can use the `tavily` optional group — `pip install "pydantic-ai-slim[tavily]"`'
    ) from _import_error

__all__ = ('tavily_search_tool',)

_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from None in factory kwargs."""


class TavilySearchResult(TypedDict):
    """A Tavily search result.

    See [Tavily Search Endpoint documentation](https://docs.tavily.com/api-reference/endpoint/search)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result.."""
    content: str
    """A short description of the search result."""
    score: float
    """The relevance score of the search result."""


tavily_search_ta = TypeAdapter(list[TavilySearchResult])


@dataclass
class TavilySearchTool:
    """The Tavily search tool."""

    client: AsyncTavilyClient
    """The Tavily search client."""

    _: KW_ONLY

    max_results: int | None = None
    """The maximum number of results. If None, the Tavily default is used."""

    async def __call__(
        self,
        query: str,
        search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = 'basic',
        topic: Literal['general', 'news', 'finance'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]:
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.
            search_depth: The depth of the search.
            topic: The category of the search.
            time_range: The time range back from the current date to filter results.
            include_domains: List of domains to specifically include in the search results.
            exclude_domains: List of domains to specifically exclude from the search results.

        Returns:
            A list of search results from Tavily.
        """
        results: dict[str, Any] = await self.client.search(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
            query,
            search_depth=search_depth,
            topic=topic,
            time_range=time_range,  # pyright: ignore[reportArgumentType]
            max_results=self.max_results,  # pyright: ignore[reportArgumentType]
            include_domains=include_domains,  # pyright: ignore[reportArgumentType]
            exclude_domains=exclude_domains,  # pyright: ignore[reportArgumentType]
        )
        return tavily_search_ta.validate_python(results['results'])


@overload
def tavily_search_tool(
    api_key: str,
    *,
    max_results: int | None = None,
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = _UNSET,
    topic: Literal['general', 'news', 'finance'] = _UNSET,
    time_range: Literal['day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


@overload
def tavily_search_tool(
    *,
    client: AsyncTavilyClient,
    max_results: int | None = None,
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = _UNSET,
    topic: Literal['general', 'news', 'finance'] = _UNSET,
    time_range: Literal['day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


def tavily_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncTavilyClient | None = None,
    max_results: int | None = None,
    search_depth: Literal['basic', 'advanced', 'fast', 'ultra-fast'] = _UNSET,
    topic: Literal['general', 'news', 'finance'] = _UNSET,
    time_range: Literal['day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]:
    """Creates a Tavily search tool.

    `max_results` is always developer-controlled and does not appear in the LLM tool schema.
    Other parameters, when provided, are fixed for all searches and hidden from the LLM's
    tool schema. Parameters left unset remain available for the LLM to set per-call.

    Args:
        api_key: The Tavily API key. Required if `client` is not provided.

            You can get one by signing up at [https://app.tavily.com/home](https://app.tavily.com/home).
        client: An existing AsyncTavilyClient. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tool instances.
        max_results: The maximum number of results. If None, the Tavily default is used.
        search_depth: The depth of the search.
        topic: The category of the search.
        time_range: The time range back from the current date to filter results.
        include_domains: List of domains to specifically include in the search results.
        exclude_domains: List of domains to specifically exclude from the search results.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncTavilyClient(api_key)
    func = TavilySearchTool(client=client, max_results=max_results).__call__

    kwargs: dict[str, Any] = {}
    if search_depth is not _UNSET:
        kwargs['search_depth'] = search_depth
    if topic is not _UNSET:
        kwargs['topic'] = topic
    if time_range is not _UNSET:
        kwargs['time_range'] = time_range
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
        name='tavily_search',
        description='Searches Tavily for the given query and returns the results.',
    )
