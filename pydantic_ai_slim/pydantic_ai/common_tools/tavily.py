from dataclasses import dataclass
from typing import Literal

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
    include_domains: list[str] | None = None
    """Default list of domains to specifically include in search results."""
    exclude_domains: list[str] | None = None
    """Default list of domains to specifically exclude from search results."""

    async def __call__(
        self,
        query: str,
        search_deep: Literal['basic', 'advanced'] = 'basic',
        topic: Literal['general', 'news'] = 'general',
        time_range: Literal['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[TavilySearchResult]:
        """Searches Tavily for the given query and returns the results.

        Args:
            query: The search query to execute with Tavily.
            search_deep: The depth of the search.
            topic: The category of the search.
            time_range: The time range back from the current date to filter results.
            include_domains: List of domains to specifically include in the search results.
                Overrides the default include_domains if provided.
            exclude_domains: List of domains to specifically exclude from the search results.
                Overrides the default exclude_domains if provided.

        Returns:
            A list of search results from Tavily.
        """
        # Use call-time values if provided, otherwise fall back to instance defaults
        effective_include = include_domains if include_domains is not None else self.include_domains
        effective_exclude = exclude_domains if exclude_domains is not None else self.exclude_domains

        results = await self.client.search(
            query,
            search_depth=search_deep,
            topic=topic,
            time_range=time_range,
            include_domains=effective_include,
            exclude_domains=effective_exclude,
        )  # type: ignore[reportUnknownMemberType]
        return tavily_search_ta.validate_python(results['results'])


def tavily_search_tool(
    api_key: str,
    *,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
):
    """Creates a Tavily search tool.

    Args:
        api_key: The Tavily API key.

            You can get one by signing up at [https://app.tavily.com/home](https://app.tavily.com/home).
        include_domains: Default list of domains to specifically include in search results.
            For example, `['arxiv.org', 'github.com']` will only return results from these domains.
        exclude_domains: Default list of domains to specifically exclude from search results.
            For example, `['medium.com']` will filter out results from these domains.
    """
    return Tool[Any](
        TavilySearchTool(
            client=AsyncTavilyClient(api_key),
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        ).__call__,
        name='tavily_search',
        description='Searches Tavily for the given query and returns the results.',
    )
