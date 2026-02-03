"""You.com search tool for Pydantic AI agents.

Provides web search capabilities using the You.com API, a search engine that
delivers unified results from web and news sources.
"""

from dataclasses import dataclass
from datetime import datetime

from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from youdotcom import You
    from youdotcom.models import (
        Contents,
        Country,
        Freshness,
        Language,
        LiveCrawl,
        LiveCrawlFormats,
        News,
        SafeSearch,
        Web,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `youdotcom` to use the You.com search tool, '
        'you can use the `you` optional group — `pip install "pydantic-ai-slim[you]"`'
    ) from _import_error

__all__ = ('you_search_tool',)


class YouSearchContents(TypedDict, total=False):
    """Contents of a page when livecrawl is enabled."""

    html: str
    """The HTML content of the page."""
    markdown: str
    """The Markdown content of the page."""


class YouSearchResult(TypedDict, total=False):
    """A You.com search result.

    See [You.com Search API documentation](https://docs.you.com/)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    description: str
    """A description or snippet of the search result."""
    snippets: list[str]
    """Text snippets from the search result, providing a preview of the content."""
    thumbnail_url: str
    """URL of the thumbnail image for the search result."""
    page_age: datetime
    """The age/publication date of the search result (ISO 8601 format)."""
    favicon_url: str
    """The URL of the favicon of the search result's domain."""
    contents: YouSearchContents
    """Contents of the page if livecrawl was enabled."""
    authors: list[str]
    """An array of authors of the search result."""


@dataclass
class YouSearchTool:
    """The You.com search tool."""

    client: You
    """The You.com client."""

    count: int | None
    """Default maximum number of results per section (web/news). Range: 1-100, default: 10."""

    offset: int | None
    """Pagination offset (0-9). Calculated in multiples of count. Not controllable by LLM."""

    freshness: Freshness | str | None
    """Default freshness of results: 'day', 'week', 'month', 'year', or date range 'YYYY-MM-DDtoYYYY-MM-DD'."""

    country: Country | None
    """Default country code for geographic focus of results."""

    language: Language | None
    """Default language of results (BCP 47 format). Default: 'EN'."""

    safesearch: SafeSearch | None
    """Default safe search filter: 'off', 'moderate', or 'strict'."""

    livecrawl: LiveCrawl | None
    """Default sections to livecrawl: 'web', 'news', or 'all'."""

    livecrawl_formats: LiveCrawlFormats | None
    """Default format for livecrawled content: 'html' or 'markdown'."""

    async def __call__(
        self,
        query: str,
        count: int | None = None,
        freshness: Freshness | str | None = None,
        country: Country | None = None,
        language: Language | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
    ) -> list[YouSearchResult]:
        """Searches You.com for the given query and returns the results.

        Args:
            query: The search query to execute with You.com.
            count: Maximum number of results per section (1-100). Only used if not configured at tool creation.
            freshness: Result freshness: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'.
                Only used if not configured at tool creation.
            country: Country code for geographic focus. Only used if not configured at tool creation.
            language: Language of results (BCP 47 format). Only used if not configured at tool creation.
            safesearch: Content moderation filter. Only used if not configured at tool creation.
            livecrawl: Sections to livecrawl for full page content. Only used if not configured at tool creation.
            livecrawl_formats: Format for livecrawled content. Only used if not configured at tool creation.

        Returns:
            A list of search results from You.com.
        """
        # Use configured defaults when set, otherwise allow LLM override
        response = await self.client.search.unified_async(
            query=query,
            count=self.count if self.count is not None else count,
            offset=self.offset,
            freshness=self.freshness if self.freshness is not None else freshness,
            country=self.country if self.country is not None else country,
            language=self.language if self.language is not None else language,
            safesearch=self.safesearch if self.safesearch is not None else safesearch,
            livecrawl=self.livecrawl if self.livecrawl is not None else livecrawl,
            livecrawl_formats=self.livecrawl_formats if self.livecrawl_formats is not None else livecrawl_formats,
        )

        results: list[YouSearchResult] = []

        if not response.results:
            return results

        # Process web results
        for result in response.results.web or []:
            search_result = self._build_result(result)
            # Web-specific fields
            if result.snippets:
                search_result['snippets'] = result.snippets
            if result.favicon_url:
                search_result['favicon_url'] = result.favicon_url
            if result.contents:
                search_result['contents'] = self._build_contents(result.contents)
            if result.authors:
                search_result['authors'] = result.authors
            results.append(search_result)

        # Process news results
        for result in response.results.news or []:
            search_result = self._build_result(result)
            # News can also have contents when livecrawl is enabled
            if hasattr(result, 'contents') and result.contents:
                search_result['contents'] = self._build_contents(result.contents)
            results.append(search_result)

        return results

    def _build_result(self, result: Web | News) -> YouSearchResult:
        """Build a YouSearchResult from a Web or News result, including only fields with values."""
        search_result: YouSearchResult = {}
        if result.title:
            search_result['title'] = result.title
        if result.url:
            search_result['url'] = result.url
        if result.description:
            search_result['description'] = result.description
        if result.thumbnail_url:
            search_result['thumbnail_url'] = result.thumbnail_url
        if result.page_age:
            search_result['page_age'] = result.page_age
        return search_result

    def _build_contents(self, contents: Contents) -> YouSearchContents:
        """Build a YouSearchContents from a SearchContents, including only fields with values."""
        result: YouSearchContents = {}
        if contents.html:
            result['html'] = contents.html
        if contents.markdown:
            result['markdown'] = contents.markdown
        return result


def you_search_tool(
    api_key: str,
    *,
    count: int | None = None,
    offset: int | None = None,
    freshness: Freshness | str | None = None,
    country: Country | None = None,
    language: Language | None = None,
    safesearch: SafeSearch | None = None,
    livecrawl: LiveCrawl | None = None,
    livecrawl_formats: LiveCrawlFormats | None = None,
) -> Tool[Any]:
    """Creates a You.com search tool.

    Args:
        api_key: The You.com API key.
            You can get one by signing up at [https://you.com/platform](https://you.com/platform).
        count: Default maximum number of results per section (1-100). Default: 10.
        offset: Pagination offset (0-9). Not controllable by LLM.
        freshness: Default result freshness: 'day', 'week', 'month', 'year', or 'YYYY-MM-DDtoYYYY-MM-DD'.
        country: Default country code for geographic focus of results.
        language: Default language of results (BCP 47 format). Default: 'EN'.
        safesearch: Default safe search filter: 'off', 'moderate', or 'strict'.
        livecrawl: Default sections to livecrawl: 'web', 'news', or 'all'.
        livecrawl_formats: Default format for livecrawled content: 'html' or 'markdown'.

    Returns:
        A [`Tool`][pydantic_ai.tools.Tool] that searches the web and news using You.com and can be used with Pydantic AI agents.
    """
    return Tool[Any](
        YouSearchTool(
            client=You(api_key_auth=api_key),
            count=count,
            offset=offset,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
        ).__call__,
        name='you_search',
        description='Leverages the You.com Search API to search the web and return web and/or news results.',
    )
