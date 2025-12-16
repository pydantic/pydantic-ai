"""Exa tools for Pydantic AI agents.

Provides web search, content retrieval, and AI-powered answer capabilities
using the Exa API, a neural search engine that finds high-quality, relevant
results across billions of web pages.
"""

from dataclasses import dataclass
from typing import Literal

from typing_extensions import Any, TypedDict

from pydantic_ai.tools import Tool

try:
    from exa_py import AsyncExa
except ImportError as _import_error:
    raise ImportError(
        'Please install `exa-py` to use the Exa tools, '
        'you can use the `exa` optional group — `pip install "pydantic-ai-slim[exa]"`'
    ) from _import_error

__all__ = (
    'exa_search_tool',
    'exa_find_similar_tool',
    'exa_get_contents_tool',
    'exa_answer_tool',
)


class ExaSearchResult(TypedDict):
    """An Exa search result with content.

    See [Exa Search API documentation](https://docs.exa.ai/reference/search)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    published_date: str | None
    """The published date of the content, if available."""
    author: str | None
    """The author of the content, if available."""
    text: str
    """The text content of the search result."""


class ExaAnswerResult(TypedDict):
    """An Exa answer result with citations.

    See [Exa Answer API documentation](https://docs.exa.ai/reference/answer)
    for more information.
    """

    answer: str
    """The AI-generated answer to the query."""
    citations: list[dict[str, Any]]
    """Citations supporting the answer."""


class ExaContentResult(TypedDict):
    """Content retrieved from a URL.

    See [Exa Contents API documentation](https://docs.exa.ai/reference/get-contents)
    for more information.
    """

    url: str
    """The URL of the content."""
    title: str
    """The title of the page."""
    text: str
    """The text content of the page."""
    author: str | None
    """The author of the content, if available."""
    published_date: str | None
    """The published date of the content, if available."""


@dataclass
class ExaSearchTool:
    """The Exa search tool."""

    client: AsyncExa
    """The Exa async client."""

    num_results: int
    """The number of results to return."""

    async def __call__(
        self,
        query: str,
        search_type: Literal['auto', 'keyword', 'neural'] = 'auto',
    ) -> list[ExaSearchResult]:
        """Searches Exa for the given query and returns the results with content.

        Args:
            query: The search query to execute with Exa.
            search_type: The type of search to perform. 'auto' automatically chooses
                the best search type, 'keyword' for exact matches, 'neural' for
                semantic search.

        Returns:
            The search results with text content.
        """
        response = await self.client.search(
            query,
            num_results=self.num_results,
            type=search_type,
            contents={'text': True},
        )

        results: list[ExaSearchResult] = []
        for result in response.results:
            results.append(
                ExaSearchResult(
                    title=result.title or '',
                    url=result.url,
                    published_date=result.published_date,
                    author=result.author,
                    text=result.text or '',
                )
            )
        return results


@dataclass
class ExaFindSimilarTool:
    """The Exa find similar tool."""

    client: AsyncExa
    """The Exa async client."""

    num_results: int
    """The number of results to return."""

    async def __call__(
        self,
        url: str,
        exclude_source_domain: bool = True,
    ) -> list[ExaSearchResult]:
        """Finds pages similar to the given URL and returns them with content.

        Args:
            url: The URL to find similar pages for.
            exclude_source_domain: Whether to exclude results from the same domain
                as the input URL. Defaults to True.

        Returns:
            Similar pages with text content.
        """
        response = await self.client.find_similar(
            url,
            num_results=self.num_results,
            exclude_source_domain=exclude_source_domain,
            contents={'text': True},
        )

        results: list[ExaSearchResult] = []
        for result in response.results:
            results.append(
                ExaSearchResult(
                    title=result.title or '',
                    url=result.url,
                    published_date=result.published_date,
                    author=result.author,
                    text=result.text or '',
                )
            )
        return results


@dataclass
class ExaGetContentsTool:
    """The Exa get contents tool."""

    client: AsyncExa
    """The Exa async client."""

    async def __call__(
        self,
        urls: list[str],
    ) -> list[ExaContentResult]:
        """Gets the content of the specified URLs.

        Args:
            urls: A list of URLs to get content for.

        Returns:
            The content of each URL.
        """
        response = await self.client.get_contents(urls, text=True)

        results: list[ExaContentResult] = []
        for result in response.results:
            results.append(
                ExaContentResult(
                    url=result.url,
                    title=result.title or '',
                    text=result.text or '',
                    author=result.author,
                    published_date=result.published_date,
                )
            )
        return results


@dataclass
class ExaAnswerTool:
    """The Exa answer tool."""

    client: AsyncExa
    """The Exa async client."""

    async def __call__(
        self,
        query: str,
    ) -> ExaAnswerResult:
        """Generates an AI-powered answer to the query with citations.

        Args:
            query: The question to answer.

        Returns:
            An answer with supporting citations from web sources.
        """
        response = await self.client.answer(query, text=True)

        citations = []
        for citation in response.citations:
            citations.append({
                'url': citation.url,
                'title': citation.title or '',
                'text': citation.text or '',
            })

        return ExaAnswerResult(
            answer=response.answer,
            citations=citations,
        )


def exa_search_tool(api_key: str, num_results: int = 5):
    """Creates an Exa search tool.

    Args:
        api_key: The Exa API key.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        num_results: The number of results to return. Defaults to 5.
    """
    return Tool[Any](
        ExaSearchTool(client=AsyncExa(api_key=api_key), num_results=num_results).__call__,
        name='exa_search',
        description='Searches Exa for the given query and returns the results with content. Exa is a neural search engine that finds high-quality, relevant results.',
    )


def exa_find_similar_tool(api_key: str, num_results: int = 5):
    """Creates an Exa find similar tool.

    Args:
        api_key: The Exa API key.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        num_results: The number of similar results to return. Defaults to 5.
    """
    return Tool[Any](
        ExaFindSimilarTool(client=AsyncExa(api_key=api_key), num_results=num_results).__call__,
        name='exa_find_similar',
        description='Finds web pages similar to a given URL. Useful for discovering related content, competitors, or alternative sources.',
    )


def exa_get_contents_tool(api_key: str):
    """Creates an Exa get contents tool.

    Args:
        api_key: The Exa API key.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
    """
    return Tool[Any](
        ExaGetContentsTool(client=AsyncExa(api_key=api_key)).__call__,
        name='exa_get_contents',
        description='Gets the full text content of specified URLs. Useful for reading articles, documentation, or any web page when you have the exact URL.',
    )


def exa_answer_tool(api_key: str):
    """Creates an Exa answer tool.

    Args:
        api_key: The Exa API key.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
    """
    return Tool[Any](
        ExaAnswerTool(client=AsyncExa(api_key=api_key)).__call__,
        name='exa_answer',
        description='Generates an AI-powered answer to a question with citations from web sources. Returns a comprehensive answer backed by real sources.',
    )
