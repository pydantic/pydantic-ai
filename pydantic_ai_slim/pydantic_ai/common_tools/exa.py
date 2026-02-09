"""Exa tools for Pydantic AI agents.

Provides web search, content retrieval, and AI-powered answer capabilities
using the Exa API, a neural search engine that finds high-quality, relevant
results across billions of web pages.
"""

from dataclasses import dataclass
from typing import Literal, overload

from typing_extensions import Any, TypedDict

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import Tool

try:
    from exa_py import AsyncExa
except ImportError as _import_error:
    raise ImportError(
        'Please install `exa-py` to use the Exa tools, '
        'you can use the `exa` optional group — `pip install "pydantic-ai-slim[exa]"`'
    ) from _import_error

__all__ = (
    'ExaToolset',
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

    max_characters: int | None
    """Maximum characters of text content per result, or None for no limit."""

    async def __call__(
        self,
        query: str,
        search_type: Literal['auto', 'keyword', 'neural', 'fast', 'deep'] = 'auto',
    ) -> list[ExaSearchResult]:
        """Searches Exa for the given query and returns the results with content.

        Args:
            query: The search query to execute with Exa.
            search_type: The type of search to perform. 'auto' automatically chooses
                the best search type, 'keyword' for exact matches, 'neural' for
                semantic search, 'fast' for speed-optimized search, 'deep' for
                comprehensive multi-query search.

        Returns:
            The search results with text content.
        """
        text_config: bool | dict[str, int] = {'maxCharacters': self.max_characters} if self.max_characters else True
        response = await self.client.search(  # pyright: ignore[reportUnknownMemberType]
            query,
            num_results=self.num_results,
            type=search_type,
            contents={'text': text_config},
        )

        return [
            ExaSearchResult(
                title=result.title or '',
                url=result.url,
                published_date=result.published_date,
                author=result.author,
                text=result.text or '',
            )
            for result in response.results
        ]


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
        response = await self.client.find_similar(  # pyright: ignore[reportUnknownMemberType]
            url,
            num_results=self.num_results,
            exclude_source_domain=exclude_source_domain,
            contents={'text': True},
        )

        return [
            ExaSearchResult(
                title=result.title or '',
                url=result.url,
                published_date=result.published_date,
                author=result.author,
                text=result.text or '',
            )
            for result in response.results
        ]


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
        response = await self.client.get_contents(urls, text=True)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        return [
            ExaContentResult(
                url=result.url,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                title=result.title or '',  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                text=result.text or '',  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                author=result.author,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                published_date=result.published_date,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            )
            for result in response.results  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        ]


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

        return ExaAnswerResult(
            answer=response.answer,  # pyright: ignore[reportUnknownMemberType,reportArgumentType,reportAttributeAccessIssue]
            citations=[
                {
                    'url': citation.url,  # pyright: ignore[reportUnknownMemberType]
                    'title': citation.title or '',  # pyright: ignore[reportUnknownMemberType]
                    'text': citation.text or '',  # pyright: ignore[reportUnknownMemberType]
                }
                for citation in response.citations  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportAttributeAccessIssue]
            ],
        )


@overload
def exa_search_tool(
    api_key: str,
    *,
    num_results: int = 5,
    max_characters: int | None = None,
) -> Tool[Any]: ...


@overload
def exa_search_tool(
    *,
    client: AsyncExa,
    num_results: int = 5,
    max_characters: int | None = None,
) -> Tool[Any]: ...


def exa_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
    num_results: int = 5,
    max_characters: int | None = None,
) -> Tool[Any]:
    """Creates an Exa search tool.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
        num_results: The number of results to return. Defaults to 5.
        max_characters: Maximum characters of text content per result. Use this to limit
            token usage. Defaults to None (no limit).
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncExa(api_key=api_key)
    return Tool[Any](
        ExaSearchTool(
            client=client,
            num_results=num_results,
            max_characters=max_characters,
        ).__call__,
        name='exa_search',
        description='Searches Exa for the given query and returns the results with content. Exa is a neural search engine that finds high-quality, relevant results.',
    )


@overload
def exa_find_similar_tool(
    api_key: str,
    *,
    num_results: int = 5,
) -> Tool[Any]: ...


@overload
def exa_find_similar_tool(
    *,
    client: AsyncExa,
    num_results: int = 5,
) -> Tool[Any]: ...


def exa_find_similar_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
    num_results: int = 5,
) -> Tool[Any]:
    """Creates an Exa find similar tool.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
        num_results: The number of similar results to return. Defaults to 5.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncExa(api_key=api_key)
    return Tool[Any](
        ExaFindSimilarTool(client=client, num_results=num_results).__call__,
        name='exa_find_similar',
        description='Finds web pages similar to a given URL. Useful for discovering related content, competitors, or alternative sources.',
    )


@overload
def exa_get_contents_tool(api_key: str) -> Tool[Any]: ...


@overload
def exa_get_contents_tool(*, client: AsyncExa) -> Tool[Any]: ...


def exa_get_contents_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
) -> Tool[Any]:
    """Creates an Exa get contents tool.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncExa(api_key=api_key)
    return Tool[Any](
        ExaGetContentsTool(client=client).__call__,
        name='exa_get_contents',
        description='Gets the full text content of specified URLs. Useful for reading articles, documentation, or any web page when you have the exact URL.',
    )


@overload
def exa_answer_tool(api_key: str) -> Tool[Any]: ...


@overload
def exa_answer_tool(*, client: AsyncExa) -> Tool[Any]: ...


def exa_answer_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
) -> Tool[Any]:
    """Creates an Exa answer tool.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
    """
    if client is None:
        if api_key is None:
            raise ValueError('Either api_key or client must be provided')
        client = AsyncExa(api_key=api_key)
    return Tool[Any](
        ExaAnswerTool(client=client).__call__,
        name='exa_answer',
        description='Generates an AI-powered answer to a question with citations from web sources. Returns a comprehensive answer backed by real sources.',
    )


class ExaToolset(FunctionToolset):
    """A toolset that provides Exa search tools with a shared client.

    This is more efficient than creating individual tools when using multiple
    Exa tools, as it shares a single API client across all tools.

    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai.common_tools.exa import ExaToolset

    toolset = ExaToolset(api_key='your-api-key')
    agent = Agent('openai:gpt-5.2', toolsets=[toolset])
    ```
    """

    def __init__(
        self,
        api_key: str,
        *,
        num_results: int = 5,
        max_characters: int | None = None,
        include_search: bool = True,
        include_find_similar: bool = True,
        include_get_contents: bool = True,
        include_answer: bool = True,
        id: str | None = None,
    ):
        """Creates an Exa toolset with a shared client.

        Args:
            api_key: The Exa API key.

                You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
            num_results: The number of results to return for search and find_similar. Defaults to 5.
            max_characters: Maximum characters of text content per result. Use this to limit
                token usage. Defaults to None (no limit).
            include_search: Whether to include the search tool. Defaults to True.
            include_find_similar: Whether to include the find_similar tool. Defaults to True.
            include_get_contents: Whether to include the get_contents tool. Defaults to True.
            include_answer: Whether to include the answer tool. Defaults to True.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        client = AsyncExa(api_key=api_key)
        tools: list[Tool[Any]] = []

        if include_search:
            tools.append(exa_search_tool(client=client, num_results=num_results, max_characters=max_characters))

        if include_find_similar:
            tools.append(exa_find_similar_tool(client=client, num_results=num_results))

        if include_get_contents:
            tools.append(exa_get_contents_tool(client=client))

        if include_answer:
            tools.append(exa_answer_tool(client=client))

        super().__init__(tools, id=id)
