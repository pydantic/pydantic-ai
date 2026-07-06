"""Exa tools for Pydantic AI agents.

Provides web search, content retrieval, and AI-powered answer capabilities
using the Exa API, a neural search engine that finds high-quality, relevant
results across billions of web pages.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from typing_extensions import Any, TypedDict, assert_never

from pydantic_ai import FunctionToolset
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.tools import Tool

try:
    from exa_py import AsyncExa
except ImportError as _import_error:
    raise ImportError(
        'Please install `exa-py` to use the Exa tools, '
        'you can use the `exa` optional group — `pip install "pydantic-ai-slim[exa]"`'
    ) from _import_error

if TYPE_CHECKING:
    from exa_py.api import ContentsOptions, Result, SearchResponse

__all__ = (
    'ExaToolset',
    'exa_search_tool',
    'exa_find_similar_tool',
    'exa_get_contents_tool',
    'exa_answer_tool',
)

SearchType = Literal['auto', 'fast', 'deep']
"""The Exa search types exposed by these tools.

`auto` automatically picks the best strategy, `fast` is speed-optimized, and `deep` runs a
comprehensive multi-query search. See the [Exa Search API documentation](https://exa.ai/docs/reference/search)
for the full set of search types supported by the underlying API.
"""

ContentType = Literal['highlights', 'text']
"""The kind of content returned for each search result.

`highlights` (the default) returns token-efficient snippets relevant to the query; `text` returns
the full page text. See the [Exa Contents API documentation](https://exa.ai/docs/reference/get-contents)
for more information.
"""


class ExaSearchResult(TypedDict):
    """An Exa search result with content.

    See [Exa Search API documentation](https://exa.ai/docs/reference/search)
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
    """The content of the result: token-efficient highlight snippets by default, or the full page
    text when the tool is configured with `content='text'`."""


class ExaAnswerResult(TypedDict):
    """An Exa answer result with citations.

    See [Exa Answer API documentation](https://exa.ai/docs/reference/answer)
    for more information.
    """

    answer: str
    """The AI-generated answer to the query."""
    citations: list[dict[str, Any]]
    """Citations supporting the answer."""


class ExaContentResult(TypedDict):
    """Content retrieved from a URL.

    See [Exa Contents API documentation](https://exa.ai/docs/reference/get-contents)
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


def _make_client(api_key: str | None, client: AsyncExa | None) -> AsyncExa:
    """Returns the provided client, or builds one from `api_key` with integration attribution."""
    if client is not None:
        return client
    if api_key is None:
        raise ValueError('Either api_key or client must be provided')
    client = AsyncExa(api_key=api_key)
    client.headers['x-exa-integration'] = 'pydantic-ai'
    return client


@dataclass
class ExaSearchTool:
    """The Exa search tool."""

    client: AsyncExa
    """The Exa async client."""

    num_results: int = 5
    """The number of results to return."""

    search_type: SearchType = 'auto'
    """The search type to use."""

    content: ContentType = 'highlights'
    """The content returned for each result: token-efficient highlight snippets or the full page text."""

    max_characters: int | None = None
    """Maximum characters of content per result, or None for no limit."""

    include_domains: list[str] | None = None
    """Developer-configured domains to restrict results to, or None for no restriction."""

    exclude_domains: list[str] | None = None
    """Developer-configured domains to exclude from results, or None for no exclusion."""

    async def __call__(
        self,
        query: str,
    ) -> list[ExaSearchResult]:
        """Searches Exa for the given query and returns the results with content.

        Args:
            query: The search query to execute with Exa.

        Returns:
            The search results with content.
        """
        contents: ContentsOptions
        if self.content == 'highlights':
            contents = (
                {'highlights': {'max_characters': self.max_characters}}
                if self.max_characters is not None
                else {'highlights': True}
            )
        elif self.content == 'text':
            contents = (
                {'text': {'max_characters': self.max_characters}} if self.max_characters is not None else {'text': True}
            )
        else:
            assert_never(self.content)

        response = await self.client.search(
            query,
            num_results=self.num_results,
            type=self.search_type,
            contents=contents,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
        )

        results: list[ExaSearchResult] = []
        for result in response.results:
            if self.content == 'highlights':
                text = ' ... '.join(result.highlights or [])
            elif self.content == 'text':
                text = result.text or ''
            else:
                assert_never(self.content)
            results.append(
                ExaSearchResult(
                    title=result.title or '',
                    url=result.url,
                    published_date=result.published_date,
                    author=result.author,
                    text=text,
                )
            )
        return results


@dataclass
class ExaFindSimilarTool:
    """The Exa find similar tool.

    !!! warning "Deprecated"
        `find_similar` is deprecated in `exa-py` and will be removed in a future major version.
        Use [`ExaSearchTool`][pydantic_ai.common_tools.exa.ExaSearchTool] instead.
    """

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
        # `find_similar` still works but `exa-py` emits its own per-call `DeprecationWarning`; suppress it
        # since callers are already warned once when the tool is created (via `PydanticAIDeprecationWarning`).
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            response = await self.client.find_similar(  # pyright: ignore[reportDeprecated]
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
        # `AsyncExa.get_contents` is typed as `(urls, **kwargs)` with no return annotation (unlike the
        # sync `Exa.get_contents` overloads), so cast to the result type we know `text=True` produces.
        response = cast(
            'SearchResponse[Result]',
            await self.client.get_contents(urls, text=True),  # pyright: ignore[reportUnknownMemberType]
        )

        return [
            ExaContentResult(
                url=result.url,
                title=result.title or '',
                text=result.text or '',
                author=result.author,
                published_date=result.published_date,
            )
            for result in response.results
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
    search_type: SearchType = 'auto',
    content: ContentType = 'highlights',
    max_characters: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> Tool[Any]: ...


@overload
def exa_search_tool(
    *,
    client: AsyncExa,
    num_results: int = 5,
    search_type: SearchType = 'auto',
    content: ContentType = 'highlights',
    max_characters: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> Tool[Any]: ...


def exa_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
    num_results: int = 5,
    search_type: SearchType = 'auto',
    content: ContentType = 'highlights',
    max_characters: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> Tool[Any]:
    """Creates an Exa search tool.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
        num_results: The number of results to return. Defaults to 5.
        search_type: The search type to use. Defaults to `auto`.
        content: The content returned for each result: token-efficient `highlights` (the default)
            or the full page `text`.
        max_characters: Maximum characters of content per result. Use this to limit
            token usage. Defaults to None (no limit).
        include_domains: Domains to restrict all results to. Defaults to None (no restriction).
        exclude_domains: Domains to exclude all results from. Defaults to None (no exclusion).
    """
    client = _make_client(api_key, client)
    return Tool[Any](
        ExaSearchTool(
            client=client,
            num_results=num_results,
            search_type=search_type,
            content=content,
            max_characters=max_characters,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
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

    !!! warning "Deprecated"
        `find_similar` is deprecated in `exa-py` and will be removed in a future major version.
        Use [`exa_search_tool`][pydantic_ai.common_tools.exa.exa_search_tool] instead.

    Args:
        api_key: The Exa API key. Required if `client` is not provided.

            You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
        client: An existing AsyncExa client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tools.
        num_results: The number of similar results to return. Defaults to 5.
    """
    warnings.warn(
        '`exa_find_similar_tool` is deprecated as `find_similar` is deprecated in `exa-py`; '
        'use `exa_search_tool` instead.',
        PydanticAIDeprecationWarning,
        stacklevel=2,
    )
    client = _make_client(api_key, client)
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
    client = _make_client(api_key, client)
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
    client = _make_client(api_key, client)
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
        api_key: str | None = None,
        *,
        client: AsyncExa | None = None,
        num_results: int = 5,
        search_type: SearchType = 'auto',
        content: ContentType = 'highlights',
        max_characters: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_search: bool = True,
        include_find_similar: bool | None = None,
        include_get_contents: bool = True,
        include_answer: bool = True,
        id: str | None = None,
    ):
        """Creates an Exa toolset with a shared client.

        Args:
            api_key: The Exa API key. Required if `client` is not provided.

                You can get one by signing up at [https://dashboard.exa.ai](https://dashboard.exa.ai).
            client: An existing AsyncExa client. If provided, `api_key` is ignored.
            num_results: The number of results to return for search and find_similar. Defaults to 5.
            search_type: The search type to use for search. Defaults to `auto`.
            content: The content returned for each search result: token-efficient `highlights`
                (the default) or the full page `text`.
            max_characters: Maximum characters of content per result. Use this to limit
                token usage. Defaults to None (no limit).
            include_domains: Domains to restrict all search results to. Defaults to None (no restriction).
            exclude_domains: Domains to exclude all search results from. Defaults to None (no exclusion).
            include_search: Whether to include the search tool. Defaults to True.
            include_find_similar: Whether to include the deprecated find_similar tool. It is included
                by default for backward compatibility, but requesting it explicitly with `True` emits a
                deprecation warning; pass `False` to exclude it.
            include_get_contents: Whether to include the get_contents tool. Defaults to True.
            include_answer: Whether to include the answer tool. Defaults to True.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        client = _make_client(api_key, client)
        tools: list[Tool[Any]] = []

        if include_search:
            tools.append(
                exa_search_tool(
                    client=client,
                    num_results=num_results,
                    search_type=search_type,
                    content=content,
                    max_characters=max_characters,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                )
            )

        if include_find_similar or include_find_similar is None:
            with warnings.catch_warnings():
                if include_find_similar is None:
                    # Included by default for backward compatibility; only an explicit opt-in warns.
                    warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
                tools.append(exa_find_similar_tool(client=client, num_results=num_results))

        if include_get_contents:
            tools.append(exa_get_contents_tool(client=client))

        if include_answer:
            tools.append(exa_answer_tool(client=client))

        super().__init__(tools, id=id)
