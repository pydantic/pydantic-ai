"""Exa tools for Pydantic AI agents.

Provides web search, content retrieval, and AI-powered answer capabilities
using the Exa API, a neural search engine that finds high-quality, relevant
results across billions of web pages.
"""

import warnings
from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
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
    'ContentType',
    'ExaToolset',
    'SearchType',
    'exa_search_tool',
    'exa_find_similar_tool',
    'exa_get_contents_tool',
    'exa_answer_tool',
)

SearchType = Literal['auto', 'fast', 'instant', 'deep-lite', 'deep', 'deep-reasoning']
"""The Exa search types exposed by these tools.

See the [Exa Search API documentation](https://exa.ai/docs/reference/search) for details about
each search type.
"""

_ModelSearchType = Literal['auto', 'fast', 'instant', 'deep-lite', 'deep', 'deep-reasoning', 'keyword', 'neural']
"""Search types accepted from models, including legacy values for backward compatibility."""

ContentType = Literal['highlights', 'text']
"""The kind of content returned for each search result.

`highlights` returns token-efficient snippets relevant to the query; `text` (the default, for
backward compatibility) returns the full page text. See the
[Exa Contents API documentation](https://exa.ai/docs/reference/get-contents) for more information.
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
    """The content of the result: full page text by default, or token-efficient highlight snippets
    when the tool is configured with `content='highlights'`."""


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


_FIND_SIMILAR_DEPRECATION_MESSAGE = (
    '`exa_find_similar_tool` is deprecated as `find_similar` is deprecated in `exa-py`; use `exa_search_tool` instead.'
)


def _warn_find_similar_deprecated(*, stacklevel: int) -> None:
    warnings.warn(
        _FIND_SIMILAR_DEPRECATION_MESSAGE,
        PydanticAIDeprecationWarning,
        stacklevel=stacklevel,
    )


@dataclass
class ExaSearchTool:
    """The Exa search tool."""

    client: AsyncExa
    """The Exa async client."""

    num_results: int
    """The number of results to return."""

    max_characters: int | None
    """Maximum characters requested from Exa for each result, or None for no limit.

    Separators added between multiple highlights are not included in Exa's limit.
    """

    _: KW_ONLY

    content: ContentType = 'text'
    """The content returned for each result: token-efficient highlight snippets or the full page text."""

    include_domains: list[str] | None = None
    """Developer-configured domains to restrict results to, or None for no restriction."""

    exclude_domains: list[str] | None = None
    """Developer-configured domains to exclude from results, or None for no exclusion."""

    async def __call__(
        self,
        query: str,
        search_type: _ModelSearchType = 'auto',
    ) -> list[ExaSearchResult]:
        """Searches Exa for the given query and returns the results with content.

        Args:
            query: The search query to execute with Exa.
            search_type: The search type to use. Legacy `keyword` and `neural` values are retained
                for backward compatibility; prefer another value for new code.

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
            type=search_type,
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

    def __post_init__(self) -> None:
        _warn_find_similar_deprecated(stacklevel=4)

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
        # `find_similar` still works but `exa-py` emits its own per-call `DeprecationWarning` synchronously; suppress
        # it since callers are already warned once when the tool is created (via `PydanticAIDeprecationWarning`)
        # and await outside the catch block.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='find_similar', category=DeprecationWarning)
            coro = self.client.find_similar(  # pyright: ignore[reportDeprecated]
                url,
                num_results=self.num_results,
                exclude_source_domain=exclude_source_domain,
                contents={'text': True},
            )
        response = await coro

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


class _ExaFindSimilarTool(ExaFindSimilarTool):
    """Internal variant used after the public factory or toolset has warned."""

    def __post_init__(self) -> None:
        pass


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
    search_type: SearchType | None = None,
    content: ContentType = 'text',
    max_characters: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> Tool[Any]: ...


@overload
def exa_search_tool(
    *,
    client: AsyncExa,
    num_results: int = 5,
    search_type: SearchType | None = None,
    content: ContentType = 'text',
    max_characters: int | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> Tool[Any]: ...


def exa_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncExa | None = None,
    num_results: int = 5,
    search_type: SearchType | None = None,
    content: ContentType = 'text',
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
        search_type: The search type to use. When provided, it is fixed for all searches and hidden
            from the model's tool schema. When omitted, the model can choose a search type per call.
        content: The content returned for each result: full page `text` (the default, for backward
            compatibility) or token-efficient `highlights`.
        max_characters: Maximum characters requested from Exa for each result. Multiple highlights
            are joined with ` ... ` separators, which are not included in Exa's limit. Defaults to None.
        include_domains: Domains to restrict all results to. Defaults to None (no restriction).
        exclude_domains: Domains to exclude all results from. Defaults to None (no exclusion).
    """
    client = _make_client(api_key, client)
    func = ExaSearchTool(
        client=client,
        num_results=num_results,
        max_characters=max_characters,
        content=content,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
    ).__call__
    if search_type is not None:
        original = func
        func = partial(func, search_type=search_type)
        func.__name__ = original.__name__  # type: ignore[union-attr]
        func.__qualname__ = original.__qualname__
        # A keyword-only partial updates the default but does not remove the parameter.
        # Remove it explicitly so developer-owned configuration stays out of the tool schema.
        orig_sig = signature(original)
        func.__signature__ = orig_sig.replace(  # type: ignore[attr-defined]
            parameters=[parameter for name, parameter in orig_sig.parameters.items() if name != 'search_type']
        )

    return Tool[Any](
        func,  # pyright: ignore[reportArgumentType]
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
    _warn_find_similar_deprecated(stacklevel=3)
    client = _make_client(api_key, client)

    return _make_find_similar_tool(client, num_results)


def _make_find_similar_tool(client: AsyncExa, num_results: int) -> Tool[Any]:
    return Tool[Any](
        _ExaFindSimilarTool(client=client, num_results=num_results).__call__,
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

    # `find_similar` is deprecated; opt out to avoid the deprecation warning.
    toolset = ExaToolset(
        api_key='your-api-key',
        search_type='auto',
        content='highlights',
        include_find_similar=False,
    )
    agent = Agent('openai:gpt-5.2', toolsets=[toolset])
    ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: AsyncExa | None = None,
        num_results: int = 5,
        search_type: SearchType | None = None,
        content: ContentType = 'text',
        max_characters: int | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_search: bool = True,
        include_find_similar: bool = True,
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
            search_type: The search type to use. When provided, it is fixed for all searches and hidden
                from the model's tool schema. When omitted, the model can choose a search type per call.
            content: The content returned for each search result: full page `text` (the default, for
                backward compatibility) or token-efficient `highlights`.
            max_characters: Maximum characters requested from Exa for each result. Multiple highlights
                are joined with ` ... ` separators, which are not included in Exa's limit. Defaults to None.
            include_domains: Domains to restrict all search results to. Defaults to None (no restriction).
            exclude_domains: Domains to exclude all search results from. Defaults to None (no exclusion).
            include_search: Whether to include the search tool. Defaults to True.
            include_find_similar: Whether to include the deprecated find_similar tool. Included by
                default (emitting a `PydanticAIDeprecationWarning`); pass `False` to exclude it.
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

        if include_find_similar:
            _warn_find_similar_deprecated(stacklevel=3)
            tools.append(_make_find_similar_tool(client, num_results))

        if include_get_contents:
            tools.append(exa_get_contents_tool(client=client))

        if include_answer:
            tools.append(exa_answer_tool(client=client))

        super().__init__(tools, id=id)
