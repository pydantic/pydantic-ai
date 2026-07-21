"""Exa toolset -- gives agents web search, page retrieval, and deep search backed by the Exa API."""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Concatenate, Literal, ParamSpec, Protocol, TypedDict, TypeVar

import httpx
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

try:
    from exa_py import AsyncExa
    from exa_py.api import (
        ContentsOptions,
        DeepOutputSchema,
        DeepSearchOutput,
        DeepTextOutputSchema,
        Result,
        SearchResponse,
        SearchType,
        TextContentsOptions,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'exa-py is required for ExaSearch. Install it with: pip install "pydantic-ai-harness[exa]"'
    ) from _import_error

EXA_MAX_NUM_RESULTS = 100
"""Largest `num_results` the Exa search API accepts."""

EXA_MAX_PAGE_TEXT_CHARS = 10_000
"""Largest per-page text budget the Exa contents API accepts."""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')


class ExaSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def _source_list(sources: Mapping[str, str | None]) -> list[ExaSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{'url': url, 'title': title} for url, title in sources.items()]


# exa-py raises a bare ValueError for any non-2xx response, embedding the HTTP
# status in the message. 401/403 mean a bad or missing API key -- configuration
# the model cannot correct, so those propagate instead of retrying.
_AUTH_STATUS_RE = re.compile(r'status code (401|403)\b')


class ExaClient(Protocol):
    """The subset of the `exa_py.AsyncExa` API that `ExaSearchToolset` calls.

    Any object with these two methods can back the toolset. Pass one via
    `ExaSearch.client` to configure authentication or the base URL explicitly,
    or to substitute a fake in tests.

    The parameter types mirror `AsyncExa`'s own signatures (including the
    `ContentsOptions` payload `exa_py` types `search` with), so a real
    `AsyncExa` instance satisfies the protocol as-is.
    """

    async def search(
        self,
        query: str,
        *,
        contents: ContentsOptions | Literal[False],
        num_results: int | None = None,
        type: SearchType | None = None,
        output_schema: DeepOutputSchema | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse[Result]:
        """Search the web and return results, optionally with page contents or a synthesized output."""
        ...  # pragma: no cover

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        """Retrieve the contents of a specific URL."""
        ...  # pragma: no cover


def _default_client() -> ExaClient:
    """Build an `AsyncExa` client from the `EXA_API_KEY` environment variable."""
    try:
        return AsyncExa()
    except ValueError as error:
        raise UserError(
            'ExaSearch needs an Exa API key: set the EXA_API_KEY environment variable, '
            'or pass a configured client, e.g. ExaSearch(client=AsyncExa(api_key=...)).'
        ) from error


def _recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient Exa API failures into `ModelRetry`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run. exa-py surfaces every non-2xx
    response as a bare `ValueError` (message embeds the status code) and network
    failures as `httpx.HTTPError`. Rate limits, transient 5xx, and rejected
    parameters are things a model can recover from (wait, rephrase, adjust), so
    they become retries; 401/403 auth failures are configuration errors and
    propagate.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except httpx.HTTPError as error:
            raise ModelRetry(f'Exa request failed: {error}') from error
        except ValueError as error:
            if _AUTH_STATUS_RE.search(str(error)):
                raise
            raise ModelRetry(f'Exa request failed: {error}') from error

    return wrapper


class ExaSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Exa search API.

    `web_search` surveys the web and returns results with their most relevant
    excerpts, and `get_page` retrieves the text of one specific URL. With
    `include_deep_search=True`, `deep_search` runs Exa's multi-step deep search
    and returns a synthesized, cited answer in one call.

    Each tool returns a `ToolReturn` whose `return_value` is the text the
    model sees and whose `metadata['sources']` lists the result URLs and
    titles (`ExaSource` dicts), so applications can render citations from
    `ToolReturnPart.metadata` without parsing the text.

    `get_page` text is capped at `max_text_chars` characters; one character of
    headroom above the cap is requested from Exa so local truncation can detect
    a longer page and append a marker (at the API ceiling of
    `EXA_MAX_PAGE_TEXT_CHARS` no headroom exists, so the marker cannot fire).
    The `web_search` result count is bounded the same way: `num_results` is
    requested from Exa and re-applied to the response. Bounds are validated by
    `ExaSearch` at construction.
    """

    def __init__(
        self,
        *,
        client: ExaClient | None,
        num_results: int,
        max_text_chars: int,
        include_deep_search: bool,
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        text_summary: bool | str = False,
    ) -> None:
        super().__init__()
        self._client = client if client is not None else _default_client()
        self._num_results = num_results
        self._max_text_chars = max_text_chars
        self._include_domains = list(include_domains) if include_domains else None
        self._exclude_domains = list(exclude_domains) if exclude_domains else None
        self._text_summary = text_summary
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')
        if include_deep_search:
            self.add_function(self.deep_search, name='deep_search')

    @_recoverable
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with its most relevant excerpts.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and excerpts.
        """
        response = await self._client.search(
            query,
            contents={'highlights': True},
            num_results=self._num_results,
            output_schema=self._summary_schema(),
            include_domains=self._include_domains,
            exclude_domains=self._exclude_domains,
        )
        if not response.results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})
        results = response.results[: self._num_results]
        sources = _source_list({result.url: result.title for result in results})
        sections = [_format_result(result, _highlights_body(result)) for result in results]
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        found = f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}'
        summary = response.output.content if response.output is not None else None
        if not isinstance(summary, str) or not summary:
            return ToolReturn(found, metadata={'sources': sources})
        return ToolReturn(f'Summary: {summary}\n\n{found}', metadata={'sources': sources})

    def _summary_schema(self) -> DeepTextOutputSchema | None:
        """The text output schema `web_search` requests when `text_summary` is on, else `None`."""
        if self._text_summary is False:
            return None
        if isinstance(self._text_summary, str):
            return {'type': 'text', 'description': self._text_summary}
        return {'type': 'text'}

    @_recoverable
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve the full text of a specific URL.

        Use it to read a promising URL from `web_search` results in full, or a
        URL the user provided.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's title, URL, and text content.
        """
        requested = min(self._max_text_chars + 1, EXA_MAX_PAGE_TEXT_CHARS)
        response = await self._client.get_contents(url, text={'max_characters': requested})
        first = response.results[0] if response.results else None
        if first is None or not first.text:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        return ToolReturn(
            _format_result(first, _truncate(first.text, self._max_text_chars)),
            metadata={'sources': _source_list({first.url: first.title})},
        )

    @_recoverable
    async def deep_search(self, question: str) -> ToolReturn[str]:
        """Run Exa's multi-step deep search and return a synthesized answer with its sources.

        A full research pass in a single call; suited to questions that need
        synthesis across many sources rather than a quick survey.

        Args:
            question: The research question to answer.

        Returns:
            The synthesized answer, followed by the sources it drew on.
        """
        response = await self._client.search(
            question,
            contents=False,
            type='deep',
            output_schema={'type': 'text'},
            include_domains=self._include_domains,
            exclude_domains=self._exclude_domains,
        )
        output = response.output
        if output is None or not output.content:
            raise ModelRetry(
                f'Deep search returned no answer for {question!r}. Rephrase the question, or use web_search.'
            )
        content = output.content
        answer = content if isinstance(content, str) else json.dumps(content)
        sources = _grounding_sources(output)
        if not sources:
            sources = {result.url: result.title or '' for result in response.results}
        return ToolReturn(_with_sources(answer, sources), metadata={'sources': _source_list(sources)})


def _format_result(result: Result, body: str | None) -> str:
    """Render one result as labelled metadata lines followed by its body text."""
    lines = [f'Title: {result.title or "(untitled)"}', f'URL: {result.url}']
    if result.published_date:
        lines.append(f'Published: {result.published_date}')
    if result.author:
        lines.append(f'Author: {result.author}')
    if body:
        lines.extend(['', body])
    return '\n'.join(lines)


def _grounding_sources(output: DeepSearchOutput) -> dict[str, str | None]:
    """Deduplicated `url -> title` sources from a synthesized output's grounding."""
    return {citation.url: citation.title for row in output.grounding for citation in row.citations}


def _with_sources(body: str, sources: Mapping[str, str | None]) -> str:
    """Append a `Sources:` block to `body`, or return `body` unchanged when there are none."""
    if not sources:
        return body
    lines = [body, '', 'Sources:']
    lines.extend(f'- {title or "(untitled)"}: {url}' for url, title in sources.items())
    return '\n'.join(lines)


def _highlights_body(result: Result) -> str | None:
    """Render a result's excerpts as a bullet list, or `None` when there are none."""
    if not result.highlights:
        return None
    return '\n'.join(f'- {highlight}' for highlight in result.highlights)


def _truncate(text: str, max_chars: int) -> str:
    """Cap page text at `max_chars`, keeping the head.

    Unlike shell output, where errors land at the end, a page's lead carries
    the substance, so the head is kept and the tail dropped.
    """
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... page text truncated at {max_chars} characters]'
