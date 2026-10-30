"""Exa toolset -- gives agents web search, page retrieval, and deep search backed by the Exa API."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, ParamSpec, Protocol, TypeVar

import httpx
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import ToolOperation, raise_retry
from pydantic_ai_harness._web_search import defer_to_native_web_search

try:
    from exa_py import AsyncExa
    from exa_py.agent.types import AgentEffort, AgentRun
    from exa_py.api import (
        ContentsOptions,
        DeepOutputSchema,
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


class ExaSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def source_map(sources: Iterable[tuple[str, str | None]]) -> dict[str, str | None]:
    """Deduplicate `(url, title)` pairs by URL, keeping each URL's first title."""
    unique: dict[str, str | None] = {}
    for url, title in sources:
        unique.setdefault(url, title)
    return unique


def source_list(sources: Mapping[str, str | None]) -> list[ExaSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{'url': url, 'title': title} for url, title in sources.items()]


# exa-py raises a bare ValueError for any non-2xx response, embedding the HTTP
# status in the message, and also for its own client-side option checks.
_STATUS_RE = re.compile(r'Request failed with status code (\d{3})\b')

# 401/403 mean a bad or missing API key and 402 means the account is out of
# credits: configuration or account states the model cannot correct.
_ACCOUNT_STATUSES = frozenset({401, 402, 403})

# Failures raised before a request reached Exa, so repeating it cannot repeat its effect.
_UNSENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


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


@dataclass(frozen=True)
class ExaSearchOperations:
    """The durable operations an `ExaSearch` capability runs its toolset's Exa requests through."""

    web_search: ToolOperation
    get_page: ToolOperation
    deep_search: ToolOperation


class DefaultExa:
    """An `AsyncExa` built from `EXA_API_KEY`, rebuilt when it's used on another event loop.

    `AsyncExa` keeps one httpx connection pool, and connections belong to the event loop that opened
    them, so an agent reused on a new loop (a second `asyncio.run`, a new test) can't use the old pool.
    As in Pydantic AI's providers, the client is only replaced once the loop it was last used on has
    stopped, so runs still going there keep it. Only that last loop is checked: with three or more loops
    taking turns, a run still going on an earlier one can see the client replaced.

    Implements both `ExaClient` and `ExaAgentRuns`, including the optional `cancel`.
    """

    def __init__(self, *, missing_key_hint: str) -> None:
        api_key = os.environ.get('EXA_API_KEY')
        if api_key is None:
            raise UserError(missing_key_hint)
        self._api_key = api_key
        self._exa = AsyncExa(api_key=api_key)
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def exa(self) -> AsyncExa:
        """The client for the running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No asyncio event loop, e.g. under Trio, so there's nothing to compare.
            return self._exa
        previous, self._loop = self._loop, loop
        if previous is not None and previous is not loop and not previous.is_running():
            self._exa = AsyncExa(api_key=self._api_key)
        return self._exa

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
        """Search with the client for the running event loop."""
        return await self._client().search(
            query,
            contents=contents,
            num_results=num_results,
            type=type,
            output_schema=output_schema,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        """Retrieve contents with the client for the running event loop."""
        return await self._client().get_contents(urls, text=text)

    def _client(self) -> ExaClient:
        """The client for the running event loop, typed by the protocol, since some `AsyncExa` methods are untyped."""
        return self.exa

    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | type[BaseModel] | None = None,
        effort: AgentEffort | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun:
        """Create an agent run with the client for the running event loop."""
        return await self.exa.agent.runs.create(
            query=query,
            system_prompt=system_prompt,
            output_schema=output_schema,
            effort=effort,
            previous_run_id=previous_run_id,
        )

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        """Poll an agent run with the client for the running event loop."""
        return await self.exa.agent.runs.poll_until_finished(run_id, poll_interval=poll_interval, timeout_ms=timeout_ms)

    async def cancel(self, run_id: str) -> AgentRun:
        """Cancel an agent run with the client for the running event loop."""
        return await self.exa.agent.runs.cancel(run_id)


def default_client() -> ExaClient:
    """Build the default client from the `EXA_API_KEY` environment variable."""
    return DefaultExa(
        missing_key_hint='ExaSearch needs an Exa API key: set the EXA_API_KEY environment variable, '
        'or pass a configured client, e.g. ExaSearch(client=AsyncExa(api_key=...)).'
    )


def is_recoverable(error: Exception, *, idempotent: bool = True) -> bool:
    """Whether the model can recover from an Exa request failure by calling the tool again.

    An HTTP error response is recoverable unless it is an account or configuration state
    (401, 402, 403). A `ValueError` that is not an HTTP error response comes from exa-py
    itself (an option it rejects, or a response it cannot parse), so it is not. A network
    failure is recoverable.

    A request with a side effect (`idempotent=False`) is only repeated when Exa can't
    have acted on it: a network failure before the request was sent, or a 4xx response.
    A 5xx response or a failure after sending may come after Exa acted.
    """
    if isinstance(error, httpx.HTTPError):
        return idempotent or isinstance(error, _UNSENT_ERRORS)
    if isinstance(error, ValueError) and (match := _STATUS_RE.match(str(error))):
        status = int(match.group(1))
        return status not in _ACCOUNT_STATUSES and (idempotent or status < 500)
    return False


def recoverable(*, idempotent: bool = True) -> Callable[[Callable[_P, Awaitable[_R]]], Callable[_P, Awaitable[_R]]]:
    """Convert recoverable Exa request failures into `ModelRetry`; see `is_recoverable`.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception propagates and aborts the whole run.
    """

    def decorate(fn: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
        @functools.wraps(fn)
        async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await fn(*args, **kwargs)
            except (httpx.HTTPError, ValueError) as error:
                if not is_recoverable(error, idempotent=idempotent):
                    raise
                raise ModelRetry(f'Exa request failed: {error}') from error

        return wrapper

    return decorate


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

    With `defer_to_native=True`, `web_search` is the local fallback for the
    model's native web search, as `ExaSearch(native=True)` sets up.

    `ExaSearch` passes `operations` so that each tool's Exa request runs as one
    of its durable operations, whose result durable execution records instead
    of making the request again on recovery.
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
        defer_to_native: bool = False,
        id: str | None = None,
        operations: ExaSearchOperations | None = None,
    ) -> None:
        super().__init__(id=id)
        self._client = client if client is not None else default_client()
        self._operations = operations
        self._num_results = num_results
        self._max_text_chars = max_text_chars
        self._include_domains = list(include_domains) if include_domains else None
        self._exclude_domains = list(exclude_domains) if exclude_domains else None
        self._text_summary = text_summary
        self.add_function(
            self.web_search, name='web_search', prepare=defer_to_native_web_search if defer_to_native else None
        )
        self.add_function(self.get_page, name='get_page')
        if include_deep_search:
            self.add_function(self.deep_search, name='deep_search')

    @recoverable()
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with its most relevant excerpts.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and excerpts.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.web_search(query))
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
        sources = source_list(source_map((result.url, result.title) for result in results))
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

    @recoverable()
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve the full text of a specific URL.

        Use it to read a promising URL from `web_search` results in full, or a
        URL the user provided.

        Args:
            url: The URL of the page to read.

        Returns:
            The page's title, URL, and text content.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.get_page(url))
        requested = min(self._max_text_chars + 1, EXA_MAX_PAGE_TEXT_CHARS)
        response = await self._client.get_contents(url, text={'max_characters': requested})
        first = response.results[0] if response.results else None
        if first is None or not first.text:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        return ToolReturn(
            _format_result(first, _truncate(first.text, self._max_text_chars)),
            metadata={'sources': source_list({first.url: first.title})},
        )

    @recoverable()
    async def deep_search(self, question: str) -> ToolReturn[str]:
        """Run Exa's multi-step deep search and return a synthesized answer with its sources.

        A full research pass in a single call; suited to questions that need
        synthesis across many sources rather than a quick survey.

        Args:
            question: The research question to answer.

        Returns:
            The synthesized answer, followed by the sources it drew on.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.deep_search(question))
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
        answer = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        sources = source_map((citation.url, citation.title) for row in output.grounding for citation in row.citations)
        if not sources:
            sources = source_map((result.url, result.title) for result in response.results)
        return ToolReturn(with_sources(answer, sources), metadata={'sources': source_list(sources)})


def one_line(text: str) -> str:
    """Collapse whitespace, so third-party text can't start a line that looks like the tool's own labels."""
    return ' '.join(text.split())


def _format_result(result: Result, body: str | None) -> str:
    """Render one result as labelled metadata lines followed by its body text."""
    lines = [f'Title: {one_line(result.title or "") or "(untitled)"}', f'URL: {one_line(result.url)}']
    if result.published_date:
        lines.append(f'Published: {one_line(result.published_date)}')
    if result.author:
        lines.append(f'Author: {one_line(result.author)}')
    if body:
        lines.extend(['', body])
    return '\n'.join(lines)


def with_sources(body: str, sources: Mapping[str, str | None]) -> str:
    """Append a `Sources:` block to `body`, or return `body` unchanged when there are none."""
    if not sources:
        return body
    lines = [body, '', 'Sources:']
    lines.extend(f'- {one_line(title or "") or "(untitled)"}: {one_line(url)}' for url, title in sources.items())
    return '\n'.join(lines)


def _highlights_body(result: Result) -> str | None:
    """Render a result's excerpts as a bullet list, or `None` when there are none."""
    if not result.highlights:
        return None
    return '\n'.join(f'- {one_line(highlight)}' for highlight in result.highlights)


def _truncate(text: str, max_chars: int) -> str:
    """Cap page text at `max_chars`, keeping the head.

    Unlike shell output, where errors land at the end, a page's lead carries
    the substance, so the head is kept and the tail dropped.
    """
    if len(text) <= max_chars:
        return text
    return f'{text[:max_chars]}\n[... page text truncated at {max_chars} characters]'
