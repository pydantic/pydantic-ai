"""Keenable toolset -- gives agents web search and page retrieval backed by the Keenable API."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx2
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import ToolOperation, raise_retry
from pydantic_ai_harness._json_api import JsonApiClient, MalformedResponse, recoverable
from pydantic_ai_harness._output import one_line, truncate_head_within, validated_budget
from pydantic_ai_harness._typeguards import is_object_list

KEENABLE_MAX_RESULTS = 50
"""The most results one Keenable search returns; the API rejects a larger `max_results`."""


class KeenableSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


class KeenableClient(Protocol):
    """The subset of the Keenable API that `KeenableSearchToolset` calls.

    Any object with these two methods can back the toolset. Pass one via
    `KeenableSearch.client` to configure authentication or the base URL
    explicitly, or to substitute a fake in tests.
    """

    async def search(self, query: str, *, max_results: int) -> Sequence[Mapping[str, object]]:
        """Search the web and return up to `max_results` raw result objects."""
        ...  # pragma: no cover

    async def fetch(self, url: str) -> Mapping[str, object]:
        """Retrieve one page and return the raw response payload."""
        ...  # pragma: no cover


class HttpKeenableClient(JsonApiClient):
    """The default `KeenableClient`, talking to the Keenable API over `httpx2`.

    Keenable needs no credentials: with no API key the public endpoints are
    used, which is why this capability works out of the box. An API key raises
    rate limits.
    """

    _DEFAULT_BASE_URL = 'https://api.keenable.ai'
    _API_KEY_ENV = 'KEENABLE_API_KEY'
    _BASE_URL_ENV = 'KEENABLE_API_URL'
    _CLIENT_TITLE = 'Pydantic AI Harness'

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The Keenable API key. Defaults to the `KEENABLE_API_KEY`
                environment variable; with neither, the keyless public
                endpoints are used.
            base_url: The API base URL. Defaults to the `KEENABLE_API_URL`
                environment variable, then to the hosted API. It must be
                `https`, or plain `http` against loopback, with no query,
                fragment, or credentials.
            http_client: The `httpx2.AsyncClient` to send every request through,
                to reuse connections or to configure a proxy, TLS, or a retrying
                transport. Its own timeouts apply, alongside the 30-second
                deadline on each request. It stays open; closing it is up to
                you. Defaults to a new client for each request, closed when the
                request ends.
        """
        self._api_key = (api_key if api_key is not None else os.environ.get(self._API_KEY_ENV, '')).strip()
        headers = {'X-Keenable-Title': self._CLIENT_TITLE}
        if self._api_key:
            headers['X-API-Key'] = self._api_key
        super().__init__(base_url=base_url, http_client=http_client, headers=headers)

    def _path(self, keyed: str, public: str) -> str:
        return keyed if self._api_key else public

    async def search(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        """Search the web via `POST /v1/search`, or `/v1/search/public` when keyless."""
        found = await self._request(
            'POST',
            self._path('/v1/search', '/v1/search/public'),
            payload={'query': query, 'max_results': max_results},
        )
        results = self._string_keyed(found).get('results')
        if not is_object_list(results):
            # Keenable always sends `results` with a 2xx, so a body without the
            # list is a gateway or proxy answering, not an empty search.
            raise MalformedResponse('`results` is missing or not a list')
        # `_string_keyed` maps anything that is not an object to `{}`, so filtering
        # on the result drops non-object entries without a second check.
        return [mapped for mapped in (self._string_keyed(entry) for entry in results) if mapped]

    async def fetch(self, url: str) -> dict[str, object]:
        """Retrieve a page via `GET /v1/fetch`, or `/v1/fetch/public` when keyless."""
        page = await self._request('GET', self._path('/v1/fetch', '/v1/fetch/public'), params={'url': url})
        return self._string_keyed(page)


@dataclass(frozen=True)
class KeenableSearchOperations:
    """The durable operations a `KeenableSearch` capability runs its toolset's Keenable requests through."""

    web_search: ToolOperation
    get_page: ToolOperation


class KeenableSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Keenable API.

    `web_search` surveys the web and returns results with a short excerpt of
    each page, and `get_page` retrieves one specific URL as markdown.

    Each tool returns a `ToolReturn` whose `return_value` is the text the model
    sees and whose `metadata['sources']` lists the result URLs and titles
    (`KeenableSource` dicts), so applications can render citations from
    `ToolReturnPart.metadata` without parsing the text.

    `web_search` asks Keenable for `num_results` results and trims each excerpt
    to `max_snippet_chars`, and `get_page` truncates at `max_page_chars` with a
    marker so the model knows the page continued; a budget too small to hold
    the marker ends the page with `…` instead. All three bounds are validated
    here as well as by `KeenableSearch`.

    `KeenableSearch` passes `operations` so that each tool's Keenable request
    runs as one of its durable operations, whose result durable execution
    records instead of making the request again on recovery.
    """

    # Titles and URLs come from third-party pages, so they are bounded like
    # excerpts. A cut URL would name a different page, and one with its inner
    # whitespace collapsed names none, so a result whose URL is longer or has
    # whitespace inside is dropped instead.
    _MAX_TITLE_CHARS = 300
    _MAX_URL_CHARS = 2048

    def __init__(
        self,
        *,
        client: KeenableClient | None,
        num_results: int,
        max_snippet_chars: int,
        max_page_chars: int,
        id: str | None = None,
        operations: KeenableSearchOperations | None = None,
    ) -> None:
        """Configure the tools.

        Args:
            client: The Keenable client the tools call. `None` uses an
                `HttpKeenableClient` configured from the `KEENABLE_API_KEY` and
                `KEENABLE_API_URL` environment variables.
            num_results: Number of results `web_search` asks Keenable for, from
                1 to 50.
            max_snippet_chars: Maximum characters of excerpt `web_search`
                returns per result.
            max_page_chars: Maximum characters of page text `get_page` returns,
                truncation marker included.
            id: An optional unique ID for the toolset, which durable execution
                needs to identify it.
            operations: The durable operations each tool's Keenable request
                runs through. `None` makes the request directly.

        Raises:
            ValueError: If a budget is not a positive integer, or
                `num_results` is above 50.
        """
        super().__init__(id=id)
        # `KeenableSearch` validates these at construction, but the toolset is
        # public too, so a direct caller gets the same guarantee. Validated
        # before the client is built, so a bad budget fails on its own terms.
        self._num_results = validated_budget('num_results', num_results, limit=KEENABLE_MAX_RESULTS)
        self._max_snippet_chars = validated_budget('max_snippet_chars', max_snippet_chars)
        self._max_page_chars = validated_budget('max_page_chars', max_page_chars)
        self._client: KeenableClient = client if client is not None else HttpKeenableClient()
        self._operations = operations
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')

    @staticmethod
    def _source_list(sources: Mapping[str, str | None]) -> list[KeenableSource]:
        """Convert a `url -> title` mapping into the metadata `sources` list."""
        return [{'url': url, 'title': title} for url, title in sources.items()]

    @recoverable('Keenable')
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with a short excerpt.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and an excerpt.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.web_search(query))
        # Keenable is asked for `num_results`, so a result without a usable URL
        # still costs a slot. Filtering before the slice matters only when a
        # server returns more than `max_results`.
        found = await self._client.search(query, max_results=self._num_results)
        results = [
            (url, result)
            for result in found
            if (url := str(result.get('url') or '').strip())
            and len(url) <= self._MAX_URL_CHARS
            and not any(char.isspace() for char in url)
        ]
        results = results[: self._num_results]
        if not results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})

        lines: list[str] = []
        sources: dict[str, str | None] = {}
        for index, (url, result) in enumerate(results, start=1):
            title = one_line(str(result.get('title') or ''))[: self._MAX_TITLE_CHARS] or None
            sources[url] = title
            lines.append(f'{index}. {title or url}\n   {url}')
            excerpt = one_line(str(result.get('snippet') or ''))[: self._max_snippet_chars]
            if excerpt:
                lines.append(f'   {excerpt}')

        return ToolReturn('\n'.join(lines), metadata={'sources': self._source_list(sources)})

    @recoverable('Keenable')
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve one web page and return its content as markdown.

        Args:
            url: The absolute URL of the page to read.

        Returns:
            The page content as markdown, truncated to the configured budget.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.get_page(url))
        page = await self._client.fetch(url)
        content = str(page.get('content') or '')
        if not content:
            raise ModelRetry(f'No readable content at {url!r}. Try a different URL.')

        page_text = truncate_head_within(content, self._max_page_chars)
        title = page.get('title') or None
        resolved = str(page.get('url') or url)
        return ToolReturn(page_text, metadata={'sources': self._source_list({resolved: str(title) if title else None})})
