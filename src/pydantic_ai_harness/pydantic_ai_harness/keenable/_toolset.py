"""Keenable toolset -- gives agents web search and page retrieval backed by the Keenable API."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar, cast

import httpx
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

KEENABLE_DEFAULT_BASE_URL = 'https://api.keenable.ai'
"""Base URL of the hosted Keenable API."""

KEENABLE_REQUEST_TIMEOUT_SECONDS = 30.0
"""Per-request timeout for Keenable HTTP calls."""

_API_KEY_ENV = 'KEENABLE_API_KEY'
_BASE_URL_ENV = 'KEENABLE_API_URL'
_CLIENT_TITLE = 'Pydantic AI Harness'
_TRUNCATION_MARKER = '\n\n[truncated]'

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')

# 401/402/403 are authentication, billing, or authorization states the model
# cannot correct, so they propagate and abort the run instead of becoming a
# retry prompt that would loop until the retry limit.
_AUTH_STATUSES = frozenset({401, 402, 403})


class KeenableSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


def _source_list(sources: Mapping[str, str | None]) -> list[KeenableSource]:
    """Convert a `url -> title` mapping into the metadata `sources` list."""
    return [{'url': url, 'title': title} for url, title in sources.items()]


def _result_text(result: Mapping[str, Any]) -> str:
    """Extract a result's text.

    Keenable returns both `snippet` and `description` on every result.
    `snippet` carries the page text and `description` is frequently empty, so
    prefer whichever has content. Snippets are raw page text with newlines in
    them, so whitespace is collapsed to keep one result on one line.
    """
    return ' '.join(str(result.get('snippet') or result.get('description') or '').split())


class KeenableClient(Protocol):
    """The subset of the Keenable API that `KeenableSearchToolset` calls.

    Any object with these two methods can back the toolset. Pass one via
    `KeenableSearch.client` to configure authentication or the base URL
    explicitly, or to substitute a fake in tests.
    """

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search the web and return the raw result dictionaries."""
        ...  # pragma: no cover

    async def fetch(self, url: str) -> dict[str, Any]:
        """Retrieve one page and return the raw response payload."""
        ...  # pragma: no cover


class HttpKeenableClient:
    """The default `KeenableClient`, talking to the Keenable API over `httpx`.

    Keenable needs no credentials: with no API key the public endpoints are
    used, which is why this capability works out of the box. An API key raises
    rate limits and is read from the `KEENABLE_API_KEY` environment variable
    unless passed explicitly.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = (api_key if api_key is not None else os.environ.get(_API_KEY_ENV, '')).strip()
        self._base_url = _normalize_base_url(base_url if base_url is not None else os.environ.get(_BASE_URL_ENV))

    def _headers(self) -> dict[str, str]:
        headers = {'X-Keenable-Title': _CLIENT_TITLE}
        if self._api_key:
            headers['X-API-Key'] = self._api_key
        return headers

    def _path(self, keyed: str, public: str) -> str:
        return keyed if self._api_key else public

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search the web via `POST /v1/search`, or `/v1/search/public` when keyless."""
        async with httpx.AsyncClient(timeout=KEENABLE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f'{self._base_url}{self._path("/v1/search", "/v1/search/public")}',
                json={'query': query},
                headers=self._headers(),
            )
        response.raise_for_status()
        return _search_results(response.json())

    async def fetch(self, url: str) -> dict[str, Any]:
        """Retrieve a page via `GET /v1/fetch`, or `/v1/fetch/public` when keyless."""
        async with httpx.AsyncClient(timeout=KEENABLE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f'{self._base_url}{self._path("/v1/fetch", "/v1/fetch/public")}',
                params={'url': url},
                headers=self._headers(),
            )
        response.raise_for_status()
        payload: object = response.json()
        return _string_keyed(payload)


def _string_keyed(payload: object) -> dict[str, Any]:
    """Narrow a decoded JSON value to a string-keyed mapping, or an empty one."""
    if not isinstance(payload, dict):
        return {}
    items = cast('dict[Any, Any]', payload)
    return {str(key): value for key, value in items.items()}


def _search_results(payload: object) -> list[dict[str, Any]]:
    """Pull the result list out of a decoded search response, tolerating an unexpected shape."""
    results = _string_keyed(payload).get('results')
    if not isinstance(results, list):
        return []
    entries = cast('list[Any]', results)
    # `_string_keyed` maps anything that is not an object to `{}`, so filtering
    # on the result drops non-object entries without a second isinstance check.
    return [mapped for mapped in (_string_keyed(entry) for entry in entries) if mapped]


def _normalize_base_url(base_url: str | None) -> str:
    """Resolve and validate the API base URL.

    The base URL is developer-controlled, never model-controlled, but it is
    validated anyway so a typo fails at construction rather than producing a
    plaintext request. Plain `http` is allowed only against loopback, for local
    development against a Keenable instance.

    A query or fragment is rejected too, even an empty one (a bare trailing
    `?` or `#`, which parses as no query at all): the endpoint path is appended
    to this string, so either one would land before it and the request would go
    somewhere other than the endpoint it names.
    """
    base = (base_url or KEENABLE_DEFAULT_BASE_URL).rstrip('/')
    parsed = httpx.URL(base)
    if parsed.host and '?' not in base and '#' not in base:
        if parsed.scheme == 'https':
            return base
        if parsed.scheme == 'http' and parsed.host in {'localhost', '127.0.0.1', '::1'}:
            return base
    raise UserError(f'{_BASE_URL_ENV} must be an https:// URL with a host and no query or fragment, got {base!r}')


def validated_budget(name: str, value: object) -> int:
    """Reject a budget that is not a positive integer.

    The annotation is advisory: a config-driven caller can pass a float or a
    string, which would only fail later as a slice bound. Negative budgets are
    worse than useless here: they slice from the end, so they would silently
    return the wrong text rather than fail.
    """
    if not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    return value


def _recoverable(
    fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
    """Convert transient Keenable API failures into `ModelRetry`.

    Pydantic AI only feeds `ModelRetry` back to the model as a retry prompt;
    any other exception propagates and aborts the run. Rate limits, transient
    5xx, rejected parameters, a 404 on a dead URL, and a 2xx whose body is not
    JSON (a proxy or gateway answering in HTML) are things a model can recover
    from (wait, rephrase, pick another URL), so they become retries. A
    401/402/403 means a bad key or an exhausted account, configuration the
    model cannot fix, so it propagates.
    """

    @functools.wraps(fn)
    async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await fn(self, *args, **kwargs)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in _AUTH_STATUSES:
                raise
            raise ModelRetry(f'Keenable request failed: {error}') from error
        except httpx.HTTPError as error:
            raise ModelRetry(f'Keenable request failed: {error}') from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # `Response.json()` decodes the bytes before parsing them, so a body
            # in a broken encoding fails as a decode error, not a JSON one.
            raise ModelRetry(f'Keenable returned a malformed response: {error}') from error

    return wrapper


class KeenableSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Keenable API.

    `web_search` surveys the web and returns results with a short excerpt of
    each page, and `get_page` retrieves one specific URL as markdown.

    Each tool returns a `ToolReturn` whose `return_value` is the text the model
    sees and whose `metadata['sources']` lists the result URLs and titles
    (`KeenableSource` dicts), so applications can render citations from
    `ToolReturnPart.metadata` without parsing the text.

    Both budgets are applied locally: `web_search` keeps `num_results` results
    and trims each excerpt to `max_snippet_chars`, and `get_page` truncates at
    `max_page_chars` and appends a marker so the model knows the page continued.
    All three bounds are validated here as well as by `KeenableSearch`.
    """

    def __init__(
        self,
        *,
        client: KeenableClient | None,
        num_results: int,
        max_snippet_chars: int,
        max_page_chars: int,
    ) -> None:
        super().__init__()
        # `KeenableSearch` validates these at construction, but the toolset is
        # public too, so a direct caller gets the same guarantee. Validated
        # before the client is built, so a bad budget fails on its own terms.
        self._num_results = validated_budget('num_results', num_results)
        self._max_snippet_chars = validated_budget('max_snippet_chars', max_snippet_chars)
        self._max_page_chars = validated_budget('max_page_chars', max_page_chars)
        self._client: KeenableClient = client if client is not None else HttpKeenableClient()
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.get_page, name='get_page')

    @_recoverable
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with a short excerpt.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and an excerpt.
        """
        # A result without a URL is unusable, so it is dropped before the limit
        # is applied rather than after; otherwise one malformed entry would cost
        # a slot and `web_search` would return fewer results than asked for.
        found = await self._client.search(query)
        results = [(url, result) for result in found if (url := str(result.get('url') or ''))]
        results = results[: self._num_results]
        if not results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})

        lines: list[str] = []
        sources: dict[str, str | None] = {}
        for index, (url, result) in enumerate(results, start=1):
            title = result.get('title') or None
            sources[url] = str(title) if title is not None else None
            lines.append(f'{index}. {title or url}\n   {url}')
            excerpt = _result_text(result)[: self._max_snippet_chars]
            if excerpt:
                lines.append(f'   {excerpt}')

        return ToolReturn('\n'.join(lines), metadata={'sources': _source_list(sources)})

    @_recoverable
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve one web page and return its content as markdown.

        Args:
            url: The absolute URL of the page to read.

        Returns:
            The page content as markdown, truncated to the configured budget.
        """
        page = await self._client.fetch(url)
        content = str(page.get('content') or '')
        if not content:
            raise ModelRetry(f'No readable content at {url!r}. Try a different URL.')

        title = page.get('title') or None
        resolved = str(page.get('url') or url)
        if len(content) > self._max_page_chars:
            # The marker counts against the budget, so make room for it; a
            # budget too small to hold it is truncated without one.
            keep = self._max_page_chars - len(_TRUNCATION_MARKER)
            content = f'{content[:keep]}{_TRUNCATION_MARKER}' if keep > 0 else content[: self._max_page_chars]
        return ToolReturn(content, metadata={'sources': _source_list({resolved: str(title) if title else None})})
