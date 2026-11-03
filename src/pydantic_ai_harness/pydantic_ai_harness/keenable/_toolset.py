"""Keenable toolset -- gives agents web search and page retrieval backed by the Keenable API."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, Protocol, TypeVar

import httpx
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import ToolOperation, raise_retry
from pydantic_ai_harness._output import truncate_head
from pydantic_ai_harness._typeguards import is_object_dict, is_object_list

KEENABLE_DEFAULT_BASE_URL = 'https://api.keenable.ai'
"""Base URL of the hosted Keenable API."""

KEENABLE_REQUEST_TIMEOUT_SECONDS = 30.0
"""httpx timeout for each connect, read, write, and pool wait of a Keenable HTTP call.

It bounds each of those operations, not the request as a whole.
"""

KEENABLE_MAX_RESULTS = 50
"""The most results one Keenable search returns; the API rejects a larger `max_results`."""

_API_KEY_ENV = 'KEENABLE_API_KEY'
_BASE_URL_ENV = 'KEENABLE_API_URL'
_CLIENT_TITLE = 'Pydantic AI Harness'

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


def _one_line(text: object) -> str:
    """Collapse whitespace, so third-party text can't start a line that looks like another result."""
    return ' '.join(str(text).split())


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


class HttpKeenableClient:
    """The default `KeenableClient`, talking to the Keenable API over `httpx`.

    Keenable needs no credentials: with no API key the public endpoints are
    used, which is why this capability works out of the box. An API key raises
    rate limits.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        """Configure the client.

        Args:
            api_key: The Keenable API key. Defaults to the `KEENABLE_API_KEY`
                environment variable; with neither, the keyless public
                endpoints are used.
            base_url: The API base URL. Defaults to the `KEENABLE_API_URL`
                environment variable, then to the hosted API. It must be
                `https`, or plain `http` against loopback, with no query,
                fragment, or credentials.
        """
        self._api_key = (api_key if api_key is not None else os.environ.get(_API_KEY_ENV, '')).strip()
        base = (base_url if base_url is not None else os.environ.get(_BASE_URL_ENV)) or KEENABLE_DEFAULT_BASE_URL
        base = base.rstrip('/')
        parsed = httpx.URL(base)
        # The base URL is developer-controlled, but a typo should fail here
        # rather than produce a plaintext request. Credentials are rejected
        # without echoing the URL: httpx puts the request URL in its error
        # messages, so they would otherwise reach logs and retry prompts.
        if parsed.userinfo:
            raise UserError(f'{_BASE_URL_ENV} must not contain credentials; pass the API key separately')
        loopback_http = parsed.scheme == 'http' and parsed.host in {'localhost', '127.0.0.1', '::1'}
        # The endpoint path is appended to the base URL, so a query or fragment,
        # even an empty one that parses as none at all, would land before it.
        if not parsed.host or '?' in base or '#' in base or not (parsed.scheme == 'https' or loopback_http):
            raise UserError(
                f'{_BASE_URL_ENV} must be an https:// URL with a host and no query or fragment, got {base!r}'
            )
        self._base_url = base

    def _headers(self) -> dict[str, str]:
        headers = {'X-Keenable-Title': _CLIENT_TITLE}
        if self._api_key:
            headers['X-API-Key'] = self._api_key
        return headers

    def _path(self, keyed: str, public: str) -> str:
        return keyed if self._api_key else public

    async def search(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        """Search the web via `POST /v1/search`, or `/v1/search/public` when keyless."""
        async with httpx.AsyncClient(timeout=KEENABLE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f'{self._base_url}{self._path("/v1/search", "/v1/search/public")}',
                json={'query': query, 'max_results': max_results},
                headers=self._headers(),
            )
        response.raise_for_status()
        results = _string_keyed(response.json()).get('results')
        if not is_object_list(results):
            return []
        # `_string_keyed` maps anything that is not an object to `{}`, so filtering
        # on the result drops non-object entries without a second check.
        return [mapped for mapped in (_string_keyed(entry) for entry in results) if mapped]

    async def fetch(self, url: str) -> dict[str, object]:
        """Retrieve a page via `GET /v1/fetch`, or `/v1/fetch/public` when keyless."""
        async with httpx.AsyncClient(timeout=KEENABLE_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(
                f'{self._base_url}{self._path("/v1/fetch", "/v1/fetch/public")}',
                params={'url': url},
                headers=self._headers(),
            )
        response.raise_for_status()
        return _string_keyed(response.json())


def _string_keyed(payload: object) -> dict[str, object]:
    """Narrow a decoded JSON value to a string-keyed mapping, or an empty one."""
    if not is_object_dict(payload):
        return {}
    return {str(key): value for key, value in payload.items()}


def validated_budget(name: str, value: object, *, limit: int | None = None) -> int:
    """Reject a budget that is not a positive integer, or that is above `limit`.

    The annotation is advisory: a config-driven caller can pass a float or a
    string, which would only fail later as a slice bound. Negative budgets are
    worse than useless here: they slice from the end, so they would return the
    wrong text rather than fail. A `bool` is an `int` too, but `True` is not a
    budget anyone means.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')
    if limit is not None and value > limit:
        raise ValueError(f'{name} must be at most {limit}, got {value!r}')
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
            response = error.response
            raise ModelRetry(
                f'Keenable request failed with HTTP {response.status_code} {response.reason_phrase}.'
            ) from error
        except httpx.HTTPError as error:
            raise ModelRetry(f'Keenable request failed: {error}') from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # `Response.json()` decodes the bytes before parsing them, so a body
            # in a broken encoding fails as a decode error, not a JSON one.
            raise ModelRetry(f'Keenable returned a malformed response: {error}') from error

    return wrapper


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
    marker so the model knows the page continued. All three bounds are
    validated here as well as by `KeenableSearch`.

    `KeenableSearch` passes `operations` so that each tool's Keenable request
    runs as one of its durable operations, whose result durable execution
    records instead of making the request again on recovery.
    """

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

    @_recoverable
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
        # A result without a URL is unusable, so it is dropped before the limit
        # is applied rather than after; otherwise one malformed entry would cost
        # a slot and `web_search` would return fewer results than asked for.
        found = await self._client.search(query, max_results=self._num_results)
        results = [(url, result) for result in found if (url := _one_line(result.get('url') or ''))]
        results = results[: self._num_results]
        if not results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})

        lines: list[str] = []
        sources: dict[str, str | None] = {}
        for index, (url, result) in enumerate(results, start=1):
            title = _one_line(result.get('title') or '') or None
            sources[url] = title
            lines.append(f'{index}. {title or url}\n   {url}')
            excerpt = _one_line(result.get('snippet') or '')[: self._max_snippet_chars]
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
        if self._operations is not None:
            return raise_retry(await self._operations.get_page(url))
        page = await self._client.fetch(url)
        content = str(page.get('content') or '')
        if not content:
            raise ModelRetry(f'No readable content at {url!r}. Try a different URL.')

        title = page.get('title') or None
        resolved = str(page.get('url') or url)
        return ToolReturn(
            truncate_head(content, self._max_page_chars),
            metadata={'sources': _source_list({resolved: str(title) if title else None})},
        )
