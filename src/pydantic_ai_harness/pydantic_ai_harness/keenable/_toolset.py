"""Keenable toolset -- gives agents web search and page retrieval backed by the Keenable API."""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Concatenate, ParamSpec, Protocol, TypeVar

import anyio
import httpx
from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import ToolOperation, raise_retry
from pydantic_ai_harness._output import truncate_head
from pydantic_ai_harness._typeguards import is_object_dict, is_object_list

KEENABLE_MAX_RESULTS = 50
"""The most results one Keenable search returns; the API rejects a larger `max_results`."""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_SelfT = TypeVar('_SelfT')


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


class HttpKeenableClient:
    """The default `KeenableClient`, talking to the Keenable API over `httpx`.

    Keenable needs no credentials: with no API key the public endpoints are
    used, which is why this capability works out of the box. An API key raises
    rate limits.
    """

    _DEFAULT_BASE_URL = 'https://api.keenable.ai'
    # Both the httpx timeout on each connect, read, write, and pool wait, and
    # the deadline for the request as a whole.
    _TIMEOUT_SECONDS = 30.0
    _MAX_RESPONSE_BYTES = 10 * 1024 * 1024
    _API_KEY_ENV = 'KEENABLE_API_KEY'
    _BASE_URL_ENV = 'KEENABLE_API_URL'
    _CLIENT_TITLE = 'Pydantic AI Harness'

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
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
            http_client: The `httpx.AsyncClient` to send every request through,
                to reuse connections or to configure a proxy, TLS, or a retrying
                transport. Its own timeouts apply, alongside the 30-second
                deadline on each request. It stays open; closing it is up to
                you. Defaults to a new client for each request, closed when the
                request ends.
        """
        self._api_key = (api_key if api_key is not None else os.environ.get(self._API_KEY_ENV, '')).strip()
        base = (base_url if base_url is not None else os.environ.get(self._BASE_URL_ENV)) or self._DEFAULT_BASE_URL
        base = base.rstrip('/')
        source = '`base_url`' if base_url is not None else f'`{self._BASE_URL_ENV}`'
        parsed = httpx.URL(base)
        # The base URL is developer-controlled, but a typo should fail here
        # rather than produce a plaintext request. Credentials are rejected
        # without echoing the URL: httpx puts the request URL in its error
        # messages, so they would otherwise reach logs and retry prompts.
        if parsed.userinfo:
            raise UserError(f'{source} must not contain credentials; pass the API key separately')
        loopback_http = parsed.scheme == 'http' and parsed.host in {'localhost', '127.0.0.1', '::1'}
        # The endpoint path is appended to the base URL, so a query or fragment,
        # even an empty one that parses as none at all, would land before it.
        if not parsed.host or '?' in base or '#' in base or not (parsed.scheme == 'https' or loopback_http):
            raise UserError(
                f'{source} must be an https:// URL, or http:// on loopback, with a host and no query or fragment, '
                f'got {base!r}'
            )
        self._base_url = base
        self._http_client = http_client

    @staticmethod
    def _string_keyed(payload: object) -> dict[str, object]:
        """Narrow a decoded JSON value to a string-keyed mapping, or an empty one."""
        if not is_object_dict(payload):
            return {}
        return {str(key): value for key, value in payload.items()}

    def _headers(self) -> dict[str, str]:
        headers = {'X-Keenable-Title': self._CLIENT_TITLE}
        if self._api_key:
            headers['X-API-Key'] = self._api_key
        return headers

    def _path(self, keyed: str, public: str) -> str:
        return keyed if self._api_key else public

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> object:
        """Send one request and decode its JSON body.

        The httpx timeout bounds each read, so a server that keeps sending a
        little at a time could hold the request open indefinitely; the deadline
        bounds the whole exchange. `base_url` can point at any server, so the
        body is read in chunks and abandoned once it passes `_MAX_RESPONSE_BYTES`.
        """
        # The default client lives for one request: nothing here would close a
        # shared one, and its connections belong to the event loop that opened
        # them, which each `run_sync` call replaces.
        client_context = (
            nullcontext(self._http_client)
            if self._http_client is not None
            else httpx.AsyncClient(timeout=self._TIMEOUT_SECONDS)
        )
        with anyio.fail_after(self._TIMEOUT_SECONDS):
            # An injected client may follow redirects, and httpx keeps the
            # `X-API-Key` header on a redirect to another host, so the key would
            # reach wherever the server points.
            async with (
                client_context as client,
                client.stream(
                    method,
                    f'{self._base_url}{path}',
                    json=payload,
                    params=params,
                    headers=self._headers(),
                    follow_redirects=False,
                ) as response,
            ):
                response.raise_for_status()
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > self._MAX_RESPONSE_BYTES:
                        raise httpx.ReadError(
                            f'response body exceeds {self._MAX_RESPONSE_BYTES} bytes', request=response.request
                        )
        return json.loads(content)

    async def search(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        """Search the web via `POST /v1/search`, or `/v1/search/public` when keyless."""
        found = await self._request(
            'POST',
            self._path('/v1/search', '/v1/search/public'),
            payload={'query': query, 'max_results': max_results},
        )
        results = self._string_keyed(found).get('results')
        if not is_object_list(results):
            return []
        # `_string_keyed` maps anything that is not an object to `{}`, so filtering
        # on the result drops non-object entries without a second check.
        return [mapped for mapped in (self._string_keyed(entry) for entry in results) if mapped]

    async def fetch(self, url: str) -> dict[str, object]:
        """Retrieve a page via `GET /v1/fetch`, or `/v1/fetch/public` when keyless."""
        page = await self._request('GET', self._path('/v1/fetch', '/v1/fetch/public'), params={'url': url})
        return self._string_keyed(page)


def validated_budget(name: str, value: object, *, limit: int | None = None) -> int:
    """Reject a budget that is not a positive integer, or that is above `limit`.

    The annotation is advisory: a config-driven caller can pass a float or a
    string, which would only fail later as a slice bound. Negative budgets are
    worse than useless here: they slice from the end, so they would return the
    wrong text rather than fail. A `bool` is an `int` too, but `True` is not a
    budget anyone means.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'`{name}` must be a positive integer, got {value!r}')
    if limit is not None and value > limit:
        raise ValueError(f'`{name}` must be at most {limit}, got {value!r}')
    return value


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
    # excerpts. A cut URL would name a different page, so a result whose URL is
    # longer is dropped rather than cut.
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

    @staticmethod
    def _one_line(text: object) -> str:
        """Collapse whitespace, so third-party text can't start a line that looks like another result."""
        return ' '.join(str(text).split())

    @staticmethod
    def _recoverable(
        fn: Callable[Concatenate[_SelfT, _P], Awaitable[_R]],
    ) -> Callable[Concatenate[_SelfT, _P], Awaitable[_R]]:
        """Convert transient Keenable API failures into `ModelRetry`.

        Pydantic AI only feeds `ModelRetry` back to the model as a retry prompt;
        any other exception propagates and aborts the run. Rate limits, transient
        5xx, rejected parameters, a 404 on a dead URL, a request past its
        deadline, and a 2xx whose body is too large or not JSON (a proxy or
        gateway answering in HTML) are things a model can recover from (wait,
        rephrase, pick another URL), so they become retries. A 401/402/403 means
        a bad key or an exhausted account, configuration the model cannot fix, so
        it propagates.
        """
        auth_statuses = frozenset({401, 402, 403})

        @functools.wraps(fn)
        async def wrapper(self: _SelfT, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return await fn(self, *args, **kwargs)
            except httpx.HTTPStatusError as error:
                if error.response.status_code in auth_statuses:
                    raise
                response = error.response
                raise ModelRetry(
                    f'Keenable request failed with HTTP {response.status_code} {response.reason_phrase}.'
                ) from error
            except httpx.HTTPError as error:
                raise ModelRetry(f'Keenable request failed: {error}') from error
            except TimeoutError as error:
                raise ModelRetry('Keenable request timed out.') from error
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                # `json.loads` decodes bytes before parsing them, so a body in a
                # broken encoding fails as a decode error, not a JSON one.
                raise ModelRetry(f'Keenable returned a malformed response: {error}') from error

        return wrapper

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
        # Keenable is asked for `num_results`, so a result without a usable URL
        # still costs a slot. Filtering before the slice matters only when a
        # server returns more than `max_results`.
        found = await self._client.search(query, max_results=self._num_results)
        results = [
            (url, result)
            for result in found
            if (url := self._one_line(result.get('url') or '')) and len(url) <= self._MAX_URL_CHARS
        ]
        results = results[: self._num_results]
        if not results:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})

        lines: list[str] = []
        sources: dict[str, str | None] = {}
        for index, (url, result) in enumerate(results, start=1):
            title = self._one_line(result.get('title') or '')[: self._MAX_TITLE_CHARS] or None
            sources[url] = title
            lines.append(f'{index}. {title or url}\n   {url}')
            excerpt = self._one_line(result.get('snippet') or '')[: self._max_snippet_chars]
            if excerpt:
                lines.append(f'   {excerpt}')

        return ToolReturn('\n'.join(lines), metadata={'sources': self._source_list(sources)})

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

        page_text = truncate_head(content, self._max_page_chars)
        if len(page_text) > self._max_page_chars:
            # `truncate_head` returns its marker alone when the marker is longer
            # than the budget; the budget is a ceiling, so a one-character
            # ellipsis marks the cut instead.
            page_text = f'{content[: self._max_page_chars - 1]}…'
        title = page.get('title') or None
        resolved = str(page.get('url') or url)
        return ToolReturn(page_text, metadata={'sources': self._source_list({resolved: str(title) if title else None})})
