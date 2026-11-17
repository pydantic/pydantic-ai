"""Firecrawl toolset -- gives agents web search and page retrieval backed by the Firecrawl API."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx2
from typing_extensions import TypedDict

from pydantic_ai._ssrf import is_private_ip
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import ToolOperation, raise_retry
from pydantic_ai_harness._json_api import JsonApiClient, MalformedResponse, recoverable
from pydantic_ai_harness._output import one_line, truncate_head_within, validated_budget
from pydantic_ai_harness._typeguards import is_mapping, is_object_list
from pydantic_ai_harness._web_search import defer_to_native_web_search

FIRECRAWL_MAX_RESULTS = 100
"""The most results one Firecrawl search returns; the API rejects a larger `limit`."""

FirecrawlExtractionMode = Literal['highlights', 'full_page']
"""How `web_search` attaches page content: query-relevant highlights or each result's full markdown."""


class FirecrawlSource(TypedDict):
    """One source behind a tool result, carried in `ToolReturn.metadata['sources']`."""

    url: str
    title: str | None


class FirecrawlClient(Protocol):
    """The subset of the Firecrawl API that `FirecrawlSearchToolset` calls.

    Any object with these two methods can back the toolset. Pass one via
    `FirecrawlSearch.client` to configure authentication or the base URL
    explicitly, or to substitute a fake in tests.
    """

    async def search(
        self,
        query: str,
        *,
        limit: int,
        full_page: bool,
        include_domains: Sequence[str],
        exclude_domains: Sequence[str],
    ) -> Sequence[Mapping[str, object]]:
        """Search the web and return up to `limit` raw web results, each with `markdown` when `full_page`."""
        ...  # pragma: no cover

    async def scrape(self, url: str) -> Mapping[str, object]:
        """Scrape one page and return the raw `data` object, with its `markdown` and `metadata`."""
        ...  # pragma: no cover


class HttpFirecrawlClient(JsonApiClient):
    """The default `FirecrawlClient`, talking to the Firecrawl v2 API over `httpx2`.

    Firecrawl's search and scrape endpoints accept keyless requests, so this
    works with no configuration; an API key raises rate limits.
    """

    _DEFAULT_BASE_URL = 'https://api.firecrawl.dev'
    _BASE_URL_ENV = 'FIRECRAWL_API_URL'
    _API_KEY_ENV = 'FIRECRAWL_API_KEY'
    # Above Firecrawl's 60-second server-side default, so a slow search that
    # scrapes every result fails there first, with a status the model sees.
    _TIMEOUT_SECONDS = 90.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Configure the client.

        Args:
            api_key: The Firecrawl API key. Defaults to the `FIRECRAWL_API_KEY`
                environment variable; with neither, requests are keyless.
            base_url: The API base URL, such as a self-hosted deployment.
                Defaults to the `FIRECRAWL_API_URL` environment variable, then
                to the hosted API. It must be `https`, or plain `http` against
                loopback, with no query, fragment, or credentials.
            http_client: The `httpx2.AsyncClient` to send every request through,
                to reuse connections or to configure a proxy, TLS, or a retrying
                transport. Its own timeouts apply, alongside the 90-second
                deadline on each request. It stays open; closing it is up to
                you. Defaults to a new client for each request, closed when the
                request ends.
        """
        key = (api_key if api_key is not None else os.environ.get(self._API_KEY_ENV, '')).strip()
        # HTTPX echoes a header value it can't send in its error, which would carry the key into a retry prompt.
        if any(char.isspace() or not char.isprintable() for char in key):
            source = '`api_key`' if api_key is not None else f'`{self._API_KEY_ENV}`'
            raise UserError(f'{source} must not contain whitespace or control characters')
        super().__init__(
            base_url=base_url, http_client=http_client, headers={'Authorization': f'Bearer {key}'} if key else {}
        )

    async def _data(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        """Post `payload` and return the response's `data` object."""
        body = self._string_keyed(await self._request('POST', path, payload=payload))
        data = body.get('data')
        # Firecrawl answers a 2xx with `success: true` and a `data` object, so
        # anything else is a gateway or proxy answering, not an empty result.
        if body.get('success') is not True or not is_mapping(data):
            raise MalformedResponse('`success` is not true or `data` is not an object')
        return {str(key): value for key, value in data.items()}

    async def search(
        self,
        query: str,
        *,
        limit: int,
        full_page: bool,
        include_domains: Sequence[str],
        exclude_domains: Sequence[str],
    ) -> list[dict[str, object]]:
        """Search the web via `POST /v2/search` and return its `web` results.

        Each result's `description` carries Firecrawl's query-relevant highlights, or the
        site's own description when Firecrawl has none for the page.
        """
        payload: dict[str, object] = {'query': query, 'limit': limit, 'highlights': True}
        if include_domains:
            payload['includeDomains'] = list(include_domains)
        if exclude_domains:
            payload['excludeDomains'] = list(exclude_domains)
        if full_page:
            payload['scrapeOptions'] = {'formats': ['markdown']}
        web = (await self._data('/v2/search', payload)).get('web', [])
        if not is_object_list(web):
            raise MalformedResponse('`data.web` is not a list')
        # `_string_keyed` maps anything that is not an object to `{}`, so filtering
        # on the result drops non-object entries without a second check.
        return [mapped for mapped in (self._string_keyed(entry) for entry in web) if mapped]

    async def scrape(self, url: str) -> dict[str, object]:
        """Scrape one page as markdown via `POST /v2/scrape`."""
        return await self._data('/v2/scrape', {'url': url, 'formats': ['markdown']})


@dataclass(frozen=True)
class FirecrawlSearchOperations:
    """The durable operations a `FirecrawlSearch` capability runs its toolset's Firecrawl requests through."""

    web_search: ToolOperation
    get_page: ToolOperation


def validate_options(
    *,
    max_results: int,
    extraction_mode: FirecrawlExtractionMode,
    max_text_chars: int,
    include_domains: Sequence[str],
    exclude_domains: Sequence[str],
) -> None:
    """Reject options the Firecrawl API or the toolset can't honor, so they fail at construction, not mid-run."""
    validated_budget('max_results', max_results, limit=FIRECRAWL_MAX_RESULTS)
    validated_budget('max_text_chars', max_text_chars)
    if extraction_mode not in ('highlights', 'full_page'):
        raise ValueError(f"`extraction_mode` must be 'highlights' or 'full_page', got {extraction_mode!r}")
    if include_domains and exclude_domains:
        raise ValueError('Specify `include_domains` or `exclude_domains`, not both.')


def _text(value: object) -> str:
    return value if isinstance(value, str) else ''


def _format_result(title: str | None, url: str, body: str) -> str:
    """Render one result as labelled metadata lines followed by its body text."""
    lines = [f'Title: {title or "(untitled)"}', f'URL: {url}']
    if body:
        lines.extend(['', body])
    return '\n'.join(lines)


_IPV4_SHORTHAND = re.compile(r'[0-9a-fx.]+')


def _domain(domain: str) -> str:
    return domain.strip().lower().strip('.')


def _host(url: str) -> str | None:
    try:
        host = httpx2.URL(url).host
    except httpx2.InvalidURL:
        return None
    return _domain(host) or None


def _in_domains(host: str, domains: Sequence[str]) -> bool:
    """Whether `host` is one of `domains` or a subdomain of one, as a `site:` operator matches."""
    return any(host == domain or host.endswith(f'.{domain}') for domain in domains)


def _check_page_url(url: str) -> None:
    """Refuse a URL a scraper should not fetch: one that is not `http(s)`, or names a private or metadata address.

    A self-hosted Firecrawl fetches from inside your network, and the model picks the URL.
    A hostname is not resolved here, since the scraper resolves it, not this process.
    """
    try:
        parsed = httpx2.URL(url)
    except httpx2.InvalidURL:
        parsed = None
    if parsed is None or parsed.scheme not in ('http', 'https') or not parsed.host:
        raise ModelRetry(f'{url!r} is not an absolute http(s) URL.')
    host = _domain(parsed.host)
    address = _ip_address(host)
    if host == 'localhost' or host.endswith('.localhost') or (address is not None and is_private_ip(address)):
        raise ModelRetry(f'{url!r} names a private or local address, which `get_page` does not read.')


def _ip_address(host: str) -> str | None:
    """The IP address `host` spells, including the shorthand IPv4 forms (`127.1`, `0x7f000001`) URL parsers accept."""
    try:
        return str(ipaddress.ip_address(host.split('%', 1)[0]))
    except ValueError:
        pass
    if not _IPV4_SHORTHAND.fullmatch(host):
        return None
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except OSError:
        return None


def _source_list(sources: Mapping[str, str | None]) -> list[FirecrawlSource]:
    return [{'url': url, 'title': title} for url, title in sources.items()]


class FirecrawlSearchToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent web research tools backed by the Firecrawl Search and Scrape APIs.

    `web_search` surveys the web and returns results with query-relevant
    highlights (or each result's full markdown, with
    `extraction_mode='full_page'`), and `get_page` scrapes one specific URL as
    markdown.

    Each tool returns a `ToolReturn` whose `return_value` is the text the model
    sees and whose `metadata['sources']` lists the result URLs and titles
    (`FirecrawlSource` dicts), so applications can render citations from
    `ToolReturnPart.metadata` without parsing the text.

    Each search result body and each page is capped at `max_text_chars`
    characters. Bounds are validated here as well as by `FirecrawlSearch`.

    With `defer_to_native=True`, `web_search` is the local fallback for the
    model's native web search, as `FirecrawlSearch(native=True)` sets up.

    `FirecrawlSearch` passes `operations` so that each tool's Firecrawl request
    runs as one of its durable operations, whose result durable execution
    records instead of making the request again on recovery.
    """

    def __init__(
        self,
        *,
        client: FirecrawlClient | None,
        max_results: int,
        extraction_mode: FirecrawlExtractionMode,
        max_text_chars: int,
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        defer_to_native: bool = False,
        id: str | None = None,
        operations: FirecrawlSearchOperations | None = None,
    ) -> None:
        """Configure the tools.

        Args:
            client: The Firecrawl client the tools call. `None` uses an
                `HttpFirecrawlClient` configured from the `FIRECRAWL_API_KEY`
                and `FIRECRAWL_API_URL` environment variables.
            max_results: Number of results `web_search` asks Firecrawl for,
                from 1 to 100.
            extraction_mode: `'highlights'` for query-relevant excerpts, or
                `'full_page'` for each result's full markdown.
            max_text_chars: Maximum characters of each search result body and
                of the page `get_page` returns. Longer text is truncated and
                marked, the marker counted against the budget; a budget too
                small to hold the marker ends the text with `…` instead.
            include_domains: If non-empty, search results only come from these
                domains.
            exclude_domains: Search results never come from these domains.
            defer_to_native: Make `web_search` the fallback for the model's
                native web search.
            id: An optional unique ID for the toolset, which durable execution
                needs to identify it.
            operations: The durable operations each tool's Firecrawl request
                runs through. `None` makes the request directly.

        Raises:
            ValueError: If a budget is not a positive integer, `max_results`
                is above 100, `extraction_mode` is unknown, or both domain
                lists are set.
        """
        super().__init__(id=id)
        validate_options(
            max_results=max_results,
            extraction_mode=extraction_mode,
            max_text_chars=max_text_chars,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        self._max_results = max_results
        self._max_text_chars = max_text_chars
        self._full_page = extraction_mode == 'full_page'
        self._include_domains = [_domain(domain) for domain in include_domains]
        self._exclude_domains = [_domain(domain) for domain in exclude_domains]
        self._client: FirecrawlClient = client if client is not None else HttpFirecrawlClient()
        self._operations = operations
        self.add_function(
            self.web_search, name='web_search', prepare=defer_to_native_web_search if defer_to_native else None
        )
        self.add_function(self.get_page, name='get_page')

    def _is_usable_result_url(self, url: str) -> bool:
        """Whether a result URL is one to show: well formed, and inside the domain lists.

        Firecrawl applies the lists as `site:` operators, which some self-hosted search
        backends treat as hints, so they are enforced on what comes back as well.
        """
        if not url or any(char.isspace() for char in url):
            return False
        host = _host(url)
        if host is None:
            return False
        if self._include_domains and not _in_domains(host, self._include_domains):
            return False
        return not _in_domains(host, self._exclude_domains)

    @recoverable('Firecrawl')
    async def web_search(self, query: str) -> ToolReturn[str]:
        """Search the web and return matching pages, each with relevant excerpts or its page content.

        Args:
            query: The search query. Natural-language questions and keyword
                queries both work.

        Returns:
            The matching pages, each with title, URL, and excerpts or page content.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.web_search(query))
        found = await self._client.search(
            query,
            limit=self._max_results,
            full_page=self._full_page,
            include_domains=self._include_domains,
            exclude_domains=self._exclude_domains,
        )
        sources: dict[str, str | None] = {}
        sections: list[str] = []
        for result in found:
            url = _text(result.get('url')).strip()
            if not self._is_usable_result_url(url):
                continue
            title = one_line(_text(result.get('title'))) or None
            # A result whose scrape failed has no markdown, so its highlight stands in. The highlight
            # goes on one line, so page text can't start a line that reads as another result.
            body = (_text(result.get('markdown')) if self._full_page else '') or one_line(
                _text(result.get('description'))
            )
            sources.setdefault(url, title)
            sections.append(_format_result(title, url, truncate_head_within(body, self._max_text_chars)))
            if len(sections) >= self._max_results:
                break
        if not sections:
            return ToolReturn(f'No results found for {query!r}.', metadata={'sources': []})
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        return ToolReturn(
            f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}',
            metadata={'sources': _source_list(sources)},
        )

    @recoverable('Firecrawl')
    async def get_page(self, url: str) -> ToolReturn[str]:
        """Retrieve the markdown of a specific URL.

        Use it to read a promising URL from `web_search` results in full, or a
        URL the user provided.

        Args:
            url: The absolute URL of the page to read.

        Returns:
            The page's title, URL, and markdown content.
        """
        if self._operations is not None:
            return raise_retry(await self._operations.get_page(url))
        _check_page_url(url)
        page = await self._client.scrape(url)
        metadata = page.get('metadata')
        details: Mapping[object, object] = metadata if is_mapping(metadata) else {}
        # Firecrawl scrapes an error page successfully and reports the site's status in the metadata.
        status = details.get('statusCode')
        if isinstance(status, int) and status >= 400:
            raise ModelRetry(f'{url!r} returned HTTP {status}. Check the URL or try another page.')
        markdown = _text(page.get('markdown'))
        if not markdown:
            raise ModelRetry(f'No content could be retrieved for {url!r}. Check the URL or try another page.')
        resolved = one_line(_text(details.get('url'))) or url
        title = one_line(_text(details.get('title'))) or None
        return ToolReturn(
            _format_result(title, resolved, truncate_head_within(markdown, self._max_text_chars)),
            metadata={'sources': _source_list({resolved: title})},
        )
