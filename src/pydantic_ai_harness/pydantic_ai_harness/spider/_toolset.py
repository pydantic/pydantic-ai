"""Spider API toolset for bounded web crawling, search, and rendering."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeGuard
from urllib.parse import urlsplit

import httpx

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._typeguards import is_object_dict, is_object_list

_DEFAULT_BASE_URL = 'https://api.spider.cloud'
_MAX_URL_CHARS = 2048


class SpiderClient(Protocol):
    """The Spider API operations used by the capability."""

    async def crawl(self, url: str, *, limit: int) -> Sequence[Mapping[str, object]]: ...

    async def search(
        self, query: str, *, search_limit: int, fetch_page_content: bool
    ) -> Sequence[Mapping[str, object]]: ...

    async def scrape(self, url: str, *, request: str) -> Sequence[Mapping[str, object]]: ...


def _is_page_list(value: object) -> TypeGuard[list[dict[str, object]]]:
    return is_object_list(value) and all(
        is_object_dict(page) and all(isinstance(key, str) for key in page) for page in value
    )


class HttpSpiderClient:
    """HTTP implementation of Spider's JSON API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv('SPIDER_API_KEY')
        self._base_url = (base_url or os.getenv('SPIDER_API_URL') or _DEFAULT_BASE_URL).rstrip('/')
        parsed = urlsplit(self._base_url)
        if (
            parsed.scheme != 'https'
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError('Spider `base_url` must be an HTTPS URL without credentials, query, or fragment')
        self._transport = transport

    async def crawl(self, url: str, *, limit: int) -> Sequence[Mapping[str, object]]:
        return await self._post('crawl', {'url': url, 'limit': limit})

    async def search(
        self, query: str, *, search_limit: int, fetch_page_content: bool
    ) -> Sequence[Mapping[str, object]]:
        return await self._post(
            'search',
            {'search': query, 'search_limit': search_limit, 'fetch_page_content': fetch_page_content},
        )

    async def scrape(self, url: str, *, request: str) -> Sequence[Mapping[str, object]]:
        return await self._post('scrape', {'url': url, 'return_format': 'markdown', 'request': request})

    async def _post(self, endpoint: str, payload: dict[str, object]) -> Sequence[Mapping[str, object]]:
        if not self._api_key:
            raise UserError('Spider requires an API key. Set `SPIDER_API_KEY` or pass `api_key` to `Spider`.')
        async with httpx.AsyncClient(transport=self._transport, timeout=30.0, follow_redirects=False) as client:
            response = await client.post(
                f'{self._base_url}/{endpoint}',
                json=payload,
                headers={'Authorization': f'Bearer {self._api_key}'},
                follow_redirects=False,
            )
            response.raise_for_status()
            body = response.json()
        if not _is_page_list(body):
            raise ModelRetry('Spider returned a malformed response; retry the request.')
        return body


class SpiderToolset(FunctionToolset[AgentDepsT]):
    """Tools that return bounded, untrusted Spider results."""

    def __init__(
        self,
        *,
        client: SpiderClient,
        max_crawl_pages: int,
        num_search_results: int,
        max_page_chars: int,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        for name, value in (
            ('max_crawl_pages', max_crawl_pages),
            ('num_search_results', num_search_results),
            ('max_page_chars', max_page_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'`{name}` must be a positive integer, got {value!r}')
        self._client = client
        self._max_crawl_pages = max_crawl_pages
        self._num_search_results = num_search_results
        self._max_page_chars = max_page_chars
        self.add_function(self.crawl_site, name='crawl_site')
        self.add_function(self.web_search, name='web_search')
        self.add_function(self.scrape_page, name='scrape_page')

    @staticmethod
    def _one_line(value: object) -> str:
        return ' '.join(str(value or '').split())

    def _page_text(self, page: Mapping[str, object], index: int) -> tuple[str, str, str]:
        url = self._one_line(page.get('url'))[:_MAX_URL_CHARS]
        title = self._one_line(page.get('title'))[:300]
        content = self._one_line(page.get('content'))
        if len(content) > self._max_page_chars:
            content = content[: self._max_page_chars - 1] + '…'
        heading = title or url or f'Page {index}'
        return heading, url, content

    async def crawl_site(self, url: str) -> ToolReturn[str]:
        """Crawl a site and return at most the configured number of pages."""
        pages = (await self._client.crawl(url, limit=self._max_crawl_pages))[: self._max_crawl_pages]
        if not pages:
            return ToolReturn('Spider found no pages.', metadata={'sources': []})
        blocks: list[str] = []
        sources: list[dict[str, str | None]] = []
        for index, page in enumerate(pages, start=1):
            title, page_url, content = self._page_text(page, index)
            if page_url:
                sources.append({'url': page_url, 'title': title})
            blocks.append(f'{index}. {title}\n{page_url}\n{content}')
        return ToolReturn('\n\n'.join(blocks), metadata={'sources': sources})

    async def web_search(self, query: str, fetch_content: bool = False) -> ToolReturn[str]:
        """Search the web; optionally ask Spider to fetch each result's page content."""
        pages = (
            await self._client.search(query, search_limit=self._num_search_results, fetch_page_content=fetch_content)
        )[: self._num_search_results]
        blocks: list[str] = []
        sources: list[dict[str, str | None]] = []
        for index, page in enumerate(pages, start=1):
            title, url, content = self._page_text(page, index)
            description = self._one_line(page.get('description') or page.get('snippet'))
            excerpt = content if fetch_content else description
            if url:
                sources.append({'url': url, 'title': title})
            blocks.append(f'{index}. {title}\n{url}\n{excerpt}')
        text = '\n\n'.join(blocks) if blocks else f'No results found for {query!r}.'
        return ToolReturn(text, metadata={'sources': sources})

    async def scrape_page(self, url: str) -> ToolReturn[str]:
        """Fetch one page as markdown using Spider's JavaScript-rendering browser mode."""
        pages = await self._client.scrape(url, request='browser')
        if not pages:
            raise ModelRetry(f'Spider returned no readable content for {url!r}.')
        title, resolved_url, content = self._page_text(pages[0], 1)
        if not content:
            raise ModelRetry(f'Spider returned no readable content for {url!r}.')
        return ToolReturn(content, metadata={'sources': [{'url': resolved_url or url, 'title': title}]})
