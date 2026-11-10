from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness import Spider as RootSpider
from pydantic_ai_harness.spider import HttpSpiderClient, Spider, SpiderToolset


def _text(returned: ToolReturn[str]) -> str:
    # `ToolReturn.return_value` is typed as the full multi-modal content union,
    # so narrow it once here rather than at every assertion.
    value = returned.return_value
    assert isinstance(value, str)
    return value


class FakeSpiderClient:
    def __init__(self) -> None:
        self.crawl_pages: list[dict[str, object]] = []
        self.search_pages: list[dict[str, object]] = []
        self.search_options: tuple[str, int, bool] | None = None
        self.scrape_pages: list[dict[str, object]] | None = None

    async def crawl(self, url: str, *, limit: int) -> Sequence[Mapping[str, object]]:
        return self.crawl_pages

    async def search(
        self, query: str, *, search_limit: int, fetch_page_content: bool
    ) -> Sequence[Mapping[str, object]]:
        self.search_options = (query, search_limit, fetch_page_content)
        return self.search_pages

    async def scrape(self, url: str, *, request: str) -> Sequence[Mapping[str, object]]:
        if self.scrape_pages is not None:
            return self.scrape_pages
        return [{'url': url, 'content': 'rendered markdown', 'title': 'Rendered'}]


async def test_spider_capability_lazy_root_export_and_agent_composition() -> None:
    assert RootSpider is Spider
    agent = Agent(TestModel(), capabilities=[Spider(client=FakeSpiderClient())])
    result = await agent.run('hello')
    assert result.output


async def test_crawl_tool_returns_capped_page_items() -> None:
    client = FakeSpiderClient()
    client.crawl_pages = [
        {'url': f'https://example.com/{i}', 'title': f'Page {i}', 'content': 'body'} for i in range(4)
    ]
    toolset = SpiderToolset(client=client, max_crawl_pages=2, num_search_results=3, max_page_chars=100)

    result = await toolset.crawl_site('https://example.com')

    text = _text(result)
    assert text.count('https://example.com/') == 2
    assert 'Page 0' in text and 'Page 1' in text
    assert 'Page 2' not in text


async def test_search_tool_returns_results_and_fetches_content() -> None:
    client = FakeSpiderClient()
    client.search_pages = [
        {'url': 'https://example.com/a', 'title': 'A', 'description': 'snippet', 'content': 'full page'}
    ]
    toolset = SpiderToolset(client=client, max_crawl_pages=2, num_search_results=3, max_page_chars=100)

    result = await toolset.web_search('query', fetch_content=True)

    assert client.search_options == ('query', 3, True)
    text = _text(result)
    assert 'full page' in text
    assert 'snippet' not in text


async def test_api_key_never_follows_redirects() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(302, headers={'Location': 'https://attacker.example/collect'})

    client = HttpSpiderClient(
        api_key='secret',
        base_url='https://api.spider.cloud',
        transport=httpx.MockTransport(respond),
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.scrape('https://example.com', request='browser')
    assert len(requests) == 1
    assert requests[0].headers['Authorization'] == 'Bearer secret'


def test_spider_rejects_non_positive_page_limits() -> None:
    with pytest.raises(ValueError, match='`max_crawl_pages` must be a positive integer'):
        Spider(max_crawl_pages=0)
    with pytest.raises(ValueError, match='`num_search_results` must be a positive integer'):
        Spider(num_search_results=True)


def test_http_client_rejects_non_https_base_url() -> None:
    with pytest.raises(ValueError, match='must be an HTTPS URL'):
        HttpSpiderClient(api_key='key', base_url='http://api.spider.cloud')


async def test_http_client_posts_and_returns_page_lists() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{'url': 'https://example.com'}, {'title': 'Another page'}])

    client = HttpSpiderClient(api_key='key', transport=httpx.MockTransport(respond))

    assert len(await client.crawl('https://example.com', limit=2)) == 2
    assert len(await client.search('query', search_limit=2, fetch_page_content=False)) == 2
    assert len(await client.scrape('https://example.com', request='browser')) == 2
    assert [request.url.path for request in requests] == ['/crawl', '/search', '/scrape']


async def test_http_client_retries_on_malformed_page_list() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'error': 'unexpected response'})

    client = HttpSpiderClient(api_key='key', transport=httpx.MockTransport(respond))

    with pytest.raises(ModelRetry, match='malformed response'):
        await client.crawl('https://example.com', limit=1)


async def test_http_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SPIDER_API_KEY', raising=False)
    client = HttpSpiderClient()

    with pytest.raises(UserError, match='requires an API key'):
        await client.crawl('https://example.com', limit=1)


def test_toolset_rejects_non_positive_page_limits() -> None:
    with pytest.raises(ValueError, match='`max_crawl_pages` must be a positive integer'):
        SpiderToolset(client=FakeSpiderClient(), max_crawl_pages=0, num_search_results=1, max_page_chars=1)


async def test_crawl_site_truncates_content_and_omits_empty_url_source() -> None:
    client = FakeSpiderClient()
    client.crawl_pages = [{'title': '', 'content': 'long content'}]
    toolset = SpiderToolset(client=client, max_crawl_pages=1, num_search_results=1, max_page_chars=8)

    result = await toolset.crawl_site('https://example.com')

    assert _text(result).endswith('long co…')
    assert result.metadata == {'sources': []}


async def test_web_search_omits_empty_url_source() -> None:
    client = FakeSpiderClient()
    client.search_pages = [{'title': 'No URL', 'snippet': 'search excerpt'}]
    toolset = SpiderToolset(client=client, max_crawl_pages=1, num_search_results=1, max_page_chars=100)

    result = await toolset.web_search('query')

    assert 'search excerpt' in _text(result)
    assert result.metadata == {'sources': []}


async def test_scrape_page_retries_without_readable_content() -> None:
    client = FakeSpiderClient()
    toolset = SpiderToolset(client=client, max_crawl_pages=1, num_search_results=1, max_page_chars=100)

    client.scrape_pages = []
    with pytest.raises(ModelRetry, match='no readable content'):
        await toolset.scrape_page('https://example.com')

    client.scrape_pages = [{'title': 'Empty page', 'content': ''}]
    with pytest.raises(ModelRetry, match='no readable content'):
        await toolset.scrape_page('https://example.com')


async def test_scrape_page_returns_markdown_and_source() -> None:
    client = FakeSpiderClient()
    toolset = SpiderToolset(client=client, max_crawl_pages=1, num_search_results=1, max_page_chars=100)

    result = await toolset.scrape_page('https://example.com')

    assert _text(result) == 'rendered markdown'
    assert result.metadata == {'sources': [{'url': 'https://example.com', 'title': 'Rendered'}]}


def test_spider_combine_keeps_one_configuration() -> None:
    capability = Spider(client=FakeSpiderClient())

    assert Spider.combine([capability]) is capability
