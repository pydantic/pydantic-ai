from __future__ import annotations

from collections.abc import Mapping, Sequence

import httpx
import pytest

from pydantic_ai import Agent
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

    async def crawl(self, url: str, *, limit: int) -> Sequence[Mapping[str, object]]:
        return self.crawl_pages

    async def search(
        self, query: str, *, search_limit: int, fetch_page_content: bool
    ) -> Sequence[Mapping[str, object]]:
        self.search_options = (query, search_limit, fetch_page_content)
        return self.search_pages

    async def scrape(self, url: str, *, request: str) -> Sequence[Mapping[str, object]]:
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
