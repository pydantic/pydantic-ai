from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest

from pydantic_ai.common_tools.bocha import BochaSearchTool, bocha_search_tool

from .conftest import ClientWithHandler


@pytest.mark.anyio
async def test_bocha_search_maps_nested_web_pages_response(client_with_handler: ClientWithHandler) -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                'data': {
                    'webPages': {
                        'value': [
                            {
                                'name': 'Pydantic AI',
                                'url': 'https://ai.pydantic.dev/',
                                'summary': 'Agent framework for production AI.',
                                'siteName': 'Pydantic',
                                'datePublished': '2026-01-02',
                            }
                        ]
                    }
                }
            },
        )

    client = client_with_handler(handler)
    search_tool = BochaSearchTool(client=client, api_key='test-key', count=3)

    results = await search_tool(
        'pydantic ai',
        freshness='oneWeek',
        summary=True,
        include_domains=['ai.pydantic.dev', 'docs.pydantic.dev'],
        exclude_domains=['example.com'],
    )

    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.url == 'https://api.bochaai.com/v1/web-search'
    assert request.headers['Authorization'] == 'Bearer test-key'
    assert json.loads(request.content) == {
        'query': 'pydantic ai',
        'freshness': 'oneWeek',
        'summary': True,
        'count': 3,
        'include': 'ai.pydantic.dev|docs.pydantic.dev',
        'exclude': 'example.com',
    }
    assert results == [
        {
            'title': 'Pydantic AI',
            'url': 'https://ai.pydantic.dev/',
            'content': 'Agent framework for production AI.',
            'site_name': 'Pydantic',
            'published_date': '2026-01-02',
        }
    ]


@pytest.mark.anyio
async def test_bocha_search_maps_top_level_web_pages_response(client_with_handler: ClientWithHandler) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'webPages': {
                    'value': [
                        {
                            'title': 'Fallback title',
                            'displayUrl': 'https://example.com/page',
                            'snippet': 'Fallback snippet.',
                            'dateLastCrawled': '2026-02-03',
                        }
                    ]
                }
            },
        )

    client = client_with_handler(handler)
    search_tool = BochaSearchTool(client=client, api_key='test-key')

    assert await search_tool('fallback fields') == [
        {
            'title': 'Fallback title',
            'url': 'https://example.com/page',
            'content': 'Fallback snippet.',
            'site_name': None,
            'published_date': '2026-02-03',
        }
    ]


@pytest.mark.anyio
async def test_bocha_search_raises_for_http_errors(client_with_handler: ClientWithHandler) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, text='unauthorized')

    client = client_with_handler(handler)
    search_tool = BochaSearchTool(client=client, api_key='bad-key')

    with pytest.raises(httpx.HTTPStatusError):
        await search_tool('secret')


def test_bocha_search_tool_requires_api_key() -> None:
    with pytest.raises(ValueError, match='api_key must be provided'):
        bocha_search_tool()


@pytest.mark.anyio
async def test_bocha_search_tool_hides_fixed_arguments_from_schema(client_with_handler: ClientWithHandler) -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(200, json={'data': {'webPages': {'value': []}}})

    client = client_with_handler(handler)

    tool = bocha_search_tool(
        'test-key',
        client=client,
        count=10,
        freshness='oneDay',
        summary=True,
        include_domains=['ai.pydantic.dev'],
        exclude_domains=['example.com'],
    )

    assert tool.name == 'bocha_search'
    assert set(tool.tool_def.parameters_json_schema['properties']) == {'query'}
    assert await cast(Any, tool.function)('schema search') == []
    assert json.loads(captured_requests[0].content) == {
        'query': 'schema search',
        'freshness': 'oneDay',
        'summary': True,
        'count': 10,
        'include': 'ai.pydantic.dev',
        'exclude': 'example.com',
    }


@pytest.mark.anyio
async def test_bocha_search_tool_creates_default_client(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[httpx.AsyncClient] = []
    async_client = httpx.AsyncClient

    def async_client_factory() -> httpx.AsyncClient:
        client = async_client()
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, 'AsyncClient', async_client_factory)

    tool = bocha_search_tool('test-key')

    assert len(clients) == 1
    assert cast(Any, tool.function).__self__.client is clients[0]
    assert set(tool.tool_def.parameters_json_schema['properties']) == {
        'query',
        'freshness',
        'summary',
        'include_domains',
        'exclude_domains',
    }
    await clients[0].aclose()
