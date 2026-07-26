"""Tests for the Nimble search tool."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from inline_snapshot import snapshot
from nimble_python import AsyncNimble
from nimble_python.types.search_response import Result, ResultMetadataSerpMetadata, SearchResponse

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.common_tools.nimble import NimbleSearchTool, nimble_search_tool
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import IsStr

pytestmark = [pytest.mark.anyio]


@pytest.mark.vcr()
async def test_basic_search(nimble_api_key: str):
    """Test basic search with default parameters."""
    tool = NimbleSearchTool(
        client=AsyncNimble(api_key=nimble_api_key, client_source='pydantic-ai'),
        max_results=3,
    )
    results = await tool('What is Pydantic AI?')
    assert len(results) == 3
    assert results == snapshot(
        [
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
        ]
    )


@pytest.mark.vcr()
async def test_search_with_include_domains(nimble_api_key: str):
    """Test search with include_domains filtering."""
    tool = NimbleSearchTool(client=AsyncNimble(api_key=nimble_api_key, client_source='pydantic-ai'), max_results=3)
    results = await tool('transformer architectures', include_domains=['arxiv.org'])
    assert len(results) == 3
    assert all('arxiv.org' in r['url'] for r in results)
    assert results == snapshot(
        [
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
            {'title': IsStr(), 'url': IsStr(), 'content': IsStr()},
        ]
    )


@pytest.mark.vcr()
async def test_search_with_exclude_domains(nimble_api_key: str):
    """Test search with exclude_domains filtering."""
    tool = NimbleSearchTool(client=AsyncNimble(api_key=nimble_api_key, client_source='pydantic-ai'), max_results=3)
    results = await tool('Pydantic AI', exclude_domains=['medium.com'])
    assert len(results) >= 1
    assert all('medium.com' not in r['url'] for r in results)


@pytest.mark.vcr()
async def test_search_with_time_range(nimble_api_key: str):
    """Test search with time_range filtering."""
    tool = NimbleSearchTool(client=AsyncNimble(api_key=nimble_api_key, client_source='pydantic-ai'), max_results=3)
    results = await tool('generative AI news', time_range='week')
    assert len(results) >= 1
    assert all(r['title'] and r['url'] for r in results)


@pytest.mark.vcr()
async def test_search_with_search_depth(nimble_api_key: str):
    """Test search with a non-default search_depth."""
    tool = NimbleSearchTool(client=AsyncNimble(api_key=nimble_api_key, client_source='pydantic-ai'), max_results=2)
    results = await tool('What is Pydantic AI?', search_depth='fast')
    assert len(results) >= 1
    assert all(r['content'] for r in results)


@pytest.mark.vcr()
async def test_factory_with_bound_params(nimble_api_key: str):
    """Test factory-bound params are forwarded through FunctionSchema.call."""
    tool = nimble_search_tool(nimble_api_key, max_results=2, include_domains=['arxiv.org'])
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    results = await tool.function_schema.call({'query': 'attention mechanisms'}, ctx)
    assert len(results) >= 1
    assert all('arxiv.org' in r['url'] for r in results)


@pytest.mark.vcr()
async def test_agent_calls_nimble_search(nimble_api_key: str):
    """Public-API path: Agent + TestModel invokes nimble_search and hits Nimble over VCR."""
    agent = Agent(
        TestModel(call_tools=['nimble_search']),
        tools=[nimble_search_tool(nimble_api_key, max_results=2)],
    )
    result = await agent.run('Search for Pydantic AI')
    assert result.output
    tool_calls = [
        p
        for m in result.all_messages()
        for p in getattr(m, 'parts', [])
        if getattr(p, 'part_kind', None) == 'tool-call'
    ]
    assert any(getattr(p, 'tool_name', None) == 'nimble_search' for p in tool_calls)


def test_client_source_header(monkeypatch: pytest.MonkeyPatch):
    """VCR matchers do not fail if X-Client-Source is dropped — pin factory client_source."""
    import pydantic_ai.common_tools.nimble as nimble_mod

    captured: dict[str, Any] = {}
    real_client = nimble_mod.AsyncNimble

    def tracking_client(*args: Any, **kwargs: Any) -> AsyncNimble:
        captured.update(kwargs)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(nimble_mod, 'AsyncNimble', tracking_client)
    tool = nimble_search_tool('test-key')
    assert tool.name == 'nimble_search'
    assert captured.get('client_source') == 'pydantic-ai'
    assert captured.get('api_key') == 'test-key'


async def test_request_body_defaults(nimble_api_key: str):
    """Cassette match-on may ignore body drift — pin lite default and max_results via httpx mock."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['headers'] = dict(request.headers)
        captured['body'] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                'request_id': '00000000-0000-0000-0000-000000000001',
                'results': [
                    {
                        'title': 'Example',
                        'url': 'https://example.com',
                        'content': 'body',
                        'description': 'desc',
                        'metadata': {
                            'country': 'US',
                            'entity_type': 'organic',
                            'locale': 'en',
                            'position': 1,
                        },
                    }
                ],
                'total_results': 1,
            },
        )

    transport = httpx.MockTransport(handler)
    client = AsyncNimble(
        api_key=nimble_api_key,
        client_source='pydantic-ai',
        http_client=httpx.AsyncClient(transport=transport),
    )
    tool = NimbleSearchTool(client=client, max_results=3)
    results = await tool('hello world')
    assert results == [{'title': 'Example', 'url': 'https://example.com', 'content': 'body'}]
    assert captured['body']['query'] == 'hello world'
    assert captured['body']['search_depth'] == 'lite'
    assert captured['body']['max_results'] == 3
    assert captured['headers'].get('x-client-source') == 'pydantic-ai'


async def test_result_content_fallback():
    """Empty content should fall back to description (lite depth) without a network call."""
    response = SearchResponse(
        request_id='00000000-0000-0000-0000-000000000002',
        total_results=1,
        results=[
            Result(
                title='Lite Result',
                url='https://example.com/lite',
                content='',
                description='snippet only',
                metadata=ResultMetadataSerpMetadata(
                    country='US',
                    entity_type='organic',
                    locale='en',
                    position=1,
                ),
            )
        ],
    )
    client = AsyncMock()
    client.search = AsyncMock(return_value=response)
    tool = NimbleSearchTool(client=cast(AsyncNimble, client))
    results = await tool('q')
    assert results == [
        {
            'title': 'Lite Result',
            'url': 'https://example.com/lite',
            'content': 'snippet only',
        }
    ]


def test_no_params_bound_exposes_all_in_schema(nimble_api_key: str):
    """Test that with no factory params, all parameters appear in the tool schema."""
    tool = nimble_search_tool(nimble_api_key)
    assert tool.name == snapshot('nimble_search')
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {
                    'description': 'The search query to execute with Nimble.',
                    'type': 'string',
                },
                'search_depth': {
                    'default': 'lite',
                    'description': 'Controls content richness and latency of search results.',
                    'enum': ['lite', 'fast', 'deep'],
                    'type': 'string',
                },
                'time_range': {
                    'anyOf': [
                        {'enum': ['hour', 'day', 'week', 'month', 'year'], 'type': 'string'},
                        {'type': 'null'},
                    ],
                    'default': None,
                    'description': 'The time range back from the current date to filter results.',
                },
                'include_domains': {
                    'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                    'default': None,
                    'description': 'List of domains to specifically include in the search results.',
                },
                'exclude_domains': {
                    'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                    'default': None,
                    'description': 'List of domains to specifically exclude from the search results.',
                },
            },
            'required': ['query'],
            'type': 'object',
        }
    )


def test_bound_params_hidden_from_schema(nimble_api_key: str):
    """Test that factory-provided params are excluded from the tool schema."""
    tool = nimble_search_tool(
        nimble_api_key,
        search_depth='deep',
        time_range='week',
        include_domains=['arxiv.org'],
        exclude_domains=['medium.com'],
    )
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {
                    'description': 'The search query to execute with Nimble.',
                    'type': 'string',
                },
            },
            'required': ['query'],
            'type': 'object',
        }
    )


def test_factory_requires_api_key_or_client():
    """Test that nimble_search_tool raises when neither api_key nor client is provided."""
    with pytest.raises(ValueError, match='Either api_key or client must be provided'):
        nimble_search_tool()  # pyright: ignore[reportCallIssue]


def test_factory_with_client():
    """Test that nimble_search_tool accepts a pre-built client."""
    client = AsyncNimble(api_key='test-key', client_source='custom')
    tool = nimble_search_tool(client=client)
    assert tool.name == 'nimble_search'
    # Pre-built client attribution is left untouched.
    assert client.client_source == 'custom'


def test_import_error_mentions_nimble_extra(monkeypatch: pytest.MonkeyPatch):
    """Missing nimble_python should raise ImportError pointing at pydantic-ai-slim[nimble]."""
    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any):
        if name == 'nimble_python' or name.startswith('nimble_python.'):
            raise ImportError('mocked missing nimble_python')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    sys.modules.pop('pydantic_ai.common_tools.nimble', None)
    sys.modules.pop('nimble_python', None)
    with pytest.raises(ImportError, match=r'pydantic-ai-slim\[nimble\]'):
        importlib.import_module('pydantic_ai.common_tools.nimble')
