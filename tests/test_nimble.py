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
from pydantic_ai.common_tools.nimble import (
    NimbleAgentRunResultTool,
    NimbleAgentRunStartTool,
    NimbleAgentRunStatusTool,
    NimbleAgentsListTool,
    NimbleAgentTemplatesListTool,
    NimbleCrawlStartTool,
    NimbleCrawlStatusTool,
    NimbleExtractTool,
    NimbleMapTool,
    NimbleSearchTool,
    NimbleToolset,
    nimble_extract_tool,
    nimble_search_tool,
)
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


async def test_extract_returns_markdown():
    """Extract projects markdown from SDK response without a network call."""
    data = AsyncMock()
    data.markdown = '# Hello'
    response = AsyncMock()
    response.data = data
    client = AsyncMock()
    client.extract = AsyncMock()
    client.extract.run = AsyncMock(return_value=response)
    tool = NimbleExtractTool(client=cast(AsyncNimble, client))
    assert await tool('https://example.com') == '# Hello'
    client.extract.run.assert_awaited_once_with(url='https://example.com', formats=['markdown'])


async def test_extract_empty_when_no_markdown():
    """Extract returns empty string when markdown is missing."""
    response = AsyncMock()
    response.data = None
    client = AsyncMock()
    client.extract = AsyncMock()
    client.extract.run = AsyncMock(return_value=response)
    tool = NimbleExtractTool(client=cast(AsyncNimble, client))
    assert await tool('https://example.com') == ''


async def test_map_projects_links():
    """Map projects link fields from the SDK response."""
    link = AsyncMock()
    link.url = 'https://example.com/a'
    link.title = 'A'
    link.description = 'desc'
    response = AsyncMock()
    response.links = [link]
    client = AsyncMock()
    client.map = AsyncMock(return_value=response)
    tool = NimbleMapTool(client=cast(AsyncNimble, client))
    assert await tool(
        'https://example.com',
        limit=10,
        domain_filter='domain',
        sitemap='include',
    ) == [{'url': 'https://example.com/a', 'title': 'A', 'description': 'desc'}]
    client.map.assert_awaited_once_with(
        url='https://example.com',
        limit=10,
        domain_filter='domain',
        sitemap='include',
    )


async def test_map_omits_unset_optional_kwargs():
    """Map only forwards optional kwargs that the caller sets."""
    response = AsyncMock()
    response.links = []
    client = AsyncMock()
    client.map = AsyncMock(return_value=response)
    await NimbleMapTool(client=cast(AsyncNimble, client))('https://example.com')
    client.map.assert_awaited_once_with(url='https://example.com')


async def test_crawl_start_and_status():
    """Crawl start/status are separate tools and do not poll."""
    start_response = AsyncMock()
    start_response.crawl_id = 'crawl_1'
    start_response.status = 'queued'
    start_response.url = 'https://example.com'
    start_response.completed = 0
    start_response.failed = 0
    start_response.pending = 1
    start_response.total = 1
    status_response = AsyncMock()
    status_response.crawl_id = 'crawl_1'
    status_response.status = 'running'
    status_response.url = 'https://example.com'
    status_response.completed = 0
    status_response.failed = 0
    status_response.pending = 1
    status_response.total = 1
    client = AsyncMock()
    client.crawl = AsyncMock()
    client.crawl.run = AsyncMock(return_value=start_response)
    client.crawl.status = AsyncMock(return_value=status_response)

    started = await NimbleCrawlStartTool(client=cast(AsyncNimble, client))(
        'https://example.com',
        limit=5,
        max_discovery_depth=2,
        include_paths=['/docs'],
        exclude_paths=['/admin'],
        sitemap='skip',
        name='docs-crawl',
    )
    assert started['crawl_id'] == 'crawl_1'
    assert started['status'] == 'queued'
    client.crawl.run.assert_awaited_once_with(
        url='https://example.com',
        limit=5,
        max_discovery_depth=2,
        include_paths=['/docs'],
        exclude_paths=['/admin'],
        sitemap='skip',
        name='docs-crawl',
    )
    status = await NimbleCrawlStatusTool(client=cast(AsyncNimble, client))('crawl_1')
    assert status['status'] == 'running'
    client.crawl.status.assert_awaited_once_with('crawl_1')


async def test_crawl_start_omits_unset_optional_kwargs():
    """Crawl start only forwards optional kwargs that the caller sets."""
    start_response = AsyncMock()
    start_response.crawl_id = 'crawl_2'
    start_response.status = 'queued'
    start_response.url = 'https://example.com'
    start_response.completed = 0
    start_response.failed = 0
    start_response.pending = 1
    start_response.total = 1
    client = AsyncMock()
    client.crawl = AsyncMock()
    client.crawl.run = AsyncMock(return_value=start_response)
    await NimbleCrawlStartTool(client=cast(AsyncNimble, client))('https://example.com')
    client.crawl.run.assert_awaited_once_with(url='https://example.com')


async def test_agent_api_lifecycle_tools():
    """Agent list/start/status/result tools map onto SDK Agent API V2 methods."""
    item = AsyncMock()
    item.model_dump = lambda mode='json': {'id': 'agent_1', 'name': 'Research'}
    list_response = AsyncMock()
    list_response.items = [item]
    template_item = AsyncMock()
    template_item.model_dump = lambda mode='json': {'template_name': 'research'}
    templates_response = AsyncMock()
    templates_response.items = [template_item]
    start_response = AsyncMock()
    start_response.model_dump = lambda mode='json': {'id': 'run_1', 'status': 'queued'}
    status_response = AsyncMock()
    status_response.model_dump = lambda mode='json': {'id': 'run_1', 'status': 'running'}
    result_response = AsyncMock()
    result_response.model_dump = lambda mode='json': {'id': 'run_1', 'output': 'done'}

    client = AsyncMock()
    client.agents = AsyncMock()
    client.agents.list = AsyncMock(return_value=list_response)
    client.agents.templates = AsyncMock()
    client.agents.templates.list = AsyncMock(return_value=templates_response)
    client.agents.runs = AsyncMock()
    client.agents.runs.create = AsyncMock(return_value=start_response)
    client.agents.runs.get = AsyncMock(return_value=status_response)
    client.agents.runs.result = AsyncMock(return_value=result_response)

    assert await NimbleAgentsListTool(client=cast(AsyncNimble, client))() == [{'id': 'agent_1', 'name': 'Research'}]
    client.agents.list.assert_awaited_once_with()
    assert await NimbleAgentsListTool(client=cast(AsyncNimble, client))(limit=10, offset=5) == [
        {'id': 'agent_1', 'name': 'Research'}
    ]
    client.agents.list.assert_awaited_with(limit=10, offset=5)
    assert await NimbleAgentTemplatesListTool(client=cast(AsyncNimble, client))() == [{'template_name': 'research'}]
    client.agents.templates.list.assert_awaited_once_with()
    assert await NimbleAgentTemplatesListTool(client=cast(AsyncNimble, client))(limit=3, offset=1) == [
        {'template_name': 'research'}
    ]
    client.agents.templates.list.assert_awaited_with(limit=3, offset=1)
    assert await NimbleAgentRunStartTool(client=cast(AsyncNimble, client))('agent_1', 'Find AI news') == {
        'id': 'run_1',
        'status': 'queued',
    }
    client.agents.runs.create.assert_awaited_once_with(agent_id='agent_1', input='Find AI news')
    assert await NimbleAgentRunStartTool(client=cast(AsyncNimble, client))('agent_1', 'Find AI news', effort='low') == {
        'id': 'run_1',
        'status': 'queued',
    }
    client.agents.runs.create.assert_awaited_with(agent_id='agent_1', input='Find AI news', effort='low')
    assert await NimbleAgentRunStatusTool(client=cast(AsyncNimble, client))('agent_1', 'run_1') == {
        'id': 'run_1',
        'status': 'running',
    }
    assert await NimbleAgentRunResultTool(client=cast(AsyncNimble, client))('agent_1', 'run_1') == {
        'id': 'run_1',
        'output': 'done',
    }
    client.agents.runs.get.assert_awaited_once_with('run_1', agent_id='agent_1')
    client.agents.runs.result.assert_awaited_once_with('run_1', agent_id='agent_1')


def test_toolset_default_and_full_surface(nimble_api_key: str):
    """NimbleToolset defaults to search+extract; opt-in flags add the rest."""
    default = NimbleToolset(nimble_api_key)
    assert sorted(default.tools) == ['nimble_extract', 'nimble_search']

    full = NimbleToolset(
        nimble_api_key,
        include_map=True,
        include_crawl=True,
        include_agents=True,
    )
    assert sorted(full.tools) == [
        'nimble_agent_run_result',
        'nimble_agent_run_start',
        'nimble_agent_run_status',
        'nimble_agent_templates_list',
        'nimble_agents_list',
        'nimble_crawl_start',
        'nimble_crawl_status',
        'nimble_extract',
        'nimble_map',
        'nimble_search',
    ]

    extract_only = NimbleToolset(nimble_api_key, include_search=False, include_extract=True)
    assert sorted(extract_only.tools) == ['nimble_extract']

    search_only = NimbleToolset(nimble_api_key, include_search=True, include_extract=False)
    assert sorted(search_only.tools) == ['nimble_search']


def test_extract_factory_name():
    """Extract factory returns a named tool."""
    tool = nimble_extract_tool('test-key')
    assert tool.name == 'nimble_extract'
