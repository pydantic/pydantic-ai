"""Tests for the Tavily search tool."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from tavily import AsyncTavilyClient

from pydantic_ai._run_context import RunContext
from pydantic_ai.common_tools.tavily import TavilySearchTool, tavily_search_tool
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import IsStr


@pytest.mark.vcr()
async def test_basic_search(tavily_api_key: str):
    """Test basic search with default parameters."""
    tool = TavilySearchTool(client=AsyncTavilyClient(tavily_api_key))
    results = await tool('What is Pydantic AI?')
    assert results == snapshot(
        [
            {
                'title': 'Pydantic AI: Agent Framework',
                'url': 'https://medium.com/ai-agent-insider/pydantic-ai-agent-framework-02b138e8db71',
                'content': IsStr(),
                'score': 0.9999875,
            },
            {
                'title': 'Pydantic AI - Pydantic AI',
                'url': 'https://ai.pydantic.dev/',
                'content': IsStr(),
                'score': 0.99997807,
            },
            {
                'title': 'Build Production-Ready AI Agents in Python with Pydantic AI',
                'url': 'https://www.youtube.com/watch?v=-WB0T0XmDrY',
                'content': IsStr(),
                'score': 0.9999398,
            },
            {
                'title': 'What is Pydantic AI?. Build production-ready AI agents with...',
                'url': 'https://medium.com/@tahirbalarabe2/what-is-pydantic-ai-15cc81dea3c3',
                'content': IsStr(),
                'score': 0.9999125,
            },
            {
                'title': 'Pydantic AI : r/LLMDevs',
                'url': 'https://www.reddit.com/r/LLMDevs/comments/1iih8az/pydantic_ai/',
                'content': IsStr(),
                'score': 0.9999001,
            },
        ]
    )


@pytest.mark.vcr()
async def test_search_with_include_domains(tavily_api_key: str):
    """Test search with include_domains filtering."""
    tool = TavilySearchTool(client=AsyncTavilyClient(tavily_api_key))
    results = await tool('transformer architectures', include_domains=['arxiv.org'])
    assert results == snapshot(
        [
            {
                'title': 'Deep Dive into Transformer Architectures for Long-Term ...',
                'url': 'https://arxiv.org/abs/2507.13043',
                'content': IsStr(),
                'score': 0.7923522,
            },
            {
                'title': '[2505.13499] Optimal Control for Transformer Architectures',
                'url': 'https://arxiv.org/abs/2505.13499',
                'content': IsStr(),
                'score': 0.783542,
            },
            {
                'title': 'Deep Dive into Transformer Architectures for Long-Term ...',
                'url': 'https://arxiv.org/html/2507.13043v1',
                'content': IsStr(),
                'score': 0.77731717,
            },
            {
                'title': 'Lightweight Transformer Architectures for Edge Devices in ...',
                'url': 'https://www.arxiv.org/abs/2601.03290',
                'content': IsStr(),
                'score': 0.76826453,
            },
            {
                'title': 'Study of Lightweight Transformer Architectures for Single ...',
                'url': 'https://arxiv.org/abs/2505.21057',
                'content': IsStr(),
                'score': 0.7663815,
            },
        ]
    )


@pytest.mark.vcr()
async def test_factory_with_bound_params(tavily_api_key: str):
    """Test factory-bound params are forwarded through FunctionSchema.call."""
    tool = tavily_search_tool(tavily_api_key, max_results=2, include_domains=['arxiv.org'])
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    results = await tool.function_schema.call({'query': 'attention mechanisms'}, ctx)
    assert results == snapshot(
        [
            {
                'title': '[2601.03329] Attention mechanisms in neural networks',
                'url': 'https://arxiv.org/abs/2601.03329',
                'content': IsStr(),
                'score': 0.81770587,
            },
            {
                'title': 'A General Survey on Attention Mechanisms in Deep ...',
                'url': 'https://arxiv.org/abs/2203.14263',
                'content': IsStr(),
                'score': 0.8138313,
            },
        ]
    )


def test_no_params_bound_exposes_all_in_schema(tavily_api_key: str):
    """Test that with no factory params, all parameters appear in the tool schema."""
    tool = tavily_search_tool(tavily_api_key)
    assert tool.name == snapshot('tavily_search')
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {
                    'description': 'The search query to execute with Tavily.',
                    'type': 'string',
                },
                'search_depth': {
                    'default': 'basic',
                    'description': 'The depth of the search.',
                    'enum': ['basic', 'advanced', 'fast', 'ultra-fast'],
                    'type': 'string',
                },
                'topic': {
                    'default': 'general',
                    'description': 'The category of the search.',
                    'enum': ['general', 'news', 'finance'],
                    'type': 'string',
                },
                'time_range': {
                    'anyOf': [
                        {'enum': ['day', 'week', 'month', 'year'], 'type': 'string'},
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


def test_factory_requires_api_key_or_client():
    """Test that tavily_search_tool raises when neither api_key nor client is provided."""
    with pytest.raises(ValueError, match='Either api_key or client must be provided'):
        tavily_search_tool()  # pyright: ignore[reportCallIssue]


def test_factory_with_client():
    """Test that tavily_search_tool accepts a pre-built client."""
    client = AsyncTavilyClient('test-key')
    tool = tavily_search_tool(client=client)
    assert tool.name == 'tavily_search'


def test_bound_params_hidden_from_schema(tavily_api_key: str):
    """Test that factory-provided params are excluded from the tool schema."""
    tool = tavily_search_tool(
        tavily_api_key,
        search_depth='advanced',
        topic='news',
        time_range='week',
        include_domains=['arxiv.org'],
        exclude_domains=['medium.com'],
    )
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {
                    'description': 'The search query to execute with Tavily.',
                    'type': 'string',
                },
            },
            'required': ['query'],
            'type': 'object',
        }
    )
