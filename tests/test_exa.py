"""Tests for the Exa tools."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from exa_py import AsyncExa
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.common_tools.exa import (
    ContentType,
    ExaAnswerTool,
    ExaGetContentsTool,
    ExaSearchTool,
    ExaToolset,
    exa_answer_tool,
    exa_find_similar_tool,
    exa_get_contents_tool,
    exa_search_tool,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import IsStr

pytestmark = pytest.mark.anyio


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


# --- Factory construction / validation (no network) ---


@pytest.mark.parametrize(
    'factory',
    [exa_search_tool, exa_get_contents_tool, exa_answer_tool, ExaToolset],
)
def test_factory_requires_api_key_or_client(factory: object):
    """Each factory raises when neither api_key nor client is provided."""
    with pytest.raises(ValueError, match='Either api_key or client must be provided'):
        factory()  # type: ignore[operator]


def test_find_similar_requires_api_key_or_client():
    """find_similar warns about deprecation and still validates that a key or client is given."""
    with pytest.warns(PydanticAIDeprecationWarning, match='find_similar'):
        with pytest.raises(ValueError, match='Either api_key or client must be provided'):
            exa_find_similar_tool()  # type: ignore[call-overload]


def test_factory_with_client():
    """Factories accept a pre-built client."""
    client = AsyncExa('test-key')
    assert exa_search_tool(client=client).name == 'exa_search'
    assert exa_get_contents_tool(client=client).name == 'exa_get_contents'
    assert exa_answer_tool(client=client).name == 'exa_answer'


def test_factory_built_client_sets_integration_attribution_header(exa_api_key: str, monkeypatch: pytest.MonkeyPatch):
    """Clients the factories build from an api_key send the attribution header; user-provided clients are untouched.

    This is a unit test because the header lives on the client rather than in any per-call payload,
    and the cassette matcher ignores request headers, so a VCR test wouldn't catch it being dropped.
    """
    created: list[AsyncExa] = []

    class _RecordingExa(AsyncExa):
        def __init__(self, api_key: str):
            super().__init__(api_key=api_key)
            created.append(self)

    monkeypatch.setattr('pydantic_ai.common_tools.exa.AsyncExa', _RecordingExa)

    exa_search_tool(exa_api_key)
    assert len(created) == 1
    assert created[0].headers['x-exa-integration'] == 'pydantic-ai'

    own = AsyncExa(exa_api_key)
    exa_search_tool(client=own)
    assert len(created) == 1
    assert 'x-exa-integration' not in own.headers


# --- Tool schema and developer-only configuration ---


def test_search_schema_exposes_only_query(exa_api_key: str):
    """The model only sees the query; search_type/content/num_results/domain filters are developer-only."""
    tool = exa_search_tool(exa_api_key, include_domains=['arxiv.org'])
    assert tool.name == snapshot('exa_search')
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {
                    'description': 'The search query to execute with Exa.',
                    'type': 'string',
                },
            },
            'required': ['query'],
            'type': 'object',
        }
    )


@pytest.mark.parametrize(
    'content,max_characters,expected_contents',
    [
        ('highlights', None, {'highlights': True}),
        ('highlights', 500, {'highlights': {'max_characters': 500}}),
        ('text', None, {'text': True}),
        ('text', 500, {'text': {'max_characters': 500}}),
    ],
)
async def test_search_forwards_developer_config(
    exa_api_key: str,
    monkeypatch: pytest.MonkeyPatch,
    content: ContentType,
    max_characters: int | None,
    expected_contents: dict[str, Any],
):
    """The developer-configured search parameters are forwarded to the Exa SDK on every search.

    This is a unit test rather than a VCR assertion because our cassette matcher isn't sensitive
    to the request body, so playback alone wouldn't catch a dropped or misshapen parameter.
    """
    client = AsyncExa(exa_api_key)
    captured: dict[str, Any] = {}

    async def fake_search(query: str, **kwargs: Any) -> SimpleNamespace:
        captured['query'] = query
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(client, 'search', fake_search)
    tool = ExaSearchTool(
        client=client,
        num_results=3,
        search_type='fast',
        content=content,
        max_characters=max_characters,
        include_domains=['arxiv.org'],
        exclude_domains=['example.com'],
    )

    assert await tool('quantum computing') == []
    assert captured == {
        'query': 'quantum computing',
        'num_results': 3,
        'type': 'fast',
        'contents': expected_contents,
        'include_domains': ['arxiv.org'],
        'exclude_domains': ['example.com'],
    }


def _stale_then_valid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simulates a model that first replays the pre-2.15 `search_type` argument, then corrects itself."""
    last_parts = messages[-1].parts
    if any(isinstance(p, RetryPromptPart) for p in last_parts):
        return ModelResponse(parts=[ToolCallPart('exa_search', {'query': 'what is pydantic ai'})])
    if any(isinstance(p, ToolReturnPart) for p in last_parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('exa_search', {'query': 'what is pydantic ai', 'search_type': 'keyword'})])


async def test_search_recovers_when_model_passes_removed_search_type(exa_api_key: str, monkeypatch: pytest.MonkeyPatch):
    """A model still passing the removed `search_type` argument gets a retry prompt and self-heals.

    Earlier versions exposed `search_type` (including the deprecated `keyword`/`neural` values) in the
    tool schema. A model replaying that argument — e.g. resuming a conversation recorded before an
    upgrade — is asked to retry rather than crashing the agent run.
    """
    client = AsyncExa(exa_api_key)

    async def fake_search(query: str, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(results=[])

    monkeypatch.setattr(client, 'search', fake_search)
    agent = Agent(FunctionModel(_stale_then_valid), tools=[exa_search_tool(client=client)])

    result = await agent.run('search for pydantic ai')

    assert result.output == 'done'
    retries = [p for m in result.all_messages() for p in m.parts if isinstance(p, RetryPromptPart)]
    assert len(retries) == 1


# --- find_similar deprecation ---


def test_find_similar_tool_is_deprecated(exa_api_key: str):
    """The find_similar factory warns, mirroring exa-py's own deprecation."""
    with pytest.warns(PydanticAIDeprecationWarning, match='find_similar'):
        tool = exa_find_similar_tool(exa_api_key)
    assert tool.name == 'exa_find_similar'


def test_toolset_includes_find_similar_by_default(exa_api_key: str):
    """The default toolset includes find_similar for backward compatibility, emitting a deprecation warning."""
    with pytest.warns(PydanticAIDeprecationWarning, match='find_similar'):
        toolset = ExaToolset(exa_api_key)
    assert set(toolset.tools) == snapshot({'exa_search', 'exa_find_similar', 'exa_get_contents', 'exa_answer'})


@pytest.mark.parametrize(
    'include_search,include_get_contents,include_answer,expected',
    [
        (True, True, False, {'exa_search', 'exa_get_contents'}),
        (False, False, True, {'exa_answer'}),
    ],
)
def test_toolset_tool_selection(
    exa_api_key: str,
    include_search: bool,
    include_get_contents: bool,
    include_answer: bool,
    expected: set[str],
):
    """include_* flags select which tools the toolset exposes (find_similar excluded to avoid its warning)."""
    toolset = ExaToolset(
        exa_api_key,
        include_search=include_search,
        include_find_similar=False,
        include_get_contents=include_get_contents,
        include_answer=include_answer,
    )
    assert set(toolset.tools) == expected


# --- Integration tests (VCR) ---


@pytest.mark.vcr()
async def test_search_returns_highlights_by_default(exa_api_key: str):
    """By default each result's text carries token-efficient highlight snippets."""
    tool = ExaSearchTool(
        client=AsyncExa(exa_api_key),
        num_results=2,
        search_type='auto',
        content='highlights',
        max_characters=None,
        include_domains=None,
        exclude_domains=None,
    )
    results = await tool('What is Pydantic AI?')
    assert results == snapshot(
        [
            {
                'title': 'Pydantic AI | Pydantic Docs',
                'url': 'https://pydantic.dev/docs/ai/overview/',
                'published_date': None,
                'author': None,
                'text': IsStr(),
            },
            {
                'title': 'AI Agent Framework, the Pydantic way - GitHub',
                'url': 'https://github.com/pydantic/pydantic-ai',
                'published_date': '2024-06-21T15:55:04.000Z',
                'author': None,
                'text': IsStr(),
            },
        ]
    )


@pytest.mark.vcr()
async def test_search_with_text_content(exa_api_key: str):
    """content='text' returns the full page text instead of highlights, scoped to a locked domain."""
    tool = ExaSearchTool(
        client=AsyncExa(exa_api_key),
        num_results=1,
        search_type='auto',
        content='text',
        max_characters=500,
        include_domains=['ai.pydantic.dev'],
        exclude_domains=None,
    )
    results = await tool('What is Pydantic AI?')
    assert results == snapshot(
        [
            {
                'title': 'Index',
                'url': 'https://ai.pydantic.dev/examples/bank-support/index.md',
                'published_date': None,
                'author': None,
                'text': IsStr(),
            }
        ]
    )


@pytest.mark.vcr()
async def test_get_contents(exa_api_key: str):
    """get_contents returns the full text content for the requested URLs."""
    tool = ExaGetContentsTool(client=AsyncExa(exa_api_key))
    results = await tool(['https://ai.pydantic.dev/'])
    assert results == snapshot(
        [
            {
                'url': 'https://ai.pydantic.dev/',
                'title': 'Pydantic AI | Pydantic Docs',
                'text': IsStr(),
                'author': None,
                'published_date': None,
            }
        ]
    )


@pytest.mark.vcr()
async def test_answer(exa_api_key: str):
    """answer returns an AI-generated answer with supporting citations."""
    tool = ExaAnswerTool(client=AsyncExa(exa_api_key))
    result = await tool('What is Pydantic AI?')
    assert result == snapshot(
        {
            'answer': IsStr(),
            'citations': [
                {
                    'url': 'https://pydantic.dev/docs/ai/overview/',
                    'title': 'Pydantic AI | Pydantic Docs',
                    'text': IsStr(),
                },
                {
                    'url': 'https://pydantic.dev/pydantic-ai?featured_on=talkpython',
                    'title': 'Pydantic AI: Type-Safe Python Framework for AI Agents & LLM Applications',
                    'text': IsStr(),
                },
                {
                    'url': 'https://web.appunite.com/blog/understanding-pydantic-ai-a-powerful-alternative-to-lang-chain-and-llama-index',
                    'title': 'Understanding Pydantic-AI: A Powerful Alternative to LangChain and LlamaIndex (Part: 1)  | Appunite Tech Blog',
                    'text': IsStr(),
                },
                {
                    'url': 'https://github.com/pydantic/pydantic-ai/',
                    'title': 'pydantic/pydantic-ai',
                    'text': IsStr(),
                },
                {
                    'url': 'https://pypi.org/project/pydantic-ai/1.84.1/',
                    'title': 'pydantic-ai v1.84.1',
                    'text': IsStr(),
                },
                {
                    'url': 'https://headofagents.ai/pydanticai',
                    'title': 'PydanticAI: Agent framework from the Pydantic team - HeadOfAgents',
                    'text': IsStr(),
                },
                {
                    'url': 'https://pydantic.dev/',
                    'title': 'Pydantic | The end-to-end AI engineering stack',
                    'text': IsStr(),
                },
                {
                    'url': 'https://pydantic.dev/docs/ai/overview/install/',
                    'title': 'Installation | Pydantic Docs',
                    'text': IsStr(),
                },
            ],
        }
    )


@pytest.mark.vcr()
async def test_find_similar(exa_api_key: str):
    """find_similar returns related pages (and the factory warns it's deprecated)."""
    with pytest.warns(PydanticAIDeprecationWarning, match='find_similar'):
        tool = exa_find_similar_tool(exa_api_key, num_results=1)
    results = await tool.function_schema.call({'url': 'https://ai.pydantic.dev/'}, _ctx())
    assert results == snapshot(
        [
            {
                'title': 'Pydantic AI | Pydantic Docs',
                'url': 'https://pydantic.dev/docs/ai/overview/',
                'published_date': None,
                'author': None,
                'text': IsStr(),
            }
        ]
    )
