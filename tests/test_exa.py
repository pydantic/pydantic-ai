"""Tests for the Exa tools."""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from exa_py import AsyncExa
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.common_tools.exa import (
    ContentType,
    ExaFindSimilarTool,
    ExaSearchTool,
    ExaToolset,
    exa_answer_tool,
    exa_find_similar_tool,
    exa_get_contents_tool,
    exa_search_tool,
)
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


def test_search_schema_preserves_model_search_type_by_default(exa_api_key: str):
    """Omitting `search_type` preserves the existing model-configured search behavior."""
    tool = exa_search_tool(exa_api_key)
    assert tool.name == snapshot('exa_search')
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'query': {'description': 'The search query to execute with Exa.', 'type': 'string'},
                'search_type': {
                    'default': 'auto',
                    'description': """\
The search type to use. Legacy `keyword` and `neural` values are retained
for backward compatibility; prefer another value for new code.\
""",
                    'enum': ['auto', 'fast', 'instant', 'deep-lite', 'deep', 'deep-reasoning', 'keyword', 'neural'],
                    'type': 'string',
                },
            },
            'required': ['query'],
            'type': 'object',
        }
    )


def test_search_schema_hides_developer_config(exa_api_key: str):
    """Developer-configured search settings stay out of the model-facing schema."""
    tool = exa_search_tool(
        exa_api_key,
        search_type='auto',
        content='highlights',
        include_domains=['arxiv.org'],
    )
    assert tool.function_schema.json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {'query': {'description': 'The search query to execute with Exa.', 'type': 'string'}},
            'required': ['query'],
            'type': 'object',
        }
    )


def test_search_tool_preserves_positional_constructor(exa_api_key: str):
    """New configuration does not shift the documented positional constructor fields."""
    tool = ExaSearchTool(AsyncExa(exa_api_key), 3, 500, content='highlights')
    assert tool.num_results == 3
    assert tool.max_characters == 500
    assert tool.content == 'highlights'


@pytest.mark.parametrize(
    'content,max_characters,expected_contents,expected_text',
    [
        ('highlights', None, {'highlights': True}, 'first highlight ... second highlight'),
        ('highlights', 500, {'highlights': {'max_characters': 500}}, 'first highlight ... second highlight'),
        ('text', None, {'text': True}, 'full page text'),
        ('text', 500, {'text': {'max_characters': 500}}, 'full page text'),
    ],
)
async def test_search_forwards_developer_config(
    exa_api_key: str,
    monkeypatch: pytest.MonkeyPatch,
    content: ContentType,
    max_characters: int | None,
    expected_contents: dict[str, Any],
    expected_text: str,
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
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    title='Result',
                    url='https://example.com',
                    published_date=None,
                    author=None,
                    highlights=['first highlight', 'second highlight'],
                    text='full page text',
                )
            ]
        )

    monkeypatch.setattr(client, 'search', fake_search)
    tool = exa_search_tool(
        client=client,
        num_results=3,
        search_type='instant',
        content=content,
        max_characters=max_characters,
        include_domains=['arxiv.org'],
        exclude_domains=['example.com'],
    )

    assert await tool.function_schema.call({'query': 'quantum computing'}, _ctx()) == [
        {
            'title': 'Result',
            'url': 'https://example.com',
            'published_date': None,
            'author': None,
            'text': expected_text,
        }
    ]
    assert captured == {
        'query': 'quantum computing',
        'num_results': 3,
        'type': 'instant',
        'contents': expected_contents,
        'include_domains': ['arxiv.org'],
        'exclude_domains': ['example.com'],
    }


async def test_search_preserves_model_configured_search_type(exa_api_key: str, monkeypatch: pytest.MonkeyPatch):
    """Legacy model-selected search types remain valid when the developer leaves the setting unset."""
    client = AsyncExa(exa_api_key)
    captured: dict[str, Any] = {}

    async def fake_search(query: str, **kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(client, 'search', fake_search)
    tool = exa_search_tool(client=client)

    assert await tool.function_schema.call({'query': 'pydantic ai', 'search_type': 'keyword'}, _ctx()) == []
    assert captured['type'] == 'keyword'


# --- find_similar deprecation ---


def test_find_similar_tool_is_deprecated(exa_api_key: str):
    """The find_similar factory warns, mirroring exa-py's own deprecation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tool = exa_find_similar_tool(exa_api_key)
    assert tool.name == 'exa_find_similar'
    assert len(caught) == 1
    assert isinstance(caught[0].message, PydanticAIDeprecationWarning)
    assert Path(caught[0].filename) == Path(__file__)


def test_find_similar_class_is_deprecated(exa_api_key: str):
    """Direct construction also warns before the SDK warning is suppressed during calls."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ExaFindSimilarTool(AsyncExa(exa_api_key), 1)
    assert len(caught) == 1
    assert isinstance(caught[0].message, PydanticAIDeprecationWarning)
    assert Path(caught[0].filename) == Path(__file__)


def test_toolset_includes_find_similar_by_default(exa_api_key: str):
    """The default toolset includes find_similar for backward compatibility, emitting a deprecation warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        toolset = ExaToolset(exa_api_key)
    assert set(toolset.tools) == snapshot({'exa_search', 'exa_find_similar', 'exa_get_contents', 'exa_answer'})
    assert len(caught) == 1
    assert isinstance(caught[0].message, PydanticAIDeprecationWarning)
    assert Path(caught[0].filename) == Path(__file__)


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


async def test_toolset_reuses_client_and_forwards_search_config(exa_api_key: str, monkeypatch: pytest.MonkeyPatch):
    """The toolset keeps the supplied client and binds developer-owned search configuration."""
    client = AsyncExa(exa_api_key)
    captured: dict[str, Any] = {}

    async def fake_search(query: str, **kwargs: Any) -> SimpleNamespace:
        captured['query'] = query
        captured.update(kwargs)
        return SimpleNamespace(results=[])

    monkeypatch.setattr(client, 'search', fake_search)
    toolset = ExaToolset(
        client=client,
        search_type='deep-reasoning',
        content='highlights',
        max_characters=400,
        include_domains=['pydantic.dev'],
        exclude_domains=['example.com'],
        include_find_similar=False,
        include_get_contents=False,
        include_answer=False,
    )

    tool = toolset.tools['exa_search']
    assert await tool.function_schema.call({'query': 'Pydantic AI'}, _ctx()) == []
    assert captured == {
        'query': 'Pydantic AI',
        'num_results': 5,
        'type': 'deep-reasoning',
        'contents': {'highlights': {'max_characters': 400}},
        'include_domains': ['pydantic.dev'],
        'exclude_domains': ['example.com'],
    }


# --- Integration tests (VCR) ---


@pytest.mark.vcr
async def test_search_returns_highlights(exa_api_key: str):
    """`content='highlights'` returns token-efficient snippets through the public factory."""
    tool = exa_search_tool(
        client=AsyncExa(exa_api_key),
        num_results=2,
        search_type='auto',
        content='highlights',
    )
    results = await tool.function_schema.call({'query': 'What is Pydantic AI?'}, _ctx())
    assert results == snapshot(
        [
            {
                'title': 'Pydantic AI | Pydantic Docs',
                'url': 'https://pydantic.dev/docs/ai/overview/',
                'published_date': None,
                'author': None,
                'text': IsStr(regex=r'(?s).+'),
            },
            {
                'title': 'AI Agent Framework, the Pydantic way - GitHub',
                'url': 'https://github.com/pydantic/pydantic-ai',
                'published_date': '2024-06-21T15:55:04.000Z',
                'author': None,
                'text': IsStr(regex=r'(?s).+'),
            },
        ]
    )


@pytest.mark.vcr()
async def test_search_with_text_content(exa_api_key: str):
    """The backward-compatible default returns full page text through the public factory."""
    tool = exa_search_tool(
        client=AsyncExa(exa_api_key),
        num_results=1,
        search_type='auto',
        max_characters=500,
        include_domains=['ai.pydantic.dev'],
    )
    results = await tool.function_schema.call({'query': 'What is Pydantic AI?'}, _ctx())
    assert results == snapshot(
        [
            {
                'title': 'Index',
                'url': 'https://ai.pydantic.dev/examples/bank-support/index.md',
                'published_date': None,
                'author': None,
                'text': IsStr(regex=r'(?s).+'),
            }
        ]
    )


@pytest.mark.vcr()
async def test_get_contents(exa_api_key: str):
    """get_contents returns the full text content for the requested URLs."""
    tool = exa_get_contents_tool(client=AsyncExa(exa_api_key))
    results = await tool.function_schema.call({'urls': ['https://ai.pydantic.dev/']}, _ctx())
    assert results == snapshot(
        [
            {
                'url': 'https://ai.pydantic.dev/',
                'title': 'Pydantic AI | Pydantic Docs',
                'text': IsStr(regex=r'(?s).+'),
                'author': None,
                'published_date': None,
            }
        ]
    )


@pytest.mark.vcr()
async def test_answer(exa_api_key: str):
    """answer returns an AI-generated answer with supporting citations."""
    tool = exa_answer_tool(client=AsyncExa(exa_api_key))
    result = await tool.function_schema.call({'query': 'What is Pydantic AI?'}, _ctx())
    assert result == snapshot(
        {
            'answer': IsStr(regex=r'(?s).+'),
            'citations': [
                {
                    'url': 'https://pydantic.dev/docs/ai/overview/',
                    'title': 'Pydantic AI | Pydantic Docs',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://pydantic.dev/pydantic-ai?featured_on=talkpython',
                    'title': 'Pydantic AI: Type-Safe Python Framework for AI Agents & LLM Applications',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://web.appunite.com/blog/understanding-pydantic-ai-a-powerful-alternative-to-lang-chain-and-llama-index',
                    'title': 'Understanding Pydantic-AI: A Powerful Alternative to LangChain and LlamaIndex (Part: 1)  | Appunite Tech Blog',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://github.com/pydantic/pydantic-ai/',
                    'title': 'pydantic/pydantic-ai',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://pypi.org/project/pydantic-ai/1.84.1/',
                    'title': 'pydantic-ai v1.84.1',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://headofagents.ai/pydanticai',
                    'title': 'PydanticAI: Agent framework from the Pydantic team - HeadOfAgents',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://pydantic.dev/',
                    'title': 'Pydantic | The end-to-end AI engineering stack',
                    'text': IsStr(regex=r'(?s).+'),
                },
                {
                    'url': 'https://pydantic.dev/docs/ai/overview/install/',
                    'title': 'Installation | Pydantic Docs',
                    'text': IsStr(regex=r'(?s).+'),
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
                'text': IsStr(regex=r'(?s).+'),
            }
        ]
    )
