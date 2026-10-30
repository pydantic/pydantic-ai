"""Tests for the OpenAI vector store backend, against the real `openai` client over a mocked transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI, AuthenticationError

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import FileSearchTool
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import Document, MetadataFilter, OpenAIVectorStore, Passage

Handler = Callable[[httpx2.Request], httpx2.Response]


def _search_page(*results: dict[str, Any]) -> dict[str, Any]:
    return {
        'object': 'vector_store.search_results.page',
        'search_query': ['q'],
        'data': list(results),
        'has_more': False,
        'next_page': None,
    }


def _hit(file_id: str = 'file-1', *, text: str = 'alpha passage', score: float = 0.8) -> dict[str, Any]:
    return {
        'file_id': file_id,
        'filename': f'{file_id}.md',
        'score': score,
        'attributes': {'tenant_id': 'acme'},
        'content': [{'type': 'text', 'text': text}],
    }


class _Server:
    """Answers the vector store endpoints and records the requests."""

    def __init__(self, routes: dict[tuple[str, str], httpx2.Response]) -> None:
        self.routes = routes
        self.requests: list[tuple[str, str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        return self.routes.get(
            (request.method, request.url.path), httpx2.Response(404, json={'error': {'message': 'nope'}})
        )

    def client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key='test-key', max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self))
        )


def _store(server: _Server, **options: Any) -> OpenAIVectorStore:
    return OpenAIVectorStore('vs_1', client=server.client(), **options)


def _ctx(model: Any) -> RunContext[object]:
    return RunContext[object](deps=None, model=model, usage=RunUsage())


class TestSearch:
    async def test_hits_become_passages_with_file_ids(self) -> None:
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(200, json=_search_page(_hit()))})
        passages = await _store(server).search('q', limit=5, filter=None)
        assert passages == [
            Passage(
                'alpha passage', source='file-1.md', score=0.8, document_id='file-1', metadata={'tenant_id': 'acme'}
            )
        ]
        assert server.requests == [
            ('POST', '/v1/vector_stores/vs_1/search', {'query': 'q', 'max_num_results': 5, 'rewrite_query': False})
        ]

    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            ({'tenant_id': 'acme'}, {'type': 'eq', 'key': 'tenant_id', 'value': 'acme'}),
            ({'source': ['manuals', 'faq']}, {'type': 'in', 'key': 'source', 'value': ['manuals', 'faq']}),
            (
                {'tenant_id': 'acme', 'year': 2026},
                {
                    'type': 'and',
                    'filters': [
                        {'type': 'eq', 'key': 'tenant_id', 'value': 'acme'},
                        {'type': 'eq', 'key': 'year', 'value': 2026},
                    ],
                },
            ),
        ],
    )
    async def test_options_and_neutral_filter_are_sent(self, filter: MetadataFilter, expected: object) -> None:
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(200, json=_search_page())})
        await _store(server, rewrite_query=True, score_threshold=0.5).search('q', limit=80, filter=filter)
        assert server.requests[0][2] == {
            'query': 'q',
            'max_num_results': 50,
            'rewrite_query': True,
            'filters': expected,
            'ranking_options': {'score_threshold': 0.5},
        }

    @pytest.mark.parametrize('status', [429, 500])
    async def test_transient_errors_become_model_retry(self, status: int) -> None:
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(status, json={'error': {}})})
        with pytest.raises(ModelRetry, match='Knowledge base search failed'):
            await _store(server).search('q', limit=5, filter=None)

    async def test_connection_errors_become_model_retry(self) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError('refused')

        client = AsyncOpenAI(
            api_key='k', max_retries=0, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(refuse))
        )
        with pytest.raises(ModelRetry, match='Knowledge base search failed'):
            await OpenAIVectorStore('vs_1', client=client).search('q', limit=5, filter=None)

    async def test_auth_errors_propagate(self) -> None:
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(401, json={'error': {}})})
        with pytest.raises(AuthenticationError):
            await _store(server).search('q', limit=5, filter=None)


_FILE = ('GET', '/v1/vector_stores/vs_1/files/file-1')
_CONTENT = ('GET', '/v1/vector_stores/vs_1/files/file-1/content')


def _file(attributes: dict[str, Any] | None = None) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            'id': 'file-1',
            'object': 'vector_store.file',
            'created_at': 0,
            'status': 'completed',
            'usage_bytes': 10,
            'vector_store_id': 'vs_1',
            'attributes': {'tenant_id': 'acme'} if attributes is None else attributes,
        },
    )


def _content(*texts: str | None) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            'object': 'vector_store.file_content.page',
            'data': [{'type': 'text', 'text': text} for text in texts],
            'has_more': False,
            'next_page': None,
        },
    )


class TestRead:
    async def test_reads_the_file_text_with_its_attributes(self) -> None:
        server = _Server({_FILE: _file(), _CONTENT: _content('first ', None, 'second')})
        assert await _store(server).read('file-1', max_chars=100, filter={'tenant_id': 'acme'}) == Document(
            'file-1', 'first second', metadata={'tenant_id': 'acme'}
        )

    async def test_file_outside_the_filter_is_not_read(self) -> None:
        """A tenant-scoped run can't open another tenant's file by id: its content is never requested."""
        server = _Server({_FILE: _file({'tenant_id': 'globex'}), _CONTENT: _content('secret')})
        assert await _store(server).read('file-1', max_chars=100, filter={'tenant_id': 'acme'}) is None
        assert [path for _, path, _ in server.requests] == ['/v1/vector_stores/vs_1/files/file-1']

    async def test_file_without_attributes_is_read_unfiltered(self) -> None:
        server = _Server({_FILE: _file({}), _CONTENT: _content('text')})
        assert await _store(server).read('file-1', max_chars=100, filter=None) == Document('file-1', 'text')

    async def test_stops_once_past_max_chars(self) -> None:
        server = _Server({_FILE: _file(), _CONTENT: _content('x' * 10, 'y' * 10)})
        document = await _store(server).read('file-1', max_chars=5, filter=None)
        assert document is not None and document.text == 'x' * 10

    async def test_unknown_file_is_none(self) -> None:
        assert await _store(_Server({})).read('file-9', max_chars=100, filter=None) is None

    async def test_transient_errors_become_model_retry(self) -> None:
        server = _Server({_FILE: _file(), _CONTENT: httpx2.Response(503, json={'error': {}})})
        with pytest.raises(ModelRetry, match='Reading the document failed'):
            await _store(server).read('file-1', max_chars=100, filter=None)


class TestNativeFileSearch:
    def test_openai_models_get_the_native_tool(self) -> None:
        model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key='k'))
        store = OpenAIVectorStore('vs_1', client=_Server({}).client())
        assert store.file_search_tool(_ctx(model), limit=80) == FileSearchTool(
            file_store_ids=['vs_1'], max_num_results=50
        )

    def test_other_models_get_none(self) -> None:
        assert (
            OpenAIVectorStore('vs_1', client=_Server({}).client()).file_search_tool(_ctx(TestModel()), limit=5) is None
        )

    async def test_non_openai_model_runs_the_local_search(self) -> None:
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(200, json=_search_page(_hit()))})
        model = TestModel(call_tools=['search_knowledge_base'])
        agent = Agent(model, capabilities=[KnowledgeBase(_store(server), native=True)])
        await agent.run('Search.')
        assert [path for _, path, _ in server.requests][0] == '/v1/vector_stores/vs_1/search'


class TestDefaultClient:
    def test_built_from_the_environment_and_rebuilt_on_a_new_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        built: list[AsyncOpenAI] = []
        server = _Server({('POST', '/v1/vector_stores/vs_1/search'): httpx2.Response(200, json=_search_page())})

        def make() -> AsyncOpenAI:
            client = server.client()
            built.append(client)
            return client

        monkeypatch.setattr('pydantic_ai_harness.knowledge_base._openai.AsyncOpenAI', make)
        store = OpenAIVectorStore('vs_1')

        async def search_twice() -> None:
            await store.search('q', limit=5, filter=None)
            await store.search('q', limit=5, filter=None)

        asyncio.run(search_twice())
        assert len(built) == 1
        asyncio.run(search_twice())
        assert len(built) == 2


class TestConfiguration:
    def test_empty_vector_store_id_rejected(self) -> None:
        with pytest.raises(ValueError, match='vector_store_id must not be empty'):
            OpenAIVectorStore('')

    @pytest.mark.parametrize('threshold', [-0.1, 1.1])
    def test_score_threshold_out_of_bounds_rejected(self, threshold: float) -> None:
        with pytest.raises(ValueError, match='score_threshold must be between 0 and 1'):
            OpenAIVectorStore('vs_1', score_threshold=threshold)

    def test_from_spec_builds_an_openai_backend(self) -> None:
        capability = KnowledgeBase[None].from_spec(backend='openai', knowledge_base_id='vs_1', native=True)
        assert capability.backend == OpenAIVectorStore('vs_1')
        assert capability.native is True

    def test_from_spec_rejects_bedrock_options_for_openai(self) -> None:
        with pytest.raises(ValueError, match='only apply to backend="bedrock"'):
            KnowledgeBase[None].from_spec(backend='openai', knowledge_base_id='vs_1', region_name='eu-central-1')
