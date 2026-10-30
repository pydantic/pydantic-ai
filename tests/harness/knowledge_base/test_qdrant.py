"""Tests for the Qdrant backend, against `qdrant-client`'s in-memory mode: real filters and queries, no server."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from pydantic_ai.embeddings import Embedder, EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.base import EmbedInputType
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import MetadataFilter, QdrantKnowledgeBase, normalize_filter

_VOCABULARY = ('return', 'refund', 'ship', 'order', 'grill', 'clean', 'warranty', 'policy')


class _WordEmbedding(EmbeddingModel):
    """Embeds text as counts of a small vocabulary, so nearest neighbours follow shared words."""

    @property
    def model_name(self) -> str:
        return 'words'  # pragma: no cover

    @property
    def system(self) -> str:
        return 'test'  # pragma: no cover

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        vectors = [[float(text.lower().count(word)) + 0.01 for word in _VOCABULARY] for text in texts]
        return EmbeddingResult(
            embeddings=vectors, inputs=texts, input_type=input_type, model_name='words', provider_name='test'
        )


_EMBEDDER = Embedder(_WordEmbedding())

_CHUNKS: list[dict[str, Any]] = [
    {
        'document_id': 'returns',
        'chunk_index': 1,
        'text': 'Refund goes to the original payment.',
        'tenant_id': 'acme',
        'year': 2025,
        'source': 'returns.md',
    },
    {
        'document_id': 'returns',
        'chunk_index': 0,
        'text': 'Return policy: return within 30 days.',
        'tenant_id': 'acme',
        'year': 2025,
        'source': 'returns.md',
    },
    {
        'document_id': 'shipping',
        'chunk_index': 0,
        'text': 'We ship every order within two days.',
        'tenant_id': 'acme',
        'year': 2022,
        'source': 'shipping.md',
        'rating': 4.5,
    },
    {
        'document_id': 'globex',
        'chunk_index': 0,
        'text': 'Return policy: no return, no refund.',
        'tenant_id': 'globex',
        'year': 2025,
        'source': 'globex.md',
        'tags': 7,
    },
    {'document_id': 'untagged', 'chunk_index': 0, 'text': 'Grill clean warranty.', 'source': 'untagged.md'},
]


async def _client() -> AsyncQdrantClient:
    client = AsyncQdrantClient(':memory:')
    await client.create_collection(
        'kb', vectors_config=models.VectorParams(size=len(_VOCABULARY), distance=models.Distance.COSINE)
    )
    vectors = (await _EMBEDDER.embed_documents([chunk['text'] for chunk in _CHUNKS])).embeddings
    await client.upsert(
        'kb',
        points=[
            models.PointStruct(id=index, vector=list(vector), payload=chunk)
            for index, (chunk, vector) in enumerate(zip(_CHUNKS, vectors))
        ],
    )
    return client


async def _backend(**options: Any) -> QdrantKnowledgeBase:
    return QdrantKnowledgeBase(await _client(), 'kb', _EMBEDDER, **options)


async def _ids(
    filter: MetadataFilter | None, query: str = 'return refund ship order grill clean warranty policy'
) -> set[str]:
    passages = await (await _backend()).search(query, limit=10, filter=normalize_filter(filter))
    return {passage.document_id or '' for passage in passages}


class TestSearch:
    async def test_nearest_chunks_come_first_as_passages(self) -> None:
        passages = await (await _backend()).search('return refund', limit=2, filter=None)
        assert passages[0].text.startswith(('Return policy', 'Refund'))
        first = passages[0]
        assert first.source in {'returns.md', 'globex.md'} and first.score is not None
        assert first.document_id in {'returns', 'globex'}
        assert 'text' not in first.metadata and 'chunk_index' not in first.metadata and 'tenant_id' in first.metadata

    @pytest.mark.parametrize(
        ('filter', 'expected'),
        [
            ({'tenant_id': 'acme'}, {'returns', 'shipping'}),
            ({'tenant_id': ['acme', 'globex']}, {'returns', 'shipping', 'globex'}),
            ({'field': 'year', 'operator': '>=', 'value': 2024}, {'returns', 'globex'}),
            ({'field': 'year', 'operator': '<', 'value': 2024}, {'shipping'}),
            ({'field': 'year', 'operator': '>', 'value': 2024}, {'returns', 'globex'}),
            ({'field': 'year', 'operator': '<=', 'value': 2022}, {'shipping'}),
            ({'field': 'tenant_id', 'operator': '!=', 'value': 'acme'}, {'globex', 'untagged'}),
            ({'field': 'tenant_id', 'operator': 'not in', 'value': ['acme']}, {'globex', 'untagged'}),
            ({'field': 'year', 'operator': 'in', 'value': [2022]}, {'shipping'}),
            ({'field': 'rating', 'operator': '==', 'value': 4.5}, {'shipping'}),
            ({'field': 'tags', 'operator': 'in', 'value': [7, 'seven']}, {'globex'}),
            (
                {
                    'operator': 'AND',
                    'conditions': [
                        {'field': 'tenant_id', 'operator': '==', 'value': 'acme'},
                        {
                            'operator': 'OR',
                            'conditions': [
                                {'field': 'year', 'operator': '>=', 'value': 2024},
                                {'field': 'source', 'operator': '==', 'value': 'shipping.md'},
                            ],
                        },
                    ],
                },
                {'returns', 'shipping'},
            ),
            (
                {'operator': 'NOT', 'conditions': [{'field': 'tenant_id', 'operator': '==', 'value': 'acme'}]},
                {'globex', 'untagged'},
            ),
        ],
    )
    async def test_filter_tree_runs_in_qdrant(self, filter: MetadataFilter, expected: set[str]) -> None:
        assert await _ids(filter) == expected

    async def test_score_threshold_is_applied_by_qdrant(self) -> None:
        backend = await _backend()
        scores = sorted(
            (p.score or 0 for p in await backend.search('return refund', limit=10, filter=None)), reverse=True
        )
        cut = (scores[0] + scores[-1]) / 2
        kept = await backend.search('return refund', limit=10, filter=None, score_threshold=cut)
        assert 0 < len(kept) < len(scores) and all((p.score or 0) >= cut for p in kept)

    @pytest.mark.parametrize('mode', ['keyword', 'hybrid'])
    async def test_only_semantic_mode(self, mode: Any) -> None:
        with pytest.raises(UserError, match="only supports mode='semantic'"):
            await (await _backend()).search('q', limit=5, filter=None, mode=mode)
        assert await (await _backend()).search('return', limit=1, filter=None, mode='semantic')

    async def test_ranges_compare_numbers_only(self) -> None:
        with pytest.raises(UserError, match="compares numbers with '>'"):
            await (await _backend()).search(
                'q', limit=5, filter=normalize_filter({'field': 'year', 'operator': '>', 'value': '2024'})
            )

    async def test_payload_without_text_source_or_id(self) -> None:
        client = AsyncQdrantClient(':memory:')
        await client.create_collection(
            'bare', vectors_config=models.VectorParams(size=len(_VOCABULARY), distance=models.Distance.COSINE)
        )
        await client.upsert(
            'bare', points=[models.PointStruct(id=1, vector=[1.0] * len(_VOCABULARY), payload={'body': 'x'})]
        )
        (passage,) = await QdrantKnowledgeBase(client, 'bare', _EMBEDDER).search('q', limit=5, filter=None)
        assert (passage.text, passage.source, passage.document_id, dict(passage.metadata)) == (
            '',
            None,
            None,
            {'body': 'x'},
        )


class TestRead:
    async def test_chunks_are_joined_in_order(self) -> None:
        document = await (await _backend()).read(
            'returns', max_chars=1000, filter=normalize_filter({'tenant_id': 'acme'})
        )
        assert document is not None
        assert document.text == 'Return policy: return within 30 days.\n\nRefund goes to the original payment.'
        assert (document.source, dict(document.metadata)) == ('returns.md', {'tenant_id': 'acme', 'year': 2025})

    async def test_reading_stops_past_max_chars(self) -> None:
        document = await (await _backend()).read('returns', max_chars=10, filter=None)
        assert document is not None and document.text == 'Return policy: return within 30 days.'

    async def test_other_tenant_and_missing_fields_fail_closed(self) -> None:
        backend = await _backend()
        acme = normalize_filter({'tenant_id': 'acme'})
        assert await backend.read('globex', max_chars=1000, filter=acme) is None
        not_globex = normalize_filter({'field': 'tenant_id', 'operator': '!=', 'value': 'globex'})
        assert await backend.read('untagged', max_chars=1000, filter=not_globex) is None
        assert await backend.read('untagged', max_chars=1000, filter=None) is not None
        assert await backend.read('nope', max_chars=1000, filter=None) is None

    async def test_without_a_document_field_there_is_no_reading(self) -> None:
        backend = await _backend(document_field=None, chunk_field=None, source_field=None)
        assert backend.can_read is False
        assert await backend.read('returns', max_chars=1000, filter=None) is None
        (passage, *_) = await backend.search('return', limit=5, filter=None)
        assert (passage.document_id, passage.source) == (None, None)
        assert list(KnowledgeBase[None](backend).get_toolset().tools) == ['search_knowledge_base']

    async def test_chunks_without_text_are_skipped(self) -> None:
        client = AsyncQdrantClient(':memory:')
        await client.create_collection(
            'gaps', vectors_config=models.VectorParams(size=len(_VOCABULARY), distance=models.Distance.COSINE)
        )
        await client.upsert(
            'gaps',
            points=[
                models.PointStruct(
                    id=1, vector=[1.0] * len(_VOCABULARY), payload={'document_id': 'd', 'chunk_index': 0}
                ),
                models.PointStruct(
                    id=2,
                    vector=[1.0] * len(_VOCABULARY),
                    payload={'document_id': 'd', 'chunk_index': 1, 'text': 'kept'},
                ),
            ],
        )
        document = await QdrantKnowledgeBase(client, 'gaps', _EMBEDDER).read('d', max_chars=100, filter=None)
        assert document is not None and document.text == 'kept'

    async def test_chunks_without_a_position_keep_scroll_order(self) -> None:
        document = await (await _backend(chunk_field=None)).read('shipping', max_chars=1000, filter=None)
        assert document is not None and document.text == 'We ship every order within two days.'


class _FailingClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def query_points(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error

    async def scroll(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


def _unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status, reason_phrase='x', content=b'detail with https://cluster.qdrant.io', headers=httpx.Headers()
    )


class TestErrors:
    @pytest.mark.parametrize(
        'error', [_unexpected(429), _unexpected(503), ResponseHandlingException(Exception('connection refused'))]
    )
    async def test_transient_errors_become_retries_without_error_text(self, error: Exception) -> None:
        backend = QdrantKnowledgeBase(_FailingClient(error), 'kb', _EMBEDDER)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ModelRetry) as raised:
            await backend.search('q', limit=5, filter=None)
        assert 'qdrant.io' not in raised.value.message and raised.value.message.endswith('try again.')
        with pytest.raises(ModelRetry):
            await backend.read('doc', max_chars=100, filter=None)

    async def test_client_errors_propagate(self) -> None:
        backend = QdrantKnowledgeBase(_FailingClient(_unexpected(404)), 'kb', _EMBEDDER)  # pyright: ignore[reportArgumentType]
        with pytest.raises(UnexpectedResponse):
            await backend.search('q', limit=5, filter=None)


class TestConfiguration:
    def test_collection_name_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match='collection_name must not be empty'):
            QdrantKnowledgeBase(AsyncQdrantClient(':memory:'), '', _EMBEDDER)

    def test_max_document_chunks_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='max_document_chunks must be positive'):
            QdrantKnowledgeBase(AsyncQdrantClient(':memory:'), 'kb', _EMBEDDER, max_document_chunks=0)

    async def test_agent_tools_search_and_read_within_the_tenant(self) -> None:
        toolset = KnowledgeBase[str](
            await _backend(), filter=lambda ctx: {'tenant_id': ctx.deps}, require_filter=True
        ).get_toolset()
        ctx = RunContext[str](deps='acme', model=TestModel(), usage=RunUsage())
        found = await toolset.search_knowledge_base(ctx, 'return refund')
        assert {record['document_id'] for record in found.metadata['results']} <= {'returns', 'shipping'}
        read = await toolset.read_document(ctx, 'returns')
        assert str(read.return_value).endswith('Refund goes to the original payment.')
