"""Tests for the pgvector backend's SQL and row handling, against a recording fake pool.

The SQL itself was run against PostgreSQL 17 with pgvector 0.8 when the backend was written; CI's
Postgres service has no pgvector, so these tests check what is sent and how rows are read back.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import MetadataFilter, Passage, PgVectorKnowledgeBase, normalize_filter

_EMBEDDER = Embedder(TestEmbeddingModel(dimensions=3))
_VECTOR = '[1.0,1.0,1.0]'


@dataclass
class _Connection:
    rows: Sequence[Sequence[object]]
    calls: list[tuple[str, tuple[object, ...]]]

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        self.calls.append((query, args))
        return self.rows


@dataclass
class _Pool:
    """Hands out one recording connection that returns canned rows."""

    rows: Sequence[Sequence[object]] = ()
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list[tuple[str, tuple[object, ...]]])

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[object]:
        yield _Connection(self.rows, self.calls)


def _backend(pool: _Pool, **options: Any) -> PgVectorKnowledgeBase:
    return PgVectorKnowledgeBase(pool, _EMBEDDER, **options)


async def _where(filter: MetadataFilter) -> tuple[str, tuple[object, ...]]:
    """The SQL condition and parameters a filter compiles to, without the vector and limit."""
    pool = _Pool()
    await _backend(pool).search('q', limit=5, filter=normalize_filter(filter))
    ((sql, args),) = pool.calls
    return sql.split(' WHERE ', 1)[1].split(' ORDER BY ', 1)[0], args[1:-1]


class TestSearch:
    async def test_query_shape_and_passages(self) -> None:
        pool = _Pool(
            rows=[
                ('alpha', 'a.md', 'doc-a', json.dumps({'tenant_id': 'acme'}), 0.91),
                ('beta', None, None, {'year': 2025}, 0.5),
                ('gamma', None, None, 'null', 0.1),
            ]
        )
        passages = await _backend(pool).search('q', limit=3, filter=None, score_threshold=0.2)
        assert passages == [
            Passage('alpha', source='a.md', score=0.91, document_id='doc-a', metadata={'tenant_id': 'acme'}),
            Passage('beta', score=0.5, metadata={'year': 2025}),
            Passage('gamma', score=0.1),
        ]
        ((sql, args),) = pool.calls
        assert sql == (
            'SELECT "text", "source", "document_id", "metadata", 1 - ("embedding" <=> $1::vector) '
            'FROM "knowledge_base_chunks" WHERE 1 - ("embedding" <=> $1::vector) >= $2 '
            'ORDER BY "embedding" <=> $1::vector LIMIT $3'
        )
        assert args == (_VECTOR, 0.2, 3)

    async def test_custom_columns_and_schema_qualified_table(self) -> None:
        pool = _Pool()
        backend = _backend(
            pool,
            table='rag.chunks',
            text_column='body',
            embedding_column='vec',
            metadata_column='attrs',
            source_column=None,
            document_column=None,
            chunk_column=None,
        )
        await backend.search('q', limit=2, filter=None)
        ((sql, args),) = pool.calls
        assert sql == (
            'SELECT "body", NULL, NULL, "attrs", 1 - ("vec" <=> $1::vector) FROM "rag"."chunks" '
            'ORDER BY "vec" <=> $1::vector LIMIT $2'
        )
        assert args == (_VECTOR, 2)

    @pytest.mark.parametrize(
        ('filter', 'sql', 'args'),
        [
            ({'tenant_id': 'acme'}, 'coalesce("metadata" -> $2 = $3::jsonb, false)', ('tenant_id', '"acme"')),
            (
                {'field': 'tenant_id', 'operator': '!=', 'value': 'acme'},
                'coalesce("metadata" -> $2 <> $3::jsonb, true)',
                ('tenant_id', '"acme"'),
            ),
            (
                {'source': ['manuals', 'faq']},
                'coalesce("metadata" -> $2 = ANY($3::jsonb[]), false)',
                ('source', ['"manuals"', '"faq"']),
            ),
            (
                {'field': 'source', 'operator': 'not in', 'value': ['drafts']},
                'coalesce(NOT ("metadata" -> $2 = ANY($3::jsonb[])), true)',
                ('source', ['"drafts"']),
            ),
            (
                {'field': 'year', 'operator': '>=', 'value': 2024},
                'coalesce(jsonb_typeof("metadata" -> $2) = jsonb_typeof($3::jsonb) AND "metadata" -> $2 >= $3::jsonb, false)',
                ('year', '2024'),
            ),
            (
                {
                    'operator': 'NOT',
                    'conditions': [
                        {
                            'operator': 'OR',
                            'conditions': [
                                {'field': 'a', 'operator': '==', 'value': True},
                                {'field': 'b', 'operator': '<', 'value': 1.5},
                            ],
                        }
                    ],
                },
                '(NOT ((coalesce("metadata" -> $2 = $3::jsonb, false) OR coalesce(jsonb_typeof("metadata" -> $4) '
                '= jsonb_typeof($5::jsonb) AND "metadata" -> $4 < $5::jsonb, false))))',
                ('a', 'true', 'b', '1.5'),
            ),
        ],
    )
    async def test_filters_compile_to_parameterised_sql(
        self, filter: MetadataFilter, sql: str, args: tuple[object, ...]
    ) -> None:
        assert await _where(filter) == (sql, args)

    async def test_field_names_and_values_never_enter_the_sql(self) -> None:
        sql, args = await _where({"tenant'; DROP TABLE x; --": "acme'; --"})
        assert 'DROP' not in sql and args == ("tenant'; DROP TABLE x; --", json.dumps("acme'; --"))

    @pytest.mark.parametrize('mode', ['keyword', 'hybrid'])
    async def test_only_semantic_mode(self, mode: Any) -> None:
        with pytest.raises(UserError, match="only supports mode='semantic'"):
            await _backend(_Pool()).search('q', limit=5, filter=None, mode=mode)

    async def test_a_pool_without_fetch_is_rejected(self) -> None:
        class _Bare:
            @asynccontextmanager
            async def acquire(self) -> AsyncGenerator[object]:
                yield object()

        with pytest.raises(TypeError, match='has no fetch method'):
            await PgVectorKnowledgeBase(_Bare(), _EMBEDDER).search('q', limit=1, filter=None)


class TestRead:
    async def test_chunks_are_joined_in_order_and_cut_past_max_chars(self) -> None:
        rows = [
            ('first part', 'doc.md', 'doc', {'tenant_id': 'acme', 'tags': ['x']}),
            ('second part', 'doc.md', 'doc', {'tenant_id': 'acme'}),
            ('third part', 'doc.md', 'doc', {'tenant_id': 'acme'}),
        ]
        pool = _Pool(rows=rows)
        document = await _backend(pool).read('doc', max_chars=15, filter=normalize_filter({'tenant_id': 'acme'}))
        assert document is not None
        assert (document.text, document.source, dict(document.metadata)) == (
            'first part\n\nsecond part',
            'doc.md',
            {'tenant_id': 'acme'},
        )
        ((sql, args),) = pool.calls
        assert sql == (
            'SELECT "text", "source", "document_id", "metadata" FROM "knowledge_base_chunks" '
            'WHERE "document_id" = $1 ORDER BY "chunk_index" LIMIT $2'
        )
        assert args == ('doc', 1000)

    async def test_any_chunk_outside_the_filter_blocks_the_read(self) -> None:
        rows: list[tuple[object, ...]] = [('a', None, 'doc', {'tenant_id': 'acme'}), ('b', None, 'doc', {})]
        acme = normalize_filter({'tenant_id': 'acme'})
        assert await _backend(_Pool(rows=rows)).read('doc', max_chars=100, filter=acme) is None
        assert await _backend(_Pool()).read('doc', max_chars=100, filter=None) is None

    async def test_without_a_chunk_column_rows_keep_table_order(self) -> None:
        pool = _Pool(rows=[('only', None, 'doc', '{}')])
        document = await _backend(pool, chunk_column=None).read('doc', max_chars=100, filter=None)
        assert document is not None and document.text == 'only'
        assert 'ORDER BY' not in pool.calls[0][0]

    async def test_without_a_document_column_there_is_no_reading(self) -> None:
        backend = _backend(_Pool(), document_column=None)
        assert backend.can_read is False
        assert await backend.read('doc', max_chars=100, filter=None) is None
        assert list(KnowledgeBase[None](backend).get_toolset().tools) == ['search_knowledge_base']


class TestConfiguration:
    @pytest.mark.parametrize(
        'options', [{'table': 'chunks; DROP'}, {'text_column': '1text'}, {'source_column': 'a.b.c'}]
    )
    def test_identifiers_are_checked(self, options: dict[str, str]) -> None:
        with pytest.raises(ValueError, match='invalid table or column name'):
            _backend(_Pool(), **options)

    def test_max_document_chunks_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match='max_document_chunks must be positive'):
            _backend(_Pool(), max_document_chunks=0)
