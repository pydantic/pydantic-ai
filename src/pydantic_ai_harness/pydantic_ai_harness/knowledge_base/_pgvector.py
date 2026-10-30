"""pgvector backend: searches a PostgreSQL table of embedded chunks, and reads documents back by stitching their chunks.

External assumptions (verified 2026-10-31 against PostgreSQL 17 with pgvector 0.8 and asyncpg 0.32):

- `embedding <=> $1::vector` is the cosine distance, so `1 - distance` is a cosine similarity; a
  vector parameter can be passed as its text form (`'[0.1,0.2]'`) without a driver codec.
- `jsonb -> key` yields `NULL` for a missing key, and comparisons with `NULL` are `NULL`, so every
  condition is wrapped in `coalesce`: missing fields fail positive comparisons and pass negative
  ones, as on the other backends.
- `jsonb` values of different types compare by type order rather than raising, so range conditions
  also require `jsonb_typeof` to match.
- asyncpg returns `jsonb` as text unless the pool registers a codec; both forms are accepted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import KW_ONLY, dataclass
from typing import Protocol, runtime_checkable

from pydantic_ai.embeddings import Embedder
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.knowledge_base._backend import (
    Document,
    FilterNode,
    Passage,
    SearchMode,
    allows_read,
)

_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,62}(\.[A-Za-z_][A-Za-z0-9_]{0,62})?')
_RANGES = {'>': '>', '>=': '>=', '<': '<', '<=': '<='}


class PostgresPool(Protocol):
    """The asyncpg-compatible pool surface `PgVectorKnowledgeBase` uses: acquire a connection."""

    def acquire(self) -> AbstractAsyncContextManager[object]:
        """Acquire one connection."""
        ...  # pragma: no cover


@runtime_checkable
class _Fetching(Protocol):
    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]: ...


@dataclass
class PgVectorKnowledgeBase:
    """Searches a PostgreSQL table of embedded chunks with [pgvector](https://github.com/pgvector/pgvector).

    Bring your own table: one row per chunk, with its text, its embedding, a `jsonb` metadata column a
    filter matches against, and optionally its source, document id and position. The column names are
    configurable. The query is embedded with `embedder`, which must be the model the rows were
    embedded with. Ingestion stays with your pipeline.

    When rows carry a document id, `read` collects a document's chunks in order and joins them,
    which gives the agent `read_document`. The pool's driver owns retries: errors propagate.
    """

    pool: PostgresPool
    """An asyncpg-compatible pool, such as `asyncpg.create_pool(...)`. You own its lifecycle."""

    embedder: Embedder
    """Embeds the query, with the same model the rows were embedded with."""

    _: KW_ONLY

    table: str = 'knowledge_base_chunks'
    """The table, optionally schema-qualified (`'rag.chunks'`)."""

    text_column: str = 'text'
    """The column holding a chunk's text."""

    embedding_column: str = 'embedding'
    """The `vector` column holding a chunk's embedding."""

    metadata_column: str = 'metadata'
    """The `jsonb` column holding a chunk's metadata, which filters match against."""

    source_column: str | None = 'source'
    """The column holding where a chunk came from; `None` when there is none."""

    document_column: str | None = 'document_id'
    """The column holding the id of the document a chunk belongs to; `None` turns reading off."""

    chunk_column: str | None = 'chunk_index'
    """The column holding a chunk's position in its document, which `read` orders chunks by."""

    max_document_chunks: int = 1_000
    """The most chunks `read` collects for one document."""

    def __post_init__(self) -> None:
        """Validate the table and column names, which go into the SQL as identifiers."""
        for name in (
            self.table,
            self.text_column,
            self.embedding_column,
            self.metadata_column,
            self.source_column,
            self.document_column,
            self.chunk_column,
        ):
            if name is not None and not _IDENTIFIER.fullmatch(name):
                raise ValueError(f'invalid table or column name: {name!r}')
        if self.max_document_chunks < 1:
            raise ValueError(f'max_document_chunks must be positive, got {self.max_document_chunks}')

    @property
    def can_read(self) -> bool:
        """Whether this backend reads documents: only when rows carry a document id."""
        return self.document_column is not None

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        """Embed `query` and return the nearest chunks matching `filter`, by cosine similarity.

        Only semantic search is supported. `score_threshold` is a cosine similarity, from -1 to 1.
        """
        if mode not in (None, 'semantic'):
            raise UserError(f"PgVectorKnowledgeBase only supports mode='semantic', not {mode!r}")
        vector = list((await self.embedder.embed_query(query)).embeddings[0])
        args: list[object] = [_vector_literal(vector)]
        conditions = [_sql_filter(filter, _quote(self.metadata_column), args)] if filter is not None else []
        distance = f'{_quote(self.embedding_column)} <=> $1::vector'
        if score_threshold is not None:
            args.append(score_threshold)
            conditions.append(f'1 - ({distance}) >= ${len(args)}')
        args.append(limit)
        where = f'WHERE {" AND ".join(conditions)} ' if conditions else ''
        sql = (
            f'SELECT {self._columns()}, 1 - ({distance}) FROM {_quote(self.table)} '
            f'{where}ORDER BY {distance} LIMIT ${len(args)}'
        )
        rows = await self._fetch(sql, args)
        return [
            Passage(
                text=str(row[0]),
                source=_text_or_none(row[1]),
                score=float(row[4]),  # pyright: ignore[reportArgumentType]
                document_id=_text_or_none(row[2]),
                metadata=_metadata(row[3]),
            )
            for row in rows
        ]

    async def read(self, document_id: str, *, max_chars: int, filter: FilterNode | None) -> Document | None:
        """Collect a document's chunks, in order, if every chunk `allows_read` under `filter`.

        Stops after `max_document_chunks` chunks; the text may also stop once it is past `max_chars`.
        """
        if self.document_column is None:
            return None
        order = f'ORDER BY {_quote(self.chunk_column)} ' if self.chunk_column is not None else ''
        sql = (
            f'SELECT {self._columns()} FROM {_quote(self.table)} '
            f'WHERE {_quote(self.document_column)} = $1 {order}LIMIT $2'
        )
        rows = await self._fetch(sql, [document_id, self.max_document_chunks])
        metadata = [_metadata(row[3]) for row in rows]
        if not rows or not all(allows_read(item, filter) for item in metadata):
            return None
        parts: list[str] = []
        size = 0
        for row in rows:
            parts.append(str(row[0]))
            size += len(parts[-1])
            if size > max_chars:
                break
        return Document(
            document_id=document_id,
            text='\n\n'.join(parts),
            source=_text_or_none(rows[0][1]),
            metadata={key: value for key, value in metadata[0].items() if _is_scalar(value)},  # pyright: ignore[reportArgumentType]
        )

    def _columns(self) -> str:
        """text, source, document id, metadata: `NULL` for a column the table doesn't have."""
        optional = [self.source_column, self.document_column]
        return ', '.join(
            [
                _quote(self.text_column),
                *(_quote(name) if name else 'NULL' for name in optional),
                _quote(self.metadata_column),
            ]
        )

    async def _fetch(self, sql: str, args: list[object]) -> Sequence[Sequence[object]]:
        async with self.pool.acquire() as connection:
            if not isinstance(connection, _Fetching):
                raise TypeError(f'pool acquired {type(connection).__name__}, which has no fetch method')
            return await connection.fetch(sql, *args)


def _quote(identifier: str | None) -> str:
    assert identifier is not None
    return '.'.join(f'"{part}"' for part in identifier.split('.'))


def _vector_literal(vector: Sequence[float]) -> str:
    return '[' + ','.join(repr(float(value)) for value in vector) + ']'


def _text_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def _metadata(value: object) -> dict[str, object]:
    """A row's metadata, from asyncpg's text form of `jsonb` or an already decoded mapping."""
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


def _sql_filter(node: FilterNode, column: str, args: list[object]) -> str:
    """Compile a filter tree to a SQL condition over `column`, appending its parameters to `args`.

    Field names and values are always parameters; only operators from fixed tables are written into
    the SQL. Every leaf is `coalesce`d to a boolean, so `NOT` and missing fields behave as documented.
    """
    if 'conditions' in node:
        children = [_sql_filter(child, column, args) for child in node['conditions']]
        joined = f' {"OR" if node["operator"] == "OR" else "AND"} '.join(children)
        return f'(NOT ({joined}))' if node['operator'] == 'NOT' else f'({joined})'
    args.append(node['field'])
    field = f'{column} -> ${len(args)}'
    operator, value = node['operator'], node['value']
    if operator in ('in', 'not in'):
        assert not isinstance(value, str | int | float | bool)
        args.append([json.dumps(item) for item in value])
        test = f'{field} = ANY(${len(args)}::jsonb[])'
        return f'coalesce({test}, false)' if operator == 'in' else f'coalesce(NOT ({test}), true)'
    args.append(json.dumps(value))
    param = f'${len(args)}::jsonb'
    if operator == '==':
        return f'coalesce({field} = {param}, false)'
    if operator == '!=':
        return f'coalesce({field} <> {param}, true)'
    return f'coalesce(jsonb_typeof({field}) = jsonb_typeof({param}) AND {field} {_RANGES[operator]} {param}, false)'
