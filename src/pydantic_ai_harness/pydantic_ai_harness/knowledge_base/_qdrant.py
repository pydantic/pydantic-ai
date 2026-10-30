"""Qdrant backend: searches a Qdrant collection of embedded chunks, and reads documents back by stitching their chunks.

External assumptions (verified 2026-10-31 against qdrant-client 1.20.0 and
<https://qdrant.tech/documentation/search/filtering/>):

- `query_points(query=<vector>, using=<vector name>, query_filter=..., score_threshold=...)` runs a dense
  search; the score is the collection's distance metric (cosine similarity for a cosine collection).
- Filters nest: `must` is AND, `should` is OR, `must_not` is NOT over its conditions, and a `Filter`
  can be a condition of another. `must_not` over several conditions means none of them, so a NOT
  of an AND wraps the AND in one nested filter.
- `MatchValue` takes a string, integer or boolean; `MatchAny` a list of strings or a list of
  integers; `Range` compares numbers. A condition on a field a point doesn't have is false, so
  `must_not` of it lets the point through, which is how negative comparisons treat a missing field.
- `scroll(scroll_filter=...)` pages through every matching point, not only the most relevant, so a
  document's chunks can be collected completely.
- `AsyncQdrantClient(':memory:')` runs the same filters and queries in process, without a server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai.embeddings import Embedder
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai_harness.knowledge_base._backend import (
    Document,
    FilterNode,
    FilterScalar,
    Passage,
    SearchMode,
    allows_read,
)

try:
    from qdrant_client import AsyncQdrantClient, models
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'qdrant-client is required for QdrantKnowledgeBase. Install it with: pip install qdrant-client'
    ) from _import_error

_RECOVERABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class QdrantKnowledgeBase:
    """Searches a [Qdrant](https://qdrant.tech) collection whose points are embedded chunks of documents.

    Each point carries its chunk text, and optionally its source, document id and position, in its
    payload; every other payload field is metadata a filter can match. The query is embedded with
    `embedder`, which must be the model the collection was embedded with. Ingestion stays with your
    pipeline.

    When points carry a document id, `read` collects all of a document's chunks with `scroll`, orders
    them by their chunk position, and joins them, which gives the agent `read_document`.
    """

    client: AsyncQdrantClient
    """The Qdrant client, for example `AsyncQdrantClient(url=..., api_key=...)`."""

    collection_name: str
    """The collection to search."""

    embedder: Embedder
    """Embeds the query, with the same model the collection's points were embedded with."""

    _: KW_ONLY

    vector_name: str | None = None
    """The named vector to search, for a collection with several; `None` for the default vector."""

    text_field: str = 'text'
    """The payload field holding a chunk's text."""

    source_field: str | None = 'source'
    """The payload field holding where a chunk came from; `None` when there is none."""

    document_field: str | None = 'document_id'
    """The payload field holding the id of the document a chunk belongs to; `None` turns reading off."""

    chunk_field: str | None = 'chunk_index'
    """The payload field holding a chunk's position in its document, which `read` orders chunks by."""

    max_document_chunks: int = 1_000
    """The most chunks `read` collects for one document."""

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not self.collection_name:
            raise ValueError('collection_name must not be empty')
        if self.max_document_chunks < 1:
            raise ValueError(f'max_document_chunks must be positive, got {self.max_document_chunks}')

    @property
    def can_read(self) -> bool:
        """Whether this backend reads documents: only when points carry a document id."""
        return self.document_field is not None

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        """Embed `query` and return the nearest chunks matching `filter`.

        Only semantic search is supported. `score_threshold` is applied by Qdrant, on the collection's
        distance metric.
        """
        if mode not in (None, 'semantic'):
            raise UserError(f"QdrantKnowledgeBase only supports mode='semantic', not {mode!r}")
        query_filter = to_qdrant_filter(filter)
        vector = list((await self.embedder.embed_query(query)).embeddings[0])
        try:
            response = await self.client.query_points(
                self.collection_name,
                query=vector,
                using=self.vector_name,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as error:
            raise _retry_or_raise('Knowledge base search failed', error) from error
        return [self._passage(point.payload or {}, point.score) for point in response.points]

    async def read(self, document_id: str, *, max_chars: int, filter: FilterNode | None) -> Document | None:
        """Collect a document's chunks, in order, if every chunk `allows_read` under `filter`.

        Stops after `max_document_chunks` chunks; the text may also stop once it is past `max_chars`.
        """
        if self.document_field is None:
            return None
        document_filter = models.Filter(
            must=[models.FieldCondition(key=self.document_field, match=models.MatchValue(value=document_id))]
        )
        try:
            points, _ = await self.client.scroll(
                self.collection_name,
                scroll_filter=document_filter,
                limit=self.max_document_chunks,
                with_payload=True,
            )
        except (UnexpectedResponse, ResponseHandlingException) as error:
            raise _retry_or_raise('Reading the document failed', error) from error
        payloads = [point.payload or {} for point in points]
        if not payloads or not all(allows_read(payload, filter) for payload in payloads):
            return None
        payloads.sort(key=self._position)
        parts: list[str] = []
        size = 0
        for payload in payloads:
            text = payload.get(self.text_field)
            if isinstance(text, str):
                parts.append(text)
                size += len(text)
            if size > max_chars:
                break
        first = payloads[0]
        source = first.get(self.source_field) if self.source_field is not None else None
        return Document(
            document_id=document_id,
            text='\n\n'.join(parts),
            source=source if isinstance(source, str) else None,
            metadata={key: value for key, value in self._metadata(first).items() if _is_scalar(value)},  # pyright: ignore[reportArgumentType]
        )

    def _passage(self, payload: Mapping[str, object], score: float) -> Passage:
        text = payload.get(self.text_field)
        source = payload.get(self.source_field) if self.source_field is not None else None
        document = payload.get(self.document_field) if self.document_field is not None else None
        return Passage(
            text=text if isinstance(text, str) else '',
            source=source if isinstance(source, str) else None,
            score=score,
            document_id=str(document) if isinstance(document, str | int) else None,
            metadata=self._metadata(payload),
        )

    def _metadata(self, payload: Mapping[str, object]) -> dict[str, object]:
        reserved = {self.text_field, self.source_field, self.document_field, self.chunk_field}
        return {key: value for key, value in payload.items() if key not in reserved}

    def _position(self, payload: Mapping[str, object]) -> float:
        position = payload.get(self.chunk_field) if self.chunk_field is not None else None
        return float(position) if isinstance(position, int | float) else 0.0


def _is_scalar(value: object) -> bool:
    return isinstance(value, str | int | float | bool)


def _retry_or_raise(what: str, error: Exception) -> Exception:
    """A retry prompt for a failure that passes (throttling, a server error, the network), else the error itself."""
    if isinstance(error, UnexpectedResponse) and error.status_code not in _RECOVERABLE_STATUSES:
        return error
    return ModelRetry(f'{what} ({type(error).__name__}); try again.')


def to_qdrant_filter(filter: FilterNode | None) -> models.Filter | None:
    """Translate a filter tree to a Qdrant `Filter`: AND to `must`, OR to `should`, NOT to `must_not`.

    Raises:
        UserError: A range compares a value that isn't a number, which Qdrant's `Range` can't.
    """
    if filter is None:
        return None
    condition = _condition(filter)
    return condition if isinstance(condition, models.Filter) else models.Filter(must=[condition])


def _condition(node: FilterNode) -> models.FieldCondition | models.Filter:
    if 'conditions' in node:
        children: list[models.Condition] = [_condition(child) for child in node['conditions']]
        if node['operator'] == 'AND':
            return models.Filter(must=children)
        if node['operator'] == 'OR':
            return models.Filter(should=children)
        return models.Filter(must_not=[models.Filter(must=children)])
    field, operator, value = node['field'], node['operator'], node['value']
    if operator == '==':
        return _equals(field, value)
    if operator == '!=':
        return models.Filter(must_not=[_equals(field, value)])
    if operator in ('in', 'not in'):
        assert not isinstance(value, str | int | float | bool)
        any_of = _any_of(field, list(value))
        return any_of if operator == 'in' else models.Filter(must_not=[any_of])
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise UserError(f'QdrantKnowledgeBase compares numbers with {operator!r}, not {value!r}')
    bounds = {'>': 'gt', '>=': 'gte', '<': 'lt', '<=': 'lte'}
    return models.FieldCondition(key=field, range=models.Range(**{bounds[operator]: value}))


def _equals(field: str, value: FilterScalar | Sequence[str | int | float]) -> models.FieldCondition:
    if isinstance(value, str | int):
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))
    assert isinstance(value, float)
    # `MatchValue` has no floats; a float matches as the one-point range.
    return models.FieldCondition(key=field, range=models.Range(gte=value, lte=value))


def _any_of(field: str, values: list[str | int | float]) -> models.FieldCondition | models.Filter:
    if all(isinstance(value, str) for value in values):
        return models.FieldCondition(key=field, match=models.MatchAny(any=[str(value) for value in values]))
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return models.FieldCondition(key=field, match=models.MatchAny(any=[int(value) for value in values]))
    # `MatchAny` takes only all-string or all-integer lists; anything else is an OR of equalities.
    return models.Filter(should=[_equals(field, value) for value in values])
