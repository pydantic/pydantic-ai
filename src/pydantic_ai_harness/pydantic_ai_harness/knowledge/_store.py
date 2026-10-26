"""Where `Knowledge` keeps embedded chunks, and the in-memory store it uses by default."""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.exceptions import UserError


@dataclass(frozen=True, kw_only=True)
class EmbeddedChunk:
    """One chunk of a document with its embedding, as `Knowledge` hands it to a `VectorStore`."""

    text: str
    """The chunk's text, returned to the model when the chunk matches a search."""

    vector: Sequence[float]
    """The chunk's document embedding."""


@dataclass(frozen=True, kw_only=True)
class KnowledgeMatch:
    """A chunk returned by `search_knowledge`, most similar first."""

    document_id: str
    """The id of the document the chunk came from: its workspace path, or the id of a `KnowledgeDocument`."""

    text: str
    """The chunk's text."""

    score: float
    """Similarity to the query, higher is closer. `InMemoryVectorStore` reports cosine similarity."""


class VectorStore(Protocol):
    """Stores embedded chunks per document and finds the chunks nearest a query vector.

    `Knowledge` replaces a document's chunks as a whole, so a store never has to merge chunk
    lists. Each document carries a fingerprint of the embedding model and chunks it was indexed
    with, which `Knowledge` compares to decide what to embed again.

    A store holds one corpus: `Knowledge` removes every document its sources no longer
    produce, so do not share a store between `Knowledge` capabilities with different sources.
    """

    async def fingerprints(self) -> Mapping[str, str]:
        """Return the id and fingerprint of every stored document."""
        ...

    async def replace_document(self, document_id: str, *, fingerprint: str, chunks: Sequence[EmbeddedChunk]) -> None:
        """Store `chunks` as the whole of `document_id`, replacing any chunks stored for it before."""
        ...

    async def remove_documents(self, document_ids: Collection[str]) -> None:
        """Remove every chunk of each document in `document_ids`; ids that are not stored are ignored."""
        ...

    async def search(self, vector: Sequence[float], *, limit: int) -> Sequence[KnowledgeMatch]:
        """Return up to `limit` chunks most similar to `vector`, most similar first."""
        ...


@dataclass(frozen=True)
class _StoredChunk:
    text: str
    vector: tuple[float, ...]
    norm: float


@dataclass(frozen=True)
class _StoredDocument:
    fingerprint: str
    chunks: tuple[_StoredChunk, ...]


class InMemoryVectorStore(VectorStore):
    """Keeps chunks in process memory and ranks them by exact cosine similarity.

    Every search compares the query with every chunk, which suits corpora up to tens of
    thousands of chunks. The index lives as long as the store object, so a new process embeds
    the corpus again on its first sync.
    """

    def __init__(self) -> None:
        self._documents: dict[str, _StoredDocument] = {}

    async def fingerprints(self) -> Mapping[str, str]:
        return {document_id: document.fingerprint for document_id, document in self._documents.items()}

    async def replace_document(self, document_id: str, *, fingerprint: str, chunks: Sequence[EmbeddedChunk]) -> None:
        stored = tuple(_StoredChunk(chunk.text, tuple(chunk.vector), math.hypot(*chunk.vector)) for chunk in chunks)
        self._documents[document_id] = _StoredDocument(fingerprint, stored)

    async def remove_documents(self, document_ids: Collection[str]) -> None:
        for document_id in document_ids:
            self._documents.pop(document_id, None)

    async def search(self, vector: Sequence[float], *, limit: int) -> list[KnowledgeMatch]:
        query_norm = math.hypot(*vector)
        scored = (
            KnowledgeMatch(document_id=document_id, text=chunk.text, score=_cosine(vector, query_norm, chunk))
            for document_id, document in self._documents.items()
            for chunk in document.chunks
        )
        return heapq.nlargest(limit, scored, key=lambda match: match.score)


def _cosine(vector: Sequence[float], norm: float, chunk: _StoredChunk) -> float:
    if len(vector) != len(chunk.vector):
        raise UserError(
            f'`InMemoryVectorStore` holds {len(chunk.vector)}-dimensional vectors, but the query has {len(vector)}. '
            'The embedding size changed under the same model name, such as through its `dimensions` setting; '
            'use a new store.'
        )
    if not norm or not chunk.norm:
        return 0.0
    return sum(a * b for a, b in zip(vector, chunk.vector)) / (norm * chunk.norm)
