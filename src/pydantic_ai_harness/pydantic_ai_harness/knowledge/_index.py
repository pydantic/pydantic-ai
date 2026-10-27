"""Keep a `VectorStore` in step with the documents `Knowledge` sources produce, and search it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cached_property

import anyio
from opentelemetry.trace import Tracer

from pydantic_ai.embeddings import Embedder
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.knowledge._chunking import Chunker
from pydantic_ai_harness.knowledge._sources import KnowledgeDocument, KnowledgeSourceLike, load_sources
from pydantic_ai_harness.knowledge._store import EmbeddedChunk, KnowledgeMatch, VectorStore

_EMBED_BATCH_SIZE = 64
"""Chunks per embedding request, within the per-request input limits of the supported providers."""


@dataclass(frozen=True, kw_only=True)
class KnowledgeSyncResult:
    """What one sync of `Knowledge` changed in its store."""

    documents: int
    """Documents the sources produced."""

    embedded: int
    """Documents embedded because they were new or changed, or because the embedding model or chunking changed."""

    removed: int
    """Documents removed from the store because the sources no longer produce them."""

    chunks_embedded: int
    """Chunks sent to the embedding model."""


@dataclass(frozen=True)
class _Pending:
    document: KnowledgeDocument
    fingerprint: str
    chunks: Sequence[str]


class KnowledgeIndex:
    """Syncs and searches one store; shared by every run of the capability that owns it."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        model_key: str,
        sources: Sequence[KnowledgeSourceLike],
        store: VectorStore,
        chunker: Chunker,
        file_suffixes: Collection[str],
    ) -> None:
        self._embedder = embedder
        self._model_key = model_key
        self._sources = sources
        self._store = store
        self._chunker = chunker
        self._file_suffixes = file_suffixes

    @cached_property
    def _lock(self) -> anyio.Lock:
        # Deferred to first use, so the lock binds to the loop that syncs, not the one that built the agent.
        return anyio.Lock()

    async def sync(self, workspace: Workspace, tracer: Tracer) -> KnowledgeSyncResult:
        """Embed new and changed documents and remove those the sources no longer produce.

        Serialized, so parallel searches do not embed the same documents twice.
        """
        async with self._lock:
            return await self._sync_in_span(workspace, tracer)

    async def search(self, workspace: Workspace, tracer: Tracer, query: str, *, limit: int) -> list[KnowledgeMatch]:
        """Sync the store with `workspace`, then return the chunks most similar to `query`.

        The sync and the store search hold one lock, so a concurrent search over another workspace
        cannot replace the store's documents in between.
        """
        vector = (await self._embedder.embed_query(query)).embeddings[0]
        async with self._lock:
            await self._sync_in_span(workspace, tracer)
            return list(await self._store.search(vector, limit=limit))

    async def _sync_in_span(self, workspace: Workspace, tracer: Tracer) -> KnowledgeSyncResult:
        with tracer.start_as_current_span('knowledge_sync') as span:
            result = await self._sync(workspace)
            if span.is_recording():
                span.set_attributes(
                    {
                        'knowledge.documents': result.documents,
                        'knowledge.documents_embedded': result.embedded,
                        'knowledge.documents_removed': result.removed,
                        'knowledge.chunks_embedded': result.chunks_embedded,
                    }
                )
            return result

    async def _sync(self, workspace: Workspace) -> KnowledgeSyncResult:
        documents = await load_sources(self._sources, workspace, self._file_suffixes)
        stored = await self._store.fingerprints()
        pending: list[_Pending] = []
        for document in documents:
            chunks = self._chunker.chunk(document.text)
            fingerprint = self._fingerprint(chunks)
            if stored.get(document.id) != fingerprint:
                pending.append(_Pending(document, fingerprint, chunks))

        current = {document.id for document in documents}
        removed = [document_id for document_id in stored if document_id not in current]
        if removed:
            await self._store.remove_documents(removed)
        chunks_embedded = await self._embed(pending)
        return KnowledgeSyncResult(
            documents=len(documents), embedded=len(pending), removed=len(removed), chunks_embedded=chunks_embedded
        )

    async def _embed(self, pending: Sequence[_Pending]) -> int:
        """Embed `pending` in batches that span documents, storing each document once all its chunks are embedded.

        Storing as batches complete keeps the progress of a sync that fails part way.
        """
        texts = [chunk for item in pending for chunk in item.chunks]
        vectors: list[Sequence[float]] = []
        stored = 0
        offset = 0
        for start in range(0, len(texts) + 1, _EMBED_BATCH_SIZE):
            if batch := texts[start : start + _EMBED_BATCH_SIZE]:
                vectors.extend((await self._embedder.embed_documents(batch)).embeddings)
            # A document with no chunks is stored too, so its fingerprint stops it being chunked into a request again.
            while stored < len(pending) and offset + len(pending[stored].chunks) <= len(vectors):
                item = pending[stored]
                chunks = [
                    EmbeddedChunk(text=text, vector=vector)
                    for text, vector in zip(item.chunks, vectors[offset : offset + len(item.chunks)], strict=True)
                ]
                await self._store.replace_document(item.document.id, fingerprint=item.fingerprint, chunks=chunks)
                offset += len(item.chunks)
                stored += 1
        return len(texts)

    def _fingerprint(self, chunks: Sequence[str]) -> str:
        """Identify what a document's stored vectors were computed from: the embedding model and the chunk texts."""
        payload = json.dumps([self._model_key, list(chunks)], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()
