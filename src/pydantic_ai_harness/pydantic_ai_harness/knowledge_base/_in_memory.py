"""A knowledge base kept in process memory and ranked with BM25: for tests, examples and small fixed corpora."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic_ai_harness.knowledge_base._backend import Document, MetadataFilter, Passage, matches

_TOKEN = re.compile(r'\w+')
_PARAGRAPH = re.compile(r'\n\s*\n')
_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def chunk(text: str, max_chars: int) -> list[str]:
    """Split `text` into chunks of whole paragraphs, each up to `max_chars` unless one paragraph is longer."""
    chunks: list[str] = []
    current = ''
    for paragraph in (part.strip() for part in _PARAGRAPH.split(text)):
        if not paragraph:
            continue
        candidate = f'{current}\n\n{paragraph}' if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class _Chunk:
    document: Document
    text: str
    terms: Counter[str]
    length: int


class InMemoryKnowledgeBase:
    """A knowledge base held in process memory, searched with BM25 keyword ranking.

    Documents are split into chunks of whole paragraphs, and a search ranks the chunks of the documents
    that match the filter. It reads whole documents too, so the agent gets `read_document`. Nothing is
    persisted and there are no embeddings: use it for tests, examples, and small corpora that ship with
    the application, and a hosted backend for anything that has to scale or match by meaning.
    """

    def __init__(self, documents: Iterable[Document] = (), *, chunk_chars: int = 1_000) -> None:
        """Index `documents`, splitting each into chunks of up to `chunk_chars` characters."""
        if chunk_chars < 1:
            raise ValueError(f'chunk_chars must be positive, got {chunk_chars}')
        self._chunk_chars = chunk_chars
        self._documents: dict[str, Document] = {}
        self._chunks: list[_Chunk] = []
        self.add(documents)

    def add(self, documents: Iterable[Document]) -> None:
        """Add `documents`, replacing any already stored under the same `document_id`."""
        for document in documents:
            self._documents[document.document_id] = document
        self._chunks = [
            _Chunk(document, text, Counter(tokens), len(tokens))
            for document in self._documents.values()
            for text in chunk(document.text, self._chunk_chars)
            if (tokens := _tokens(text))
        ]

    async def search(self, query: str, *, limit: int, filter: MetadataFilter | None) -> Sequence[Passage]:
        """Rank the matching documents' chunks against `query` and return the best `limit` that share a term."""
        terms = set(_tokens(query))
        candidates = [item for item in self._chunks if matches(item.document.metadata, filter)]
        if not terms or not candidates:
            return []
        average = sum(item.length for item in candidates) / len(candidates)
        frequency = {term: sum(1 for item in candidates if term in item.terms) for term in terms}
        scored: list[tuple[float, _Chunk]] = []
        for item in candidates:
            score = 0.0
            for term in terms:
                count = item.terms[term]
                if count:
                    idf = math.log(1 + (len(candidates) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                    score += idf * count * (_K1 + 1) / (count + _K1 * (1 - _B + _B * item.length / average))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Passage(
                text=item.text,
                source=item.document.source,
                score=score,
                document_id=item.document.document_id,
                metadata=dict(item.document.metadata),
            )
            for score, item in scored[:limit]
        ]

    async def read(self, document_id: str, *, max_chars: int, filter: MetadataFilter | None) -> Document | None:
        """Return the stored document if it matches `filter`, else `None`; the caller cuts the text at `max_chars`."""
        document = self._documents.get(document_id)
        return document if document is not None and matches(document.metadata, filter) else None
