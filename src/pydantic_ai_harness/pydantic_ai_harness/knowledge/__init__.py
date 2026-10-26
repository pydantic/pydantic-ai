"""Knowledge capability: retrieval-augmented generation over workspace files and documents."""

from pydantic_ai_harness.knowledge._capability import Knowledge
from pydantic_ai_harness.knowledge._chunking import Chunker, RecursiveChunker
from pydantic_ai_harness.knowledge._index import KnowledgeSyncResult
from pydantic_ai_harness.knowledge._sources import DEFAULT_FILE_SUFFIXES, KnowledgeDocument, KnowledgeSource
from pydantic_ai_harness.knowledge._store import EmbeddedChunk, InMemoryVectorStore, KnowledgeMatch, VectorStore

__all__ = [
    'DEFAULT_FILE_SUFFIXES',
    'Chunker',
    'EmbeddedChunk',
    'InMemoryVectorStore',
    'Knowledge',
    'KnowledgeDocument',
    'KnowledgeMatch',
    'KnowledgeSource',
    'KnowledgeSyncResult',
    'RecursiveChunker',
    'VectorStore',
]
