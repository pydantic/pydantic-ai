"""Memory capability: a persistent, injected notebook plus on-demand memory files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai_harness.memory._capability import Memory, WriteMetadataFunc
from pydantic_ai_harness.memory._file import FileStore
from pydantic_ai_harness.memory._in_memory import InMemoryStore
from pydantic_ai_harness.memory._postgres import PostgresConnection, PostgresMemoryStore, PostgresPool
from pydantic_ai_harness.memory._sqlite import SqliteMemoryStore
from pydantic_ai_harness.memory._store import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchMatch,
    MemorySearchResult,
    MemoryStore,
    MetadataMemoryStore,
    SearchableMemoryStore,
)
from pydantic_ai_harness.memory._toolset import (
    MemoryDeleteResult,
    MemorySearchMatchResult,
    MemorySearchResponse,
    MemoryToolset,
    MemoryWriteResult,
)

__all__ = [
    'FileStore',
    'InMemoryStore',
    'Memory',
    'MemoryConflictError',
    'MemoryDeleteResult',
    'MemoryFile',
    'MemoryMutation',
    'MemoryOperation',
    'MemoryOperationConflictError',
    'MemorySearchMatch',
    'MemorySearchMatchResult',
    'MemorySearchResponse',
    'MemorySearchResult',
    'MemoryStore',
    'MemoryToolset',
    'MemoryWriteResult',
    'MetadataMemoryStore',
    'PostgresConnection',
    'PostgresMemoryStore',
    'PostgresPool',
    'SearchableMemoryStore',
    'SqliteMemoryStore',
    'WriteMetadataFunc',
]


if TYPE_CHECKING:
    from pydantic_ai_harness.memory._pixeltable import PixeltableMemoryStore as PixeltableMemoryStore


def __getattr__(name: str) -> Any:
    if name == 'PixeltableMemoryStore':
        from pydantic_ai_harness.memory._pixeltable import PixeltableMemoryStore

        return PixeltableMemoryStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
