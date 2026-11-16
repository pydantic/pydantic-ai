"""Memory capability: a persistent, injected notebook plus on-demand memory files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.memory._capability import Memory, WriteMetadataFunc
from pydantic_ai_harness.memory._file import FileMemoryStore
from pydantic_ai_harness.memory._in_memory import InMemoryMemoryStore
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
    'FileMemoryStore',
    'InMemoryMemoryStore',
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


_RENAMED = {'FileStore': FileMemoryStore, 'InMemoryStore': InMemoryMemoryStore}


def __getattr__(name: str) -> object:
    renamed = _RENAMED.get(name)
    if renamed is not None:
        warn_class_renamed(name, renamed.__name__, __name__)
        return renamed
    if name == 'PixeltableMemoryStore':
        from pydantic_ai_harness.memory._pixeltable import PixeltableMemoryStore

        return PixeltableMemoryStore
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
