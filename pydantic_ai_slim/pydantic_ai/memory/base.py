"""Base protocol and types for memory providers."""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from ..messages import ModelMessage

__all__ = (
    'MemoryProvider',
    'RetrievedMemory',
    'StoredMemory',
)


class RetrievedMemory:
    """Represents a memory retrieved from the memory provider.

    Attributes:
        id: Unique identifier for the memory.
        memory: The actual memory content/text.
        score: Relevance score (0.0 to 1.0).
        metadata: Additional metadata associated with the memory.
        created_at: When the memory was created.
    """

    def __init__(
        self,
        id: str,
        memory: str,
        score: float = 1.0,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
    ):
        self.id = id
        self.memory = memory
        self.score = score
        self.metadata = metadata or {}
        self.created_at = created_at

    def __repr__(self) -> str:
        return f'RetrievedMemory(id={self.id!r}, memory={self.memory!r}, score={self.score})'


class StoredMemory:
    """Represents a memory that was stored.

    Attributes:
        id: Unique identifier for the stored memory.
        memory: The memory content that was stored.
        event: The type of event (ADD, UPDATE, DELETE).
        metadata: Additional metadata.
    """

    def __init__(
        self,
        id: str,
        memory: str,
        event: str = 'ADD',
        metadata: dict[str, Any] | None = None,
    ):
        self.id = id
        self.memory = memory
        self.event = event
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f'StoredMemory(id={self.id!r}, memory={self.memory!r}, event={self.event!r})'


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory providers.

    Memory providers handle storage and retrieval of agent memories.
    This protocol allows for different memory backend implementations
    (e.g., Mem0, custom databases, vector stores, etc.).
    """

    async def retrieve_memories(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve relevant memories based on a query.

        Args:
            query: The search query to find relevant memories.
            user_id: Optional user identifier to scope the search.
            agent_id: Optional agent identifier to scope the search.
            run_id: Optional run identifier to scope the search.
            top_k: Maximum number of memories to retrieve.
            metadata: Additional metadata filters for retrieval.

        Returns:
            List of retrieved memories sorted by relevance.
        """
        ...

    async def store_memories(
        self,
        messages: list[ModelMessage],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMemory]:
        """Store conversation messages as memories.

        Args:
            messages: The conversation messages to store.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            metadata: Additional metadata to store with memories.

        Returns:
            List of stored memories with their IDs and events.
        """
        ...

    async def get_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedMemory]:
        """Get all memories for given identifiers.

        Args:
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            limit: Optional limit on number of memories to return.

        Returns:
            List of all memories matching the filters.
        """
        ...

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        ...


class BaseMemoryProvider(ABC):
    """Abstract base class for memory providers.

    Provides a concrete base that can be extended instead of implementing
    the Protocol directly.
    """

    @abstractmethod
    async def retrieve_memories(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve relevant memories based on a query."""
        raise NotImplementedError

    @abstractmethod
    async def store_memories(
        self,
        messages: list[ModelMessage],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMemory]:
        """Store conversation messages as memories."""
        raise NotImplementedError

    @abstractmethod
    async def get_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedMemory]:
        """Get all memories for given identifiers."""
        raise NotImplementedError

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        raise NotImplementedError
