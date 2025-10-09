"""Memory context for use in agent runs."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

from .base import RetrievedMemory, StoredMemory

if TYPE_CHECKING:
    from .base import MemoryProvider

__all__ = ('MemoryContext',)


class MemoryContext:
    """Context for memory operations within an agent run.

    This class provides access to the memory provider and tracks
    memories retrieved and stored during the current run.

    Attributes:
        provider: The memory provider instance.
        retrieved_memories: List of memories retrieved in this run.
        stored_memories: List of memories stored in this run.
    """

    def __init__(self, provider: MemoryProvider):
        """Initialize memory context.

        Args:
            provider: The memory provider to use.
        """
        self.provider = provider
        self.retrieved_memories: list[RetrievedMemory] = []
        self.stored_memories: list[StoredMemory] = []

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[RetrievedMemory]:
        """Search for memories.

        Args:
            query: The search query.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            top_k: Maximum number of memories to retrieve.
            metadata: Additional metadata filters.

        Returns:
            List of retrieved memories.
        """
        memories = await self.provider.retrieve_memories(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            top_k=top_k,
            metadata=metadata,
        )
        self.retrieved_memories.extend(memories)
        return memories

    async def add(
        self,
        messages: list[Any],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMemory]:
        """Add new memories.

        Args:
            messages: Messages to store as memories.
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            metadata: Additional metadata.

        Returns:
            List of stored memories.
        """
        stored = await self.provider.store_memories(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
        )
        self.stored_memories.extend(stored)
        return stored

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedMemory]:
        """Get all memories.

        Args:
            user_id: Optional user identifier.
            agent_id: Optional agent identifier.
            run_id: Optional run identifier.
            limit: Optional limit on results.

        Returns:
            List of all memories.
        """
        return await self.provider.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
        )

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if successful, False otherwise.
        """
        return await self.provider.delete_memory(memory_id)
