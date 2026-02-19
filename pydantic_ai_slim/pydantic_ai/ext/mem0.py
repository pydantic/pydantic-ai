from __future__ import annotations

from typing import Any, Protocol

from pydantic_ai import FunctionToolset


class Mem0Client(Protocol):
    """Minimal Mem0Client interface needed for the toolset."""

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> Any: ...


__all__ = ('Mem0Toolset',)


class Mem0Toolset(FunctionToolset):
    """Toolset Providing Mem0 memory tools."""

    def __init__(
        self,
        client: Mem0Client,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        id: str | None = None,
    ):
        self._client = client
        self._user_id = user_id
        self._agent_id = agent_id
        super().__init__([], id=id)

        def save_memory(
            content: str,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Save important information to Mem0 memory.

            Args:
                content: The content to store in memory.
                metadata: Optional metadata to associate with the memory.
            """
            result = self._client.add(
                [{'role': 'user', 'content': content}],
                user_id=self._user_id,
                agent_id=self._agent_id,
                metadata=metadata,
            )
            return {'result': result}

        def search_memory(
            query: str,
            limit: int = 5,
        ) -> dict[str, Any]:
            """Search Mem0 memories.

            Args:
                query: What to search for.
                limit: Max number of results to return.
            """
            results = self._client.search(
                query,
                user_id=self._user_id,
                agent_id=self._agent_id,
                limit=limit,
            )
            return {'results': results}

        self.add_function(save_memory, name='save_memory')
        self.add_function(search_memory, name='search_memory')
