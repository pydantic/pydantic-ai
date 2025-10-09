"""Mem0 memory provider implementation."""

from __future__ import annotations as _annotations

import logging
from typing import TYPE_CHECKING, Any

from ..base import BaseMemoryProvider, RetrievedMemory, StoredMemory
from ..config import MemoryConfig

if TYPE_CHECKING:
    from ...messages import ModelMessage

try:
    from mem0 import AsyncMemoryClient, MemoryClient

    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    AsyncMemoryClient = None  # type: ignore[assignment,misc]
    MemoryClient = None  # type: ignore[assignment,misc]

__all__ = ('Mem0Provider',)

logger = logging.getLogger(__name__)


class Mem0Provider(BaseMemoryProvider):
    """Memory provider using Mem0 platform.

    This provider integrates with Mem0's hosted platform for memory storage
    and retrieval.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.memory import MemoryConfig
        from pydantic_ai.memory.providers import Mem0Provider

        # Create mem0 provider
        memory = Mem0Provider(api_key="your-mem0-api-key")

        # Create agent with memory
        agent = Agent(
            'openai:gpt-4o',
            memory_provider=memory
        )

        # Use agent - memories are automatically managed
        result = await agent.run(
            'My name is Alice',
            deps={'user_id': 'user_123'}
        )
        ```

    Attributes:
        client: The Mem0 client instance (sync or async).
        config: Memory configuration settings.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        host: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        config: MemoryConfig | None = None,
        client: AsyncMemoryClient | MemoryClient | None = None,  # type: ignore[valid-type]
        version: str = '2',
    ):
        """Initialize Mem0 provider.

        Args:
            api_key: Mem0 API key. If not provided, will look for MEM0_API_KEY env var.
            host: Mem0 API host. Defaults to https://api.mem0.ai
            org_id: Organization ID for mem0 platform.
            project_id: Project ID for mem0 platform.
            config: Memory configuration. Uses defaults if not provided.
            client: Optional pre-configured Mem0 client (sync or async).
            version: API version to use. Defaults to '2' (recommended).

        Raises:
            ImportError: If mem0 package is not installed.
            ValueError: If no API key is provided.
        """
        if not MEM0_AVAILABLE:
            raise ImportError(
                'mem0 is not installed. Install it with: pip install mem0ai\n'
                'Or install pydantic-ai with mem0 support: pip install pydantic-ai[mem0]'
            )

        self.config = config or MemoryConfig()
        self.version = version

        if client is not None:
            self.client = client
            self._is_async = isinstance(client, AsyncMemoryClient)  # type: ignore[arg-type,misc]
        else:
            # Create async client by default for better performance
            self.client = AsyncMemoryClient(  # type: ignore[misc]
                api_key=api_key,
                host=host,
                org_id=org_id,
                project_id=project_id,
            )
            self._is_async = True

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
        """Retrieve relevant memories from Mem0.

        Args:
            query: The search query.
            user_id: User identifier.
            agent_id: Agent identifier.
            run_id: Run/session identifier.
            top_k: Maximum number of memories to retrieve.
            metadata: Additional metadata filters.

        Returns:
            List of retrieved memories sorted by relevance.
        """
        # Build search parameters
        search_kwargs: dict[str, Any] = {
            'query': query,
            'top_k': top_k,
        }

        # Add identifiers
        if user_id:
            search_kwargs['user_id'] = user_id
        if agent_id:
            search_kwargs['agent_id'] = agent_id
        if run_id:
            search_kwargs['run_id'] = run_id
        if metadata:
            search_kwargs['metadata'] = metadata

        # Perform search
        try:
            if self._is_async:
                response = await self.client.search(**search_kwargs)
            else:
                response = self.client.search(**search_kwargs)

            # Parse response - handle both v1.1 format (dict) and raw list
            if isinstance(response, dict):
                results = response.get('results', [])
            elif isinstance(response, list):
                results = response
            else:
                logger.warning(f'Unexpected response type from Mem0: {type(response)}')
                results = []

            # Convert to RetrievedMemory objects
            memories = []
            for result in results:
                # Handle both dict and direct memory objects
                if isinstance(result, dict):
                    memory = RetrievedMemory(
                        id=result.get('id', ''),
                        memory=result.get('memory', ''),
                        score=result.get('score', 1.0),
                        metadata=result.get('metadata', {}),
                        created_at=result.get('created_at'),
                    )
                else:
                    # Skip non-dict results
                    continue

                # Apply relevance score filter
                if memory.score >= self.config.min_relevance_score:
                    memories.append(memory)

            logger.debug(f'Retrieved {len(memories)} memories from Mem0 for query: {query[:50]}...')
            return memories

        except Exception as e:
            logger.error(f'Error retrieving memories from Mem0: {e}')
            # Return empty list on error to not break agent execution
            return []

    def _convert_messages_to_mem0_format(self, messages: list[ModelMessage]) -> list[dict[str, str]]:
        """Convert ModelMessage objects to mem0 format.

        Args:
            messages: Messages to convert.

        Returns:
            List of message dicts in mem0 format.
        """
        mem0_messages = []
        for msg in messages:
            msg_dict = self._extract_message_content(msg)
            if msg_dict['content']:
                mem0_messages.append(msg_dict)
        return mem0_messages

    def _extract_message_content(self, msg: ModelMessage) -> dict[str, str]:  # type: ignore[misc]
        """Extract content and role from a ModelMessage.

        Args:
            msg: Message to extract from.

        Returns:
            Dict with 'role' and 'content' keys.
        """
        msg_dict = {'role': 'user', 'content': ''}

        if not hasattr(msg, 'parts'):  # type: ignore[misc]
            return msg_dict

        for part in msg.parts:  # type: ignore[attr-defined]
            # Extract content
            if hasattr(part, 'content'):
                content_value = part.content  # type: ignore[attr-defined]
                msg_dict['content'] += str(content_value) if not isinstance(content_value, str) else content_value

            # Determine role from part type
            part_type = type(part).__name__
            if 'User' in part_type:
                msg_dict['role'] = 'user'
            elif 'Text' in part_type or 'Assistant' in part_type:
                msg_dict['role'] = 'assistant'
            elif 'System' in part_type:
                msg_dict['role'] = 'system'

        return msg_dict

    def _parse_mem0_response(self, response: Any) -> list[Any]:  # type: ignore[misc]
        """Parse Mem0 API response to extract results.

        Args:
            response: Raw response from Mem0 API.

        Returns:
            List of result items.
        """
        if isinstance(response, dict):
            return response.get('results', [])  # type: ignore[return-value]
        if isinstance(response, list):
            return response  # type: ignore[return-value]
        logger.warning(f'Unexpected response type from Mem0: {type(response)}')
        return []

    async def store_memories(
        self,
        messages: list[ModelMessage],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[StoredMemory]:
        """Store conversation messages as memories in Mem0.

        Args:
            messages: Conversation messages to store.
            user_id: User identifier.
            agent_id: Agent identifier.
            run_id: Run/session identifier.
            metadata: Additional metadata.

        Returns:
            List of stored memories.
        """
        # Convert messages to mem0 format
        mem0_messages = self._convert_messages_to_mem0_format(messages)
        if not mem0_messages:
            logger.warning('No valid messages to store in Mem0')
            return []

        # Build add parameters
        add_kwargs: dict[str, Any] = {'messages': mem0_messages}
        if user_id:
            add_kwargs['user_id'] = user_id
        if agent_id:
            add_kwargs['agent_id'] = agent_id
        if run_id:
            add_kwargs['run_id'] = run_id
        if metadata:
            add_kwargs['metadata'] = metadata

        # Store in Mem0
        try:
            response = await self.client.add(**add_kwargs) if self._is_async else self.client.add(**add_kwargs)  # type: ignore[misc]
            results = self._parse_mem0_response(response)

            # Convert to StoredMemory objects
            stored = [
                StoredMemory(
                    id=result.get('id', ''),  # type: ignore[union-attr]
                    memory=result.get('memory', ''),  # type: ignore[union-attr]
                    event=result.get('event', 'ADD'),  # type: ignore[union-attr]
                    metadata=result.get('metadata', {}),  # type: ignore[union-attr]
                )
                for result in results
                if isinstance(result, dict)
            ]

            logger.debug(f'Stored {len(stored)} memories in Mem0')
            return stored

        except Exception as e:
            logger.error(f'Error storing memories in Mem0: {e}')
            return []

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
            user_id: User identifier.
            agent_id: Agent identifier.
            run_id: Run identifier.
            limit: Optional limit on results.

        Returns:
            List of all memories.
        """
        get_kwargs: dict[str, Any] = {}

        if user_id:
            get_kwargs['user_id'] = user_id
        if agent_id:
            get_kwargs['agent_id'] = agent_id
        if run_id:
            get_kwargs['run_id'] = run_id

        try:
            if self._is_async:
                response = await self.client.get_all(**get_kwargs)
            else:
                response = self.client.get_all(**get_kwargs)

            # Parse response - handle both v1.1 format (dict) and raw list
            if isinstance(response, dict):
                results = response.get('results', [])
            elif isinstance(response, list):
                results = response
            else:
                logger.warning(f'Unexpected response type from Mem0: {type(response)}')
                results = []

            # Apply limit if specified
            if limit:
                results = results[:limit]

            memories = []
            for result in results:
                if isinstance(result, dict):
                    memories.append(
                        RetrievedMemory(
                            id=result.get('id', ''),
                            memory=result.get('memory', ''),
                            score=1.0,  # No score for get_all
                            metadata=result.get('metadata', {}),
                            created_at=result.get('created_at'),
                        )
                    )

            return memories

        except Exception as e:
            logger.error(f'Error getting all memories from Mem0: {e}')
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Mem0.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self._is_async:
                await self.client.delete(memory_id)
            else:
                self.client.delete(memory_id)

            logger.debug(f'Deleted memory {memory_id} from Mem0')
            return True

        except Exception as e:
            logger.error(f'Error deleting memory {memory_id} from Mem0: {e}')
            return False
