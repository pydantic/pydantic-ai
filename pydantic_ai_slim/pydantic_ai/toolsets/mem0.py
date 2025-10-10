"""Mem0 memory toolset for Pydantic AI.

This toolset provides memory capabilities using the Mem0 platform,
allowing agents to save and search through conversation memories.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

from .._run_context import AgentDepsT, RunContext
from .function import FunctionToolset

if TYPE_CHECKING:
    from mem0 import AsyncMemoryClient, MemoryClient

try:
    from mem0 import AsyncMemoryClient, MemoryClient
except ImportError as _e:
    _import_error = _e
    AsyncMemoryClient = None
    MemoryClient = None
else:
    _import_error = None

__all__ = ('Mem0Toolset',)


class Mem0Toolset(FunctionToolset[AgentDepsT]):
    """A toolset that provides Mem0 memory capabilities to agents.

    This toolset adds two tools to your agent:
    - `save_memory`: Save information to memory for later retrieval
    - `search_memory`: Search through stored memories

    Example:
        ```python test="skip"
        from pydantic_ai import Agent
        from pydantic_ai.toolsets import Mem0Toolset

        # Create toolset with Mem0 API key
        mem0_toolset = Mem0Toolset(api_key='your-mem0-api-key')

        # Add to agent
        agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])

        async def main():
            # The agent can now use memory tools automatically
            await agent.run(
                'Remember that my favorite color is blue',
                deps='user_123'
            )
        ```

    The toolset expects the agent's `deps` to be either:
    - A string representing the user_id
    - An object with a `user_id` attribute
    - An object with a `get_user_id()` method

    Attributes:
        client: The Mem0 client instance (sync or async).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        host: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        client: AsyncMemoryClient | MemoryClient | None = None,  # type: ignore[valid-type]
        id: str | None = None,
        limit: int = 5,
    ):
        """Initialize the Mem0 toolset.

        Args:
            api_key: Mem0 API key. If not provided, will look for MEM0_API_KEY env var.
            host: Mem0 API host. Defaults to https://api.mem0.ai
            org_id: Organization ID for Mem0 platform.
            project_id: Project ID for Mem0 platform.
            client: Optional pre-configured Mem0 client (sync or async).
            id: Optional unique ID for the toolset.
            limit: Default number of memories to retrieve in search. Defaults to 5.

        Raises:
            ImportError: If mem0 package is not installed.
        """
        if _import_error is not None:
            raise ImportError(
                'mem0 is not installed. Install it with: pip install mem0ai\n'
                'Or install pydantic-ai with mem0 support: pip install pydantic-ai[mem0]'
            ) from _import_error

        if client is not None:
            self.client = client
            # Check if client is async by looking for async methods
            self._is_async = hasattr(client, 'search') and hasattr(getattr(client, 'search'), '__call__')
            if self._is_async:
                import inspect

                self._is_async = inspect.iscoroutinefunction(client.search)
        else:
            # Create async client by default for better performance
            self.client = AsyncMemoryClient(  # type: ignore[misc]
                api_key=api_key,
                host=host,
                org_id=org_id,
                project_id=project_id,
            )
            self._is_async = True

        self._limit = limit

        # Initialize parent FunctionToolset
        super().__init__(id=id)

        # Register memory tools
        self.tool(self._search_memory_impl)
        self.tool(self._save_memory_impl)

    def _extract_user_id(self, deps: Any) -> str:
        """Extract user_id from deps.

        Args:
            deps: The agent dependencies.

        Returns:
            The user_id as a string.

        Raises:
            ValueError: If user_id cannot be extracted.
        """
        if isinstance(deps, str):
            return deps
        elif hasattr(deps, 'user_id'):
            user_id = deps.user_id
            if isinstance(user_id, str):
                return user_id
            raise ValueError(f'deps.user_id must be a string, got {type(user_id).__name__}')
        elif hasattr(deps, 'get_user_id'):
            user_id = deps.get_user_id()
            if isinstance(user_id, str):
                return user_id
            raise ValueError(f'deps.get_user_id() must return a string, got {type(user_id).__name__}')
        else:
            raise ValueError(
                'Cannot extract user_id from deps. '
                'Deps must be a string, have a user_id attribute, or have a get_user_id() method. '
                f'Got {type(deps).__name__}'
            )

    async def _search_memory_impl(self, ctx: RunContext[AgentDepsT], query: str) -> str:
        """Search through stored memories.

        Args:
            ctx: The run context containing user information.
            query: The search query to find relevant memories.

        Returns:
            A formatted string of relevant memories or a message if none found.
        """
        user_id = self._extract_user_id(ctx.deps)

        try:
            if self._is_async:
                response = await self.client.search(query=query, user_id=user_id, limit=self._limit)
            else:
                response = self.client.search(query=query, user_id=user_id, limit=self._limit)

            # Parse response - handle both dict format and raw list
            if isinstance(response, dict):
                results = response.get('results', [])
            elif isinstance(response, list):
                results = response
            else:
                return 'Error: Unexpected response format from Mem0'

            if not results:
                return 'No relevant memories found.'

            # Format memories for the agent
            memory_lines = ['Found relevant memories:']
            for mem in results:
                if isinstance(mem, dict):
                    memory_text = mem.get('memory', '')
                    score = mem.get('score', 0)
                    memory_lines.append(f'- {memory_text} (relevance: {score:.2f})')

            return '\n'.join(memory_lines)

        except Exception as e:
            return f'Error searching memories: {str(e)}'

    async def _save_memory_impl(self, ctx: RunContext[AgentDepsT], content: str) -> str:
        """Save information to memory.

        Args:
            ctx: The run context containing user information.
            content: The content to save as a memory.

        Returns:
            A confirmation message.
        """
        user_id = self._extract_user_id(ctx.deps)

        try:
            messages = [{'role': 'user', 'content': content}]

            if self._is_async:
                await self.client.add(messages=messages, user_id=user_id)
            else:
                self.client.add(messages=messages, user_id=user_id)

            return f'Successfully saved to memory: {content}'

        except Exception as e:
            return f'Error saving to memory: {str(e)}'
