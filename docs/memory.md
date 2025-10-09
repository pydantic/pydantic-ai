# Memory

PydanticAI provides a pluggable memory system that allows agents to store and retrieve information across conversations. This enables agents to maintain context, remember user preferences, and build upon previous interactions.

## Overview

The memory system in PydanticAI consists of several key components:

- **Memory Providers**: Backend implementations for storing and retrieving memories (e.g., Mem0, custom databases)
- **Memory Configuration**: Settings that control how memories are stored and retrieved
- **Memory Context**: Runtime context for memory operations within an agent run

## Memory Providers

Memory providers implement the [`MemoryProvider`][pydantic_ai.memory.MemoryProvider] protocol, which defines the interface for storing and retrieving memories.

### Built-in Providers

#### Mem0 Provider

PydanticAI includes a built-in provider for [Mem0](https://mem0.ai), a hosted memory platform:

```python test="skip"
from pydantic_ai import Agent
from pydantic_ai.memory.providers import Mem0Provider


async def main():
    # Create memory provider
    memory = Mem0Provider(api_key='your-mem0-api-key')

    # Create agent
    agent = Agent('openai:gpt-4o')

    # Run agent
    result = await agent.run('My name is Alice')

    # Store memories
    await memory.store_memories(
        messages=result.all_messages(),
        user_id='user_123',
    )

    # Retrieve memories
    memories = await memory.retrieve_memories(
        query='user name',
        user_id='user_123',
    )
    print(f'Found {len(memories)} memories')
```

### Custom Providers

You can implement your own memory provider by creating a class that implements the [`MemoryProvider`][pydantic_ai.memory.MemoryProvider] protocol or extends [`BaseMemoryProvider`][pydantic_ai.memory.BaseMemoryProvider]:

```python test="skip"
from typing import Any

from pydantic_ai.memory import BaseMemoryProvider, RetrievedMemory, StoredMemory
from pydantic_ai.messages import ModelMessage


class CustomMemoryProvider(BaseMemoryProvider):
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
        # Your retrieval logic here
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
        # Your storage logic here
        return []

    async def get_all_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RetrievedMemory]:
        # Your get all logic here
        return []

    async def delete_memory(self, memory_id: str) -> bool:
        # Your deletion logic here
        return True
```

## Memory Configuration

Configure memory behavior using [`MemoryConfig`][pydantic_ai.memory.MemoryConfig]:

```python
from pydantic_ai.memory import MemoryConfig, MemoryScope, RetrievalStrategy

config = MemoryConfig(
    auto_store=True,  # Automatically store conversations
    auto_retrieve=True,  # Automatically retrieve relevant memories
    retrieval_strategy=RetrievalStrategy.SEMANTIC_SEARCH,
    top_k=5,  # Retrieve top 5 most relevant memories
    min_relevance_score=0.7,  # Minimum relevance threshold
    store_after_turns=1,  # Store after each conversation turn
    memory_summary_in_system=True,  # Include memories in system prompt
    scope=MemoryScope.USER,  # Scope memories to user
    metadata={'app_version': '1.0'},  # Custom metadata
)
```

### Retrieval Strategies

The [`RetrievalStrategy`][pydantic_ai.memory.RetrievalStrategy] enum defines how memories are retrieved:

- **`SEMANTIC_SEARCH`**: Use semantic similarity to find relevant memories (default)
- **`RECENCY`**: Retrieve the most recent memories
- **`HYBRID`**: Combine semantic search with recency

### Memory Scope

The [`MemoryScope`][pydantic_ai.memory.MemoryScope] enum defines the scope of memory operations:

- **`USER`**: Memories scoped to a specific user (default)
- **`AGENT`**: Memories scoped to a specific agent
- **`RUN`**: Memories scoped to a specific run/session
- **`GLOBAL`**: Global memories not scoped to any identifier

## Memory Context

The [`MemoryContext`][pydantic_ai.memory.MemoryContext] provides access to memory operations within an agent run:

```python test="skip"
from pydantic_ai.memory import MemoryContext
from pydantic_ai.memory.providers import Mem0Provider


async def main():
    # Create memory provider
    memory_provider = Mem0Provider(api_key='your-api-key')

    # Create memory context
    memory_context = MemoryContext(memory_provider)

    # Search for memories
    memories = await memory_context.search(
        'user preferences',
        user_id='user_123',
    )
    print(f'Found {len(memories)} memories')

    # Add new memories (assuming result is defined elsewhere)
    # stored = await memory_context.add(
    #     messages=result.all_messages(),
    #     user_id='user_123',
    # )

    # Get all memories
    all_memories = await memory_context.get_all(user_id='user_123')
    print(f'Total memories: {len(all_memories)}')

    # Delete a memory
    deleted = await memory_context.delete('mem_id')
    print(f'Deleted: {deleted}')
```

## Memory Data Types

### RetrievedMemory

The [`RetrievedMemory`][pydantic_ai.memory.RetrievedMemory] class represents a memory retrieved from the provider:

```python
from pydantic_ai.memory import RetrievedMemory

memory = RetrievedMemory(
    id='mem_123',
    memory='User prefers concise responses',
    score=0.95,
    metadata={'category': 'preference'},
    created_at='2024-01-01T00:00:00Z',
)
```

### StoredMemory

The [`StoredMemory`][pydantic_ai.memory.StoredMemory] class represents a memory that was stored:

```python
from pydantic_ai.memory import StoredMemory

stored = StoredMemory(
    id='mem_456',
    memory='User is interested in Python',
    event='ADD',
    metadata={'importance': 'high'},
)
```

## Use Cases

### Personalized Conversations

```python test="skip"
from pydantic_ai import Agent
from pydantic_ai.memory.providers import Mem0Provider


async def main():
    memory = Mem0Provider(api_key='your-api-key')
    agent = Agent('openai:gpt-4o')

    # First conversation
    result1 = await agent.run('I love Python programming')
    await memory.store_memories(
        messages=result1.all_messages(),
        user_id='alice',
    )

    # Later conversation - agent can recall preferences
    memories = await memory.retrieve_memories(
        query='programming preferences',
        user_id='alice',
    )
    result2 = await agent.run(
        f'Suggest a project for me. Context: {memories[0].memory}',
    )
    print(result2.output)
```

### Multi-Session Context

```python test="skip"
from pydantic_ai.memory.providers import Mem0Provider


async def session_example(result):
    memory = Mem0Provider(api_key='your-api-key')

    # Store memories with session context
    await memory.store_memories(
        messages=result.all_messages(),
        user_id='alice',
        run_id='session_1',
    )

    # Retrieve session-specific memories
    session_memories = await memory.retrieve_memories(
        query='what did we discuss?',
        user_id='alice',
        run_id='session_1',
    )
    print(f'Found {len(session_memories)} session memories')
```

### Agent-Specific Knowledge

```python test="skip"
from pydantic_ai.memory.providers import Mem0Provider


async def agent_knowledge_example(training_messages):
    memory = Mem0Provider(api_key='your-api-key')

    # Store agent-specific knowledge
    await memory.store_memories(
        messages=training_messages,
        agent_id='support_agent',
        metadata={'category': 'product_knowledge'},
    )

    # Retrieve when running the agent
    knowledge = await memory.retrieve_memories(
        query='product features',
        agent_id='support_agent',
    )
    print(f'Found {len(knowledge)} knowledge items')
```

## Installation

To use the Mem0 provider, install PydanticAI with the `mem0` extra:

```bash
pip install 'pydantic-ai[mem0]'
```

Or install Mem0 separately:

```bash
pip install mem0ai
```

## Best Practices

1. **Scope Appropriately**: Use the right [`MemoryScope`][pydantic_ai.memory.MemoryScope] for your use case
   - User-specific preferences: `MemoryScope.USER`
   - Agent training data: `MemoryScope.AGENT`
   - Session context: `MemoryScope.RUN`

2. **Filter by Relevance**: Set appropriate `min_relevance_score` to avoid retrieving irrelevant memories

3. **Manage Memory Growth**: Use `limit` parameters when retrieving memories to control response size

4. **Add Metadata**: Include meaningful metadata to enable better filtering and organization

5. **Handle Errors Gracefully**: Memory operations should not break agent execution - providers should return empty lists on errors

## API Reference

For detailed API documentation, see:

- [`MemoryProvider`][pydantic_ai.memory.MemoryProvider]
- [`BaseMemoryProvider`][pydantic_ai.memory.BaseMemoryProvider]
- [`MemoryConfig`][pydantic_ai.memory.MemoryConfig]
- [`MemoryContext`][pydantic_ai.memory.MemoryContext]
- [`RetrievedMemory`][pydantic_ai.memory.RetrievedMemory]
- [`StoredMemory`][pydantic_ai.memory.StoredMemory]
- [`RetrievalStrategy`][pydantic_ai.memory.RetrievalStrategy]
- [`MemoryScope`][pydantic_ai.memory.MemoryScope]
