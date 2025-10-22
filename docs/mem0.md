# Mem0 Memory Integration

PydanticAI provides a lightweight integration with [Mem0](https://mem0.ai) through the [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset]. This toolset adds memory capabilities to your agents, allowing them to save and search through conversation memories.

## Overview

The [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset] is a simple toolset that provides two memory tools:

- **`_search_memory_impl`**: Search through stored memories
- **`_save_memory_impl`**: Save information to memory

This integration follows the same pattern as other third-party integrations like [LangChain tools](third-party-tools.md#langchain-tools).

## Installation

To use the Mem0Toolset, install PydanticAI with the `mem0` extra:

```bash
pip install 'pydantic-ai[mem0]'
```

Or install Mem0 separately:

```bash
pip install mem0ai
```

## Quick Start

Here's a simple example of using the Mem0Toolset:

```python test="skip"
import os

from pydantic_ai import Agent, Mem0Toolset

# Create Mem0 toolset
mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))

# Create agent with memory capabilities
agent = Agent(
    'openai:gpt-4o',
    toolsets=[mem0_toolset],
    instructions='You are a helpful assistant with memory capabilities.',
)


async def main():
    # Use the agent - it can now save and search memories
    result = await agent.run(
        'My name is Alice and I love Python. Please remember this.',
        deps='user_alice',
    )
    print(result.output)
```

## How It Works

### User Identification

The [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset] requires a user identifier to scope memories. It extracts the `user_id` from the agent's `deps` in three ways:

1. **String deps**: If `deps` is a string, it's used directly as the `user_id`
   ```python test="skip"
   import os

   from pydantic_ai import Agent, Mem0Toolset

   mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
   agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


   async def example():
       await agent.run('Remember this', deps='user_123')
   ```

2. **Object with `user_id` attribute**: If `deps` has a `user_id` attribute
   ```python test="skip"
   import os
   from dataclasses import dataclass

   from pydantic_ai import Agent, Mem0Toolset


   @dataclass
   class UserSession:
       user_id: str
       session_id: str


   mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
   agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


   async def example():
       await agent.run(
           'Remember this', deps=UserSession(user_id='user_123', session_id='session_1')
       )
   ```

3. **Object with `get_user_id()` method**: If `deps` has a `get_user_id()` method
   ```python test="skip"
   import os

   from pydantic_ai import Agent, Mem0Toolset


   class UserContext:
       def get_user_id(self) -> str:
           return 'user_123'


   mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
   agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


   async def example():
       await agent.run('Remember this', deps=UserContext())
   ```

### Memory Tools

The agent automatically uses the memory tools when appropriate:

```python test="skip"
import os

from pydantic_ai import Agent, Mem0Toolset

mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


async def example():
    # Agent can save memories
    await agent.run(
        'I prefer concise responses and dark mode.',
        deps='user_alice',
    )

    # Agent can search memories
    await agent.run(
        'What are my preferences?',
        deps='user_alice',
    )
```

## Configuration

### API Key

Provide your Mem0 API key in one of two ways:

1. **Environment variable** (recommended):
   ```bash
   export MEM0_API_KEY=your-api-key
   ```

2. **Constructor parameter**:
   ```python test="skip"
   from pydantic_ai.toolsets import Mem0Toolset

   mem0_toolset = Mem0Toolset(api_key='your-api-key')
   ```

### Custom Configuration

You can customize the toolset behavior:

```python test="skip"
import os

from pydantic_ai.toolsets import Mem0Toolset

mem0_toolset = Mem0Toolset(
    api_key=os.getenv('MEM0_API_KEY'),
    limit=10,  # Return top 10 memories (default: 5)
    host='https://api.mem0.ai',  # Custom host
    org_id='your-org-id',  # Organization ID
    project_id='your-project-id',  # Project ID
    id='my-mem0-toolset',  # Custom toolset ID
)
```

### Using a Custom Client

You can also provide a pre-configured Mem0 client:

```python test="skip"
import os

from mem0 import AsyncMemoryClient

from pydantic_ai.toolsets import Mem0Toolset

client = AsyncMemoryClient(api_key=os.getenv('MEM0_API_KEY'))
mem0_toolset = Mem0Toolset(client=client)
```

## Examples

### Basic Memory Usage

```python test="skip"
import os

from pydantic_ai import Agent, Mem0Toolset

mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


async def main():
    # Save information
    await agent.run(
        'My name is Alice and I love Python.',
        deps='user_alice',
    )

    # Recall information
    await agent.run(
        'What programming language do I like?',
        deps='user_alice',
    )
```

### Multi-User Isolation

Memories are automatically isolated per user:

```python test="skip"
import os

from pydantic_ai import Agent, Mem0Toolset

mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


async def main():
    # Alice's memories
    await agent.run('My favorite color is blue.', deps='user_alice')

    # Bob's memories
    await agent.run('My favorite color is red.', deps='user_bob')

    # Each user gets their own memories back
    await agent.run('What is my favorite color?', deps='user_alice')
    # Output: "Your favorite color is blue."

    await agent.run('What is my favorite color?', deps='user_bob')
    # Output: "Your favorite color is red."
```

### Using with Dataclass Deps

```python test="skip"
import os
from dataclasses import dataclass

from pydantic_ai import Agent, Mem0Toolset


@dataclass
class UserSession:
    user_id: str
    session_id: str


mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])


async def main():
    session = UserSession(user_id='user_charlie', session_id='session_123')

    await agent.run(
        'I work as a data scientist.',
        deps=session,
    )
```

### Personalized Assistant

```python test="skip"
import os

from pydantic_ai import Agent, Mem0Toolset

mem0_toolset = Mem0Toolset(api_key=os.getenv('MEM0_API_KEY'))
agent = Agent(
    'openai:gpt-4o',
    toolsets=[mem0_toolset],
    instructions=(
        'You are a helpful assistant with memory. '
        'Remember user preferences and provide personalized assistance.'
    ),
)


async def main():
    # First conversation - learn preferences
    await agent.run(
        'I prefer concise responses with Python code examples.',
        deps='user_diana',
    )

    # Later conversation - agent recalls preferences
    result = await agent.run(
        'How do I read a CSV file?',
        deps='user_diana',
    )
    print(result.output)
    # Agent will provide concise response with Python examples
```

## Comparison with Other Approaches

The [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset] is designed to be lightweight and follow the same pattern as other third-party tool integrations like LangChain tools.

### Similar to LangChain Tools

Just like you can use [LangChain tools](third-party-tools.md#langchain-tools) with PydanticAI:

```python test="skip"
from langchain_community.tools import WikipediaQueryRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

toolset = LangChainToolset([WikipediaQueryRun()])
agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

You can use Mem0 tools the same way:

```python test="skip"
from pydantic_ai import Agent, Mem0Toolset

toolset = Mem0Toolset(api_key='your-api-key')
agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

## API Reference

For detailed API documentation, see [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset].

## Complete Example

See the [complete example](examples/mem0-toolset.md) for a full demonstration of the Mem0Toolset capabilities.
