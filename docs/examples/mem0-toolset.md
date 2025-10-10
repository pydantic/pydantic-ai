# Mem0 Toolset Memory Integration

Example demonstrating how to use the [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset] to add memory capabilities to your Pydantic AI agents.

Demonstrates:

* [Using third-party toolsets](../mem0.md)
* [Memory save and search operations](../mem0.md#memory-tools)
* [Multi-user memory isolation](../mem0.md#multi-user-isolation)
* [Different deps patterns for user identification](../mem0.md#user-identification)

This example shows how to integrate [Mem0](https://mem0.ai) memory capabilities into your agents using a simple toolset approach, allowing agents to remember and recall information across conversations.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
pip install pydantic-ai mem0ai
export MEM0_API_KEY=your-mem0-api-key
export OPENAI_API_KEY=your-openai-api-key
python/uv-run -m pydantic_ai_examples.mem0_toolset
```

## Example Code

The example demonstrates several use cases:

1. **Basic Memory Usage**: Saving and searching memories
2. **Multi-User Isolation**: Separate memories for different users
3. **Dataclass Deps**: Using structured dependency objects
4. **Personalized Assistant**: Remembering user preferences

```snippet {path="/examples/pydantic_ai_examples/mem0_toolset.py"}```

## Key Features

### Simple Integration

The [`Mem0Toolset`][pydantic_ai.toolsets.Mem0Toolset] provides an easy way to add memory to any agent:

```python
from pydantic_ai import Agent, Mem0Toolset

mem0_toolset = Mem0Toolset(api_key='your-api-key')
agent = Agent('openai:gpt-4o', toolsets=[mem0_toolset])
```

### Automatic Memory Management

The agent automatically decides when to save or search memories based on the conversation context. Users don't need to explicitly call memory functions.

### User Isolation

Memories are automatically scoped to individual users through the `deps` parameter, ensuring privacy and personalization:

```python
# Alice's memories
await agent.run('My favorite color is blue', deps='user_alice')

# Bob's memories (completely separate)
await agent.run('My favorite color is red', deps='user_bob')
```

## Learn More

For detailed documentation on the Mem0 integration, see the [Mem0 Memory Integration](../mem0.md) guide.
