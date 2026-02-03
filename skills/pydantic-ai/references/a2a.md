# Agent-to-Agent (A2A) Protocol Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/`, FastA2A library

## Overview

The [Agent2Agent (A2A) Protocol](https://google.github.io/A2A/) is an open standard for inter-agent communication. PydanticAI includes FastA2A for exposing agents as HTTP servers.

## Quick Start

```python {title="agent_to_a2a.py" test="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5', instructions='Be helpful!')
app = agent.to_a2a()

# Run with: uvicorn agent_to_a2a:app --host 0.0.0.0 --port 8000
```

The `app` is an ASGI application compatible with any ASGI server.

## Installation

```bash
pip install 'pydantic-ai-slim[a2a]'
# or
uv add 'pydantic-ai-slim[a2a]'
```

## FastA2A Architecture

FastA2A is framework-agnostic and requires three components:

```
HTTP Server ←→ TaskManager ←→ Storage
                    ↓
                 Broker → Worker
```

| Component | Purpose |
|-----------|---------|
| `Storage` | Persist tasks and conversation context |
| `Broker` | Schedule and queue tasks |
| `Worker` | Execute tasks (your agent logic) |

## Task and Context Concepts

- **Task**: One complete agent execution (request → response)
- **Context**: Conversation thread spanning multiple tasks
  - `context_id` maintains conversation continuity
  - New messages without `context_id` create new conversations
  - Same `context_id` continues existing conversation

## Storage Architecture

Storage serves two purposes:

1. **Task Storage**: A2A-formatted tasks (status, artifacts, messages)
2. **Context Storage**: Agent-specific state (tool calls, reasoning traces)

This separation allows rich internal state while maintaining A2A compliance.

## Automatic Behaviors with to_a2a()

When using `agent.to_a2a()`, PydanticAI automatically:

- Stores complete conversation history (including tool calls)
- Ensures `context_id` continuity across requests
- Persists results as A2A artifacts:
  - String → `TextPart` artifact + message history
  - Structured data → `DataPart` artifact as `{"result": <data>}`
  - Includes metadata with type info and JSON schema

## Custom FastA2A Configuration

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')

# to_a2a accepts same args as FastA2A constructor
app = agent.to_a2a(
    # Custom storage, broker, worker can be provided
    # See FastA2A documentation for options
)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `FastA2A` | `fasta2a.FastA2A` | A2A server application |
| `Storage` | `fasta2a.Storage` | Task/context persistence |
| `Broker` | `fasta2a.Broker` | Task scheduling |
| `Worker` | `fasta2a.Worker` | Task execution |

## Example: Calling an A2A Agent

```python
import httpx


async def call_a2a_agent():
    async with httpx.AsyncClient() as client:
        # Send message to A2A server
        response = await client.post(
            'http://localhost:8000/tasks',
            json={
                'message': {'text': 'Hello!'},
                # Optional: include context_id to continue conversation
                # 'context_id': 'existing-context-id',
            },
        )
        result = response.json()
        print(result)
```

## See Also

- [agents.md](agents.md) — Agent configuration
- [streaming.md](streaming.md) — Streaming responses
- [A2A Protocol Spec](https://google.github.io/A2A/) — Official documentation
- [observability.md](observability.md) — Logfire debugging
