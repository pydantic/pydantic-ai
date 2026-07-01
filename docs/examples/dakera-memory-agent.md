# Dakera Memory Agent

Persistent cross-session memory agent using [Dakera](https://dakera.ai) — a self-hosted, decay-weighted vector memory server.

Demonstrates:

- [tool definitions](../tools.md)
- [agent dependencies](../dependencies.md)
- persistent memory with decay-weighted recall
- cross-session context retention

## What Dakera Provides

Dakera is a self-hosted REST server that stores agent memories as embeddings in a persistent vector index. Unlike in-process memory, Dakera memories survive process restarts, container rebuilds, and horizontal scaling.

**Decay-weighted scoring**: Each memory has an importance score that decays exponentially over time without access. Memories recalled frequently stay ranked higher; stale context fades naturally — matching how human memory works.

**Three REST endpoints cover all operations:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/memories` | Store text as an embedding |
| `POST` | `/v1/memories/search` | Semantic recall with decay scoring |
| `DELETE` | `/v1/memories` | Forget memories by filter |

## Setup

Start the Dakera server with Docker:

```bash
docker run -d \
  --name dakera \
  -p 3300:3300 \
  -e DAKERA_API_KEY=demo \
  -v dakera_data:/data \
  ghcr.io/dakera-ai/dakera:latest
```

Verify it's running:

```bash
curl http://localhost:3300/health
```

## Running the Example

Store a fact (directly, without the agent):

```bash
DAKERA_API_KEY=demo uv run -m pydantic_ai_examples.dakera_memory_agent \
  store "My name is Alice and I prefer concise technical answers."
```

Recall semantically similar memories:

```bash
DAKERA_API_KEY=demo uv run -m pydantic_ai_examples.dakera_memory_agent \
  recall "What do you know about my communication preferences?"
```

Chat with the agent — it automatically recalls relevant context and stores new facts:

```bash
OPENAI_API_KEY=your-key DAKERA_API_KEY=demo \
  uv run -m pydantic_ai_examples.dakera_memory_agent \
  chat "What are my preferences?"

# New process, same memories:
OPENAI_API_KEY=your-key DAKERA_API_KEY=demo \
  uv run -m pydantic_ai_examples.dakera_memory_agent \
  chat "Do you remember who I am?"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAKERA_BASE_URL` | `http://localhost:3300` | Dakera server URL |
| `DAKERA_API_KEY` | `demo` | Dakera API key |
| `DAKERA_NAMESPACE` | `pydantic-ai-agent` | Namespace to isolate memories |
| `PYDANTIC_AI_MODEL` | `openai:gpt-4o` | Model to use for the agent |

## Example Code

```snippet {path="/examples/pydantic_ai_examples/dakera_memory_agent.py"}```
