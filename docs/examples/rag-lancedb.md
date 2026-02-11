# RAG with LanceDB

A simpler RAG example using LanceDB — no Docker or external database required.

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- [embeddings](../embeddings.md) with pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder]
- [chunking strategies](../embeddings.md#chunking-strategies) with [chonkie](https://github.com/chonkie-inc/chonkie)
- RAG search with [LanceDB](https://lancedb.github.io/lancedb/) (embeddable vector database)

This example is the same as the [pgvector RAG example](./rag.md), but uses LanceDB instead of PostgreSQL — no Docker, no configuration, just `pip install lancedb`.

## Setup

With [dependencies installed and environment variables set](./setup.md#usage), build the search database:

!!! warning "API Costs"
    This requires the `OPENAI_API_KEY` env variable and will call the OpenAI embedding API to generate embeddings for each chunk of documentation.

```bash
python/uv-run -m pydantic_ai_examples.rag_lancedb build
```

## Querying

Ask the agent a question:

```bash
python/uv-run -m pydantic_ai_examples.rag_lancedb search "How do I configure logfire to work with FastAPI?"
```

## LanceDB vs pgvector

| Feature | LanceDB | pgvector |
|---------|---------|----------|
| Setup complexity | None (local file) | Requires Docker |
| Persistence | Local directory | PostgreSQL database |
| Scalability | Good for small-medium | Better for large scale |
| SQL support | Limited | Full PostgreSQL |
| Best for | Prototyping, small apps | Production, complex queries |

## Example Code

```snippet {path="/examples/pydantic_ai_examples/rag_lancedb.py"}```
