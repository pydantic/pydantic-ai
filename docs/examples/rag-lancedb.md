# RAG with LanceDB

A simpler RAG example using LanceDB — no Docker or external database required.

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- [embeddings](../embeddings.md) with pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder]
- [chunking strategies](../embeddings.md#chunking-strategies) with [chonkie](https://github.com/chonkie-inc/chonkie)
- RAG search with [LanceDB](https://lancedb.github.io/lancedb/) (embeddable vector database)

This example is the same as the [pgvector RAG example](./rag.md), but uses LanceDB instead of PostgreSQL. LanceDB is an embeddable vector database that stores data locally — perfect for getting started quickly.

## Why LanceDB?

- **No Docker required** — data is stored in a local directory
- **Zero configuration** — just `pip install lancedb`
- **Fast** — optimized for vector similarity search
- **Serverless** — runs in your process, no external service needed

## Setup

No external services needed! LanceDB stores data in a `.lancedb` directory.

With [dependencies installed and environment variables set](./setup.md#usage), build the search database:

!!! warning "API Costs"
    This requires the `OPENAI_API_KEY` env variable and will call the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation.

```bash
python/uv-run -m pydantic_ai_examples.rag_lancedb build
```

The build process:

1. Fetches documentation sections from a JSON file
2. Chunks each section using [`RecursiveChunker`](https://github.com/chonkie-inc/chonkie) (512 tokens, 50 token overlap)
3. Generates embeddings using pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder] with OpenAI's `text-embedding-3-small`
4. Stores everything in a local LanceDB table

## Querying

Ask the agent a question:

```bash
python/uv-run -m pydantic_ai_examples.rag_lancedb search "How do I configure logfire to work with FastAPI?"
```

The agent uses a retrieval tool that:

1. Embeds the search query
2. Finds similar chunks using LanceDB's vector search
3. Returns the most relevant documentation sections as context

## Comparison with pgvector

| Feature | LanceDB | pgvector |
|---------|---------|----------|
| Setup complexity | None (local file) | Requires Docker |
| Persistence | Local directory | PostgreSQL database |
| Scalability | Good for small-medium | Better for large scale |
| SQL support | Limited | Full PostgreSQL |
| Best for | Prototyping, small apps | Production, complex queries |

Use LanceDB for quick prototyping and small applications. Consider pgvector for production systems that need SQL queries, joins, or scale to millions of documents.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/rag_lancedb.py"}```

## Next Steps

- See the [pgvector RAG example](./rag.md) for a production-ready setup
- Learn more about [chunking strategies](../embeddings.md#chunking-strategies)
- Explore [embeddings providers](../embeddings.md#providers)
