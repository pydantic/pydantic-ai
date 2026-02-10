# RAG

RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- [embeddings](../embeddings.md) with pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder]
- [chunking strategies](../embeddings.md#chunking-strategies) with [chonkie](https://github.com/chonkie-inc/chonkie)
- RAG search with [pgvector](https://github.com/pgvector/pgvector)

This is done by creating a database containing each section of the markdown documentation, then registering
the search tool with the Pydantic AI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in
[this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

## Setup

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database. The easiest way to download and run pgvector is using Docker:

```bash
mkdir postgres-data
docker run --rm \
  -e POSTGRES_PASSWORD=postgres \
  -p 54320:5432 \
  -v `pwd`/postgres-data:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

As with the [SQL gen](./sql-gen.md) example, we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running.
We also mount the PostgreSQL `data` directory locally to persist the data if you need to stop and restart the container.

## Building the Database

With pgvector running and [dependencies installed and environment variables set](./setup.md#usage), build the search database:

!!! warning "API Costs"
    This requires the `OPENAI_API_KEY` env variable and will call the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation.

```bash
python/uv-run -m pydantic_ai_examples.rag build
```

The build process:

1. Fetches documentation sections from a JSON file
2. Chunks each section using [`RecursiveChunker`](https://github.com/chonkie-inc/chonkie) (512 tokens)
3. Generates embeddings using pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder] with OpenAI's `text-embedding-3-small`
4. Stores chunks and embeddings in pgvector

## Querying

You can then ask the agent a question:

```bash
python/uv-run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

The agent uses a retrieval tool that:

1. Embeds the search query
2. Finds similar chunks using pgvector's vector similarity search
3. Returns the most relevant documentation sections as context

## Example Code

```snippet {path="/examples/pydantic_ai_examples/rag.py"}```

## Next Steps

- See the [LanceDB RAG example](./rag-lancedb.md) for a simpler setup without Docker
- Learn more about [chunking strategies](../embeddings.md#chunking-strategies)
- Explore [embeddings providers](../embeddings.md#providers)
