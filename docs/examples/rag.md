# RAG

RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- RAG search

This is done by creating a database containing each section of the markdown documentation, then registering
the search tool with the Pydantic AI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in
[this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database, the easiest way to download and run pgvector is using Docker:

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

With that running and [dependencies installed and environment variables set](./index.md#usage), we can build the search database with (**WARNING**: this requires the `OPENAI_API_KEY` env variable and will calling the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation):

```bash
python/uv-run -m pydantic_ai_examples.rag build
```

(Note building the database doesn't use Pydantic AI right now, instead it uses the OpenAI SDK directly.)

You can then ask the agent a question with:

```bash
python/uv-run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/rag.py"}```
