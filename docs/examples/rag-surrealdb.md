# RAG with SurrealDB

RAG search example using SurrealDB. This demo allows you to ask questions about the [logfire](https://pydantic.dev/logfire) documentation.

Demonstrates:

- [tools](../tools.md)
- [Web Chat UI](../web.md)
- RAG search with SurrealDB

This is done by creating a database containing each section of the markdown documentation, then registering
the search tool with the Pydantic AI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in
[this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

Set up your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key
```

Or store it in a `.env` file and add `--env-file .env` to your `uv run` commands.

Build the search database (**warning**: this calls the OpenAI embedding API for every documentation section from the [Logfire docs JSON gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992)):

```bash
uv run -m pydantic_ai_examples.rag_surrealdb build
```

Ask the agent a question with:

```bash
uv run -m pydantic_ai_examples.rag_surrealdb search "How do I configure logfire to work with FastAPI?"
```

Or use the web UI:

```bash
uv run -m pydantic_ai_examples.rag_surrealdb web
```

This example runs SurrealDB embedded. To run it in a separate process (useful if you want to explore the database with [Surrealist](https://surrealdb.com/surrealist)), follow the [installation instructions](https://surrealdb.com/docs/surrealdb/installation) or [run with docker](https://surrealdb.com/docs/surrealdb/installation/running/docker):

```bash
surreal start -u root -p root rocksdb:database
```

With docker

```bash
docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start -u root -p root rocksdb:database
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/rag_surrealdb.py"}```
