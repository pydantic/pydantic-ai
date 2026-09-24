---
title: Answer from private documents with citations
description: Retrieve relevant private text, expose it through a typed tool, and require source identifiers in the answer.
---

# Answer from private documents with citations

Keep retrieval in application code and return source IDs with every chunk so the final answer can be checked against the retrieved evidence.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_ai import Agent, Embedder, RunContext


class Answer(BaseModel):
    text: str
    sources: list[str]


@dataclass
class Document:
    source_id: str
    text: str
    embedding: list[float]


@dataclass
class SearchIndex:
    embedder: Embedder
    documents: list[Document]
    retrieved_source_ids: set[str]


agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=SearchIndex,
    output_type=Answer,
    instructions='Answer only from retrieved passages and include their source IDs.',
)


@agent.tool
async def search_documents(ctx: RunContext[SearchIndex], query: str) -> list[dict[str, str]]:
    """Return the private passages most relevant to the question."""
    query_vector = (await ctx.deps.embedder.embed_query(query)).embeddings[0]

    def score(document: Document) -> float:
        return sum(a * b for a, b in zip(query_vector, document.embedding, strict=True))

    matches = sorted(ctx.deps.documents, key=score, reverse=True)[:3]
    ctx.deps.retrieved_source_ids = {item.source_id for item in matches}
    return [{'source_id': item.source_id, 'text': item.text} for item in matches]


async def main() -> None:
    embedder = Embedder('openai:text-embedding-3-small')
    passages = {
        'runbook.md#rollback': 'Rollback starts by shifting traffic to the previous release.',
        'runbook.md#database': 'Database migrations require a compatibility check before rollback.',
    }
    vectors = (await embedder.embed_documents(list(passages.values()))).embeddings
    index = SearchIndex(
        embedder=embedder,
        documents=[
            Document(source_id=source_id, text=text, embedding=vector)
            for (source_id, text), vector in zip(passages.items(), vectors, strict=True)
        ],
        retrieved_source_ids=set(),
    )

    result = await agent.run('What is the first rollback step?', deps=index)
    if not result.output.sources or not set(result.output.sources) <= index.retrieved_source_ids:
        raise ValueError('The answer cited a source that was not retrieved.')
    print(result.output.text)
    #> Shift traffic to the previous release.
    print(result.output.sources)
    #> ['runbook.md#rollback']


if __name__ == '__main__':
    asyncio.run(main())
```

Persist document vectors in a real vector store for larger collections, but keep the same boundary: retrieval returns inspectable evidence, while typed output makes citations available for validation and evaluation. Checking that a citation was retrieved does not prove that it supports the claim; add a citation-correctness evaluator when that distinction matters.

## Related

See [Embeddings](../embeddings.md) for model and vector-store options, and the [RAG example](../examples/rag.md) for a complete application backed by PostgreSQL.
