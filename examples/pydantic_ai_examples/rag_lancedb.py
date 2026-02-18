"""RAG example with pydantic-ai and LanceDB — no Docker required.

This example demonstrates a simpler RAG setup using:
- pydantic-ai's Embedder for generating embeddings
- chonkie for chunking documents
- LanceDB as an embeddable vector database (no external service needed)

Build the search DB with:

    uv run -m pydantic_ai_examples.rag_lancedb build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag_lancedb search "How do I configure logfire to work with FastAPI?"
"""

# pyright: reportUnknownMemberType=false
from __future__ import annotations as _annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import lancedb
import logfire
import pyarrow as pa
from chonkie import RecursiveChunker

from pydantic_ai import Agent, Embedder, RunContext

from ._rag_common import DOCS_JSON, sections_ta

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

embedder = Embedder('openai:text-embedding-3-small')
chunker = RecursiveChunker(chunk_size=512)

DB_PATH = Path('.lancedb')
TABLE_NAME = 'doc_sections'


@dataclass
class Deps:
    embedder: Embedder
    table: lancedb.table.Table


agent = Agent('openai:gpt-5.2', deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        result = await context.deps.embedder.embed_query(search_query)

    results = context.deps.table.search(result.embeddings[0]).limit(8).to_pandas()

    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for _, row in results.iterrows()
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    logfire.info('Asking "{question}"', question=question)

    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    deps = Deps(embedder=embedder, table=table)
    answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database.                                    #
#######################################################


async def build_search_db():
    """Build the search database using LanceDB."""
    print('Fetching documentation...')
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    print(f'Processing {len(sections)} sections...')
    records: list[dict[str, str]] = []
    for section in sections:
        content = section.embedding_content()
        for chunk in chunker.chunk(content):
            records.append(
                {
                    'url': section.url(),
                    'title': section.title,
                    'content': chunk.text,
                }
            )

    print(f'Generating embeddings for {len(records)} chunks...')
    chunk_texts = [r['content'] for r in records]
    all_embeddings: list[list[float]] = []
    batch_size = 100
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        with logfire.span(f'embed batch {i // batch_size + 1}'):
            result = await embedder.embed_documents(batch)
            all_embeddings.extend(list(e) for e in result.embeddings)
        print(f'  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)}')

    print('Creating LanceDB table...')
    db = lancedb.connect(DB_PATH)
    schema = pa.schema(
        [
            pa.field('url', pa.string()),
            pa.field('title', pa.string()),
            pa.field('content', pa.string()),
            pa.field('vector', pa.list_(pa.float32(), 1536)),
        ]
    )
    data = [
        {'url': r['url'], 'title': r['title'], 'content': r['content'], 'vector': emb}
        for r, emb in zip(records, all_embeddings, strict=False)
    ]

    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    db.create_table(TABLE_NAME, data, schema=schema)
    print(f'Created table with {len(data)} chunks at {DB_PATH}')


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == 'build':
        asyncio.run(build_search_db())
    elif action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'How do I configure logfire to work with FastAPI?'
        asyncio.run(run_agent(q))
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag_lancedb build|search',
            file=sys.stderr,
        )
        sys.exit(1)
