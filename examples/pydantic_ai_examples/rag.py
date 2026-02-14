"""RAG example with pydantic-ai — using vector search to augment a chat agent.

This example demonstrates:
- Using pydantic-ai's Embedder for generating embeddings
- Chunking documents with chonkie before embedding
- Storing embeddings in pgvector for similarity search
- Building a RAG agent that retrieves relevant context

Run pgvector with:

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

Build the search DB with:

    uv run -m pydantic_ai_examples.rag build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import pydantic_core
from anyio import create_task_group
from chonkie import RecursiveChunker
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, Embedder, RunContext

from ._rag_common import DOCS_JSON, DocsSection, sections_ta

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()

embedder = Embedder('openai:text-embedding-3-small')
chunker = RecursiveChunker(chunk_size=512)


@dataclass
class Deps:
    embedder: Embedder
    pool: asyncpg.Pool


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

    embedding_json = pydantic_core.to_json(result.embeddings[0]).decode()
    rows = await context.deps.pool.fetch(
        'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
        embedding_json,
    )
    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(embedder=embedder, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database.                                    #
#######################################################


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    async with database_connect(True) as pool:
        with logfire.span('create schema'):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with create_task_group() as tg:
            for section in sections:
                tg.start_soon(insert_doc_section, sem, pool, section)


async def insert_doc_section(
    sem: asyncio.Semaphore,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
        if exists:
            logfire.info('Skipping {url=}', url=url)
            return

        content = section.embedding_content()
        chunks = chunker.chunk(content)
        chunk_texts = [c.text for c in chunks] if chunks else [content]

        with logfire.span('create embedding for {url=}', url=url):
            result = await embedder.embed_documents(chunk_texts)

        for chunk_text, embedding in zip(chunk_texts, result.embeddings, strict=False):
            embedding_json = pydantic_core.to_json(embedding).decode()
            await pool.execute(
                'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
                url,
                section.title,
                chunk_text,
                embedding_json,
            )


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        'postgresql://postgres:postgres@localhost:54320',
        'pydantic_ai_rag',
    )
    if create_db:
        with logfire.span('check and create DB'):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    'SELECT 1 FROM pg_database WHERE datname = $1', database
                )
                if not db_exists:
                    await conn.execute(f'CREATE DATABASE {database}')
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL,
    title text NOT NULL,
    content text NOT NULL,
    -- text-embedding-3-small returns a vector of 1536 floats
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


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
            'uv run --extra examples -m pydantic_ai_examples.rag build|search',
            file=sys.stderr,
        )
        sys.exit(1)
