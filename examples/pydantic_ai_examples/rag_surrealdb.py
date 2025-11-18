"""RAG example with pydantic-ai — using vector search to augment a chat agent.

Uses SurrealDB with HNSW vector indexes for persistent storage.

Start SurrealDB locally with:

    surreal start -u root -p root rocksdb:database

Build the search DB with:

    uv run -m pydantic_ai_examples.rag_surrealdb build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag_surrealdb search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import logfire
import ollama
from anyio import create_task_group
from pydantic import TypeAdapter
from surrealdb import (
    AsyncHttpSurrealConnection,
    AsyncSurreal,
    AsyncWsSurrealConnection,
    RecordID,
)
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Deps:
    # openai: AsyncOpenAI
    db: AsyncWsSurrealConnection | AsyncHttpSurrealConnection


# agent = Agent('openai:gpt-5', deps_type=Deps)
agent = Agent('ollama:llama3.2', deps_type=Deps)


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
        embedding = ollama.embed(
            input=search_query,
            # model='text-embedding-3-small',
            model='all-minilm:22m',
            truncate=True,
        )

    assert len(embedding.embeddings) == 1, (
        f'Expected 1 embedding, got {len(embedding.embeddings)}, doc query: {search_query!r}'
    )
    embedding_vector = list(embedding.embeddings[0])

    # SurrealDB vector search using HNSW index
    result = await context.deps.db.query(
        """
        SELECT url, title, content, vector::distance::knn() AS dist
        FROM doc_sections
        WHERE embedding <|8, 40|> $vector
        ORDER BY dist ASC
        LIMIT 8;
        """,
        {'vector': embedding_vector},
    )

    # Process SurrealDB query result
    rows = []
    if isinstance(result, list):
        for record in result:
            if isinstance(record, dict) and 'url' in record:
                rows.append(record)

    return '\n\n'.join(
        f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    # openai = AsyncOpenAI()
    # logfire.instrument_openai(openai)

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as db:
        deps = Deps(db=db)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    # openai = AsyncOpenAI()
    # logfire.instrument_openai(openai)

    async with database_connect(True) as db:
        with logfire.span('create schema'):
            await db.query(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with create_task_group() as tg:
            for section in sections:
                tg.start_soon(insert_doc_section, sem, db, section)


async def insert_doc_section(
    sem: asyncio.Semaphore,
    # openai: AsyncOpenAI,
    db: AsyncWsSurrealConnection | AsyncHttpSurrealConnection,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        # Create a URL-safe record ID
        url_slug = slugify(url, '_')
        # record_id = f'doc_sections:{url_slug}'
        record_id = RecordID('doc_sections', url_slug)

        # Check if record exists
        existing = await db.select(record_id)
        if existing:
            logfire.info('Skipping {url=}', url=url)
            return

        with logfire.span('create embedding for {url=}', url=url):
            embedding = ollama.embed(
                input=section.embedding_content(),
                # model='text-embedding-3-small',
                model='all-minilm:22m',
            )
        assert len(embedding.embeddings) == 1, (
            f'Expected 1 embedding, got {len(embedding.embeddings)}, doc section: {section}'
        )
        embedding_vector = embedding.embeddings[0]

        # Create record with embedding as array, using record ID directly
        _res = await db.create(
            record_id,
            {
                'url': url,
                'title': section.title,
                'content': section.content,
                'embedding': list(embedding_vector),
            },
        )


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r'\.md$', '', self.path)
        return (
            f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'
        )

    def embedding_content(self) -> str:
        return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))


sections_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[
    Any, None
]:  # Returns AsyncWsSurrealConnection | AsyncHttpSurrealConnection
    """Connect to SurrealDB local instance.

    Connects to a local SurrealDB server running on localhost:8000.
    Make sure to start SurrealDB with: surreal start -u root -p root rocksdb:database
    """
    db_url = 'ws://localhost:8000/rpc'
    namespace = 'rag'
    database = 'docs'
    username = 'root'
    password = 'root'

    async with AsyncSurreal(db_url) as db:
        # Sign in to the database
        await db.signin({'username': username, 'password': password})

        # Set namespace and database
        await db.use(namespace, database)

        # Initialize schema if creating database
        if create_db:
            with logfire.span('create schema'):
                await db.query(DB_SCHEMA)

        yield db


DB_SCHEMA = """
DEFINE TABLE doc_sections SCHEMALESS;

DEFINE FIELD embedding ON doc_sections TYPE array<float>;

DEFINE INDEX hnsw_idx_doc_sections ON doc_sections
    FIELDS embedding
    HNSW DIMENSION 384
    DIST COSINE
    TYPE F32;
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(rf'[{separator}\s]+', separator, value)


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
            'uv run --extra examples -m pydantic_ai_examples.rag_surrealdb build|search',
            file=sys.stderr,
        )
        sys.exit(1)
