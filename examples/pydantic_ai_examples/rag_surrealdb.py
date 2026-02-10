"""RAG example with pydantic-ai — using vector search to augment a chat agent.

Uses SurrealDB with HNSW vector indexes for persistent storage.

Set up your OpenAI API key:

    export OPENAI_API_KEY=your-api-key

Or, store it in a .env file, and add `--env-file .env` to your `uv run` commands.

Build the search DB with:

    uv run -m pydantic_ai_examples.rag_surrealdb build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag_surrealdb search "How do I configure logfire to work with FastAPI?"

Or use the web UI:

    uv run -m pydantic_ai_examples.rag_surrealdb web

This example runs SurrealDB embedded. If you want to run it in a separate process (useful to explore the db using Surrealist) you can start it with (or with docker):

    surreal start -u root -p root rocksdb:database
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import httpx
import logfire
import uvicorn
from pydantic import BaseModel, TypeAdapter
from surrealdb import (
    AsyncEmbeddedSurrealConnection,
    AsyncHttpSurrealConnection,
    AsyncSurreal,
    AsyncWsSurrealConnection,
    RecordID,
    Value,
)
from typing_extensions import AsyncGenerator

from pydantic_ai import Agent, Embedder

SurrealConn = (
    AsyncWsSurrealConnection
    | AsyncHttpSurrealConnection
    | AsyncEmbeddedSurrealConnection
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
# TODO: add this once logfire is updated to at least 4.19.0
# logfire.instrument_surrealdb()

THIS_DIR = Path(__file__).parent

SURREALDB_NS = 'pydantic_ai_examples'
SURREALDB_DB = 'rag_surrealdb'
SURREALDB_USER = 'root'
SURREALDB_PASS = 'root'

embedder = Embedder('openai:text-embedding-3-small')
agent = Agent('openai:gpt-5.2')

RecordType = TypeVar('RecordType')


class RetrievalQueryResult(BaseModel):
    url: str
    title: str
    content: str
    dist: float


async def query(
    conn: SurrealConn,
    query_: str,
    vars_: dict[str, Value],
    record_type: type[RecordType],
) -> list[RecordType]:
    result = await conn.query(query_, vars_)
    result_ta = TypeAdapter(list[record_type])
    rows = result_ta.validate_python(result)
    return rows


@agent.tool_plain
async def retrieve(search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        search_query: The search query.
    """
    with logfire.span(
        'create embedding for {search_query=}', search_query=search_query
    ):
        result = await embedder.embed_query(search_query)
        embedding = result.embeddings

    # Embedder method guarantees there's one item here
    embedding_vector = embedding[0]

    # SurrealDB vector search using HNSW index
    async with database_connect(False) as db:
        rows = await query(
            db,
            """
            SELECT url, title, content, vector::distance::knn() AS dist
            FROM doc_sections
            WHERE embedding <|8, 40|> $vector
            ORDER BY dist ASC
            """,
            {'vector': list(embedding_vector)},
            RetrievalQueryResult,
        )

    return '\n\n'.join(
        f'# {row.title}\nDocumentation URL:{row.url}\n\n{row.content}' for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    logfire.info('Asking "{question}"', question=question)
    answer = await agent.run(question)
    print(answer.output)


# Web chat UI
app = agent.to_web()

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


def build_doc_rec_id(url: str) -> RecordID:
    return RecordID('doc_sections', slugify(url, '_'))


async def build_search_db():
    """Build the search database."""
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    async with database_connect(True) as db:
        missing_sections: list[DocsSection] = []
        for section in sections:
            url = section.url()
            record_id = build_doc_rec_id(url)
            existing = await db.select(record_id)
            if existing:
                logfire.info('Skipping {url=}', url=url)
                continue
            missing_sections.append(section)

        if missing_sections:
            with logfire.span('create embeddings'):
                result = await embedder.embed_documents(
                    [section.embedding_content() for section in missing_sections]
                )
                embeddings = result.embeddings

            for section, embedding_vector in zip(
                missing_sections, embeddings, strict=True
            ):
                await insert_doc_section(db, section, embedding_vector)
        else:
            logfire.info('All documents already exist; skipping embedding generation')


async def insert_doc_section(
    db: SurrealConn,
    section: DocsSection,
    embedding_vector: Sequence[float],
) -> None:
    url = section.url()
    record_id = build_doc_rec_id(url)

    # Create record with embedding, using record ID directly
    res = await db.create(
        record_id,
        {
            'url': url,
            'title': section.title,
            'content': section.content,
            'embedding': list(embedding_vector),
        },
    )
    if not isinstance(res, dict):
        raise ValueError(f'Unexpected response from database: {res}')


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


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[SurrealConn, None]:
    # Running SurrealDB embedded
    db_path = THIS_DIR / f'.{SURREALDB_DB}'
    db_url = f'file://{db_path}'
    requires_auth = False

    # Running SurrealDB in a separate process, connect with URL
    # db_url = 'ws://localhost:8000/rpc'
    # requires_auth = True

    async with AsyncSurreal(db_url) as db:
        # Sign in to the database
        if requires_auth:
            await db.signin({'username': SURREALDB_USER, 'password': SURREALDB_PASS})

        # Set namespace and database
        await db.use(SURREALDB_NS, SURREALDB_DB)

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
    HNSW DIMENSION 1536
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
    elif action == 'web':
        uvicorn.run(app, host='127.0.0.1', port=7932)
    else:
        print(
            'uv run --extra examples -m pydantic_ai_examples.rag_surrealdb build|search',
            file=sys.stderr,
        )
        sys.exit(1)
