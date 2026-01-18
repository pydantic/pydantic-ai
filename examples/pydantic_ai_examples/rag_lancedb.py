"""RAG example with pydantic-ai and LanceDB — no Docker required.

This example demonstrates a simpler RAG setup using:
- pydantic-ai's Embedder for generating embeddings
- chonkie for chunking documents
- LanceDB as an embeddable vector database (no external service needed)

This is a great starting point for RAG applications — everything runs locally
without Docker or external databases.

Build the search DB with:

    uv run -m pydantic_ai_examples.rag_lancedb build

Ask the agent a question with:

    uv run -m pydantic_ai_examples.rag_lancedb search "How do I configure logfire to work with FastAPI?"
"""

from __future__ import annotations as _annotations

import asyncio
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import lancedb
import logfire
import pyarrow as pa
from chonkie import RecursiveChunker
from pydantic import TypeAdapter

from pydantic_ai import Agent, Embedder, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# Create an embedder using pydantic-ai's unified interface
embedder = Embedder('openai:text-embedding-3-small')

# LanceDB stores data locally — no external service required
DB_PATH = Path('.lancedb')
TABLE_NAME = 'doc_sections'


@dataclass
class Deps:
    embedder: Embedder
    table: lancedb.table.Table


agent = Agent('openai:gpt-4o', deps_type=Deps)


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

    # Search using LanceDB's vector search
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
# search database, and some utilities.                #
#######################################################

# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
    'https://gist.githubusercontent.com/'
    'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
    '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

# Create a chunker for splitting large content
chunker = RecursiveChunker(chunk_size=512)


async def build_search_db():
    """Build the search database using LanceDB."""
    print('Fetching documentation...')
    async with httpx.AsyncClient() as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sections_ta.validate_json(response.content)

    print(f'Processing {len(sections)} sections...')

    # Prepare data for embedding
    records = []
    for section in sections:
        url = section.url()
        content = section.embedding_content()

        # Chunk the content if it's large
        chunks = chunker.chunk(content)
        chunk_text = chunks[0].text if chunks else content

        records.append(
            {
                'url': url,
                'title': section.title,
                'content': section.content,
                'chunk_text': chunk_text,
            }
        )

    print('Generating embeddings...')
    # Embed all chunks in batches
    chunk_texts = [r['chunk_text'] for r in records]

    # Process in batches of 100 to avoid rate limits
    embeddings = []
    batch_size = 100
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        with logfire.span(f'embed batch {i // batch_size + 1}'):
            result = await embedder.embed_documents(batch)
            embeddings.extend(result.embeddings)
        print(f'  Embedded {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)}')

    # Create LanceDB table
    print('Creating LanceDB table...')
    db = lancedb.connect(DB_PATH)

    # Define schema with vector dimension matching text-embedding-3-small
    schema = pa.schema(
        [
            pa.field('url', pa.string()),
            pa.field('title', pa.string()),
            pa.field('content', pa.string()),
            pa.field('vector', pa.list_(pa.float32(), 1536)),
        ]
    )

    # Prepare data with embeddings
    data = [
        {
            'url': r['url'],
            'title': r['title'],
            'content': r['content'],
            'vector': list(emb),
        }
        for r, emb in zip(records, embeddings, strict=False)
    ]

    # Create or overwrite table
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    db.create_table(TABLE_NAME, data, schema=schema)
    print(f'Created table with {len(data)} records at {DB_PATH}')


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
            'uv run --extra examples -m pydantic_ai_examples.rag_lancedb build|search',
            file=sys.stderr,
        )
        sys.exit(1)
