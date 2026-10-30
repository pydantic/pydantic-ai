"""OpenAI vector store backend: searches and reads a vector store with the OpenAI API.

External assumptions (verified 2026-10-30 against openai 3.31.0 and
<https://platform.openai.com/docs/api-reference/vector-stores/search>):

- `vector_stores.search` takes `max_num_results` from 1 to 50 and `filters` built from comparison
  filters (equality, ranges, `in` and its negation) and compound `and` / `or` filters over file
  attributes. Each result carries `file_id`, `filename`, `score`, `attributes` and text `content`.
- `vector_stores.files.content` pages through a file's parsed text, which `read` uses.
- The Responses API's `file_search` tool searches the same vector stores natively; core sends it as
  `FileSearchTool(file_store_ids=[...])` on `OpenAIResponsesModel`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.native_tools import FileSearchTool
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.knowledge_base._backend import Document, MetadataFilter, Passage, matches

try:
    from openai import APIConnectionError, AsyncOpenAI, InternalServerError, NotFoundError, RateLimitError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'openai is required for OpenAIVectorStore. Install it with: pip install "pydantic-ai-slim[openai]"'
    ) from _import_error

if TYPE_CHECKING:
    from openai.types.vector_store_search_params import Filters

OPENAI_MAX_NUM_RESULTS = 50
"""Largest `max_num_results` the vector store search API accepts."""


@dataclass
class OpenAIVectorStore:
    """Searches and reads an [OpenAI vector store](https://platform.openai.com/docs/guides/retrieval).

    Upload and attribute files with the OpenAI API; this backend only queries the store. Searches
    return at most 50 passages, the API maximum. A file's id is its document id, so the agent can open
    a whole file with `read_document`.

    With `KnowledgeBase(native=True)`, OpenAI Responses models search the store with the provider's
    native `file_search` tool instead, and every other model uses this backend's search.
    """

    vector_store_id: str
    """The vector store to search, such as `'vs_abc123'`."""

    rewrite_query: bool = False
    """Let OpenAI rewrite the query for vector search before searching."""

    score_threshold: float | None = None
    """Drop results scoring below this, from 0 to 1."""

    client: AsyncOpenAI | None = field(default=None, repr=False)
    """The OpenAI client; when `None`, one is built from the `OPENAI_API_KEY` environment variable.

    The default client is rebuilt when the backend is used on a new event loop after the previous one
    stopped, since its connection pool belongs to the loop that opened it. A client you pass is used as is.
    """

    _default: AsyncOpenAI | None = field(default=None, init=False, repr=False, compare=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration against the API's documented bounds."""
        if not self.vector_store_id:
            raise ValueError('vector_store_id must not be empty')
        if self.score_threshold is not None and not 0 <= self.score_threshold <= 1:
            raise ValueError(f'score_threshold must be between 0 and 1, got {self.score_threshold}')

    def _openai(self) -> AsyncOpenAI:
        if self.client is not None:
            return self.client
        loop = asyncio.get_running_loop()
        previous, self._loop = self._loop, loop
        if self._default is None or (previous is not None and previous is not loop and not previous.is_running()):
            self._default = AsyncOpenAI()
        return self._default

    async def search(self, query: str, *, limit: int, filter: MetadataFilter | None) -> Sequence[Passage]:
        """Search the vector store and turn each matching file chunk into a passage."""
        options: dict[str, Any] = {}
        if (filters := to_openai_filter(filter)) is not None:
            options['filters'] = filters
        if self.score_threshold is not None:
            options['ranking_options'] = {'score_threshold': self.score_threshold}
        try:
            page = await self._openai().vector_stores.search(
                self.vector_store_id,
                query=query,
                max_num_results=min(limit, OPENAI_MAX_NUM_RESULTS),
                rewrite_query=self.rewrite_query,
                **options,
            )
        except (APIConnectionError, RateLimitError, InternalServerError) as error:
            raise ModelRetry(f'Knowledge base search failed: {error}') from error
        return [
            Passage(
                text='\n'.join(part.text for part in result.content),
                source=result.filename,
                score=result.score,
                document_id=result.file_id,
                metadata=dict(result.attributes or {}),
            )
            for result in page.data
        ]

    async def read(self, document_id: str, *, max_chars: int, filter: MetadataFilter | None) -> Document | None:
        """Read a file's parsed text, stopping once it is past `max_chars`, if its attributes match `filter`.

        The file's attributes are fetched first, so a file outside the run's filter is never read.
        """
        parts: list[str] = []
        size = 0
        client = self._openai()
        try:
            file = await client.vector_stores.files.retrieve(document_id, vector_store_id=self.vector_store_id)
            attributes = dict(file.attributes or {})
            if not matches(attributes, filter):
                return None
            async for part in client.vector_stores.files.content(document_id, vector_store_id=self.vector_store_id):
                if part.text:
                    parts.append(part.text)
                    size += len(part.text)
                if size > max_chars:
                    break
        except NotFoundError:
            return None
        except (APIConnectionError, RateLimitError, InternalServerError) as error:
            raise ModelRetry(f'Reading the document failed: {error}') from error
        return Document(document_id=document_id, text=''.join(parts), metadata=attributes)

    def file_search_tool(self, ctx: RunContext[object], *, limit: int) -> FileSearchTool | None:
        """The native `file_search` tool for OpenAI models, which can search this store themselves."""
        if ctx.model.system != 'openai':
            return None
        return FileSearchTool(file_store_ids=[self.vector_store_id], max_num_results=min(limit, OPENAI_MAX_NUM_RESULTS))


def to_openai_filter(filter: MetadataFilter | None) -> Filters | None:
    """Translate a neutral filter: `eq` for a scalar, `in` for a list, and `and` over two or more keys."""
    if not filter:
        return None
    clauses: list[Filters] = [
        {'type': 'eq', 'key': key, 'value': value}
        if isinstance(value, str | int | float | bool)
        else {'type': 'in', 'key': key, 'value': list(value)}
        for key, value in filter.items()
    ]
    return clauses[0] if len(clauses) == 1 else {'type': 'and', 'filters': clauses}
