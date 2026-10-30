"""OpenAI vector store backend: searches and reads a vector store with the OpenAI API.

External assumptions (verified 2026-10-30 against openai 3.31.0 and
<https://platform.openai.com/docs/api-reference/vector-stores/search>):

- `vector_stores.search` takes `max_num_results` from 1 to 50 and `filters` built from comparison
  filters (equality, ranges, `in` and its negation) and compound `and` / `or` filters over file
  attributes; there is no `not`. `ranking_options.score_threshold` runs from 0 to 1. The search is
  semantic, with no keyword-only or hybrid mode on this endpoint. Each result carries `file_id`, `filename`, `score`, `attributes` and text `content`.
- `vector_stores.files.content` pages through a file's parsed text, which `read` uses.
- The Responses API's `file_search` tool searches the same vector stores natively; core sends it as
  `FileSearchTool(file_store_ids=[...])` on `OpenAIResponsesModel`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.native_tools import FileSearchTool
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.knowledge_base._backend import Document, FilterNode, Passage, SearchMode, matches

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

    def _openai(self) -> AsyncOpenAI:
        if self.client is not None:
            return self.client
        loop = asyncio.get_running_loop()
        previous, self._loop = self._loop, loop
        if self._default is None or (previous is not None and previous is not loop and not previous.is_running()):
            self._default = AsyncOpenAI()
        return self._default

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filter: FilterNode | None,
        mode: SearchMode | None = None,
        score_threshold: float | None = None,
    ) -> Sequence[Passage]:
        """Search the vector store and turn each matching file chunk into a passage.

        The search is semantic, so `mode` can only be `None` or `'semantic'`. `score_threshold` is on
        OpenAI's 0 to 1 scale and applied by the API.
        """
        if mode not in (None, 'semantic'):
            raise UserError(f"OpenAIVectorStore only supports mode='semantic', not {mode!r}")
        options: dict[str, Any] = {}
        if (filters := to_openai_filter(filter)) is not None:
            options['filters'] = filters
        if score_threshold is not None:
            if not 0 <= score_threshold <= 1:
                raise UserError(f'OpenAIVectorStore takes a score_threshold from 0 to 1, got {score_threshold}')
            options['ranking_options'] = {'score_threshold': score_threshold}
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

    async def read(self, document_id: str, *, max_chars: int, filter: FilterNode | None) -> Document | None:
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


_OPERATORS: dict[str, Literal['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'nin']] = {
    '==': 'eq',
    '!=': 'ne',
    '>': 'gt',
    '>=': 'gte',
    '<': 'lt',
    '<=': 'lte',
    'in': 'in',
    'not in': 'nin',
}


def to_openai_filter(filter: FilterNode | None) -> Filters | None:
    """Translate a filter tree: comparisons map one to one, `AND` and `OR` to compound filters.

    Raises:
        UserError: The filter uses `NOT`, which the vector store API has no operator for. Negate the
            comparisons instead (`!=`, `not in`).
    """
    return None if filter is None else _translate(filter)


def _translate(node: FilterNode) -> Filters:
    if 'conditions' not in node:
        value = node['value']
        return {
            'type': _OPERATORS[node['operator']],
            'key': node['field'],
            'value': value if isinstance(value, str | int | float | bool) else list(value),
        }
    if node['operator'] == 'NOT':
        raise UserError("OpenAIVectorStore can't apply NOT; negate the conditions instead, with '!=' or 'not in'")
    children = [_translate(child) for child in node['conditions']]
    if len(children) == 1:
        return children[0]
    return {'type': 'and' if node['operator'] == 'AND' else 'or', 'filters': children}
