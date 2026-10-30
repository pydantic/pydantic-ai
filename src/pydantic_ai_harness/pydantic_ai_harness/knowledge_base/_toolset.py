"""Knowledge base toolset: the `search_knowledge_base` and `read_document` tools over any backend."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TypeAlias

from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolReturn
from pydantic_ai.native_tools import FileSearchTool
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import RetryRequest, raise_retry
from pydantic_ai_harness.knowledge_base._backend import (
    FilterNode,
    KnowledgeBaseBackend,
    KnowledgeBaseResult,
    MetadataFilter,
    Passage,
    ReadableKnowledgeBase,
    SearchMode,
    is_readable,
    normalize_filter,
)

FilterFunc: TypeAlias = Callable[[RunContext[AgentDepsT]], MetadataFilter | None | Awaitable[MetadataFilter | None]]
"""Builds the filter for one run from its `RunContext`, for example the tenant in `ctx.deps`; sync or async."""

SearchOperation: TypeAlias = Callable[[str, FilterNode | None], Awaitable[ToolReturn[str] | RetryRequest]]
"""A durable operation that runs one search for a query and its resolved filter and renders the result."""

ReadOperation: TypeAlias = Callable[[str, FilterNode | None, int], Awaitable[ToolReturn[str] | RetryRequest]]
"""A durable operation that reads one document within the resolved filter, from an offset, and renders it."""

_SEARCH_DESCRIPTION = 'Search the knowledge base for passages relevant to a query.'


@dataclass(frozen=True)
class KnowledgeBaseOperations:
    """The durable operations a `KnowledgeBase` capability runs its toolset's requests through."""

    search: SearchOperation
    read: ReadOperation


async def _defer_to_file_search(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
    """Mark the local search as the fallback for the native `file_search` tool, so only one is sent."""
    return replace(tool_def, unless_native=FileSearchTool.kind)


class KnowledgeBaseToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent `search_knowledge_base` over a knowledge base backend, and `read_document` where it reads.

    A search returns a `ToolReturn` whose `return_value` is the text the model sees and whose
    `metadata['results']` lists the passages as `KnowledgeBaseResult` records, so applications can
    render citations without parsing the text. `read_document` returns a document in windows of
    `max_document_chars`, and says which `offset` continues it. `KnowledgeBase` passes `operations`
    so that each backend call runs as one of its durable operations.
    """

    def __init__(
        self,
        backend: KnowledgeBaseBackend,
        *,
        number_of_results: int = 5,
        filter: MetadataFilter | FilterFunc[AgentDepsT] | None = None,
        require_filter: bool = False,
        search_mode: SearchMode | None = None,
        score_threshold: float | None = None,
        show_sources: bool = True,
        max_document_chars: int = 20_000,
        description: str | None = None,
        defer_to_native: bool = False,
        id: str | None = None,
        operations: KnowledgeBaseOperations | None = None,
    ) -> None:
        super().__init__(id=id)
        self._backend = backend
        self._number_of_results = number_of_results
        self._filter = filter
        self._require_filter = require_filter
        self._search_mode: SearchMode | None = search_mode
        self._score_threshold = score_threshold
        self._show_sources = show_sources
        self._max_document_chars = max_document_chars
        self._operations = operations
        self._readable = is_readable(backend)
        search_description = (
            f'{_SEARCH_DESCRIPTION} The knowledge base holds {description}.' if description else _SEARCH_DESCRIPTION
        )
        self.add_function(
            self.search_knowledge_base,
            name='search_knowledge_base',
            description=search_description,
            prepare=_defer_to_file_search if defer_to_native else None,
        )
        if self._readable:
            self.add_function(self.read_document, name='read_document')

    async def search_knowledge_base(self, ctx: RunContext[AgentDepsT], query: str) -> ToolReturn[str]:
        """Search the knowledge base for passages relevant to a query.

        Args:
            ctx: The run context, which a callable `filter` builds the run's filter from.
            query: What to look for, phrased as a statement or keyword phrase rather than a chat question.

        Returns:
            The best matching passages, each under the source it came from.
        """
        filter = await self._resolve_filter(ctx)
        if self._operations is not None:
            return raise_retry(await self._operations.search(query, filter))
        return await self.search(query, filter)

    async def read_document(self, ctx: RunContext[AgentDepsT], document_id: str, offset: int = 0) -> ToolReturn[str]:
        """Read a whole document from the knowledge base, by the `Document:` id a search result showed.

        Args:
            ctx: The run context, which a callable `filter` builds the run's filter from.
            document_id: The id of the document to read.
            offset: The character to start at. A long document comes back in parts; each part says the
                offset that continues it.

        Returns:
            The document text from `offset`, up to the configured length.
        """
        filter = await self._resolve_filter(ctx)
        if self._operations is not None:
            return raise_retry(await self._operations.read(document_id, filter, offset))
        return await self.read(document_id, filter, offset)

    async def search(self, query: str, filter: FilterNode | None) -> ToolReturn[str]:
        """Run one backend search and render it. The request a capability's durable operation records."""
        passages = list(
            await self._backend.search(
                query,
                limit=self._number_of_results,
                filter=filter,
                mode=self._search_mode,
                score_threshold=self._score_threshold,
            )
        )
        passages = passages[: self._number_of_results]
        if not passages:
            return ToolReturn(f'No results found for {query!r}.', metadata={'results': []})
        records = [_record(passage) for passage in passages]
        sections = [self._format(passage) for passage in passages]
        plural = 's' if len(sections) != 1 else ''
        joined = '\n\n---\n\n'.join(sections)
        return ToolReturn(
            f'Found {len(sections)} result{plural} for {query!r}:\n\n{joined}', metadata={'results': records}
        )

    async def read(self, document_id: str, filter: FilterNode | None, offset: int = 0) -> ToolReturn[str]:
        """Read one document within `filter` from `offset` and render up to `max_document_chars` of it.

        A document outside the filter reads as missing, so the model learns nothing about it.
        """
        if offset < 0:
            raise ModelRetry(f'offset must be 0 or more, got {offset}.')
        backend = self._backend
        assert isinstance(backend, ReadableKnowledgeBase)
        document = await backend.read(document_id, max_chars=offset + self._max_document_chars, filter=filter)
        if document is None:
            raise ModelRetry(f'No document with id {document_id!r}. Use an id from a search result.')
        if offset and offset >= len(document.text):
            raise ModelRetry(f'offset {offset} is past the end of document {document_id!r}.')
        end = offset + self._max_document_chars
        text = document.text[offset:end]
        next_offset = end if len(document.text) > end else None
        if next_offset is not None:
            text += f'\n[... the document continues; call read_document with offset={next_offset}]'
        lines = [f'Document: {_one_line(document.document_id)}']
        if self._show_sources and document.source:
            lines.append(f'Source: {_one_line(document.source)}')
        if offset:
            lines.append(f'From character: {offset}')
        return ToolReturn(
            '\n'.join([*lines, '', text]),
            metadata={
                'document': {
                    'document_id': document.document_id,
                    'source': document.source,
                    'metadata': dict(document.metadata),
                    'offset': offset,
                    'next_offset': next_offset,
                }
            },
        )

    async def _resolve_filter(self, ctx: RunContext[AgentDepsT]) -> FilterNode | None:
        if callable(self._filter):
            filter = self._filter(ctx)
            if inspect.isawaitable(filter):
                filter = await filter
        else:
            filter = self._filter
        node = normalize_filter(filter)
        if node is None and self._require_filter:
            raise UserError('The knowledge base filter resolved to nothing for this run, and require_filter is set.')
        return node

    def _format(self, passage: Passage) -> str:
        lines: list[str] = []
        if self._show_sources:
            source = _one_line(passage.source or '') or '(unknown)'
            score = passage.score
            lines.append(f'Source: {source}' if score is None else f'Source: {source} (score {score:.2f})')
        if self._readable and passage.document_id is not None:
            lines.append(f'Document: {_one_line(passage.document_id)}')
        return '\n'.join([*lines, passage.text])


def _one_line(text: str) -> str:
    """Collapse whitespace, so a source or id can't start a line that looks like the tool's own labels."""
    return ' '.join(text.split())


def _record(passage: Passage) -> KnowledgeBaseResult:
    return {
        'source': passage.source,
        'score': passage.score,
        'document_id': passage.document_id,
        'metadata': dict(passage.metadata),
    }
