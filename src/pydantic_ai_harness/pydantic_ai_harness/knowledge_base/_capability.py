"""Knowledge base capability: retrieval from a document corpus the application provides, over a pluggable backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import ToolReturn
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._durable import RetryRequest, retry_as_result
from pydantic_ai_harness.knowledge_base._backend import (
    FilterNode,
    KnowledgeBaseBackend,
    MetadataFilter,
    NativeFileSearchBackend,
    ReadableKnowledgeBase,
    SearchMode,
    normalize_filter,
)
from pydantic_ai_harness.knowledge_base._toolset import FilterFunc, KnowledgeBaseOperations, KnowledgeBaseToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_MAX_NUMBER_OF_RESULTS = 100

_INSTRUCTIONS = (
    'You have a `search_knowledge_base` tool that searches a knowledge base{contents}. '
    'Search it before answering a question its contents could cover, and base the answer on the passages '
    'it returns, citing their sources. Phrase queries as statements or keyword phrases rather than chat '
    'questions, resolve references to earlier turns first, and run separate searches for unrelated '
    'questions.{read} When nothing relevant comes back, say so instead of guessing.'
)
_READ_INSTRUCTIONS = (
    ' When a passage is not enough, open its whole document with `read_document` and the `Document:` id it shows.'
)


@dataclass
class KnowledgeBase(AbstractCapability[AgentDepsT]):
    """Retrieval from a knowledge base: a corpus of documents the application provides and the agent searches.

    Adds `search_knowledge_base`, which returns the passages most relevant to a query with their
    sources, and `read_document` when the backend can return whole documents, plus short instructions
    on when to search. Where the documents live and how they are matched is the backend's job:

    - `InMemoryKnowledgeBase`: documents held in process, BM25 keyword ranking.
    - `BedrockKnowledgeBase`: an Amazon Bedrock Knowledge Base (needs `boto3`).
    - `OpenAIVectorStore`: an OpenAI vector store, optionally searched natively on OpenAI models (needs `openai`).

    Any object with an async `search(query, *, limit, filter)` method is a backend; see `KnowledgeBaseBackend`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import KnowledgeBase
    from pydantic_ai_harness.knowledge_base import Document, InMemoryKnowledgeBase

    docs = InMemoryKnowledgeBase([Document('returns', 'Items can be returned within 30 days.')])
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[KnowledgeBase(docs)])
    ```

    Each backend call runs as a durable operation, so under durable execution a recovered run reuses
    the recorded result instead of searching again.
    """

    backend: KnowledgeBaseBackend
    """Where the documents live and how a search matches them."""

    _: KW_ONLY

    number_of_results: int = 5
    """How many passages one search returns (1 to 100; a backend may cap it lower, OpenAI at 50)."""

    filter: MetadataFilter | FilterFunc[AgentDepsT] | None = None
    """A metadata filter every search and document read applies, or a callable that builds it per run.

    The shorthand maps metadata keys to a value (equal) or a list of values (any of), and every key
    must match: `{'tenant_id': 'acme', 'source': ['manuals', 'faq']}`. The tree form adds `!=`, ranges,
    `not in`, `OR` and `NOT`: see `MetadataFilter`. The callable form scopes a shared knowledge base
    to the run's tenant or user, for example `lambda ctx: {'tenant_id': ctx.deps.tenant_id}`; it may
    be async, and returning `None` searches unfiltered. Each backend translates the filter to its own
    syntax and raises for parts it can't apply.
    """

    search_mode: SearchMode | None = None
    """How searches match: `'semantic'`, `'keyword'` or `'hybrid'`; `None` for the backend's default.

    Backends support a subset: in-memory is keyword, Bedrock semantic or hybrid, OpenAI semantic.
    An unsupported mode raises on the first search.
    """

    score_threshold: float | None = None
    """Drop passages scoring below this, on the backend's own scale (Bedrock and OpenAI 0 to 1, in-memory BM25)."""

    show_sources: bool = True
    """Whether each passage's `Source:` line (location and score) is in the text the model sees.

    Off, passages are rendered without it and provenance stays in the tool result's
    `metadata['results']`, for an application that renders citations itself or must not show the model
    internal storage paths.
    """

    max_document_chars: int = 20_000
    """How much of a document one `read_document` call returns, in characters.

    A longer document ends with a marker naming the `offset` that continues it, so the model can read on.
    """

    native: bool = False
    """Let models whose provider can search the backend's index natively do so, with `FileSearchTool`.

    Needs a backend that offers it (`OpenAIVectorStore`), and no `filter`, `search_mode` or
    `score_threshold`, since the native tool uses the provider's own settings. Models without native
    file search use `search_knowledge_base` as usual; with native search, passages come back as the
    provider's own tool result instead of `metadata['results']`.
    """

    description: str | None = None
    """What the knowledge base holds, for example `'product manuals and the support FAQ'`.

    Added to the tool description, the default instructions and the deferred-loading catalog line.
    """

    guidance: str | None = None
    """Custom instructions for the system prompt: `None` for the default guidance, `''` for none."""

    id: str | None = 'knowledge_base'
    """Stable identity for durable execution, which records each search and read under it.

    Two instances under one `id` must agree on every field; a second knowledge base on the same agent
    needs its own `id` and `PrefixTools`.
    """

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if not 1 <= self.number_of_results <= _MAX_NUMBER_OF_RESULTS:
            raise ValueError(
                f'number_of_results must be between 1 and {_MAX_NUMBER_OF_RESULTS}, got {self.number_of_results}'
            )
        if self.max_document_chars < 1:
            raise ValueError(f'max_document_chars must be positive, got {self.max_document_chars}')
        if self.filter is not None and not callable(self.filter):
            normalize_filter(self.filter)
        if self.native:
            if not isinstance(self.backend, NativeFileSearchBackend):
                raise ValueError(f'native needs a backend the provider can search natively, not {self.backend!r}')
            if self.filter is not None or self.search_mode is not None or self.score_threshold is not None:
                raise ValueError(
                    "native search uses the provider's own settings and can't apply a filter, "
                    'search_mode or score_threshold; use one or the other'
                )

    def get_description(self) -> str:
        """The catalog line shown with `load_capability` when `defer_loading=True`: what to load it for."""
        contents = f' of {self.description}' if self.description else ''
        return f'Search a knowledge base{contents} for passages before answering questions it could cover.'

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static guidance: search before answering, phrase queries well, cite sources, admit misses.

        A non-`None` `guidance` replaces the default; `''` disables instructions entirely.
        """
        if self.guidance is not None:
            return self.guidance or None
        contents = f' of {self.description}' if self.description else ''
        read = _READ_INSTRUCTIONS if isinstance(self.backend, ReadableKnowledgeBase) else ''
        return _INSTRUCTIONS.format(contents=contents, read=read)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one configuration stated twice; two that disagree raise rather than merge.

        A merge could widen a `filter` that scopes the knowledge base to a tenant, so this narrows.
        """
        return one_per_id(capabilities)

    def get_toolset(self) -> KnowledgeBaseToolset[AgentDepsT]:
        """Build the toolset, routed through the durable operations."""
        return self._build_toolset(id=self.id, operations=KnowledgeBaseOperations(search=self._search, read=self._read))

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        """The native file search tool, per run and only for models that can search the backend's index."""
        return [self._native_file_search] if self.native else []

    def _native_file_search(self, ctx: RunContext[AgentDepsT]) -> AbstractNativeTool | None:
        backend = self.backend
        assert isinstance(backend, NativeFileSearchBackend)
        return backend.file_search_tool(ctx, limit=self.number_of_results)

    @durable_operation('search_knowledge_base')
    async def _search(self, query: str, filter: FilterNode | None) -> ToolReturn[str] | RetryRequest:
        return await retry_as_result(self._requests.search(query, filter))

    @durable_operation('read_document')
    async def _read(self, document_id: str, filter: FilterNode | None, offset: int) -> ToolReturn[str] | RetryRequest:
        return await retry_as_result(self._requests.read(document_id, filter, offset))

    @cached_property
    def _requests(self) -> KnowledgeBaseToolset[AgentDepsT]:
        """The toolset whose `search` and `read` make the requests that the durable operations run."""
        return self._build_toolset()

    def _build_toolset(
        self, *, id: str | None = None, operations: KnowledgeBaseOperations | None = None
    ) -> KnowledgeBaseToolset[AgentDepsT]:
        return KnowledgeBaseToolset[AgentDepsT](
            self.backend,
            number_of_results=self.number_of_results,
            filter=self.filter,
            search_mode=self.search_mode,
            score_threshold=self.score_threshold,
            show_sources=self.show_sources,
            max_document_chars=self.max_document_chars,
            description=self.description,
            defer_to_native=self.native,
            id=id,
            operations=operations,
        )

    @classmethod
    def from_spec(
        cls,
        *,
        backend: Literal['bedrock', 'openai'],
        knowledge_base_id: str,
        region_name: str | None = None,
        reranking_model: str | None = None,
        fetch_results: int | None = None,
        number_of_results: int = 5,
        filter: MetadataFilter | None = None,
        search_mode: SearchMode | None = None,
        score_threshold: float | None = None,
        show_sources: bool = True,
        max_document_chars: int = 20_000,
        native: bool = False,
        description: str | None = None,
        guidance: str | None = None,
        id: str | None = 'knowledge_base',
    ) -> KnowledgeBase[AgentDepsT]:
        """Construct the capability from serializable spec options.

        `knowledge_base_id` is the Bedrock knowledge base id or the OpenAI vector store id. The Bedrock
        options (`region_name`, `reranking_model`, `fetch_results`) only apply to
        `backend='bedrock'`. A spec takes a static `filter`; a callable filter, a custom client and an
        in-memory backend are Python-only.
        """
        store: KnowledgeBaseBackend
        if backend == 'bedrock':
            from pydantic_ai_harness.knowledge_base._bedrock import BedrockKnowledgeBase

            store = BedrockKnowledgeBase(
                knowledge_base_id,
                region_name=region_name,
                reranking_model=reranking_model,
                fetch_results=fetch_results,
            )
        else:
            if (region_name, reranking_model, fetch_results) != (None, None, None):
                raise ValueError('region_name, reranking_model and fetch_results only apply to backend="bedrock"')
            from pydantic_ai_harness.knowledge_base._openai import OpenAIVectorStore

            store = OpenAIVectorStore(knowledge_base_id)
        return cls(
            store,
            number_of_results=number_of_results,
            filter=filter,
            search_mode=search_mode,
            score_threshold=score_threshold,
            show_sources=show_sources,
            max_document_chars=max_document_chars,
            native=native,
            description=description,
            guidance=guidance,
            id=id,
        )
