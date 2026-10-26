"""The `Knowledge` capability: semantic search over documents the agent is given."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import KW_ONLY, dataclass, field

from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.embeddings import Embedder, EmbeddingModel
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._workspace import require_workspace, secondary_workspace
from pydantic_ai_harness.knowledge._chunking import Chunker, RecursiveChunker
from pydantic_ai_harness.knowledge._index import KnowledgeIndex, KnowledgeSyncResult
from pydantic_ai_harness.knowledge._sources import DEFAULT_FILE_SUFFIXES, KnowledgeSourceLike, is_path_source
from pydantic_ai_harness.knowledge._store import InMemoryVectorStore, VectorStore
from pydantic_ai_harness.knowledge._toolset import KnowledgeToolset

_INSTRUCTIONS = (
    'Search {subject} before answering a question it may cover, rather than answering from memory, '
    'and say which document the answer comes from.'
)


@dataclass
class Knowledge(AbstractCapability[AgentDepsT]):
    """Give the model a `search_knowledge` tool over documents: retrieval-augmented generation (RAG).

    Sources are files and directories in the run's workspace, inline `KnowledgeDocument`s, and
    custom `KnowledgeSource`s. They are chunked with `chunker`, embedded with `embedder`, and kept
    in `store`. Every search first syncs the store with the sources, embedding only documents that
    are new or changed since they were stored and removing those the sources no longer produce, so
    the model searches the documents as they are when it calls the tool.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    from pydantic_ai_harness import Knowledge

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            LocalWorkspace('.'),
            Knowledge(['docs'], embedder='openai:text-embedding-3-small'),
        ],
    )
    ```
    """

    sources: Sequence[KnowledgeSourceLike]
    """What to index, in order. A `str` or `Path` is a file or directory in the workspace, relative to its
    working directory; a directory contributes every file below it with one of `file_suffixes`, skipping
    hidden files and directories. A `KnowledgeDocument` is indexed as given, and a `KnowledgeSource` is
    asked for its documents on every sync. Document ids must be unique across sources."""

    _: KW_ONLY

    embedder: Embedder | EmbeddingModel | str
    """The embedding model: an `Embedder`, an `EmbeddingModel`, or a model name such as
    `'openai:text-embedding-3-small'`. Its embeddings are not counted in the run's usage."""

    store: VectorStore = field(default_factory=InMemoryVectorStore)
    """Where embedded chunks are kept. The default `InMemoryVectorStore` lasts as long as this capability."""

    chunker: Chunker = field(default_factory=RecursiveChunker)
    """How documents are split into the passages that are embedded and returned."""

    description: str | None = None
    """What the documents are about, such as `'the product manual'`, named in the instructions so the model
    knows when to search. Without it they say "the knowledge base"."""

    max_results: int = 5
    """Passages returned per search. Must be positive."""

    file_suffixes: Collection[str] = DEFAULT_FILE_SUFFIXES
    """File suffixes read from a directory source, compared case-insensitively. A file listed by path
    in `sources` is read whatever its suffix."""

    workspace: WorkspaceBackend | None = None
    """Where path sources and custom sources read from instead of the run's workspace, such as
    `LocalWorkspaceBackend('/app')` for documents shipped with an app whose agent works in a sandbox."""

    id: str | None = 'knowledge'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    An agent has one `search_knowledge` tool, so two `Knowledge` sharing this `id` are one knowledge base
    stated twice when they agree, and an error when they differ. To keep two, give each its own `id` and
    wrap them in `PrefixTools`, since their tool names are the same.
    """

    _own_workspace: Workspace | None = field(init=False, repr=False, compare=False)
    _index: KnowledgeIndex = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.sources:
            raise UserError('`Knowledge` needs at least one source.')
        if self.max_results <= 0:
            raise UserError('`Knowledge.max_results` must be positive.')
        if invalid := sorted(suffix for suffix in self.file_suffixes if not suffix.startswith('.')):
            raise UserError(f'`Knowledge.file_suffixes` entries start with a dot, such as `.md`: {invalid!r}.')
        self._own_workspace = secondary_workspace(self.workspace, 'Knowledge')
        if isinstance(self.embedder, Embedder):
            embedder, model = self.embedder, self.embedder.model
        else:
            embedder, model = Embedder(self.embedder), self.embedder
        self._index = KnowledgeIndex(
            embedder=embedder,
            # Fixed at construction: `Embedder` replaces a model name with the model it infers on first use.
            model_key=model if isinstance(model, str) else f'{model.system}:{model.model_name}',
            sources=tuple(self.sources),
            store=self.store,
            chunker=self.chunker,
            file_suffixes=frozenset(suffix.lower() for suffix in self.file_suffixes),
        )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one knowledge base stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when a path source has no workspace to be read from."""
        if self.workspace is None and any(is_path_source(source) for source in self.sources):
            require_workspace(ctx.workspace, 'Knowledge', ctx.messages)

    async def sync(
        self, workspace: WorkspaceBackend | None = None, *, tracer: Tracer | None = None
    ) -> KnowledgeSyncResult:
        """Bring the store up to date with the sources outside a run, such as when an app starts.

        A search syncs the store anyway; syncing ahead keeps the first search of a large corpus from
        waiting while it is embedded.

        Args:
            workspace: Where path sources are read from when the capability has no `workspace=` of its own,
                as the run's workspace would be, such as `LocalWorkspaceBackend('.')`.
            tracer: Tracer the `knowledge_sync` span is started on. Defaults to a no-op tracer.

        Raises:
            UserError: A path source needs a workspace, and none was passed.
        """
        run_workspace = secondary_workspace(workspace, 'Knowledge.sync')
        target = self._own_workspace or run_workspace
        if target is None:
            if any(is_path_source(source) for source in self.sources):
                raise UserError(
                    '`Knowledge.sync()` needs the workspace the path sources are in, such as `LocalWorkspaceBackend(".")`.'
                )
            target = Workspace(NO_WORKSPACE)
        return await self._index.sync(target, tracer if tracer is not None else NoOpTracer())

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_knowledge` tool."""
        return KnowledgeToolset[AgentDepsT](
            self._index,
            subject=self._subject,
            workspace=self._own_workspace,
            max_results=self.max_results,
            id=self.id or 'knowledge',
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model to search before answering. The tool is not named, since `PrefixTools` may rename it."""
        return _INSTRUCTIONS.format(subject=self._subject)

    @property
    def _subject(self) -> str:
        return self.description or 'the knowledge base'
