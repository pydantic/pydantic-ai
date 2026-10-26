"""The `ConversationSearch` capability: BM25 recall over persisted step history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._warn import warn_default_changed
from pydantic_ai_harness.conversation_search._source import HistorySource
from pydantic_ai_harness.conversation_search._toolset import (
    SCOPE_DEFAULT_CHANGE_IMPACT,
    ConversationSearchToolset,
    SearchScope,
)

_INSTRUCTIONS_PREFIX = (
    'A `search_conversation_history` tool can retrieve exact details from persisted history: '
    'earlier turns that context compaction has since dropped from the live context, and past '
)
_INSTRUCTIONS_SUFFIX = ' Reach for it when the current context, or a compaction summary, lacks a detail you need.'

_CONVERSATION_INSTRUCTIONS = f'{_INSTRUCTIONS_PREFIX}runs in the same conversation.{_INSTRUCTIONS_SUFFIX}'
"""Instruction text under `scope='conversation'`, where the corpus stops at the conversation."""

_ALL_INSTRUCTIONS = f'{_INSTRUCTIONS_PREFIX}runs persisted in the same store.{_INSTRUCTIONS_SUFFIX}'
"""Instruction text under `scope='all'`. Naming the conversation boundary here would talk the
model out of the cross-conversation recall that opting into `all` exists to enable."""


@dataclass
class ConversationSearch(AbstractCapability[AgentDepsT]):
    """Search persisted conversation history with a dependency-free BM25 tool.

    This capability persists nothing itself: it reads whatever history a persistence
    capability already stores, through a `HistorySource`. Pair it with
    `StepPersistence` sharing the same store, and the model can recall what
    compaction dropped from the live context as well as anything from past runs in
    the same conversation:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.compaction import SlidingWindowCompaction
    from pydantic_ai_harness.conversation_search import ConversationSearch, SnapshotHistorySource
    from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

    store = SqliteStepStore(database='sessions.db')
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            StepPersistence(store=store),
            ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
            SlidingWindowCompaction(max_messages=40),
        ],
    )


    async def ask(question: str, conversation_id: str) -> str:
        result = await agent.run(question, conversation_id=conversation_id)
        return result.output
    ```

    Search is conversation-scoped by default (see `scope`), so reaching a past run
    requires both runs to share a `conversation_id`. A run inherits the id from
    `message_history`, so threading history through follow-up runs keeps them in one
    conversation. History a run inherits is indexed under the run that first
    persisted it, except a request a later run adds an instruction change to, which is
    indexed again with that change.

    `StepPersistence` snapshots each step boundary before the next compaction runs, so
    the union of a run's snapshots still holds messages a compaction later replaced.
    The capabilities can be listed in any order: the tool reads the store when called.
    """

    source: HistorySource
    """Where the search corpus comes from. Use `SnapshotHistorySource` over the
    store a `StepPersistence` capability writes to."""

    scope: SearchScope | None = None
    """How much of the store one search may reach.

    `conversation` restricts the corpus to runs whose `conversation_id` matches the
    calling run's. A run with no `conversation_id` searches nothing and the tool says
    so, rather than falling back to every other unlabelled run. `all` searches every
    run the source enumerates and must only be used when the store is already isolated
    to one principal.

    `None` means the caller did not choose. It resolves to `conversation` and emits a
    `HarnessDeprecationWarning` once per instance, because this default was `all` in
    earlier releases and a store-wide caller would otherwise get a narrower corpus with no
    error. Pass either value to silence it; neither is deprecated. The warning will be
    removed in the next breaking release; the conversation-scoped default will remain.
    """

    tool_id: str = 'conversation-search'
    """Toolset id for the `search_conversation_history` tool."""

    max_matches: int = 10
    """Maximum number of matching excerpts the search tool returns. Must be non-negative."""

    context_lines: int = 5
    """Number of surrounding messages shown around each search match, within the match's run. Must be non-negative."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation, non-negative. This capability's default; Lucene's
    `BM25Similarity` uses `1.2`."""

    bm25_b: float = 0.75
    """BM25 length-normalization, between `0.0` and `1.0` (Lucene/Elasticsearch default)."""

    add_instructions: bool = True
    """Emit a short instruction telling the model the recall tool exists."""

    _: KW_ONLY
    id: str | None = 'conversation_search'
    """Stable capability ID, so its instructions are addressable and `defer_loading=True` needs none.

    Separate from `tool_id`, which identifies the toolset. Two `ConversationSearch` sharing this `id` are one
    search stated twice when they agree, and an error when they differ, since a merge could widen the corpus
    one of them was given. To keep two, give each its own `id` and `tool_id` and wrap them in `PrefixTools`,
    since their tool names are the same.
    """

    def __post_init__(self) -> None:
        """Warn once, at construction, when the caller left `scope` to the changed default.

        Construction is where the caller's choice (or non-choice) is expressed, so warning
        here fires once per instance instead of once per `search_conversation_history` call.
        """
        if self.scope is None:
            warn_default_changed(
                owner='ConversationSearch',
                option='scope',
                old='all',
                new='conversation',
                impact=SCOPE_DEFAULT_CHANGE_IMPACT,
            )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one search stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    @property
    def effective_scope(self) -> SearchScope:
        """The scope actually applied: the caller's choice, or `conversation` when unset."""
        return 'conversation' if self.scope is None else self.scope

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the source."""
        return ConversationSearchToolset[AgentDepsT](
            self.source,
            tool_id=self.tool_id,
            max_matches=self.max_matches,
            context_lines=self.context_lines,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            scope=self.effective_scope,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the recall tool exists, unless `add_instructions` is false.

        The wording follows `scope` so it never describes a corpus narrower than the
        one configured.
        """
        if not self.add_instructions:
            return None
        if self.effective_scope == 'conversation':
            return _CONVERSATION_INSTRUCTIONS
        return _ALL_INSTRUCTIONS
