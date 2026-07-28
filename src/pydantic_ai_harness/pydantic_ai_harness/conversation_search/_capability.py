"""The `ConversationSearch` capability: BM25 recall over persisted step history."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.agent.abstract import AgentInstructions
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.conversation_search._source import HistorySource
from pydantic_ai_harness.conversation_search._toolset import ConversationSearchToolset, SearchScope

_INSTRUCTIONS = (
    'A `search_conversation_history` tool can retrieve exact details from persisted history: '
    'earlier turns that context compaction has since dropped from the live context, and past '
    'runs persisted in the same store. Reach for it when the current context, or a compaction '
    'summary, lacks a detail you need.'
)


@dataclass
class ConversationSearch(AbstractCapability[AgentDepsT]):
    """Search persisted conversation history with a dependency-free BM25 tool.

    This capability persists nothing itself: it reads whatever history a persistence
    capability already stores, through a `HistorySource`. Pair it with
    `StepPersistence` sharing the same store, and the model can recall what
    compaction dropped from the live context as well as anything from past runs:

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
            ConversationSearch(SnapshotHistorySource(store)),
            SlidingWindowCompaction(max_messages=40),
        ],
    )
    ```

    Some compaction strategies persist their edits into the run's durable message
    history (`SummarizingCompaction` replaces summarized prefixes for good; a
    `SlidingWindowCompaction` trim only narrows what each request sends). Either way,
    `StepPersistence` snapshots each step boundary before the next compaction runs,
    so the union of a run's snapshots still holds the originals --
    `SnapshotHistorySource` recovers them. No ordering or hook coordination between
    the capabilities is required; the search tool reads the store lazily at call
    time.
    """

    source: HistorySource
    """Where the search corpus comes from. Use `SnapshotHistorySource` over the
    store a `StepPersistence` capability writes to."""

    scope: SearchScope = 'all'
    """How much of the store one search may reach.

    `all` searches every run the source enumerates. `conversation` restricts the
    corpus to runs whose `conversation_id` matches the calling run's, which a store
    shared across users or tenants needs: with `all`, any run reading that store can
    retrieve verbatim excerpts from every other conversation in it. Under
    `conversation`, a run with no `conversation_id` searches nothing and the tool says
    so, rather than falling back to every other unlabelled run.
    """

    tool_id: str = 'conversation-search'
    """Toolset id for the `search_conversation_history` tool."""

    max_matches: int = 10
    """Maximum number of matching excerpts the search tool returns. Must be non-negative."""

    context_lines: int = 5
    """Number of surrounding lines shown around each search match. Must be non-negative."""

    bm25_k1: float = 1.5
    """BM25 term-frequency saturation, non-negative. This capability's default; Lucene's
    `BM25Similarity` uses `1.2`."""

    bm25_b: float = 0.75
    """BM25 length-normalization, between `0.0` and `1.0` (Lucene/Elasticsearch default)."""

    add_instructions: bool = True
    """Emit a short instruction telling the model the recall tool exists."""

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_conversation_history` tool over the source."""
        return ConversationSearchToolset[AgentDepsT](
            self.source,
            tool_id=self.tool_id,
            max_matches=self.max_matches,
            context_lines=self.context_lines,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
            scope=self.scope,
        )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Tell the model the recall tool exists, unless `add_instructions` is false."""
        if not self.add_instructions:
            return None
        return _INSTRUCTIONS
