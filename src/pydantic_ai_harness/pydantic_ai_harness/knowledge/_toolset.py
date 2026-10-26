"""The `search_knowledge` tool."""

from __future__ import annotations

from opentelemetry import trace

from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness._workspace import raise_tool_failure
from pydantic_ai_harness.knowledge._index import KnowledgeIndex
from pydantic_ai_harness.knowledge._store import KnowledgeMatch

_DURABLE_TRACER = trace.get_tracer('pydantic-ai-harness.knowledge')
"""Tracer for a search run inside a durable unit, such as a Temporal activity, where the run's tracer is not carried."""


class KnowledgeToolset(FunctionToolset[AgentDepsT]):
    """Exposes one `search_knowledge` tool over a `Knowledge` capability's store.

    Each call syncs the store with the sources first, so it searches the documents as they are
    when it is made.
    """

    def __init__(
        self, index: KnowledgeIndex, *, subject: str, workspace: Workspace | None, max_results: int, id: str
    ) -> None:
        super().__init__(id=id)
        self._index = index
        self._workspace = workspace
        self._max_results = max_results
        self.add_function(
            self.search_knowledge,
            name='search_knowledge',
            description=(
                f'Search {subject} for the passages most relevant to `query`, most relevant first, each with the '
                'id of the document it came from. Matching is by meaning, not exact words, so describe what you '
                'are looking for in a sentence or a question, and search again with other wording if the passages '
                'do not answer it.'
            ),
        )

    async def search_knowledge(self, ctx: RunContext[AgentDepsT], query: str) -> list[KnowledgeMatch]:  # noqa: D417
        """Search the knowledge base; the description is set per toolset in `__init__`.

        Args:
            query: What to look for.
        """
        tracer = _DURABLE_TRACER if isinstance(ctx, DurableRunContext) else ctx.tracer
        try:
            return await self._index.search(self._workspace or ctx.workspace, tracer, query, limit=self._max_results)
        except WorkspaceError as error:
            raise_tool_failure(error)
