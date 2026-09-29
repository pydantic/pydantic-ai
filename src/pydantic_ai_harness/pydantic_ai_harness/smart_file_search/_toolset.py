"""The `smart_file_search` tool: plain-English code search over the run's workspace."""

from __future__ import annotations

from pydantic_ai.exceptions import ModelAPIError, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness._workspace import raise_tool_failure, require_workspace, supports_commands
from pydantic_ai_harness.smart_file_search._index import MAX_CACHED_INDEXES, SnippetIndexes
from pydantic_ai_harness.smart_file_search._judge import JudgeModel, resolve_judge_model
from pydantic_ai_harness.smart_file_search._search import DEFAULT_CANDIDATES, SmartFileSearchResult, search_code

TOOL_NAME = 'smart_file_search'


class SmartFileSearchToolset(FunctionToolset[AgentDepsT]):
    """Registers `smart_file_search`, which searches the run's workspace and judges snippets with a model.

    Files are listed with `rg` and read through `ctx.workspace`, so the tool is only offered when the
    workspace can run commands. A run with no workspace fails at its start. With `cache_index`, each
    searched directory's index is kept on this toolset between searches and runs.
    """

    def __init__(
        self,
        *,
        model: JudgeModel | None,
        threshold: float,
        concurrency: int,
        cache_index: bool = False,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._model = model
        self._threshold = threshold
        self._concurrency = concurrency
        self._indexes = SnippetIndexes(MAX_CACHED_INDEXES if cache_index else 0)
        self.add_function(self.smart_file_search, name=TOOL_NAME)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer no tools when the workspace cannot execute commands; fail a run with no workspace."""
        if not ctx.workspace.attached:
            require_workspace(ctx.workspace, 'SmartFileSearch', ctx.messages)
        if not supports_commands(ctx.workspace):
            return {}
        return await super().get_tools(ctx)

    async def smart_file_search(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
        directory: str = '.',
        glob: str | None = None,
        limit: int = 5,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> SmartFileSearchResult:
        """Find code by what it DOES, described in plain English.

        Use for behaviour-based discovery when you don't know the symbol names,
        e.g. "where do we reject expired sessions?" or "retry a failed network
        request". Keep using regular grep for exact symbols, regexes and exhaustive
        reference lists -- this tool ranks, it does not enumerate.

        A local lexical shortlist of `candidates` snippets is judged by a
        relevance model; matches come back ranked with file/line ranges and a
        short excerpt. An empty result does not prove absence. Tests are
        labelled kind="test", not demoted.

        Args:
            ctx: The current agent run context.
            query: Plain-English description of the behaviour to find.
            directory: Root to search (respects .gitignore; skips hidden files).
            glob: Optional ripgrep glob to restrict files, e.g. "*.py".
            limit: Max matches to return (1-100).
            candidates: Snippets to judge after the lexical shortlist (1-256).
                Defaults to 128. Higher = better recall, slower and costlier.
        """
        model = resolve_judge_model(self._model, ctx.model)
        try:
            return await search_code(
                ctx.workspace,
                model,
                query,
                directory,
                indexes=self._indexes,
                glob=glob,
                limit=limit,
                candidates=candidates,
                threshold=self._threshold,
                concurrency=self._concurrency,
            )
        except WorkspaceError as error:
            raise_tool_failure(error)
        except (ModelAPIError, UnexpectedModelBehavior) as error:
            # Report the judge's failure to the model, which can fall back to regular search.
            raise ToolFailed(f'smart_file_search failed: {type(error).__name__}: {error}') from error
