"""The `smart_search` tool: plain-English code search over the run's workspace."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from pydantic_ai.exceptions import ModelAPIError, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceError
from pydantic_ai_harness._workspace import raise_tool_failure, require_workspace, supports_commands
from pydantic_ai_harness.smart_search._index import MAX_CACHED_INDEXES, SnippetIndexes
from pydantic_ai_harness.smart_search._judge import JudgeModel
from pydantic_ai_harness.smart_search._search import (
    DEFAULT_CANDIDATES,
    MAX_CANDIDATES,
    MAX_LIMIT,
    SmartSearchResult,
    search_code,
)

TOOL_NAME = 'smart_search'
DEFAULT_THRESHOLD = 0.5
DEFAULT_CONCURRENCY = 8

_INSTRUCTIONS = """\
## Code discovery: smart_search first
Use `smart_search` as your first tool for gathering code context about an unfamiliar implementation.
Describe the behavior you need to find in plain English, scoped to the relevant directory. Do this
instead of guessing symbol names, running exploratory grep chains, or listing and reading many files
just to locate the implementation.

Use the returned excerpts and line ranges to decide which files need a targeted read. Do not
automatically read every match or repeat searches when the excerpts already answer the question.
Read the relevant code before editing; `smart_search` is discovery, not a substitute for
understanding it.

Use regular text search for exact symbols, regexes, and exhaustive references, and read a known file
directly when its location is already established.

The lexical shortlist is not exhaustive, and no matches do not prove absence. If results are weak,
rephrase or narrow the query, or fall back to exact search. If `smart_search` fails, fall back to
regular search and targeted reads rather than retrying. `smart_search` sends selected source, paths and
the query to the judge model's provider, so honor requests to keep code local by using local search
instead."""


def validate_settings(threshold: float, concurrency: int) -> None:
    """Reject a `threshold` outside `[0, 1]` and a `concurrency` that would never let a judgment start."""
    if not 0 <= threshold <= 1:
        raise ValueError(f'`threshold` must be between 0 and 1, got {threshold}')
    # Negated so NaN, which an agent spec can pass through unvalidated, fails too instead of hanging the search.
    if not concurrency >= 1:
        raise ValueError(f'`concurrency` must be a positive integer, got {concurrency}')


class SmartSearchToolset(FunctionToolset[AgentDepsT]):
    """Registers `smart_search`, which searches the run's workspace and judges snippets with a model.

    Files are listed with `rg` and read through `ctx.workspace`, so the tool is only offered when the
    workspace can run commands. A run with no workspace fails at its start. With `cache_index`, each
    searched directory's index is kept on this toolset between searches and runs.
    """

    def __init__(
        self,
        *,
        model: JudgeModel,
        threshold: float = DEFAULT_THRESHOLD,
        concurrency: int = DEFAULT_CONCURRENCY,
        cache_index: bool = False,
        guidance: str | None = None,
        id: str | None = None,
    ) -> None:
        validate_settings(threshold, concurrency)
        super().__init__(id=id)
        self._guidance = _INSTRUCTIONS if guidance is None else guidance
        self._model = model
        self._threshold = threshold
        self._concurrency = concurrency
        self._indexes = SnippetIndexes(MAX_CACHED_INDEXES if cache_index else 0)
        self.add_function(self.smart_search, name=TOOL_NAME)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer no tools when the workspace cannot execute commands; fail a run with no workspace."""
        if not ctx.workspace.attached:
            require_workspace(ctx.workspace, 'SmartSearch', ctx.messages)
        if not supports_commands(ctx.workspace):
            return {}
        return await super().get_tools(ctx)

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> list[InstructionPart] | None:
        """The discovery policy, offered only with the tool: `smart_search` first, exact search for exact symbols."""
        if not self._guidance or not supports_commands(ctx.workspace):
            return None
        return [InstructionPart(self._guidance)]

    async def smart_search(
        self,
        ctx: RunContext[AgentDepsT],
        query: str,
        directory: str = '.',
        glob: str | None = None,
        limit: Annotated[int, Field(ge=1, le=MAX_LIMIT)] = 5,
        candidates: Annotated[int, Field(ge=1, le=MAX_CANDIDATES)] = DEFAULT_CANDIDATES,
    ) -> SmartSearchResult:
        """Find code by what it DOES, described in plain English.

        Use for behavior-based discovery when you don't know the symbol names,
        e.g. "where do we reject expired sessions?" or "retry a failed network
        request". Keep using regular grep for exact symbols, regexes and exhaustive
        reference lists -- this tool ranks, it does not enumerate.

        A local lexical shortlist of `candidates` snippets is judged by a
        relevance model; matches come back ranked with file/line ranges and a
        short excerpt. An empty result does not prove absence. Tests are
        labeled kind="test", not demoted.

        Args:
            ctx: The current agent run context.
            query: Plain-English description of the behavior to find.
            directory: Root to search (respects .gitignore; skips hidden files).
            glob: Optional ripgrep glob to restrict files, e.g. "*.py".
            limit: Max matches to return (1-100).
            candidates: Snippets to judge after the lexical shortlist (1-256).
                Defaults to 128. Higher = better recall, slower and costlier. Each
                judged snippet counts as one request against the run's request limit.
        """
        model = self._model
        attributes: dict[str, str] = {
            'smart_search.judge_model': model if isinstance(model, str) else f'{model.system}:{model.model_name}'
        }
        if ctx.trace_include_content:
            attributes['smart_search.query'] = query
        # The judge runs are not instrumented (their prompts carry source), so this span is what a trace shows.
        # Failures are left to the enclosing tool span, which redacts them by `include_content`: an exception
        # message here can carry the arguments or a judge's error body.
        with ctx.tracer.start_as_current_span(
            'smart search', attributes=attributes, record_exception=False, set_status_on_exception=False
        ) as span:
            try:
                result = await search_code(
                    ctx,
                    model,
                    query,
                    directory,
                    indexes=self._indexes,
                    glob=glob,
                    limit=limit,
                    candidates=candidates,
                    threshold=self._threshold,
                    concurrency=self._concurrency,
                    span=span,
                )
            except WorkspaceError as error:
                raise_tool_failure(error)
            except (ModelAPIError, UnexpectedModelBehavior) as error:
                # Report the judge's failure to the model, which can fall back to regular search.
                raise ToolFailed(f'`smart_search` failed: {type(error).__name__}: {error}') from error
            if span.is_recording():
                coverage = result.coverage
                span.set_attributes(
                    {
                        'smart_search.files': coverage.files,
                        'smart_search.skipped_files': coverage.skipped_files,
                        'smart_search.snippets': coverage.snippets,
                        'smart_search.evaluated': coverage.evaluated,
                        'smart_search.selection_complete': coverage.selection_complete,
                        'smart_search.matches': len(result.matches),
                        'smart_search.omitted_matches': result.omitted_matches,
                    }
                )
            return result
