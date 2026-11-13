"""The `search_work_log` and `read_work_log` tools, and the shapes they return.

The tools return `TypedDict`s, so CodeMode renders their return types for code that calls them.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from typing_extensions import TypedDict

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import RetryRequest, raise_retry
from pydantic_ai_harness._line_slice import check_window
from pydantic_ai_harness._work_log_kinds import WORK_LOG_READ_KIND, WORK_LOG_SEARCH_KIND
from pydantic_ai_harness.work_log._store import WorkLogEntryKind, WorkLogOutcome


class WorkLogMatch(TypedDict):
    """One `search_work_log` match: where the entry is, and an excerpt of it."""

    seq: int
    context: int
    kind: WorkLogEntryKind
    tool_name: str | None
    args: str | None
    """The call's arguments, clipped to about 200 characters."""
    excerpt: str
    """About 300 characters of the content, around the first query term it contains."""
    chars: int
    """The full content's length, to tell whether reading it needs more than one page."""


class WorkLogSearchResult(TypedDict):
    """What `search_work_log` returns."""

    total_entries: int
    """Entries in the whole log, before any filter."""
    matches: list[WorkLogMatch]
    capped: bool
    """Whether matches were left out to keep the result under the read size cap."""


class WorkLogRead(TypedDict):
    """What `read_work_log` returns: one entry, with a window of its content's lines."""

    seq: int
    context: int
    kind: WorkLogEntryKind
    tool_name: str | None
    tool_call_id: str | None
    parent_tool_call_id: str | None
    args: str | None
    outcome: WorkLogOutcome | None
    text: str
    """The selected lines of the content."""
    offset: int
    next_offset: int | None
    """The `offset` that reads on, or `None` when this window reached the end (the start, with `from_end`)."""
    total_lines: int
    """Lines matching `pattern`, or all lines without one."""


class WorkLogToolset(FunctionToolset[AgentDepsT]):
    """The work log's tools, each running through one of `WorkLog`'s durable operations."""

    def __init__(
        self,
        *,
        search: Callable[
            [RunContext[AgentDepsT], str | None, str | None, str | None, int], Awaitable[WorkLogSearchResult]
        ],
        read: Callable[
            [RunContext[AgentDepsT], int, int, int, bool, str | None], Awaitable[WorkLogRead | RetryRequest]
        ],
        tool_id: str,
    ) -> None:
        super().__init__(id=tool_id)
        self._search = search
        self._read = read
        self.add_function(self.search_work_log, tool_kind=WORK_LOG_SEARCH_KIND)
        self.add_function(self.read_work_log, tool_kind=WORK_LOG_READ_KIND)

    async def search_work_log(
        self,
        ctx: RunContext[AgentDepsT],
        query: str | None = None,
        tool_name: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
    ) -> WorkLogSearchResult:
        """Search the work log of this conversation.

        The log holds every tool call made so far with its full result, including calls made inside
        code, and your earlier replies. Without `query`, returns the most recent entries, newest
        first. With `query`, ranks entries by BM25 over the tool name, arguments and result: rare
        terms and exact words score higher. Each match shows an excerpt; read an entry in full by
        its `seq`.

        Args:
            ctx: The run context (supplied by the agent).
            query: Words to search for. Omit it to list the most recent entries.
            tool_name: Only entries for calls to this tool.
            run_id: Only entries from this agent run.
            limit: Maximum number of matches (>= 1).
        """
        if limit < 1:
            raise ModelRetry('`limit` must be >= 1.')
        return await self._search(ctx, query, tool_name, run_id, limit)

    async def read_work_log(
        self,
        ctx: RunContext[AgentDepsT],
        seq: int,
        offset: int = 0,
        limit: int = 200,
        from_end: bool = False,
        pattern: str | None = None,
    ) -> WorkLogRead:
        """Read one work log entry in full: the call, its arguments and outcome, and its result by lines.

        Args:
            ctx: The run context (supplied by the agent).
            seq: The entry's `seq`, from a search match.
            offset: Number of matching lines to skip from the start (or end). Must be >= 0.
            limit: Maximum number of lines to return (>= 1; clamped to a built-in cap).
            from_end: Count `offset`/`limit` from the end of the result.
            pattern: Optional literal substring; only lines containing it are returned.
        """
        limit = check_window(offset, limit)
        return raise_retry(await self._read(ctx, seq, offset, limit, from_end, pattern))
