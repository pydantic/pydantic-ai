"""The `WorkLog` capability: a queryable record of every tool call with its full result."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic_core import PydanticSerializationError

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
    durable_operation,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import (
    CachePoint,
    ModelResponse,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturn,
    is_multi_modal_content,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness._durable import RetryRequest
from pydantic_ai_harness._history_bm25 import Bm25Params, bm25_rank, clip, tokenize
from pydantic_ai_harness._line_slice import MAX_READ_CHARS, ReadUnit, slice_bytes, slice_lines
from pydantic_ai_harness._typeguards import is_object_list
from pydantic_ai_harness._work_log_kinds import WORK_LOG_READ_KIND, WORK_LOG_SEARCH_KIND
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.compaction._reset_compaction import reset_count
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits, indented_json
from pydantic_ai_harness.tool_output_limits._payload import to_text
from pydantic_ai_harness.work_log._store import (
    WorkLogEntry,
    WorkLogEntryKind,
    WorkLogOutcome,
    WorkLogRecord,
    WorkLogStore,
    WorkspaceWorkLogStore,
)
from pydantic_ai_harness.work_log._toolset import WorkLogMatch, WorkLogRead, WorkLogSearchResult, WorkLogToolset

_OWN_KINDS = frozenset({WORK_LOG_SEARCH_KIND.id, WORK_LOG_READ_KIND.id})
"""The work log's own tools, whose calls it does not record."""

_BM25 = Bm25Params(k1=1.5, b=0.75)
"""`ConversationSearch`'s default BM25 tuning."""

_ARGS_CHARS = 200

DEFAULT_MAX_SCANNED_ENTRIES = 2_000
"""How many of the newest matching entries one search scores by default: `ConversationSearch`'s `max_scanned_messages`."""
_EXCERPT_CHARS = 300


@dataclass
class WorkLog(AbstractCapability[AgentDepsT]):
    """Record every tool call with its full result, and model replies, in a log the model can search.

    Each conversation has one log, keyed by its `conversation_id`, so a run continued from earlier
    history appends to the same log. Tool results are recorded before `ToolOutputLimits` reduces
    them. Calls made from CodeMode's `run_code` are recorded too, each with `run_code`'s call id as
    its `parent_tool_call_id`. The model reads the log with `search_work_log` and `read_work_log`,
    as tools or from code:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.work_log import FileWorkLogStore, WorkLog

    agent = Agent('openai:gpt-5', capabilities=[WorkLog(FileWorkLogStore('work-logs'))])
    ```

    `WorkLog` adds no instructions: the tool descriptions say what the log holds.
    """

    store: WorkLogStore | WorkspaceWorkLogStore
    """Where the log is kept: an `InMemoryWorkLogStore`, which lasts as long as the process, a
    `FileWorkLogStore`, a `WorkspaceWorkLogStore`, or your own `WorkLogStore`."""

    record_text: bool = True
    """Record the model's text replies."""

    record_thinking: bool = False
    """Record the model's thinking."""

    _: KW_ONLY
    max_scanned_entries: int = DEFAULT_MAX_SCANNED_ENTRIES
    """How many of the newest matching entries one search with a `query` scores. Must be positive.

    Ranking runs on the event loop, so this bounds how long one search can hold it. When more
    entries match the filters, older ones are not searched and the result's `unsearched_entries`
    says how many. Raise it to search further back at the cost of slower searches.
    """

    id: str | None = 'work_log'
    """Stable capability ID: durable execution names the log's operations by it, and it is the id of the toolset
    holding `search_work_log` and `read_work_log`.

    Two `WorkLog`s sharing this `id` are one log stated twice when they agree, and an error when
    they differ.
    """

    def __post_init__(self) -> None:
        if self.max_scanned_entries < 1:
            raise ValueError(f'max_scanned_entries must be positive, got {self.max_scanned_entries!r}.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one log stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: `store` is a live object, and a store's spec form is not designed yet."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        # Inside `ToolOutputLimits`: inner `after_tool_execute` hooks run first, so the log sees a
        # result before it is reduced.
        return CapabilityOrdering(wrapped_by=[ToolOutputLimits])

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when the log belongs in the run's workspace and none is attached."""
        store = self.store
        if isinstance(store, WorkspaceWorkLogStore) and store.workspace is None:
            require_workspace(ctx.workspace, 'WorkLog', ctx.messages)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide `search_work_log` and `read_work_log`, each running through a durable operation."""
        return WorkLogToolset[AgentDepsT](search=self._search, read=self._read, toolset_id=self.id)

    # --- recording ---

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Record a tool's result as the model sees it, before an outer capability reduces it."""
        if tool_def.tool_kind not in _OWN_KINDS:
            await self._append(ctx, _tool_record(ctx, call, 'success', _result_text(result)))
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Record a tool call that failed or asked for a retry, then re-raise.

        Failures skip `after_tool_execute`, so they are recorded here. A deferral or a skipped call
        has no result yet; its later execution is recorded.
        """
        try:
            return await handler(args)
        except (CallDeferred, ApprovalRequired, SkipToolExecution):
            raise
        except Exception as error:
            if tool_def.tool_kind not in _OWN_KINDS:
                if isinstance(error, (ModelRetry, ToolRetryError)):
                    record = _tool_record(ctx, call, 'retried', str(error))
                elif isinstance(error, ToolFailedError):
                    # Core wraps a tool's `ToolFailed` before it reaches a wrap hook; name what the tool raised.
                    record = _tool_record(ctx, call, 'failed', f'ToolFailed: {error}')
                else:
                    # The type only, as `ReportToolErrors` reports it: the model never saw this message, which can
                    # carry file paths, hostnames or connection strings.
                    record = _tool_record(ctx, call, 'failed', type(error).__name__)
                await self._append(ctx, record)
            raise

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Record the response's text, and its thinking when `record_thinking` is set."""
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart) and self.record_text:
                kind: WorkLogEntryKind = 'text'
            elif isinstance(part, ThinkingPart) and self.record_thinking:
                kind = 'thinking'
            else:
                continue
            key = f'{ctx.run_id}:{ctx.run_step}:response:{index}'
            await self._append(ctx, _record(ctx, key=key, kind=kind, content=part.content))
        return response

    # --- durable operations ---

    def _target(self, ctx: RunContext[AgentDepsT]) -> tuple[WorkLogStore, str]:
        """The store, bound to the run's workspace when it keeps logs there, and this conversation's log id."""
        log_id = ctx.conversation_id
        assert log_id is not None, 'every run resolves a conversation id at its start'
        store = self.store
        return (store.bind(ctx.workspace) if isinstance(store, WorkspaceWorkLogStore) else store), log_id

    # One operation per recorded part, so a replay or a retried unit does not write again; the
    # record's key makes a retried write inside a durable unit, such as `run_code`'s, write once.
    @durable_operation('append')
    async def _append(self, ctx: RunContext[AgentDepsT], record: WorkLogRecord) -> int:
        store, log_id = self._target(ctx)
        return await store.append(log_id, record)

    @durable_operation('search')
    async def _search(
        self, ctx: RunContext[AgentDepsT], query: str | None, tool_name: str | None, run_id: str | None, limit: int
    ) -> WorkLogSearchResult:
        store, log_id = self._target(ctx)
        entries = await store.read(log_id)
        candidates = [
            entry
            for entry in entries
            if (tool_name is None or entry['tool_name'] == tool_name) and (run_id is None or entry['run_id'] == run_id)
        ]
        # Listing the newest entries ranks nothing, so only a search with a query is cut to the newest.
        unsearched = max(0, len(candidates) - self.max_scanned_entries) if query and query.strip() else 0
        ranked = _ranked(candidates[unsearched:], query)
        matches, capped = _matches(ranked[:limit], frozenset(tokenize(query or '')))
        return WorkLogSearchResult(
            total_entries=len(entries), unsearched_entries=unsearched, matches=matches, capped=capped
        )

    @durable_operation('read')
    async def _read(
        self,
        ctx: RunContext[AgentDepsT],
        seq: int,
        offset: int,
        limit: int,
        from_end: bool,
        pattern: str | None,
        unit: ReadUnit,
    ) -> WorkLogRead | RetryRequest:
        store, log_id = self._target(ctx)
        found = await store.read(log_id, seq, seq + 1)
        if not found:
            count = len(await store.read(log_id))
            valid = f'entries run from 1 to {count}' if count else 'the work log is empty'
            return RetryRequest(f'No work log entry has `seq` {seq}: {valid}.')
        entry = found[0]
        if unit == 'bytes':
            chunk = slice_bytes(entry['content'], offset=offset, limit=limit, from_end=from_end)
            reached = chunk.start == 0 if from_end else chunk.end >= chunk.total
            text, total, capped, cut_at = chunk.text, chunk.total, False, None
            next_offset = None if reached else offset + chunk.end - chunk.start
        else:
            window = slice_lines(
                entry['content'], offset=offset, limit=limit, from_end=from_end, pattern=pattern, cap_keeps_end=from_end
            )
            # A capped window reads on from its first line not returned whole. A line the cap cut is passed over:
            # `cut_at` says where a `unit='bytes'` read returns the rest of it.
            end = offset + (max(window.whole, 1) if window.capped else window.shown)
            text, total, capped, cut_at = window.text, window.total, window.capped, window.cut_at
            next_offset = end if end < window.total else None
        return WorkLogRead(
            seq=entry['seq'],
            context=entry['context'],
            kind=entry['kind'],
            tool_name=entry['tool_name'],
            tool_call_id=entry['tool_call_id'],
            parent_tool_call_id=entry['parent_tool_call_id'],
            args=entry['args'],
            outcome=entry['outcome'],
            text=text,
            unit=unit,
            offset=offset,
            next_offset=next_offset,
            total=total,
            capped=capped,
            cut_at=cut_at,
        )


def _record(
    ctx: RunContext[Any],
    *,
    key: str,
    kind: WorkLogEntryKind,
    content: str,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    parent_tool_call_id: str | None = None,
    args: str | None = None,
    outcome: WorkLogOutcome | None = None,
) -> WorkLogRecord:
    """Build a record, numbering its context from the latest reset notice in the run's messages."""
    return WorkLogRecord(
        key=key,
        run_id=ctx.run_id,
        step=ctx.run_step,
        context=reset_count(ctx.messages),
        kind=kind,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        parent_tool_call_id=parent_tool_call_id,
        args=args,
        outcome=outcome,
        content=content,
        chars=len(content),
        timestamp=datetime.now(UTC).isoformat(),
    )


def _tool_record(ctx: RunContext[Any], call: ToolCallPart, outcome: WorkLogOutcome, content: str) -> WorkLogRecord:
    # The step is in the key because models reuse tool call ids across turns, the tool name because `ToolBridge`
    # runs a bridged tool under its `tool_call`'s id, and the outcome because an outer `after_tool_execute` can
    # reject a result already recorded as a success, which `wrap_tool_execute` then records as retried or failed.
    # A nested `ToolManager`, such as `run_code`'s, carries the context of the call that dispatched it;
    # a top-level call's manager carries the step's, which has no tool call id.
    dispatcher = ctx.tool_manager.ctx if ctx.tool_manager is not None else None
    return _record(
        ctx,
        key=f'{ctx.run_id}:{ctx.run_step}:{call.tool_call_id}:{call.tool_name}:{ctx.retry}:{outcome}',
        kind='tool',
        content=content,
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        parent_tool_call_id=dispatcher.tool_call_id if dispatcher is not None else None,
        args=call.args_as_json_str(),
        outcome=outcome,
    )


def _ranked(entries: list[WorkLogEntry], query: str | None) -> list[WorkLogEntry]:
    """`entries` by BM25 relevance to `query` over tool name, arguments and content, or newest first without one."""
    if query is None or not query.strip():
        return entries[::-1]
    documents = [
        ' '.join(text for text in (entry['tool_name'], entry['args'], entry['content']) if text) for entry in entries
    ]
    return [entries[index] for index, _ in bm25_rank(query, documents, _BM25)]


def _matches(entries: list[WorkLogEntry], terms: frozenset[str]) -> tuple[list[WorkLogMatch], bool]:
    """Excerpt each entry around `terms`, stopping before the matches pass `MAX_READ_CHARS` as JSON.

    Also returns whether it stopped early.
    """
    matches: list[WorkLogMatch] = []
    size = 0
    for entry in entries:
        args = entry['args']
        match = WorkLogMatch(
            seq=entry['seq'],
            context=entry['context'],
            kind=entry['kind'],
            tool_name=entry['tool_name'],
            args=None if args is None else clip(args, _ARGS_CHARS, terms),
            excerpt=clip(entry['content'], _EXCERPT_CHARS, terms),
            chars=entry['chars'],
        )
        size += len(json.dumps(match, ensure_ascii=False))
        if size > MAX_READ_CHARS:
            return matches, True
        matches.append(match)
    return matches, False


def _result_text(result: object) -> str:
    """A tool result as the model sees it: the value as text, then a `ToolReturn`'s `content` when set.

    Files are named by media type rather than logged.
    """
    value: object = result.return_value if isinstance(result, ToolReturn) else result
    items: list[object] = (
        list(value) if is_object_list(value) and any(is_multi_modal_content(item) for item in value) else [value]
    )
    if isinstance(result, ToolReturn) and (content := result.content) is not None:
        # The model gets `content` as a user message after the tool result, so it follows the value.
        items.extend([content] if isinstance(content, str) else content)
    return '\n'.join(_item_text(item) for item in items if not isinstance(item, CachePoint))


def _item_text(item: object) -> str:
    """One item of a tool result as text; kept apart from `_result_text` to hold it under the complexity limit."""
    if isinstance(item, str):
        return item
    if isinstance(item, TextContent):
        return item.content
    if not is_multi_modal_content(item):
        # One field per line, so `read_work_log` pages a large structured result by line.
        try:
            return indented_json(item)
        except PydanticSerializationError:
            # A leaf with no JSON form, such as a type: `to_text` renders it as its `repr`.
            return to_text(item)
    try:
        media_type = item.media_type
    except ValueError:
        # A URL whose media type cannot be inferred from it.
        media_type = item.kind
    return f'[{media_type} content not logged]'
