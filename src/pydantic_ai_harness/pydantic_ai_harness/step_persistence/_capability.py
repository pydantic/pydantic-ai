"""StepPersistence capability: append-only event log + continuable snapshots."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic_ai import CallToolsNode, ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from pydantic_ai_harness.step_persistence._context import (
    current_run_id,
    live_run_history,
    snapshot_saved,
)
from pydantic_ai_harness.step_persistence._helpers import is_provider_valid
from pydantic_ai_harness.step_persistence._store import InMemoryStepStore, StepStore
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    SnapshotState,
    StepEvent,
    ToolEffectRecord,
)


def _empty_metadata() -> dict[str, str]:
    return {}


def _has_model_response(messages: list[ModelMessage]) -> bool:
    """A history worth rescuing as a resume point on error.

    A history without a model response is equivalent to restarting the run,
    so it is not worth persisting.
    """
    return any(isinstance(message, ModelResponse) for message in messages)


@dataclass
class StepPersistence(AbstractCapability[AgentDepsT]):
    """Append-only step log + continuable snapshots + tool-effect ledger.

    The capability emits a `StepEvent` at every interesting boundary
    (run/model-request/tool-call start, completion, failure), records a
    `ToolEffectRecord` per tool call so the orchestrator can decide whether
    replay is safe, and saves a `ContinuableSnapshot` at every settled
    `CallToolsNode` boundary -- folding in the pending tool-return request, so
    the point is durable the moment the tool completes -- plus a fallback save
    at `after_run` when the run ends past that boundary. A run that *fails*
    saves the live at-failure history (see `on_run_error`), classified by its
    tool-work state: `complete` when every tool call is resolved,
    `interrupted` otherwise.

    A run that crashes between `before_tool_execute` and `after_tool_execute`
    leaves a visible event trail, a `started` tool-effect record (the
    `unknown_after_crash` signal), and an `interrupted` snapshot carrying
    every completed cycle. The default `latest_snapshot` / `continue_run`
    read path only returns `complete` snapshots; pass
    `include_interrupted=True` to resume from the interrupted frontier after
    consulting `list_unresolved_tool_effects`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.step_persistence import StepPersistence, InMemoryStepStore

    store = InMemoryStepStore()
    librarian = Agent(
        'openai:gpt-5',
        capabilities=[StepPersistence(store=store, agent_name='code_librarian')],
    )
    await librarian.run('Find ThinkingPartDelta and confirm the callable allowance')
    ```

    Use `continue_run(store, run_id=...)` / `fork_run(store, run_id=...)`
    to load a prior snapshot, then pass the result to
    `Agent.run(..., message_history=...)`.
    """

    # Override the inherited default ID because durable-operation recovery needs a stable identity.
    id: str | None = field(default='step_persistence', kw_only=True)

    store: StepStore = field(default_factory=InMemoryStepStore)
    """Backend that records events, snapshots, and tool effects."""

    agent_name: str | None = None
    """Logical agent name (e.g. `code_librarian`, `reproducer`).

    Used as a stable prefix for the context-derived `run_id` so store
    inspection identifies both the agent and the durable run.
    """

    run_id: str | None = None
    """Identifier for this one `Agent.run` call.

    `run_id` is per-call, matching `pydantic_ai.RunContext.run_id`. For
    multi-turn logical grouping use `conversation_id` on `Agent.run(...)` --
    that is the pyai-native primitive for it.

    Resolution order (materialised in `for_run`):

    1. **Explicit value** → used as-is. Single-shot use cases:
       deterministic id for testing, replay, debugging. Reusing the
       capability across multiple `.run()` calls with the same explicit
       `run_id` raises `ValueError` in `before_run` -- the tool-effect
       ledger keys on `(run_id, tool_call_id)` and providers reuse
       deterministic tool-call ids, so a silent collision would erase
       the `unknown_after_crash` signal. Use `conversation_id=` on
       `Agent.run` for multi-turn grouping.
    2. **`agent_name` set, `run_id` unset** -> path-safe encoding of both values.
       Reusing the capability instance yields distinct ids because the agent
       graph assigns each run its own id.
    3. **Neither set** -> `ctx.run_id` per `.run()`.
    """

    parent_run_id: str | None = None
    """Run that spawned this one.

    Auto-inferred from the enclosing `StepPersistence` `wrap_run` scope --
    when an orchestrator's tool synchronously calls a delegate's
    `Agent.run(...)`, the delegate picks up the orchestrator's `run_id`
    here without manual threading. Set explicitly to override (e.g. for
    cross-process delegation where `ContextVar`s do not propagate).
    """

    metadata: dict[str, str] = field(default_factory=_empty_metadata)
    """Free-form metadata stored on the `RunRecord` and on each event."""

    _event_sequence: int = field(default=0, init=False, repr=False)
    _snapshot_sequence: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> StepPersistence[Any]:
        """Construct from a serialised spec.

        Supports `backend='memory'` (default), `backend='file'` (with
        `directory`), or `backend='sqlite'` (with `database`). Raises
        `ValueError` for any other `backend` value -- silently falling
        back to in-memory storage would turn a typo into accidental
        non-durability.

        `max_snapshots_per_run` (default `None`, unbounded) is forwarded to
        the constructed store to bound per-run snapshot growth.
        """
        backend = kwargs.pop('backend', 'memory')
        max_snapshots_per_run = kwargs.pop('max_snapshots_per_run', None)
        if backend == 'memory':
            return cls(store=InMemoryStepStore(max_snapshots_per_run=max_snapshots_per_run), **kwargs)
        if backend == 'file':
            from pydantic_ai_harness.step_persistence._store import FileStepStore

            directory = kwargs.pop('directory', '.step-persistence')
            return cls(store=FileStepStore(directory, max_snapshots_per_run=max_snapshots_per_run), **kwargs)
        if backend == 'sqlite':
            from pydantic_ai_harness.step_persistence._store import SqliteStepStore

            database = kwargs.pop('database', '.step-persistence.db')
            return cls(store=SqliteStepStore(database=database, max_snapshots_per_run=max_snapshots_per_run), **kwargs)
        raise ValueError(f'unknown backend {backend!r}; expected `memory`, `file`, or `sqlite`')

    def compaction_transcript_handle(self) -> str | None:
        """Retrieval handle to this run's transcript, for compaction receipts.

        Satisfies the compaction `TranscriptHandleProvider` protocol structurally (no import
        coupling). A compaction strategy discovers this capability via `RunContext.capabilities`
        and records its run id. Returns `None` before `for_run` has materialised the id.
        """
        return self.run_id

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Materialise `run_id` and `parent_run_id` for this `Agent.run` call.

        Reads the contextvar set by any enclosing `StepPersistence.wrap_run`
        before the local run overwrites it, so a delegate's `parent_run_id`
        ends up pointing at its orchestrator's `run_id`.

        A separate `ContextVar` is needed because pydantic_ai's own
        cross-run signals (`RUN_ID_BAGGAGE_KEY` via OTel baggage,
        `RunContext.run_id`, and `_CURRENT_RUN_CONTEXT`) are single-slot:
        the inner `Instrumentation.wrap_run` overwrites them before any
        nested capability sees the parent. The harness-local contextvar
        lets us snapshot the parent here, *before* the local `wrap_run`
        rebinds it.
        """
        inferred_parent = self.parent_run_id if self.parent_run_id is not None else current_run_id.get()
        resolved_run_id = self.run_id or self._derive_run_id(ctx)
        return replace(self, run_id=resolved_run_id, parent_run_id=inferred_parent)

    def _derive_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if ctx.run_id is None:
            raise RuntimeError('StepPersistence needs the agent graph to assign `RunContext.run_id`.')
        if self.agent_name is not None:
            agent_name_bytes = self.agent_name.encode()
            payload = f'{len(agent_name_bytes)}:{self.agent_name}{ctx.run_id}'.encode()
            encoded = base64.urlsafe_b64encode(payload).decode().rstrip('=')
            derived = f'sp-{encoded}'
            if len(derived) > 200:
                raise ValueError("derived StepPersistence run_id exceeds FileStepStore's 200-character limit")
            return derived
        return ctx.run_id

    def _effective_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if self.run_id is not None:
            return self.run_id
        if ctx.run_id is not None:
            return ctx.run_id
        raise RuntimeError('StepPersistence run id was not materialized by `for_run`.')

    @durable_operation('register_run')
    async def _register_run(
        self,
        *,
        run_id: str,
        conversation_id: str | None,
        parent_run_id: str | None,
        agent_name: str | None,
        metadata: dict[str, str],
        registration_id: str,
    ) -> None:
        existing = await self.store.get_run(run_id=run_id)
        if existing is not None and existing.registration_id == registration_id:
            return
        if existing is not None:
            raise ValueError(
                f'StepPersistence: run_id {run_id!r} is already in the store. '
                '`run_id` is single-shot; pass `conversation_id=` to '
                '`Agent.run` for multi-turn grouping instead.'
            )
        await self.store.register_run(
            RunRecord(
                run_id=run_id,
                conversation_id=conversation_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                metadata=metadata,
                registration_id=registration_id,
            )
        )

    @durable_operation('registration_id')
    async def _registration_id(self) -> str:
        return str(uuid4())

    @durable_operation('append_event')
    async def _append_event(
        self,
        *,
        run_id: str,
        kind: EventKind,
        step_index: int,
        conversation_id: str | None,
        parent_run_id: str | None,
        agent_name: str | None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
        metadata: dict[str, str],
        event_index: int,
    ) -> None:
        await self.store.append_event(
            StepEvent(
                run_id=run_id,
                kind=kind,
                step_index=step_index,
                timestamp=datetime.now(timezone.utc),
                conversation_id=conversation_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=error,
                metadata=metadata,
                # Hook order is deterministic under replay. The sequence distinguishes repeated
                # events within one step, while the other fields keep the key inspectable.
                idempotency_key=f'{event_index}:{step_index}:{kind}:{tool_call_id or ""}',
            )
        )

    async def _record_event(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        kind: EventKind,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> None:
        event_index = self._event_sequence
        self._event_sequence += 1
        await self._append_event(
            run_id=self._effective_run_id(ctx),
            kind=kind,
            step_index=ctx.run_step,
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            error=error,
            metadata=dict(self.metadata),
            event_index=event_index,
        )

    @durable_operation('save_snapshot')
    async def _persist_snapshot(
        self,
        *,
        run_id: str,
        step_index: int,
        messages: list[ModelMessage],
        conversation_id: str | None,
        parent_run_id: str | None,
        agent_name: str | None,
        state: SnapshotState = 'complete',
        snapshot_index: int,
    ) -> None:
        await self.store.save_snapshot(
            ContinuableSnapshot(
                run_id=run_id,
                step_index=step_index,
                messages=messages,
                conversation_id=conversation_id,
                parent_run_id=parent_run_id,
                agent_name=agent_name,
                timestamp=datetime.now(timezone.utc),
                state=state,
                # The store scopes keys by run id. Within that scope the sequence distinguishes
                # separate save calls, step_index identifies the graph position, and state keeps
                # settled and interrupted histories distinct. `for_run` creates the capability
                # instance that owns this zero-based counter, so each run starts at zero. Replay
                # executes the same saves in order and regenerates the claimed key; a new save
                # advances to a key no earlier write has used.
                idempotency_key=f'{snapshot_index}:{step_index}:{state}',
            )
        )

    async def _save_snapshot(
        self,
        *,
        run_id: str,
        step_index: int,
        messages: list[ModelMessage],
        conversation_id: str | None,
        parent_run_id: str | None,
        agent_name: str | None,
        state: SnapshotState = 'complete',
    ) -> None:
        snapshot_index = self._snapshot_sequence
        self._snapshot_sequence += 1
        await self._persist_snapshot(
            run_id=run_id,
            step_index=step_index,
            messages=messages,
            conversation_id=conversation_id,
            parent_run_id=parent_run_id,
            agent_name=agent_name,
            state=state,
            snapshot_index=snapshot_index,
        )

    @durable_operation('record_tool_effect')
    async def _record_tool_effect(self, run_id: str, tool_call_id: str, tool_name: str) -> None:
        await self.store.record_tool_effect(
            ToolEffectRecord(tool_call_id=tool_call_id, tool_name=tool_name, run_id=run_id, status='started')
        )

    @durable_operation('finish_tool_effect')
    async def _finish_tool_effect(
        self,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        status: Literal['completed', 'failed'],
        error: str | None = None,
    ) -> None:
        prior = await self.store.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)
        now = datetime.now(timezone.utc)
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
                status=status,
                started_at=prior.started_at if prior is not None else now,
                ended_at=now,
                idempotency_key=prior.idempotency_key if prior is not None else None,
                effect_summary=prior.effect_summary
                if prior is not None and prior.effect_summary is not None
                else error,
            )
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Push this run's id onto the contextvar so nested delegates can read it."""
        token = current_run_id.set(self._effective_run_id(ctx))
        saved_token = snapshot_saved.set(0)
        history_token = live_run_history.set(None)
        try:
            return await handler()
        finally:
            live_run_history.reset(history_token)
            snapshot_saved.reset(saved_token)
            current_run_id.reset(token)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Register run lineage and emit `run_started`.

        Reject reuse by a distinct registration -- the
        tool-effect ledger keys on `(run_id, tool_call_id)` and providers
        reuse deterministic tool-call ids, so a second `Agent.run` with
        the same `run_id` would silently collide. A journaled registration id
        distinguishes a retry of this durable operation from a new run.
        """
        run_id = self._effective_run_id(ctx)
        registration_id = await self._registration_id()
        await self._register_run(
            run_id=run_id,
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            metadata=dict(self.metadata),
            registration_id=registration_id,
        )
        await self._record_event(ctx, kind='run_started')

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Emit `run_completed`, saving a final snapshot only as a fallback.

        When a terminal `CallToolsNode` already saved the final history via
        `after_node_run` it carries the correct `step_index`, whereas by
        `after_run` `ctx.run_step` is reset to 0 -- so re-saving would both
        duplicate the tail and stamp a misleading `step_index`. We save only
        when the run ended past the newest boundary snapshot.

        That covers a run which reached no provider-valid boundary at all, and
        `Agent.run_stream`, which ends through `SetFinalResult` rather than a
        terminal `CallToolsNode` and appends its closing response after the last
        boundary -- leaving `after_run` the only hook that sees the full run.
        """
        messages = result.all_messages()
        if len(messages) > snapshot_saved.get():
            if is_provider_valid(messages):
                await self._save_snapshot(
                    run_id=self._effective_run_id(ctx),
                    step_index=ctx.run_step,
                    messages=list(messages),
                    conversation_id=ctx.conversation_id,
                    parent_run_id=self.parent_run_id,
                    agent_name=self.agent_name,
                )
        await self._record_event(ctx, kind='run_completed')
        return result

    def _stash_live_history(self, ctx: RunContext[AgentDepsT]) -> None:
        """Hold the live message list by reference so `on_run_error` can read the at-failure history.

        `on_run_error` cannot read the live history itself: its `RunContext`
        holds the start-of-run list, which `UserPromptNode.run` replaces with
        the run's working list. This leans on a pydantic-ai core invariant:
        that rebind happens exactly once, and every later change to
        `ctx.state.message_history` is an in-place mutation (`append` /
        `[:]=` -- the discipline `capture_run_messages` requires, stated in
        `pydantic_ai._agent_graph`). If core ever rebinds the list mid-run
        again, this stash silently goes stale and the error-path snapshot
        persists a pre-failure history while believing it is the at-failure
        one -- `tests/step_persistence` pins the invariant so that surfaces
        as a test failure, not silent data loss.

        Re-stashed at every boundary because the first `after_node_run`
        (the `UserPromptNode` boundary) still sees the pre-rebind list; from
        the next boundary on, the stashed reference is the live list. The
        stashed `run_step` is the last completed boundary's, so an error
        snapshot's `step_index` can lag the failing request by one.
        """
        live_run_history.set((ctx.messages, ctx.run_step))

    async def _save_continuable_snapshot(
        self,
        ctx: RunContext[AgentDepsT],
        messages: list[ModelMessage],
        step_index: int,
        state: SnapshotState = 'complete',
    ) -> None:
        await self._save_snapshot(
            run_id=self._effective_run_id(ctx),
            step_index=step_index,
            messages=messages,
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            state=state,
        )

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Persist the live at-failure history as the run's last resume point, then emit `run_failed`.

        The single error-path save site: reads the list reference stashed by
        `after_node_run` (see `_stash_live_history`), whose content at this
        point is the full history the run had built when it failed -- including
        a failing model request's payload and any partial tool returns captured
        by the graph during unwind. Nothing is compared against the store:
        the live history is by definition the newest state, so an earlier
        boundary snapshot is simply superseded, and a history a sticky
        processor trimmed is persisted as trimmed -- exactly what the next
        request would have sent.

        The history is saved whenever it contains a model response (a bare
        prompt equals restarting the run), classified `complete` when every
        tool call is resolved and `interrupted` otherwise. Interrupted
        snapshots stay off the default `latest_snapshot` read path.
        """
        stashed = live_run_history.get()
        if stashed is not None:
            messages, step_index = stashed
            captured = list(messages)
            if _has_model_response(captured):
                state: SnapshotState = 'complete' if is_provider_valid(captured) else 'interrupted'
                await self._save_continuable_snapshot(ctx, captured, step_index, state)
        await self._record_event(ctx, kind='run_failed', error=repr(error))
        raise error

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        await self._record_event(ctx, kind='model_request_started')
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        await self._record_event(ctx, kind='model_request_completed')
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        """Emit `model_request_failed` and re-raise.

        No snapshot is saved here: the failing request's payload already sits
        in the live history (the graph appends the request before sending), so
        `on_run_error`'s save covers it. A failure the model layer recovers
        from (retry, fallback) needs no rescue at all.
        """
        await self._record_event(ctx, kind='model_request_failed', error=repr(error))
        raise error

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = self._effective_run_id(ctx)
        await self._record_tool_effect(run_id, call.tool_call_id, tool_def.name)
        await self._record_event(
            ctx,
            kind='tool_call_started',
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
        )
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        run_id = self._effective_run_id(ctx)
        await self._finish_tool_effect(run_id, call.tool_call_id, tool_def.name, 'completed')
        await self._record_event(
            ctx,
            kind='tool_call_completed',
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
        )
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        run_id = self._effective_run_id(ctx)
        await self._finish_tool_effect(run_id, call.tool_call_id, tool_def.name, 'failed', repr(error))
        await self._record_event(
            ctx,
            kind='tool_call_failed',
            tool_call_id=call.tool_call_id,
            tool_name=tool_def.name,
            error=repr(error),
        )
        raise error

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Save a continuable snapshot after a settled `CallToolsNode`, and refresh the live-history stash.

        At that boundary every tool call from the preceding `ModelRequestNode`
        has a matching tool return, so the history is provider-valid. The
        returned `ModelRequestNode` carries those returns and is not yet in
        `ctx.messages`, so its request is folded in before validation --
        without it a worker killed right after a completed tool call would
        leave no resume point at all (#373). `is_provider_valid` doubles as a
        defense in case a custom node reshapes history, and the saved count
        goes to `snapshot_saved` so `after_run` can tell whether the run ended
        past this boundary.

        This save is the durable one: it lands in the store while the run is
        still healthy, so it survives a hard kill that fires no hook. The
        error path (`on_run_error`) only rescues histories that a raise unwinds
        through.

        Every node boundary also re-stashes the live message list so that
        `on_run_error` can persist the at-failure history when a later node
        raises before its own `after_node_run` fires. The stash holds that list
        by reference, so the snapshot candidate rebinds to a new list rather
        than appending to it -- an append would leak `result.request` into the
        history the error path later reads, duplicating it once the graph
        appends the request itself.
        """
        self._stash_live_history(ctx)
        messages = list(ctx.messages)
        if isinstance(node, CallToolsNode):
            if isinstance(result, ModelRequestNode):
                messages = [*messages, result.request]
            if is_provider_valid(messages):
                await self._save_snapshot(
                    run_id=self._effective_run_id(ctx),
                    step_index=ctx.run_step,
                    messages=messages,
                    conversation_id=ctx.conversation_id,
                    parent_run_id=self.parent_run_id,
                    agent_name=self.agent_name,
                )
                snapshot_saved.set(len(messages))
        return result
