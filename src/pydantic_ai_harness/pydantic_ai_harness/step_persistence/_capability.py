"""StepPersistence capability: append-only event log + continuable snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic_ai import CallToolsNode
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from pydantic_ai_harness.step_persistence._context import current_run_id, snapshot_saved
from pydantic_ai_harness.step_persistence._helpers import is_provider_valid
from pydantic_ai_harness.step_persistence._store import InMemoryStepStore, StepStore
from pydantic_ai_harness.step_persistence._types import (
    ContinuableSnapshot,
    EventKind,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)


def _empty_metadata() -> dict[str, str]:
    return {}


@dataclass
class StepPersistence(AbstractCapability[AgentDepsT]):
    """Append-only step log + continuable snapshots + tool-effect ledger.

    The capability emits a `StepEvent` at every interesting boundary
    (run/model-request/tool-call start, completion, failure), records a
    `ToolEffectRecord` per tool call so the orchestrator can decide whether
    replay is safe, and saves a `ContinuableSnapshot` at every
    provider-valid boundary -- the end of each `CallToolsNode` -- plus a
    fallback save at `after_run` if the run reached no such boundary.

    A run that crashes between `before_tool_execute` and `after_tool_execute`
    leaves a visible event trail and a `started` tool-effect record, but no
    new continuable snapshot -- the latest snapshot reflects the last
    provider-valid state.

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

    store: StepStore = field(default_factory=InMemoryStepStore)
    """Backend that records events, snapshots, and tool effects."""

    agent_name: str | None = None
    """Logical agent name (e.g. `code_librarian`, `reproducer`).

    Used as a stable prefix for the auto-derived `run_id` so store
    inspection shows readable IDs like `code_librarian-a3b2`.
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
    2. **`agent_name` set, `run_id` unset** → `{agent_name}-{short-uuid}`,
       freshly materialised per `.run()`. Reusing the capability instance
       yields distinct ids. Recommended default for delegate capabilities.
    3. **Neither set** → `ctx.run_id` per `.run()`, falling back to UUID4.
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

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> StepPersistence[Any]:
        """Construct from a serialised spec.

        Supports `backend='memory'` (default), `backend='file'` (with
        `directory`), or `backend='sqlite'` (with `database`). Raises
        `ValueError` for any other `backend` value -- silently falling
        back to in-memory storage would turn a typo into accidental
        non-durability.
        """
        backend = kwargs.pop('backend', 'memory')
        if backend == 'memory':
            return cls(store=InMemoryStepStore(), **kwargs)
        if backend == 'file':
            from pydantic_ai_harness.step_persistence._store import FileStepStore

            directory = kwargs.pop('directory', '.step-persistence')
            return cls(store=FileStepStore(directory), **kwargs)
        if backend == 'sqlite':
            from pydantic_ai_harness.step_persistence._store import SqliteStepStore

            database = kwargs.pop('database', '.step-persistence.db')
            return cls(store=SqliteStepStore(database=database), **kwargs)
        raise ValueError(f'unknown backend {backend!r}; expected `memory`, `file`, or `sqlite`')

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
        if resolved_run_id == self.run_id and inferred_parent == self.parent_run_id:
            return self
        return replace(self, run_id=resolved_run_id, parent_run_id=inferred_parent)

    def _derive_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if self.agent_name is not None:
            return f'{self.agent_name}-{uuid4().hex[:8]}'
        return ctx.run_id or str(uuid4())

    def _effective_run_id(self, ctx: RunContext[AgentDepsT]) -> str:
        if self.run_id is not None:
            return self.run_id
        return ctx.run_id or str(uuid4())

    def _make_event(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        kind: EventKind,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        error: str | None = None,
    ) -> StepEvent:
        return StepEvent(
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
        )

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Push this run's id onto the contextvar so nested delegates can read it."""
        token = current_run_id.set(self._effective_run_id(ctx))
        saved_token = snapshot_saved.set(False)
        try:
            return await handler()
        finally:
            snapshot_saved.reset(saved_token)
            current_run_id.reset(token)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Register run lineage and emit `run_started`.

        When the caller pinned an explicit `run_id`, reject reuse -- the
        tool-effect ledger keys on `(run_id, tool_call_id)` and providers
        reuse deterministic tool-call ids, so a second `Agent.run` with
        the same explicit `run_id` would silently collide. The auto-derived
        cases cannot trigger this check because each call materialises a
        fresh id in `for_run`.
        """
        run_id = self._effective_run_id(ctx)
        if self.run_id is not None and await self.store.get_run(run_id=run_id) is not None:
            raise ValueError(
                f'StepPersistence: run_id {run_id!r} is already in the store. '
                'Explicit `run_id` is single-shot; pass `conversation_id=` to '
                '`Agent.run` for multi-turn grouping instead.'
            )
        await self.store.register_run(
            RunRecord(
                run_id=run_id,
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                metadata=dict(self.metadata),
            )
        )
        await self.store.append_event(self._make_event(ctx, kind='run_started'))

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Emit `run_completed`, saving a final snapshot only as a fallback.

        The terminal `CallToolsNode` already saved the final provider-valid
        snapshot via `after_node_run`, carrying the correct `step_index`. By
        `after_run`, `ctx.run_step` is reset to 0, so re-saving here would both
        duplicate the tail and stamp a misleading `step_index`. We only save
        when the run produced no snapshot at all (no provider-valid node
        boundary was reached), as a last-resort capture of the final state.
        """
        if not snapshot_saved.get():
            messages = result.all_messages()
            if is_provider_valid(messages):
                await self.store.save_snapshot(
                    ContinuableSnapshot(
                        run_id=self._effective_run_id(ctx),
                        step_index=ctx.run_step,
                        messages=list(messages),
                        conversation_id=ctx.conversation_id,
                        parent_run_id=self.parent_run_id,
                        agent_name=self.agent_name,
                    )
                )
        await self.store.append_event(self._make_event(ctx, kind='run_completed'))
        return result

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Emit `run_failed` so a killed run leaves a visible event trail."""
        await self.store.append_event(self._make_event(ctx, kind='run_failed', error=repr(error)))
        raise error

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        await self.store.append_event(self._make_event(ctx, kind='model_request_started'))
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        await self.store.append_event(self._make_event(ctx, kind='model_request_completed'))
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        await self.store.append_event(self._make_event(ctx, kind='model_request_failed', error=repr(error)))
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
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='started',
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_started',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
            )
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
        prior = await self.store.get_tool_effect(run_id=run_id, tool_call_id=call.tool_call_id)
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='completed',
                started_at=prior.started_at if prior is not None else datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                idempotency_key=prior.idempotency_key if prior is not None else None,
                effect_summary=prior.effect_summary if prior is not None else None,
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_completed',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
            )
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
        prior = await self.store.get_tool_effect(run_id=run_id, tool_call_id=call.tool_call_id)
        prior_summary = prior.effect_summary if prior is not None else None
        await self.store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                run_id=run_id,
                status='failed',
                started_at=prior.started_at if prior is not None else datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                idempotency_key=prior.idempotency_key if prior is not None else None,
                effect_summary=prior_summary if prior_summary is not None else repr(error),
            )
        )
        await self.store.append_event(
            self._make_event(
                ctx,
                kind='tool_call_failed',
                tool_call_id=call.tool_call_id,
                tool_name=tool_def.name,
                error=repr(error),
            )
        )
        raise error

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Save a mid-run continuable snapshot after `CallToolsNode` succeeds.

        At that boundary every tool call from the preceding `ModelRequestNode`
        has a matching tool return, so the history is provider-valid.
        Snapshots are filtered through `is_provider_valid` defensively in case
        a custom node reshapes history.
        """
        if isinstance(node, CallToolsNode):
            messages = list(ctx.messages)
            if is_provider_valid(messages):
                await self.store.save_snapshot(
                    ContinuableSnapshot(
                        run_id=self._effective_run_id(ctx),
                        step_index=ctx.run_step,
                        messages=messages,
                        conversation_id=ctx.conversation_id,
                        parent_run_id=self.parent_run_id,
                        agent_name=self.agent_name,
                    )
                )
                snapshot_saved.set(True)
        return result
