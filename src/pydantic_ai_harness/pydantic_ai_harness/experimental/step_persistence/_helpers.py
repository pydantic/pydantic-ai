"""Helpers for continuation, forking, provider-validity checks, and tool-effect annotation."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.experimental.step_persistence._context import current_run_id
from pydantic_ai_harness.experimental.step_persistence._store import StepStore
from pydantic_ai_harness.experimental.step_persistence._types import ToolEffectRecord


def is_provider_valid(messages: list[ModelMessage]) -> bool:
    """Return True when `messages` can be safely passed to `Agent.run(message_history=...)`.

    A history is provider-valid when:

    1. Every `ToolCallPart` has a matching `ToolReturnPart` or
       tool-bound `RetryPromptPart` later in the conversation, and
    2. Every tool return / tool-bound retry resolves a currently-open
       tool call (no orphans, duplicates, or out-of-order returns).

    A `RetryPromptPart` with `tool_name is None` is an output-validation
    retry -- providers map it as a regular user message, not a tool result,
    so it does not need to resolve an open call.
    """
    open_calls: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    open_calls.add(part.tool_call_id)
        else:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    if part.tool_call_id not in open_calls:
                        return False
                    open_calls.discard(part.tool_call_id)
                elif isinstance(part, RetryPromptPart) and part.tool_name is not None:
                    if part.tool_call_id not in open_calls:
                        return False
                    open_calls.discard(part.tool_call_id)
    return not open_calls


async def continue_run(store: StepStore, *, run_id: str) -> list[ModelMessage]:
    """Load the latest continuable snapshot for `run_id` as a message history.

    Pass the return value to `Agent.run(message_history=...)` to continue
    a delegate's prior investigation instead of starting fresh.

    Raises `LookupError` if no continuable snapshot exists for `run_id` -- the
    run may have crashed mid-tool-call, in which case there is event-log data
    but no safe resume point.
    """
    snapshot = await store.latest_snapshot(run_id=run_id)
    if snapshot is None:
        raise LookupError(f'no continuable snapshot for run_id {run_id!r}')
    return list(snapshot.messages)


async def fork_run(store: StepStore, *, run_id: str) -> list[ModelMessage]:
    """Return a copy of the latest snapshot's messages, intended for a new logical run.

    Semantically identical to `continue_run` at the data layer; the
    distinction is in how the caller treats the returned history (new
    `run_id`, new lineage entry, branching off prior context).
    """
    return await continue_run(store, run_id=run_id)


async def annotate_tool_effect(
    store: StepStore,
    ctx: RunContext[Any],
    *,
    idempotency_key: str | None = None,
    effect_summary: str | None = None,
) -> None:
    """Attach `idempotency_key` and / or `effect_summary` to the in-flight tool's effect record.

    Call from inside a tool body when the tool writes external state
    (artifacts, labels, PRs, network mutations) so an orchestrator
    inspecting `list_unresolved_tool_effects` after a crash can tell
    whether replay is safe.

    Resolves the active run from the `StepPersistence` `ContextVar` and
    the tool identity from `ctx.tool_call_id` / `ctx.tool_name`. No-op
    when called outside a step-persistence-wrapped tool call. The
    capability's `after_tool_execute` preserves these fields when it
    writes the terminal `completed` / `failed` record.
    """
    run_id = current_run_id.get()
    tool_call_id = ctx.tool_call_id
    tool_name = ctx.tool_name
    if run_id is None or tool_call_id is None or tool_name is None:
        return
    prior = await store.get_tool_effect(run_id=run_id, tool_call_id=tool_call_id)
    if prior is None:
        return
    await store.record_tool_effect(
        ToolEffectRecord(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=run_id,
            status=prior.status,
            started_at=prior.started_at,
            ended_at=prior.ended_at,
            idempotency_key=idempotency_key if idempotency_key is not None else prior.idempotency_key,
            effect_summary=effect_summary if effect_summary is not None else prior.effect_summary,
        )
    )
