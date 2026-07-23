"""Data types for step-event persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from pydantic_ai.messages import ModelMessage

EventKind = Literal[
    'run_started',
    'run_completed',
    'run_failed',
    'model_request_started',
    'model_request_completed',
    'model_request_failed',
    'tool_call_started',
    'tool_call_completed',
    'tool_call_failed',
]
"""Boundary that produced a `StepEvent`.

Choose `kind` from this set so consumers can route on event type without
string typos. Append-only: never mutate an emitted event; record corrections
as a follow-up event.
"""

ToolEffectStatus = Literal['started', 'completed', 'failed']
"""Lifecycle status of a tool call recorded in the effect ledger.

A `tool_call_id` whose latest record is `started` was in flight when the
process last wrote. Treat it as `unknown_after_crash` when replaying -- the
external side effect may or may not have happened.
"""

SnapshotState = Literal['complete', 'interrupted']
"""Whether a snapshot's tool work was settled when it was captured.

`complete`: every `ToolCallPart` has a matching result; resuming starts a
fresh model turn. `interrupted`: the history carries unsettled tool work (an
open tool call, or a partially-returned batch) captured at failure time.
pydantic-ai (>= 2.10) makes either shape sendable on resume, but an
`interrupted` point may re-execute pending tool calls or close them out with
synthesized returns -- check `list_unresolved_tool_effects` before resuming
from one. Vocabulary mirrors `pydantic_ai.messages.ModelRequestState`.
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _empty_str_dict() -> dict[str, str]:
    return {}


@dataclass(kw_only=True)
class StepEvent:
    """A single append-only event recorded during agent execution.

    Events describe boundaries (run start/end, model request, tool call,
    failure) but never carry recoverable state on their own. Pair with a
    `ContinuableSnapshot` for resume; pair with `ToolEffectRecord` for
    side-effect status.

    `conversation_id` mirrors pydantic_ai's three-level identity stack --
    conversation (the dialogue) → run (one `Agent.run` call) → step
    (one graph node). Two `Agent.run` calls that share a
    `conversation_id` produce two separate `run_id`s.
    """

    run_id: str
    kind: EventKind
    step_index: int
    timestamp: datetime = field(default_factory=_utcnow)
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    error: str | None = None
    metadata: dict[str, str] = field(default_factory=_empty_str_dict)


@dataclass(kw_only=True)
class ContinuableSnapshot:
    """A message-history snapshot a run can be resumed from.

    Pass `messages` to `Agent.run(..., message_history=...)` to continue or
    fork the run. `state` says whether the captured tool work was settled;
    stores hide `interrupted` snapshots from `latest_snapshot` unless the
    caller opts in (see `SnapshotState`).
    """

    run_id: str
    step_index: int
    messages: list[ModelMessage]
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    state: SnapshotState = 'complete'


@dataclass(kw_only=True)
class ToolEffectRecord:
    """Ledger entry for a tool call's side-effect status.

    Read-only tools and side-effectful tools share the same record shape;
    the orchestrator decides whether replay is safe based on
    `idempotency_key` and `effect_summary`. A record without a matching
    `completed` or `failed` update after process restart should be treated
    as `unknown_after_crash`.
    """

    tool_call_id: str
    tool_name: str
    run_id: str
    status: ToolEffectStatus
    started_at: datetime = field(default_factory=_utcnow)
    ended_at: datetime | None = None
    idempotency_key: str | None = None
    effect_summary: str | None = None


@dataclass(kw_only=True)
class RunRecord:
    """Lineage metadata for an agent run.

    `conversation_id` groups runs of the same dialogue (the user-visible
    sequence). `parent_run_id` is the hierarchical link: which run spawned
    this one. The two are independent axes -- a delegate may share a
    conversation across attempts (different `run_id`s, same `conversation_id`)
    while pointing at a different orchestrator run via `parent_run_id`.
    """

    run_id: str
    conversation_id: str | None = None
    parent_run_id: str | None = None
    agent_name: str | None = None
    metadata: dict[str, str] = field(default_factory=_empty_str_dict)
    started_at: datetime = field(default_factory=_utcnow)
