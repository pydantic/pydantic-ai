"""Recovery facts, not replay permission. External side effects require reconciliation."""

from dataclasses import dataclass

from ._store import StepStore
from ._types import ContinuableSnapshot, ToolEffectRecord


@dataclass(frozen=True, kw_only=True)
class RecoveryInspection:
    """The newest and settled frontiers, plus tool activity recorded for the run.

    A completed effect is not proof its normalized result reached a checkpoint.
    A failed effect is not proof nothing happened. No field here authorizes replay.
    """

    latest: ContinuableSnapshot | None
    settled: ContinuableSnapshot | None
    unresolved: tuple[ToolEffectRecord, ...]
    completed_tools: tuple[str, ...]
    failed_tools: tuple[str, ...]


async def inspect_recovery(*, store: StepStore, run_id: str) -> RecoveryInspection:
    """Read store facts without executing tools or rewriting history."""
    events = await store.list_events(run_id=run_id)
    return RecoveryInspection(
        latest=await store.latest_snapshot(run_id=run_id, include_interrupted=True),
        settled=await store.latest_snapshot(run_id=run_id),
        unresolved=tuple(await store.list_unresolved_tool_effects(run_id=run_id)),
        completed_tools=tuple(
            dict.fromkeys(e.tool_name for e in events if e.kind == 'tool_call_completed' and e.tool_name)
        ),
        failed_tools=tuple(dict.fromkeys(e.tool_name for e in events if e.kind == 'tool_call_failed' and e.tool_name)),
    )
