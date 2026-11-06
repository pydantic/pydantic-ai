"""The task gate checks that board tools call."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.planning import OwnedPlanItem
from pydantic_ai_harness.teams._events import TeamTaskCompletingEvent, TeamTaskCreatingEvent


def trace_rejection(
    ctx: RunContext[AgentDepsT], gate: str, feedback: Sequence[str], *, member: str | None, item_id: str | None
) -> None:
    attributes: dict[str, str | int] = {'team.gate': gate, 'team.feedback_count': len(feedback)}
    if member is not None:
        attributes['team.member'] = member
    if item_id is not None:
        attributes['team.item_id'] = item_id
    if ctx.trace_include_content:
        attributes['team.feedback'] = joined(feedback)
    ctx.tracer.start_span('team gate rejected', attributes=attributes).end()


def joined(feedback: Sequence[str]) -> str:
    return '\n\n'.join(feedback)


async def check_task_creating(ctx: RunContext[AgentDepsT], item: OwnedPlanItem) -> None:
    """Ask the team's gates whether a tool call may add `item` to the team board.

    Call it from a board tool, before the item is stored. It emits a `TeamTaskCreatingEvent`, so
    it needs a `RunContext` whose tool or hook belongs to a capability of the run.

    Raises:
        ModelRetry: A gate rejected the item; the message carries the gates' feedback.
    """
    event = await ctx.emit(TeamTaskCreatingEvent(item=item))
    if event.rejected:
        trace_rejection(ctx, 'task_creating', event.feedback, member=None, item_id=item.id)
        raise ModelRetry(f'The task was not added to the board.\n\n{joined(event.feedback)}')


async def check_task_completing(ctx: RunContext[AgentDepsT], item: OwnedPlanItem, *, owner: str) -> None:
    """Ask the team's gates whether a tool call by `owner` may mark `item` completed.

    Call it from a board tool, before the status change is stored. It emits a
    `TeamTaskCompletingEvent`, so it needs a `RunContext` whose tool or hook belongs to a capability
    of the run.

    Raises:
        ModelRetry: A gate rejected the completion; the message carries the gates' feedback.
    """
    event = await ctx.emit(TeamTaskCompletingEvent(item=item, owner=owner))
    if event.rejected:
        trace_rejection(ctx, 'task_completing', event.feedback, member=owner, item_id=item.id)
        raise ModelRetry(f'The task was not marked completed.\n\n{joined(event.feedback)}')
