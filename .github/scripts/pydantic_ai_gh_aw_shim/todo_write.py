"""Claude's `TodoWrite` tool -- record the agent's task checklist.

Backed by pydantic-ai-harness's `planning` capability: the adapter
maps Claude's todo schema onto the harness `PlanItem` list and renders it with
the harness's `render_plan`, the checklist its `write_plan` tool returns, with the
same advisory note when more than one step is `in_progress`. `write_plan` itself
can't be called here: it emits plan events, which only a capability-owned tool
inside an agent run may do, and `TodoWrite` is a plain tool.

The Claude `TodoWrite` signature (`content` / `status` /
`activeForm` items) is preserved; `activeForm` is the present-tense label Claude
shows while a step runs and has no harness equivalent, so it's dropped (the
headless shim renders nothing live anyway).
"""

from pydantic_ai_harness.planning import PlanItem, TaskStatus, render_plan
from typing_extensions import TypedDict


class TodoItem(TypedDict):
    """One entry for `TodoWrite` (Claude's todo schema)."""

    content: str
    status: str
    activeForm: str


_CLAUDE_STATUSES = frozenset({TaskStatus.pending, TaskStatus.in_progress, TaskStatus.completed})
"""Claude's todo statuses. The harness's `blocked` needs its subtask dependencies, which `TodoWrite` lacks."""


def _to_status(value: str) -> TaskStatus:
    """Map a Claude todo status onto a harness `TaskStatus`, defaulting to `pending`."""
    try:
        status = TaskStatus(value)
    except ValueError:
        return TaskStatus.pending
    return status if status in _CLAUDE_STATUSES else TaskStatus.pending


async def todo_write(todos: list[TodoItem]) -> str:
    """Record the agent's task checklist."""
    items = [PlanItem(content=t.get('content', ''), status=_to_status(t.get('status', ''))) for t in todos]
    # Claude resends the full list every time, so no plan state is retained across calls.
    in_progress = sum(1 for item in items if item.status is TaskStatus.in_progress)
    note = '' if in_progress <= 1 else '\n\nNote: keep only one step in_progress at a time.'
    return f'Plan updated: {len(items)} step(s).\n\n{render_plan(items)}{note}'
