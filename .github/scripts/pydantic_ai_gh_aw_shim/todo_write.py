"""Claude's `TodoWrite` tool -- record the agent's task checklist.

Backed by pydantic-ai-harness's `experimental.planning` capability: the adapter
maps Claude's todo schema onto the harness `PlanItem` list and calls the same
`write_plan` the capability exposes, so the checklist is rendered (with the
harness's advisory note when more than one step is `in_progress`) instead of a
hand-rolled ack.

`planning` is experimental -- its API may change without a deprecation period --
which is acceptable here; the warning is silenced at import so it doesn't leak
into the agent's stdout. The Claude `TodoWrite` signature (`content` / `status` /
`activeForm` items) is preserved; `activeForm` is the present-tense label Claude
shows while a step runs and has no harness equivalent, so it's dropped (the
headless shim renders nothing live anyway).
"""

import warnings

from pydantic_ai_harness.experimental import HarnessExperimentalWarning
from typing_extensions import TypedDict

with warnings.catch_warnings():
    warnings.simplefilter('ignore', HarnessExperimentalWarning)
    from pydantic_ai_harness.experimental.planning import PlanItem, Planning, PlanningToolset, TaskStatus


class TodoItem(TypedDict):
    """One entry for `TodoWrite` (Claude's todo schema)."""

    content: str
    status: str
    activeForm: str


def _to_status(value: str) -> TaskStatus:
    """Map a Claude todo status onto a harness `TaskStatus`, defaulting to `pending`."""
    try:
        return TaskStatus(value)
    except ValueError:
        return TaskStatus.pending


async def todo_write(todos: list[TodoItem]) -> str:
    """Record the agent's task checklist."""
    items = [PlanItem(content=t.get('content', ''), status=_to_status(t.get('status', ''))) for t in todos]
    # A fresh capability per call gives a fresh plan state; Claude resends the
    # full list every time, so no cross-call state needs to be retained.
    # `get_toolset()` is typed `AgentToolset | None` but always returns the
    # planning toolset, so narrow to reach `write_plan` without a private import.
    toolset = Planning[None]().get_toolset()
    assert isinstance(toolset, PlanningToolset)
    return await toolset.write_plan(items)
