"""Planning toolset: a single `write_plan` tool over a shared, per-run plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset


class TaskStatus(str, Enum):
    """Lifecycle status of a single plan step."""

    pending = 'pending'
    in_progress = 'in_progress'
    completed = 'completed'
    cancelled = 'cancelled'


_STATUS_ICONS = {
    TaskStatus.pending: '[ ]',
    TaskStatus.in_progress: '[~]',
    TaskStatus.completed: '[x]',
    TaskStatus.cancelled: '[-]',
}


class PlanItem(BaseModel):
    """A single step in the plan."""

    content: str = Field(description='Imperative description of the step, e.g. "Add the database migration".')
    status: TaskStatus = Field(default=TaskStatus.pending, description='Current status of this step.')


@dataclass
class PlanState:
    """Mutable per-run plan storage shared between the toolset and the capability hook."""

    items: list[PlanItem] = field(default_factory=list[PlanItem])


def render_plan(items: list[PlanItem]) -> str:
    """Render the plan as a checklist with a one-line progress summary."""
    if not items:
        return 'No plan yet.'
    lines = [f'{i + 1}. {_STATUS_ICONS[item.status]} {item.content}' for i, item in enumerate(items)]
    completed = sum(1 for item in items if item.status is TaskStatus.completed)
    lines.append(f'({completed}/{len(items)} completed)')
    return '\n'.join(lines)


class PlanningToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single `write_plan` tool that overwrites the shared `PlanState`."""

    def __init__(self, state: PlanState) -> None:
        super().__init__()
        self._state = state
        self.add_function(self.write_plan, name='write_plan')

    async def write_plan(self, items: list[PlanItem]) -> str:
        """Create or replace the full task plan.

        Pass the entire ordered plan every time -- including steps that are
        unchanged, completed, or cancelled. Keep exactly one step `in_progress`.
        Call this when you start and when you finish a step so your progress
        stays visible.

        Args:
            items: The complete ordered list of plan steps.
        """
        self._state.items = list(items)
        in_progress = sum(1 for item in items if item.status is TaskStatus.in_progress)
        note = '' if in_progress <= 1 else '\n\nNote: keep only one step in_progress at a time.'
        return f'Plan updated: {len(items)} step(s).\n\n{render_plan(self._state.items)}{note}'
