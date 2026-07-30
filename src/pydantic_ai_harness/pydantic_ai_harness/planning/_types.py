"""Data types for the `Planning` capability."""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle status of a single plan step.

    `blocked` is only reachable when the capability runs with
    `enable_subtasks=True`; otherwise the toolset rejects it.
    """

    pending = 'pending'
    in_progress = 'in_progress'
    completed = 'completed'
    cancelled = 'cancelled'
    blocked = 'blocked'


class PlanItem(BaseModel):
    """One step in the plan.

    Used both as the `write_plan` input and as the stored item. `id` is
    auto-generated when omitted, so the model can create a plan without ids and
    reference existing steps by id when restructuring.

    Attributes:
        id: Stable identifier (auto-generated 8-char hex string).
        content: Imperative description of the step, e.g. `Add the database migration`.
        status: Current status of this step.
        active_form: Optional present-continuous label shown while the step runs,
            e.g. `Adding the database migration`.
        parent_id: Parent step id for subtask hierarchies (subtasks mode only).
        depends_on: Ids of steps that must complete before this one (subtasks mode only).
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:8], description='Step id. Auto-generated if omitted.')
    content: str = Field(description='Imperative description of the step, e.g. "Add the database migration".')
    status: TaskStatus = Field(
        default=TaskStatus.pending,
        description='Current status of this step. "blocked" requires subtasks to be enabled.',
    )
    active_form: str = Field(
        default='',
        description='Present-continuous label shown while the step runs, e.g. "Adding the database migration".',
    )
    parent_id: str | None = Field(
        default=None, description='Parent step id for a subtask hierarchy. Requires subtasks to be enabled.'
    )
    depends_on: list[str] = Field(
        default_factory=lambda: [],
        description='Ids of steps that must complete before this one. Requires subtasks to be enabled.',
    )


class PlanStatusUpdate(BaseModel):
    """One entry of the `update_task_statuses` batch tool."""

    task_id: str = Field(description='Id of the step to update (from `read_plan`).')
    status: TaskStatus = Field(
        description='New status: pending, in_progress, completed, or cancelled -- '
        'or blocked, when subtasks are enabled.'
    )
