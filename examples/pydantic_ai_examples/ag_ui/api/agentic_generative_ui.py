"""Agentic Generative UI feature."""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Literal

from pydantic import BaseModel, Field

from ag_ui.core import EventType, StateDeltaEvent, StateSnapshotEvent
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui.app import AGUIApp

StepStatus = Literal['pending', 'completed']


class Step(BaseModel):
    """Represents a step in a plan."""

    description: str = Field(description='The description of the step')
    status: StepStatus = Field(
        default='pending',
        description='The status of the step (e.g., pending, completed)',
    )


class Plan(BaseModel):
    """Represents a plan with multiple steps."""

    steps: list[Step] = Field(default_factory=list, description='The steps in the plan')


class JSONPatchOp(BaseModel):
    """A class representing a JSON Patch operation (RFC 6902)."""

    op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test'] = Field(
        description='The operation to perform: add, remove, replace, move, copy, or test',
    )
    path: str = Field(description='JSON Pointer (RFC 6901) to the target location')
    value: Any = Field(
        default=None,
        description='The value to apply (for add, replace operations)',
    )
    from_: str | None = Field(
        default=None,
        alias='from',
        description='Source path (for move, copy operations)',
    )


agent = Agent(
    'openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning use tools only, without any other messages.
        IMPORTANT:
        - Use the `create_plan` tool to set the initial state of the steps
        - Use the `update_plan_step` tool to update the status of each step
        - Do NOT repeat the plan or summarise it in a message
        - Do NOT confirm the creation or updates in a message
        - Do NOT ask the user for additional information or next steps

        Only one plan can be active at a time, so do not call the `create_plan` tool
        again until all the steps in current plan are completed.
        """
    ),
)


@agent.tool_plain
async def create_plan(steps: list[str]) -> StateSnapshotEvent:
    """Create a plan with multiple steps.

    Args:
        steps: List of step descriptions to create the plan.

    Returns:
        StateSnapshotEvent containing the initial state of the steps.
    """
    plan: Plan = Plan(
        steps=[Step(description=step) for step in steps],
    )
    return StateSnapshotEvent(
        type=EventType.STATE_SNAPSHOT,
        snapshot=plan.model_dump(),
    )


@agent.tool_plain
async def update_plan_step(
    index: int, description: str | None = None, status: StepStatus | None = None
) -> StateDeltaEvent:
    """Update the plan with new steps or changes.

    Args:
        index: The index of the step to update.
        description: The new description for the step.
        status: The new status for the step.

    Returns:
        StateDeltaEvent containing the changes made to the plan.
    """
    changes: list[JSONPatchOp] = []
    if description is not None:
        changes.append(
            JSONPatchOp(
                op='replace', path=f'/steps/{index}/description', value=description
            )
        )
    if status is not None:
        changes.append(
            JSONPatchOp(op='replace', path=f'/steps/{index}/status', value=status)
        )
    return StateDeltaEvent(
        type=EventType.STATE_DELTA,
        delta=changes,
    )


app = AGUIApp(agent)
