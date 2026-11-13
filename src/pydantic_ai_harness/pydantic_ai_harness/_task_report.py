"""`SubAgents` background task reports and questions, as the capability parts the parent run receives them as.

Kept outside `subagents` so that `guardrails` can recognize a report without importing the capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard

from pydantic_ai.messages import DataCapabilityPart, ModelRequestPart, ModelSource, UserPromptPart
from pydantic_ai_harness._provenance import tagged_text

_TASK_REPORT_ORIGIN = 'pydantic-ai-harness.subagents.report.v1'
"""The `origin` in the metadata of the `TextContent` an earlier version carried a background task's output in."""

_TASK_QUESTION_ORIGIN = 'pydantic-ai-harness.subagents.question.v1'
"""The `origin` in the metadata of the `TextContent` an earlier version carried a task's question to its parent in."""


@dataclass(kw_only=True)
class TaskOutputPart(
    DataCapabilityPart[str],
    namespace='subagents',
    name='task_output',
    tag='task_output',
    attributes=('task_id', 'agent_name', 'outcome'),
):
    """A settled background task's report: the child's output, as its parent receives it.

    Data on the user channel in a `<task_output>` tag naming the task, as the child's model wrote it, with a note that
    the sub-agent wrote it, not the user or the application. Find reports in history by this type, or with
    [`is_task_report`][pydantic_ai_harness.subagents.is_task_report].
    """

    task_id: str
    """The task's id."""

    agent_name: str
    """The name of the sub-agent that ran the task."""

    outcome: str | None = None
    """How the task settled, when it didn't succeed: like a background tool's `<tool_result>`, the frame only says so then."""

    generation: int = 0
    """Which run of the task this reports on. Never sent to the model."""


@dataclass(kw_only=True)
class TaskQuestionPart(
    DataCapabilityPart[str],
    namespace='subagents',
    name='task_question',
    tag='task_question',
    attributes=('task_id', 'agent_name'),
):
    """A running task's question to its parent, as the parent receives it.

    The question was written by the child's model, so like its output it's data, never in the application's voice.
    Find questions in history by this type, or with [`is_task_report`][pydantic_ai_harness.subagents.is_task_report].
    """

    task_id: str
    """The asking task's id."""

    agent_name: str
    """The name of the sub-agent that asks."""

    question_id: str
    """The question's id, which its answer names. Never sent to the model."""


def _sub_agent(agent_name: str) -> ModelSource:
    """Who wrote a task's output or question: the sub-agent's model, so it has no authority."""
    return ModelSource(f'sub-agent `{agent_name}`')


def make_task_output(body: str, *, task_id: str, agent_name: str, outcome: str, generation: int) -> TaskOutputPart:
    """The report of a task that settled with `outcome` and `body` as its output."""
    return TaskOutputPart(
        content=body,
        source=_sub_agent(agent_name),
        task_id=task_id,
        agent_name=agent_name,
        outcome=None if outcome == 'ok' else outcome,
        generation=generation,
    )


def make_task_question(question: str, *, task_id: str, agent_name: str, question_id: str) -> TaskQuestionPart:
    """A task's `question` to its parent."""
    return TaskQuestionPart(
        content=question,
        source=_sub_agent(agent_name),
        task_id=task_id,
        agent_name=agent_name,
        question_id=question_id,
    )


def is_task_report(part: ModelRequestPart) -> TypeGuard[TaskOutputPart | TaskQuestionPart | UserPromptPart]:
    """Whether `part` carries a background task's output or a task's question, as `DelegationReports` delivers them.

    That's a `TaskOutputPart` or `TaskQuestionPart`, or the user text an earlier version delivered them as.
    """
    return isinstance(part, TaskOutputPart | TaskQuestionPart) or (
        isinstance(part, UserPromptPart)
        and any(
            tagged_text(part.content, origin) is not None for origin in (_TASK_REPORT_ORIGIN, _TASK_QUESTION_ORIGIN)
        )
    )
