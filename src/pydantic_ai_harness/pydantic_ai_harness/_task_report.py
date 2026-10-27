"""Private helpers for `SubAgents` background task report provenance.

Kept outside `subagents` so that `guardrails` can recognize a report without importing the capability.
"""

from __future__ import annotations

import re
from typing import TypeGuard

from pydantic_ai.framing import wrap
from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._framing_tags import TASK_OUTPUT_TAG, TASK_QUESTION_TAG
from pydantic_ai_harness._provenance import tagged_text

_TASK_REPORT_ORIGIN = 'pydantic-ai-harness.subagents.report.v1'
"""The `origin` in the metadata of the `TextContent` that carries a background task's output."""

_TASK_QUESTION_ORIGIN = 'pydantic-ai-harness.subagents.question.v1'
"""The `origin` in the metadata of the `TextContent` that carries a task's question to its parent."""

# TODO: register `task-output` too, or drop this, once #11497 settles whether framing tag names may have hyphens.
# The `<` of an opening or closing `task-output`, the tag older reports used, so a body can't forge one of those;
# `wrap` neutralizes every registered tag, `task_output` and `task_question` among them.
_LEGACY_TASK_OUTPUT_TAG_OPENER = re.compile(r'<(?=\s*(?:/\s*)?task-output(?=[\s/>]|$))', re.IGNORECASE)


def make_task_output(body: str, *, task_id: str, agent_name: str, outcome: str, generation: int) -> UserPromptPart:
    """Frame `body` as `<task_output>` data that can't end its own element or open a framing tag, with escaped attributes.

    Every opening or closing framing tag, core's (`<system>`, `<tool_result>` and the rest) and every registered one
    (`<task_output>` itself and `<task_question>` among them), is neutralized in the body.

    Like a background tool's `<tool_result>`, the frame carries `outcome` only when the task didn't succeed.
    """
    text = wrap(
        TASK_OUTPUT_TAG,
        f'\n{_LEGACY_TASK_OUTPUT_TAG_OPENER.sub("&lt;", body)}\n',
        task_id=task_id,
        agent_name=agent_name,
        outcome=None if outcome == 'ok' else outcome,
    )
    metadata = {'origin': _TASK_REPORT_ORIGIN, 'task_id': task_id, 'generation': generation}
    return UserPromptPart([TextContent(text, metadata=metadata)])


def make_task_question(question: str, *, task_id: str, agent_name: str, question_id: str) -> UserPromptPart:
    """Frame a task's `question` to its parent as `<task_question>` data, escaped as `make_task_output` escapes output.

    The question was written by the child's model, so like its output it never enters system voice.
    """
    text = wrap(
        TASK_QUESTION_TAG,
        f'\n{_LEGACY_TASK_OUTPUT_TAG_OPENER.sub("&lt;", question)}\n',
        task_id=task_id,
        agent_name=agent_name,
    )
    metadata = {'origin': _TASK_QUESTION_ORIGIN, 'task_id': task_id, 'question_id': question_id}
    return UserPromptPart([TextContent(text, metadata=metadata)])


def is_task_report(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether `part` carries a background task's output or a task's question, as `DelegationReports` delivers them."""
    return isinstance(part, UserPromptPart) and any(
        tagged_text(part.content, origin) is not None for origin in (_TASK_REPORT_ORIGIN, _TASK_QUESTION_ORIGIN)
    )
