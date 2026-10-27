"""Private helpers for `SubAgents` background task report provenance.

Kept outside `subagents` so that `guardrails` can recognize a report without importing the capability.
"""

from __future__ import annotations

import html
from typing import TypeGuard

from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._provenance import neutralize_harness_tags, tagged_text

_TASK_REPORT_ORIGIN = 'pydantic-ai-harness.subagents.report.v1'
"""The `origin` in the metadata of the `TextContent` that carries a background task's output."""

_TASK_QUESTION_ORIGIN = 'pydantic-ai-harness.subagents.question.v1'
"""The `origin` in the metadata of the `TextContent` that carries a task's question to its parent."""

# TODO: switch to `wrap()` from the public `pydantic_ai.framing` module once it lands (#10957), registering
# `task_output` as a framing tag there.


def make_task_output(body: str, *, task_id: str, agent_name: str, outcome: str, generation: int) -> UserPromptPart:
    """Frame `body` as `<task_output>` data that can't end its own element or open a framing tag, with escaped attributes.

    Every opening or closing framing tag core writes (`<system>`, `<context_diff>`, `<tool_result>` and the rest)
    is neutralized in the body, as are `<task_output>` itself, `<task_question>`, and `<peer_message>`.

    Like a background tool's `<tool_result>`, the frame carries `outcome` only when the task didn't succeed.
    """
    attributes = {'task_id': task_id, 'agent_name': agent_name, **({'outcome': outcome} if outcome != 'ok' else {})}
    attrs = ''.join(f' {name}="{html.escape(value)}"' for name, value in attributes.items())
    body = neutralize_harness_tags(body)
    metadata = {'origin': _TASK_REPORT_ORIGIN, 'task_id': task_id, 'generation': generation}
    return UserPromptPart([TextContent(f'<task_output{attrs}>\n{body}\n</task_output>', metadata=metadata)])


def make_task_question(question: str, *, task_id: str, agent_name: str, question_id: str) -> UserPromptPart:
    """Frame a task's `question` to its parent as `<task_question>` data, escaped as `make_task_output` escapes output.

    The question was written by the child's model, so like its output it never enters system voice.
    """
    attrs = f' task_id="{html.escape(task_id)}" agent_name="{html.escape(agent_name)}"'
    body = neutralize_harness_tags(question)
    metadata = {'origin': _TASK_QUESTION_ORIGIN, 'task_id': task_id, 'question_id': question_id}
    return UserPromptPart([TextContent(f'<task_question{attrs}>\n{body}\n</task_question>', metadata=metadata)])


def is_task_report(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether `part` carries a background task's output or a task's question, as `DelegationReports` delivers them."""
    return isinstance(part, UserPromptPart) and any(
        tagged_text(part.content, origin) is not None for origin in (_TASK_REPORT_ORIGIN, _TASK_QUESTION_ORIGIN)
    )
