"""Private helpers for `SubAgents` background task report provenance.

Kept outside `subagents` so that `guardrails` can recognize a report without importing the capability.
"""

from __future__ import annotations

import html
import re
from typing import TypeGuard

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.framing import register_tag, wrap
from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._provenance import tagged_text

_TASK_REPORT_ORIGIN = 'pydantic-ai-harness.subagents.report.v1'
"""The `origin` in the metadata of the `TextContent` that carries a background task's output."""

_TASK_QUESTION_ORIGIN = 'pydantic-ai-harness.subagents.question.v1'
"""The `origin` in the metadata of the `TextContent` that carries a task's question to its parent."""

_TEAM_MESSAGE_ORIGIN = 'pydantic-ai-harness.subagents.message.v1'
"""The `origin` in the metadata of the `TextContent` that carries a team member's message to another agent."""

_PEER_MESSAGE = register_tag('peer_message')

# TODO: switch to `wrap()` from the public `pydantic_ai.framing` module once it lands (#10957), registering
# `task_output` as a framing tag there.
# The `<` of an opening or closing `task_output` (or the hyphenated `task-output` older reports used) or
# `task_question`, which could end a frame early or open a forged report or question. Any case, spacing or attributes
# match, as does a tag cut off at the end of the body, where the frame's own end tag would complete it; core's own
# tags are left to `neutralize_tags`.
_TASK_FRAME_TAG_OPENER = re.compile(r'<(?=\s*(?:/\s*)?(?:task[-_]output|task_question)(?=[\s/>]|$))', re.IGNORECASE)


def make_task_output(
    body: str, *, task_id: str, agent_name: str, outcome: str, generation: int, name: str | None = None
) -> UserPromptPart:
    """Frame `body` as `<task_output>` data that can't end its own element or open a framing tag, with escaped attributes.

    Every opening or closing framing tag core writes (`<system>`, `<context_diff>`, `<tool_result>` and the rest)
    is neutralized in the body, as are `<task_output>` itself and `<task_question>`.

    Like a background tool's `<tool_result>`, the frame carries `outcome` only when the task didn't succeed, and
    `name` only for a team member.
    """
    attributes = {
        'task_id': task_id,
        **({'name': name} if name is not None else {}),
        'agent_name': agent_name,
        **({'outcome': outcome} if outcome != 'ok' else {}),
    }
    attrs = ''.join(f' {name}="{html.escape(value)}"' for name, value in attributes.items())
    body = _TASK_FRAME_TAG_OPENER.sub('&lt;', neutralize_tags(body))
    metadata = {'origin': _TASK_REPORT_ORIGIN, 'task_id': task_id, 'generation': generation}
    return UserPromptPart([TextContent(f'<task_output{attrs}>\n{body}\n</task_output>', metadata=metadata)])


def make_task_question(question: str, *, task_id: str, agent_name: str, question_id: str) -> UserPromptPart:
    """Frame a task's `question` to its parent as `<task_question>` data, escaped as `make_task_output` escapes output.

    The question was written by the child's model, so like its output it never enters system voice.
    """
    attrs = f' task_id="{html.escape(task_id)}" agent_name="{html.escape(agent_name)}"'
    body = _TASK_FRAME_TAG_OPENER.sub('&lt;', neutralize_tags(question))
    metadata = {'origin': _TASK_QUESTION_ORIGIN, 'task_id': task_id, 'question_id': question_id}
    return UserPromptPart([TextContent(f'<task_question{attrs}>\n{body}\n</task_question>', metadata=metadata)])


def make_team_message(text: str, *, sender: str, message_id: str) -> UserPromptPart:
    """Frame a team member's message to another agent as `<peer_message>` data, escaped like a task's output.

    The message was written by another agent's model, so like a task's output it never enters system voice, and the
    recipient never reads it as its user's words.
    """
    body = wrap(_PEER_MESSAGE, _TASK_FRAME_TAG_OPENER.sub('&lt;', text), sender=sender, id=message_id)
    metadata = {'origin': _TEAM_MESSAGE_ORIGIN, 'sender': sender, 'message_id': message_id}
    return UserPromptPart([TextContent(body, metadata=metadata)])


def is_task_report(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether `part` carries a background task's output, a task's question or a team member's message.

    These are the parts `DelegationReports` delivers.
    """
    return isinstance(part, UserPromptPart) and any(
        tagged_text(part.content, origin) is not None
        for origin in (_TASK_REPORT_ORIGIN, _TASK_QUESTION_ORIGIN, _TEAM_MESSAGE_ORIGIN)
    )
