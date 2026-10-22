"""Private helpers for `SubAgents` background task report provenance.

Kept outside `subagents` so that `guardrails` can recognize a report without importing the capability.
"""

from __future__ import annotations

import html
import re
from typing import TypeGuard

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._provenance import tagged_text

_TASK_REPORT_ORIGIN = 'pydantic-ai-harness.subagents.report.v1'
"""The `origin` in the metadata of the `TextContent` that carries a background task's output."""

# TODO: switch to `wrap()` from the public `pydantic_ai.framing` module once it lands (#10957), registering
# `task_output` as a framing tag there.
# The `<` of an opening or closing `task_output`, or the hyphenated `task-output` older reports used, which could end
# this frame early or open a forged one. Any case, spacing or attributes match, as does a tag cut off at the end of
# the body, where the frame's own end tag would complete it; core's own tags are left to `neutralize_tags`.
_TASK_OUTPUT_TAG_OPENER = re.compile(r'<(?=\s*(?:/\s*)?task[-_]output(?=[\s/>]|$))', re.IGNORECASE)


def make_task_output(body: str, *, task_id: str, agent_name: str, outcome: str, generation: int) -> UserPromptPart:
    """Frame `body` as `<task_output>` data that can't end its own element or open a framing tag, with escaped attributes.

    Every opening or closing framing tag core writes (`<system>`, `<context_diff>`, `<tool_result>` and the rest)
    is neutralized in the body, as is `<task_output>` itself.

    Like a background tool's `<tool_result>`, the frame carries `outcome` only when the task didn't succeed.
    """
    attributes = {'task_id': task_id, 'agent_name': agent_name, **({'outcome': outcome} if outcome != 'ok' else {})}
    attrs = ''.join(f' {name}="{html.escape(value)}"' for name, value in attributes.items())
    body = _TASK_OUTPUT_TAG_OPENER.sub('&lt;', neutralize_tags(body))
    metadata = {'origin': _TASK_REPORT_ORIGIN, 'task_id': task_id, 'generation': generation}
    return UserPromptPart([TextContent(f'<task_output{attrs}>\n{body}\n</task_output>', metadata=metadata)])


def is_task_report(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether `part` carries a background task's output, as delivered by `DelegationReports`."""
    return isinstance(part, UserPromptPart) and tagged_text(part.content, _TASK_REPORT_ORIGIN) is not None
