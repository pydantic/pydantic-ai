"""Every framing tag Pydantic AI Harness writes into prompt text, registered with core in one place.

Registering a tag makes core's own data frames, such as `<tool_result>`, and every other registered
frame neutralize it, so content in one frame can't close another. The registry is process-wide, so
the tags are registered together here rather than by each capability's module, and every harness
module that renders or escapes text with `pydantic_ai.framing` imports this one, even when it uses no
harness tag itself: the first of them registers them all, so which tags are registered doesn't depend
on which capabilities a process imports, or when, and the same text always renders the same bytes. It
isn't imported by `pydantic_ai_harness` itself, which must stay importable without loading
`pydantic_ai`.

Render them with `pydantic_ai.framing.wrap`, or `open_tag`/`close_tag` around a body that is escaped
some other way, never by hand.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic_ai.framing import register_tag

MEMORY_TAG: Final = register_tag('memory')
"""Frames a `Memory` snapshot."""

JUDGE_FEEDBACK_TAG: Final = register_tag('judge_feedback')
"""Frames a `TrajectoryJudge` course correction."""

TRAJECTORY_TAG: Final = register_tag('trajectory')
"""Frames the run's trajectory in a `TrajectoryJudge` prompt."""

TASK_OUTPUT_TAG: Final = register_tag('task_output')
"""Frames a background `SubAgents` task's report."""

TASK_QUESTION_TAG: Final = register_tag('task_question')
"""Frames a `SubAgents` task's question to its parent."""

GOAL_TAG: Final = register_tag('goal')
"""Frames the original goal in a `SystemReminders` generator prompt."""

SUMMARY_TAG: Final = register_tag('summary')
"""Frames a summary of earlier conversation in a `SystemReminders` generator prompt."""

GENERATED_REMINDER_TAG: Final = register_tag('generated_reminder')
"""Frames a reminder another model wrote for `SystemReminders`."""

MESSAGE_TAG: Final = register_tag('message')
"""Frames one message of a transcript, in the `SystemReminders` generator and `SummarizingCompaction` prompts."""

MESSAGES_TAG: Final = register_tag('messages')
"""Frames the `SummarizingCompaction` transcript, in its default summary prompt."""

HARNESS_DATA_TAG: Final = register_tag('harness_data')
"""Frames content a harness capability added on the user channel, in the `SummarizingCompaction` transcript."""

TOOL_CALL_TAG: Final = register_tag('tool_call')
"""Frames the call under review in a `ToolCallJudge` or `ApprovalReviewer` prompt."""

CONVERSATION_TAG: Final = register_tag('conversation')
"""Frames the conversation leading up to the call under review, in the same prompt."""

CURRENT_TASK_TAG: Final = register_tag('current_task')
"""Frames the agent's current task in a `RelevanceCompaction` judge prompt."""

RECENT_CONTEXT_TAG: Final = register_tag('recent_context')
"""Frames the most recent part of the conversation in the same prompt."""

EARLIER_STEP_TAG: Final = register_tag('earlier_step')
"""Frames the earlier step being judged in the same prompt."""

CONTEXT_FILE_TAG: Final = register_tag('context_file')
"""Frames an instruction file `RepoContext` loaded, or names one the model can read."""

PREVIOUS_SUMMARY_TAG: Final = register_tag('previous_summary')
"""Frames the summary an incremental `SummarizingCompaction` updates."""

_LEGACY_TAG_OPENERS: Final = re.compile(
    r'<(?=\s*(?:/\s*)?(?:context-file|previous-summary|task-output)(?=[\s/>]|$))', re.IGNORECASE
)
"""The `<` of an opening or closing tag in a spelling earlier releases wrote, before tag names were snake_case."""


def neutralize_legacy_tags(text: str) -> str:
    """Escape the `<` of `context-file`, `previous-summary` and `task-output` tags, which `neutralize_tags` doesn't know.

    Histories saved by earlier releases still carry those frames, so content in a frame now must not be able to close
    or forge one. Apply it in the bodies of the frames that replaced them.
    """
    return _LEGACY_TAG_OPENERS.sub('&lt;', text)
