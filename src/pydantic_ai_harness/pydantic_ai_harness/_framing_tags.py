"""Every framing tag Pydantic AI Harness writes into prompt text, registered with core in one place.

Registering a tag makes core's own data frames, such as `<tool_result>`, and every other registered
frame neutralize it, so content in one frame can't close another. The registry is process-wide, so
the tags are registered together here rather than by each capability's module: the first capability
that frames text imports this module and registers them all, so which tags are registered doesn't
depend on which other capabilities a process imports, or when, and the same text always renders the
same bytes. It isn't imported by `pydantic_ai_harness` itself, which must stay importable without
loading `pydantic_ai`.

Render them with `pydantic_ai.framing.wrap`, or `open_tag`/`close_tag` around a body that is escaped
some other way, never by hand.
"""

from __future__ import annotations

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
