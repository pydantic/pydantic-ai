"""Where a conversation can be rewound to: the prompts that started its runs, as the terminal and web UI offer them."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    RetryFeedbackPart,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.compaction._summary import summary_time
from pydantic_clai2.runtime.imported_history import is_typed_prompt


@dataclass(kw_only=True)
class RewindPoint:
    """An editable user prompt at a run boundary in retained history."""

    message_index: int
    text: str
    images: tuple[BinaryContent, ...]
    reason: str = ''
    """Why it cannot be rewound to; empty when it can."""


def rewind_points(messages: Sequence[ModelMessage]) -> list[RewindPoint]:
    """Run-start prompts, oldest first, never a steering message amid unfinished tool calls."""
    # Compaction inserts newer context ahead of older preserved prompts; core may
    # merge them into one request. Those prompts no longer have their original boundary.
    # Harness pins this shape in `tests/harness/compaction/test_summary_provenance.py`.
    context_time = max(
        (
            *(
                part.timestamp
                for message in messages
                if isinstance(message, ModelRequest) and message.run_id is None
                for part in message.parts
                if isinstance(part, SystemPromptPart)
            ),
            # Any compaction part (provider-native, app-written or the harness's) rewrote what came before it.
            *(
                compacted_at
                for message in messages
                if message.run_id is None
                for part in message.parts
                if isinstance(part, CompactionPart) and (compacted_at := _compaction_time(part, message)) is not None
            ),
        ),
        default=None,
    )
    points: list[RewindPoint] = []
    seen_runs: set[str] = set()
    for index, message in enumerate(messages):
        if message.run_id is not None:
            if message.run_id in seen_runs:
                continue
            seen_runs.add(message.run_id)
        if not isinstance(message, ModelRequest) or any(
            isinstance(part, (ToolReturnPart, RetryFeedbackPart)) for part in message.parts
        ):
            continue
        prompts = [part for part in message.parts if is_typed_prompt(part)]
        if not prompts:
            continue
        points.append(_rewind_point(message_index=index, prompts=prompts, context_time=context_time))
    return points


def _rewind_point(*, message_index: int, prompts: list[UserPromptPart], context_time: datetime | None) -> RewindPoint:
    """The point for the typed `prompts` of one request, with why it cannot be rewound to when it cannot."""
    content = [item for part in prompts for item in ([part.content] if isinstance(part.content, str) else part.content)]
    text = '\n'.join(item for item in content if isinstance(item, str))
    images = tuple(item for item in content if isinstance(item, BinaryContent))
    reason = ''
    if context_time is not None and any(part.timestamp < context_time for part in prompts):
        reason = 'predates rewritten context'
    elif not all(isinstance(item, (str, BinaryContent)) for item in content):
        reason = 'unsupported attachment'
    return RewindPoint(message_index=message_index, text=text, images=images, reason=reason)


def _compaction_time(part: CompactionPart, message: ModelMessage) -> datetime | None:
    """When *part* rewrote the history before it.

    The harness's summary records that on the part: merging its request into the preserved one after it
    leaves the message with that older request's timestamp.
    """
    return summary_time(part) or message.timestamp
