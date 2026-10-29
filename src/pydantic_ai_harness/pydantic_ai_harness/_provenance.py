"""Private helpers for recognizing harness-authored content by its metadata."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai._utils import is_str_dict
from pydantic_ai.messages import TextContent, UserContent

# Interim, deliberately generic rule: a harness capability that tags what it adds, user-channel data or a system
# prompt, with an `origin` under this prefix is covered by one check, without per-capability code. Today that is
# `BackgroundTools` results, `TrajectoryJudge` feedback, `SubAgents` task reports, `Memory` snapshots and
# `SystemReminders` reminders. The rule is temporary: once core provenance (`CapabilityPart`) lands, recognize that
# instead of this metadata convention.
HARNESS_ORIGIN_PREFIX = 'pydantic-ai-harness.'
"""The prefix of the `origin` metadata harness capabilities put on user-channel data and system prompts they add."""


def harness_origin(metadata: object) -> str | None:
    """The harness `origin` that *metadata* names, or `None` when it names none."""
    if (
        is_str_dict(metadata)
        and isinstance(origin := metadata.get('origin'), str)
        and origin.startswith(HARNESS_ORIGIN_PREFIX)
    ):
        return origin
    return None


def tagged_text(content: str | Sequence[UserContent] | None, origin: str) -> TextContent | None:
    """The `TextContent` that is all of `content`, when its metadata names `origin`."""
    if isinstance(content, Sequence) and len(content) == 1 and isinstance(content[0], TextContent):
        text_content = content[0]
        metadata = text_content.metadata
        if isinstance(metadata, dict) and 'origin' in metadata and metadata['origin'] == origin:
            return text_content
    return None
