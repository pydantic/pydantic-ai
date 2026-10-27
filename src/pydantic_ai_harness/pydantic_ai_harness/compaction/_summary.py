"""How `SummarizingCompaction` stores its summary, and how it recognizes one it wrote."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from typing_extensions import TypeIs

from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelRequestPart, ModelResponsePart

SUMMARY_METADATA_KEY = 'pydantic-ai-harness.compaction.summary.v1'
"""`CompactionPart.metadata` key marking a summary `SummarizingCompaction` wrote.

Metadata never reaches the model, so a summary is recognized by this marker rather than by its text
or its part type, and a `CompactionPart` an application wrote itself is never mistaken for one. The
marker is on the part, not the message, so it survives the history cleanup that merges adjacent
requests.
"""

SUMMARY_TIME_METADATA_KEY = 'pydantic-ai-harness.compaction.summary.created_at'
"""`CompactionPart.metadata` key holding when `SummarizingCompaction` wrote the summary, as an ISO 8601 string.

The summary's request takes the timestamp of the request it is merged with, which can be an older prompt the
compaction preserved, so the time it was written is kept on the part.
"""


@dataclass(frozen=True)
class WrittenSummary:
    """A summary and when it was written, recorded together by a durable engine so a replay restores both."""

    text: str
    created_at: datetime


def summary_request(summary: str, created_at: datetime | None = None) -> ModelRequest:
    """The request holding *summary*: a provider-less `CompactionPart`, marked as this capability's own.

    Core's provider models send such a part as framed user-voice text. *created_at* is recorded on the part
    when given; a summary replayed from a result recorded before the time was kept goes without it.
    """
    metadata: dict[str, Any] = {SUMMARY_METADATA_KEY: True}
    if created_at is not None:
        metadata[SUMMARY_TIME_METADATA_KEY] = created_at.isoformat()
    return ModelRequest(
        parts=[CompactionPart(content=summary, metadata=metadata)], timestamp=created_at or datetime.now(UTC)
    )


def summary_time(part: ModelRequestPart | ModelResponsePart) -> datetime | None:
    """When `SummarizingCompaction` wrote *part*, or `None` when it isn't a summary it wrote or the time is unreadable."""
    if not is_summary_part(part):
        return None
    assert part.metadata is not None
    value = part.metadata.get(SUMMARY_TIME_METADATA_KEY)
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def is_summary_part(part: ModelRequestPart | ModelResponsePart) -> TypeIs[CompactionPart]:
    """Whether *part* is a summary written by `SummarizingCompaction`.

    That is the whole record: the marker, no provider data, and a summary to read. Anything else
    carrying the marker is left alone as ordinary content.
    """
    return (
        isinstance(part, CompactionPart)
        and part.metadata is not None
        and part.metadata.get(SUMMARY_METADATA_KEY) is True
        and part.provider_name is None
        and part.id is None
        and part.provider_details is None
        and bool(part.content)
    )


def without_summaries(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """*messages* without the summaries `SummarizingCompaction` wrote, dropping a request left with nothing else."""
    out: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest) and any(is_summary_part(part) for part in message.parts):
            parts = [part for part in message.parts if not is_summary_part(part)]
            if not parts:
                continue
            message = replace(message, parts=parts)
        out.append(message)
    return out
