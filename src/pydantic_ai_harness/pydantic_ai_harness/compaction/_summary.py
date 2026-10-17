"""How a compaction summary written outside any provider is recognized."""

from __future__ import annotations

from typing_extensions import TypeIs

from pydantic_ai.messages import CompactionPart, ModelRequestPart, ModelResponsePart


def is_summary_part(part: ModelRequestPart | ModelResponsePart) -> TypeIs[CompactionPart]:
    """Whether *part* is a compaction summary no provider produced, like `SummarizingCompaction`'s own.

    Core sends such a part to every request-response model as framed assistant text; one carrying
    `provider_name` belongs to that provider's native compaction instead.
    """
    return isinstance(part, CompactionPart) and part.provider_name is None
