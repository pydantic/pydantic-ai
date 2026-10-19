"""How a compaction summary written outside any provider is recognized."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from typing_extensions import TypeIs

from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequestPart, ModelResponse, ModelResponsePart


def is_summary_part(part: ModelRequestPart | ModelResponsePart) -> TypeIs[CompactionPart]:
    """Whether *part* is a compaction summary no provider produced, like `SummarizingCompaction`'s own.

    Core's provider models send such a part as framed assistant text; one carrying
    `provider_name` belongs to that provider's native compaction instead.
    """
    return isinstance(part, CompactionPart) and part.provider_name is None


def without_summaries(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """*messages* with every summary part removed, dropping any response left with nothing else."""
    out: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse) and any(is_summary_part(part) for part in message.parts):
            parts = [part for part in message.parts if not is_summary_part(part)]
            if not parts:
                continue
            message = replace(message, parts=parts)
        out.append(message)
    return out
