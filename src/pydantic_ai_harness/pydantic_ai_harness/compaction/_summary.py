"""How `SummarizingCompaction` stores its summary, and how it recognizes one it wrote."""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import TypeIs

from pydantic_ai.messages import CompactionPart, ModelMessage, ModelResponse

SUMMARY_METADATA_KEY = 'pydantic-ai-harness.compaction.summary.v1'
"""`ModelResponse.metadata` key marking the response that holds a `SummarizingCompaction` summary.

Metadata never reaches the model, so a summary is recognized by this marker rather than by its text
or its part type, and a `CompactionPart` an application wrote itself is never mistaken for one.
"""


def summary_response(summary: str) -> ModelResponse:
    """The response holding *summary*: a provider-less `CompactionPart`, marked as this capability's own.

    Core's provider models send such a part as framed user-voice text.
    """
    return ModelResponse(parts=[CompactionPart(content=summary)], metadata={SUMMARY_METADATA_KEY: True})


def is_summary_message(message: ModelMessage) -> TypeIs[ModelResponse]:
    """Whether *message* is a summary response written by `SummarizingCompaction`."""
    return (
        isinstance(message, ModelResponse)
        and message.metadata is not None
        and message.metadata.get(SUMMARY_METADATA_KEY) is True
    )


def summary_text(message: ModelResponse) -> str | None:
    """The summary a summary response holds, or `None` when it holds none."""
    return next((part.content for part in message.parts if isinstance(part, CompactionPart) and part.content), None)


def without_summaries(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """*messages* without the summary responses `SummarizingCompaction` wrote; everything else is kept."""
    return [message for message in messages if not is_summary_message(message)]
