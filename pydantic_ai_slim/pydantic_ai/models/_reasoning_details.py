"""Codec for the `reasoning_details` wire format used by OpenAI-compatible APIs to carry reasoning.

The format originated with OpenRouter and has been adopted by other providers (e.g. Snowflake
Cortex). Each detail carries a `type` (`reasoning.text`, `reasoning.summary`, or
`reasoning.encrypted`) and a `format` naming the upstream provider's reasoning representation
(`anthropic-claude-v1`, `openai-responses-v1`, ...), so the same codec round-trips reasoning for
any provider that speaks it.
"""

from __future__ import annotations as _annotations

from typing import Literal

from pydantic import BaseModel
from typing_extensions import assert_never

from ..messages import ThinkingPart


class BaseReasoningDetail(BaseModel, frozen=True):
    """Common fields shared across all reasoning detail types."""

    id: str | None = None
    format: (
        Literal['unknown', 'openai-responses-v1', 'anthropic-claude-v1', 'xai-responses-v1', 'google-gemini-v1']
        | str
        | None
    ) = None
    index: int | None = None
    type: Literal['reasoning.text', 'reasoning.summary', 'reasoning.encrypted']


class ReasoningSummary(BaseReasoningDetail, frozen=True):
    """Represents a high-level summary of the reasoning process."""

    type: Literal['reasoning.summary']
    summary: str = ''


class ReasoningEncrypted(BaseReasoningDetail, frozen=True):
    """Represents encrypted reasoning data."""

    type: Literal['reasoning.encrypted']
    data: str = ''


class ReasoningText(BaseReasoningDetail, frozen=True):
    """Represents raw text reasoning."""

    type: Literal['reasoning.text']
    text: str = ''
    signature: str | None = None


ReasoningDetail = ReasoningSummary | ReasoningEncrypted | ReasoningText


def from_reasoning_detail(reasoning: ReasoningDetail, provider_name: str) -> ThinkingPart:
    provider_details = reasoning.model_dump(include={'format', 'index', 'type'})
    if isinstance(reasoning, ReasoningText):
        return ThinkingPart(
            id=reasoning.id,
            content=reasoning.text,
            signature=reasoning.signature,
            provider_name=provider_name,
            provider_details=provider_details,
        )
    elif isinstance(reasoning, ReasoningSummary):
        return ThinkingPart(
            id=reasoning.id, content=reasoning.summary, provider_name=provider_name, provider_details=provider_details
        )
    elif isinstance(reasoning, ReasoningEncrypted):
        return ThinkingPart(
            id=reasoning.id,
            content='',
            signature=reasoning.data,
            provider_name=provider_name,
            provider_details=provider_details,
        )
    else:
        assert_never(reasoning)


def into_reasoning_detail(thinking_part: ThinkingPart) -> ReasoningDetail | None:
    if thinking_part.provider_details is None:  # pragma: lax no cover
        return None

    data = BaseReasoningDetail.model_validate(thinking_part.provider_details)

    if data.type == 'reasoning.text':
        return ReasoningText(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            text=thinking_part.content,
            signature=thinking_part.signature,
        )
    elif data.type == 'reasoning.summary':
        return ReasoningSummary(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            summary=thinking_part.content,
        )
    elif data.type == 'reasoning.encrypted':
        assert thinking_part.signature is not None
        return ReasoningEncrypted(
            type=data.type,
            id=thinking_part.id,
            format=data.format,
            index=data.index,
            data=thinking_part.signature,
        )
    else:
        assert_never(data.type)
