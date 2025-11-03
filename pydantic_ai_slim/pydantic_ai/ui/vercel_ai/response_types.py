"""Vercel AI response types (SSE chunks).

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.59/packages/ai/src/ui-message-stream/ui-message-chunks.ts
"""

from abc import ABC
from typing import Annotated, Any, Literal

from pydantic import Field

from ._utils import CamelBaseModel

# Technically this is recursive union of JSON types; for simplicity, we call it Any
JSONValue = Any
ProviderMetadata = dict[str, dict[str, JSONValue]]
"""Provider metadata."""


class BaseChunk(CamelBaseModel, ABC):
    """Abstract base class for response SSE events."""

    def encode(self) -> str:
        return self.model_dump_json(by_alias=True, exclude_none=True)


class TextStartChunk(BaseChunk):
    """Text start chunk."""

    type: Literal['text-start'] = 'text-start'
    id: str
    provider_metadata: ProviderMetadata | None = None


class TextDeltaChunk(BaseChunk):
    """Text delta chunk."""

    type: Literal['text-delta'] = 'text-delta'
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = None


class TextEndChunk(BaseChunk):
    """Text end chunk."""

    type: Literal['text-end'] = 'text-end'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningStartChunk(BaseChunk):
    """Reasoning start chunk."""

    type: Literal['reasoning-start'] = 'reasoning-start'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningDeltaChunk(BaseChunk):
    """Reasoning delta chunk."""

    type: Literal['reasoning-delta'] = 'reasoning-delta'
    id: str
    delta: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningEndChunk(BaseChunk):
    """Reasoning end chunk."""

    type: Literal['reasoning-end'] = 'reasoning-end'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ErrorChunk(BaseChunk):
    """Error chunk."""

    type: Literal['error'] = 'error'
    error_text: str


class ToolInputStartChunk(BaseChunk):
    """Tool input start chunk."""

    type: Literal['tool-input-start'] = 'tool-input-start'
    tool_call_id: str
    tool_name: str
    provider_executed: bool | None = None
    dynamic: bool | None = None


class ToolInputDeltaChunk(BaseChunk):
    """Tool input delta chunk."""

    type: Literal['tool-input-delta'] = 'tool-input-delta'
    tool_call_id: str
    input_text_delta: str


class ToolOutputAvailableChunk(BaseChunk):
    """Tool output available chunk."""

    type: Literal['tool-output-available'] = 'tool-output-available'
    tool_call_id: str
    output: Any
    provider_executed: bool | None = None
    dynamic: bool | None = None
    preliminary: bool | None = None


class ToolInputAvailableChunk(BaseChunk):
    """Tool input available chunk."""

    type: Literal['tool-input-available'] = 'tool-input-available'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None


class ToolInputErrorChunk(BaseChunk):
    """Tool input error chunk."""

    type: Literal['tool-input-error'] = 'tool-input-error'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None
    error_text: str


class ToolOutputErrorChunk(BaseChunk):
    """Tool output error chunk."""

    type: Literal['tool-output-error'] = 'tool-output-error'
    tool_call_id: str
    error_text: str
    provider_executed: bool | None = None
    dynamic: bool | None = None


class SourceUrlChunk(BaseChunk):
    """Source URL chunk."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None


class SourceDocumentChunk(BaseChunk):
    """Source document chunk."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None


class FileChunk(BaseChunk):
    """File chunk."""

    type: Literal['file'] = 'file'
    url: str
    media_type: str


class DataChunk(BaseChunk):
    """Data chunk with dynamic type."""

    type: Annotated[str, Field(pattern=r'^data-')]
    data: Any


class StartStepChunk(BaseChunk):
    """Start step chunk."""

    type: Literal['start-step'] = 'start-step'


class FinishStepChunk(BaseChunk):
    """Finish step chunk."""

    type: Literal['finish-step'] = 'finish-step'


class StartChunk(BaseChunk):
    """Start chunk."""

    type: Literal['start'] = 'start'
    message_id: str | None = None
    message_metadata: Any | None = None


class FinishChunk(BaseChunk):
    """Finish chunk."""

    type: Literal['finish'] = 'finish'
    message_metadata: Any | None = None


class AbortChunk(BaseChunk):
    """Abort chunk."""

    type: Literal['abort'] = 'abort'


class MessageMetadataChunk(BaseChunk):
    """Message metadata chunk."""

    type: Literal['message-metadata'] = 'message-metadata'
    message_metadata: Any


class DoneChunk(BaseChunk):
    """Done chunk."""

    type: Literal['done'] = 'done'

    def encode(self) -> str:
        return '[DONE]'
