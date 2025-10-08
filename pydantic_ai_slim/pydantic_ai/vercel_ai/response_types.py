"""Convert to Python from.

https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts

Mostly with Claude.
"""

from typing import Any, Literal

from ._utils import CamelBaseModel, ProviderMetadata


class AbstractSSEChunk(CamelBaseModel):
    """Abstract base class for response SSE even."""

    def sse(self) -> str:
        return self.model_dump_json(exclude_none=True, by_alias=True)


class TextStartChunk(AbstractSSEChunk):
    """Text start chunk."""

    type: Literal['text-start'] = 'text-start'
    id: str
    provider_metadata: ProviderMetadata | None = None


class TextDeltaChunk(AbstractSSEChunk):
    """Text delta chunk."""

    type: Literal['text-delta'] = 'text-delta'
    delta: str
    id: str
    provider_metadata: ProviderMetadata | None = None


class TextEndChunk(AbstractSSEChunk):
    """Text end chunk."""

    type: Literal['text-end'] = 'text-end'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningStartChunk(AbstractSSEChunk):
    """Reasoning start chunk."""

    type: Literal['reasoning-start'] = 'reasoning-start'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningDeltaChunk(AbstractSSEChunk):
    """Reasoning delta chunk."""

    type: Literal['reasoning-delta'] = 'reasoning-delta'
    id: str
    delta: str
    provider_metadata: ProviderMetadata | None = None


class ReasoningEndChunk(AbstractSSEChunk):
    """Reasoning end chunk."""

    type: Literal['reasoning-end'] = 'reasoning-end'
    id: str
    provider_metadata: ProviderMetadata | None = None


class ErrorChunk(AbstractSSEChunk):
    """Error chunk."""

    type: Literal['error'] = 'error'
    error_text: str


class ToolInputAvailableChunk(AbstractSSEChunk):
    """Tool input available chunk."""

    type: Literal['tool-input-available'] = 'tool-input-available'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None


class ToolInputErrorChunk(AbstractSSEChunk):
    """Tool input error chunk."""

    type: Literal['tool-input-error'] = 'tool-input-error'
    tool_call_id: str
    tool_name: str
    input: Any
    provider_executed: bool | None = None
    provider_metadata: ProviderMetadata | None = None
    dynamic: bool | None = None
    error_text: str


class ToolOutputAvailableChunk(AbstractSSEChunk):
    """Tool output available chunk."""

    type: Literal['tool-output-available'] = 'tool-output-available'
    tool_call_id: str
    output: Any
    provider_executed: bool | None = None
    dynamic: bool | None = None
    preliminary: bool | None = None


class ToolOutputErrorChunk(AbstractSSEChunk):
    """Tool output error chunk."""

    type: Literal['tool-output-error'] = 'tool-output-error'
    tool_call_id: str
    error_text: str
    provider_executed: bool | None = None
    dynamic: bool | None = None


class ToolInputStartChunk(AbstractSSEChunk):
    """Tool input start chunk."""

    type: Literal['tool-input-start'] = 'tool-input-start'
    tool_call_id: str
    tool_name: str
    provider_executed: bool | None = None
    dynamic: bool | None = None


class ToolInputDeltaChunk(AbstractSSEChunk):
    """Tool input delta chunk."""

    type: Literal['tool-input-delta'] = 'tool-input-delta'
    tool_call_id: str
    input_text_delta: str


# Source chunk types
class SourceUrlChunk(AbstractSSEChunk):
    """Source URL chunk."""

    type: Literal['source-url'] = 'source-url'
    source_id: str
    url: str
    title: str | None = None
    provider_metadata: ProviderMetadata | None = None


class SourceDocumentChunk(AbstractSSEChunk):
    """Source document chunk."""

    type: Literal['source-document'] = 'source-document'
    source_id: str
    media_type: str
    title: str
    filename: str | None = None
    provider_metadata: ProviderMetadata | None = None


class FileChunk(AbstractSSEChunk):
    """File chunk."""

    type: Literal['file'] = 'file'
    url: str
    media_type: str


class DataUIMessageChunk(AbstractSSEChunk):
    """Data UI message chunk with dynamic type."""

    type: str  # Will be f"data-{NAME}"
    data: Any


class StartStepChunk(AbstractSSEChunk):
    """Start step chunk."""

    type: Literal['start-step'] = 'start-step'


class FinishStepChunk(AbstractSSEChunk):
    """Finish step chunk."""

    type: Literal['finish-step'] = 'finish-step'


# Message lifecycle chunk types
class StartChunk(AbstractSSEChunk):
    """Start chunk."""

    type: Literal['start'] = 'start'
    message_id: str | None = None
    message_metadata: Any | None = None


class FinishChunk(AbstractSSEChunk):
    """Finish chunk."""

    type: Literal['finish'] = 'finish'
    message_metadata: Any | None = None


class AbortChunk(AbstractSSEChunk):
    """Abort chunk."""

    type: Literal['abort'] = 'abort'


class MessageMetadataChunk(AbstractSSEChunk):
    """Message metadata chunk."""

    type: Literal['message-metadata'] = 'message-metadata'
    message_metadata: Any
