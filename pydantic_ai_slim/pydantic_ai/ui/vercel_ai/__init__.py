"""Vercel AI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the Vercel AI protocol,
enabling streaming event-based communication for interactive AI applications.

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts
"""

from ._adapter import VercelAIAdapter
from ._event_stream import VercelAIEventStream
from ._request_types import (
    RequestData,
    SubmitMessage,
    TextUIPart,
    ToolOutputAvailablePart,
    UIMessage,
    UIPart,
    request_data_ta,
)
from ._response_types import (
    AbortChunk,
    BaseChunk,
    DataUIMessageChunk,
    ErrorChunk,
    FileChunk,
    FinishChunk,
    FinishStepChunk,
    MessageMetadataChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    SourceDocumentChunk,
    SourceUrlChunk,
    StartChunk,
    StartStepChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputErrorChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    ToolOutputErrorChunk,
)
from ._utils import ProviderMetadata

__all__ = [
    # Utilities
    'ProviderMetadata',
    # Request types
    'RequestData',
    'TextUIPart',
    'UIMessage',
    'ToolOutputAvailablePart',
    'UIPart',
    'SubmitMessage',
    'request_data_ta',
    # Response types
    'BaseChunk',
    'TextStartChunk',
    'TextDeltaChunk',
    'TextEndChunk',
    'ReasoningStartChunk',
    'ReasoningDeltaChunk',
    'ReasoningEndChunk',
    'ErrorChunk',
    'ToolInputStartChunk',
    'ToolInputDeltaChunk',
    'ToolInputAvailableChunk',
    'ToolInputErrorChunk',
    'ToolOutputAvailableChunk',
    'ToolOutputErrorChunk',
    'SourceUrlChunk',
    'SourceDocumentChunk',
    'FileChunk',
    'DataUIMessageChunk',
    'StartStepChunk',
    'FinishStepChunk',
    'StartChunk',
    'FinishChunk',
    'AbortChunk',
    'MessageMetadataChunk',
    # Event stream and adapter
    'VercelAIEventStream',
    'VercelAIAdapter',
]
