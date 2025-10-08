"""Vercel AI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the Vercel AI protocol,
enabling streaming event-based communication for interactive AI applications.

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts
"""

from ._utils import VERCEL_AI_DSP_HEADERS, CamelBaseModel, JSONValue, ProviderMetadata
from .adapter import VercelAIAdapter
from .event_stream import VercelAIEventStream, protocol_messages_to_pai_messages
from .request_types import (
    RequestData,
    SubmitMessage,
    TextUIPart,
    ToolOutputAvailablePart,
    UIMessage,
    UIPart,
    request_data_ta,
)
from .response_types import (
    AbortChunk,
    AbstractSSEChunk,
    DataUIMessageChunk,
    DoneChunk,
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

__all__ = [
    # Utilities
    'CamelBaseModel',
    'ProviderMetadata',
    'JSONValue',
    'VERCEL_AI_DSP_HEADERS',
    # Request types
    'RequestData',
    'TextUIPart',
    'UIMessage',
    'ToolOutputAvailablePart',
    'UIPart',
    'SubmitMessage',
    'request_data_ta',
    # Response types
    'AbstractSSEChunk',
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
    'DoneChunk',
    # Event stream and adapter
    'VercelAIEventStream',
    'VercelAIAdapter',
    'protocol_messages_to_pai_messages',
]
