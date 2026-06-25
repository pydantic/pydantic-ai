"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Gemini Live,
Amazon Nova Sonic, ...) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules
(e.g. `pydantic_ai.realtime.openai`). The high-level entry point is
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session].
"""

from ._base import (
    AudioDelta,
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RateLimit,
    RateLimits,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeSessionEvent,
    Reconnected,
    SessionError,
    Sources,
    SpeechStarted,
    SpeechStopped,
    TextInput,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnComplete,
    Usage,
    WebSource,
)
from ._session import RealtimeSession, ToolRunner

__all__ = (
    'AudioDelta',
    'AudioInput',
    'CancelResponse',
    'ClearAudio',
    'CommitAudio',
    'CreateResponse',
    'ImageInput',
    'InputTranscript',
    'RateLimit',
    'RateLimits',
    'RealtimeConnection',
    'RealtimeEvent',
    'RealtimeInput',
    'RealtimeModel',
    'RealtimeSession',
    'RealtimeSessionEvent',
    'Reconnected',
    'SessionError',
    'Sources',
    'SpeechStarted',
    'SpeechStopped',
    'TextInput',
    'ToolCall',
    'ToolCallCompleted',
    'ToolCallStarted',
    'ToolResult',
    'ToolRunner',
    'Transcript',
    'TruncateOutput',
    'TurnComplete',
    'Usage',
    'WebSource',
)
