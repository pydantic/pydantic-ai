"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Gemini Live,
Amazon Nova Sonic, ...) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules
(e.g. `pydantic_ai.realtime.openai`). The high-level entry point is
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session].

A session translates the low-level codec events (the connection-facing `RealtimeEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), re-exported here for convenience, plus the realtime control-plane events defined below.
"""

from ..messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
)
from ._base import (
    AudioDelta,
    AudioInput,
    AudioRetention,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    Grounding,
    ImageInput,
    InputTranscript,
    RateLimit,
    RateLimits,
    RealtimeCapabilities,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeSessionEvent,
    Reconnected,
    ReconnectPolicy,
    SessionError,
    SessionUsage,
    Sources,
    SpeechStarted,
    SpeechStopped,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnComplete,
    WebSource,
)
from ._session import RealtimeSession, ToolRunner

__all__ = (
    # Shared message/part events (re-exported from `pydantic_ai.messages`) that a session yields.
    'SpeechPart',
    'SpeechPartDelta',
    'FunctionToolCallEvent',
    'FunctionToolResultEvent',
    'PartDeltaEvent',
    'PartEndEvent',
    'PartStartEvent',
    # Realtime codec events, control-plane events, inputs, and ABCs.
    'AudioDelta',
    'AudioInput',
    'AudioRetention',
    'CancelResponse',
    'ClearAudio',
    'CommitAudio',
    'CreateResponse',
    'Grounding',
    'ImageInput',
    'InputTranscript',
    'RateLimit',
    'RateLimits',
    'RealtimeCapabilities',
    'RealtimeConnection',
    'RealtimeEvent',
    'RealtimeInput',
    'RealtimeModel',
    'RealtimeSession',
    'RealtimeSessionEvent',
    'ReconnectPolicy',
    'Reconnected',
    'SessionError',
    'SessionUsage',
    'Sources',
    'SpeechStarted',
    'SpeechStopped',
    'TextInput',
    'ToolCall',
    'ToolResult',
    'ToolRunner',
    'Transcript',
    'TruncateOutput',
    'TurnComplete',
    'WebSource',
)
