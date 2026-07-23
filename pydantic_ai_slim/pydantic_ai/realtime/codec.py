"""Low-level *codec* vocabulary for realtime providers.

Most users only need the session-level API in [`pydantic_ai.realtime`][pydantic_ai.realtime]
([`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session], the events a session yields,
and the content passed to [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send]). This
submodule holds the lower-level vocabulary used when *implementing* a realtime provider or consuming a
[`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] directly: the raw codec events a
connection yields to the session, the turn-control verbs and inputs a connection accepts, and the
model-profile merge helpers.
"""

from __future__ import annotations as _annotations

from ._base import (
    DEFAULT_REALTIME_PROFILE,
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    Transcript,
    TruncateOutput,
    merge_realtime_profile,
)

__all__ = (
    # Connection ABC and the event/input unions it exchanges with a session.
    'RealtimeConnection',
    'RealtimeCodecEvent',
    'RealtimeInput',
    # Codec events a connection yields.
    'AudioDelta',
    'Transcript',
    'InputTranscript',
    'ToolCall',
    'ToolResult',
    'ToolCallCancelled',
    'ConversationCreated',
    'ConversationItemCreated',
    # Turn-control verbs a connection accepts.
    'CommitAudio',
    'ClearAudio',
    'CreateResponse',
    'CancelResponse',
    'TruncateOutput',
    # Model-profile helpers for provider implementations.
    'merge_realtime_profile',
    'DEFAULT_REALTIME_PROFILE',
)
