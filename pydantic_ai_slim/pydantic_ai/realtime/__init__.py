"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Azure OpenAI,
Gemini Live, and xAI Grok Voice) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic pieces mirror the request-response layout: `pydantic_ai.realtime.model` holds
[`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] and model inference,
`pydantic_ai.realtime.settings` the settings vocabulary, `pydantic_ai.realtime.profiles` the model
profiles, and [`pydantic_ai.realtime.codec`][pydantic_ai.realtime.codec] the low-level
connection vocabulary; concrete providers live in submodules (e.g. `pydantic_ai.realtime.openai`).
The high-level entry point is [`Agent.realtime`][pydantic_ai.agent.Agent.realtime], followed by
[`AgentRealtime.session`][pydantic_ai.agent.AgentRealtime.session].

A session translates the low-level codec events (the connection-facing `RealtimeCodecEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), plus the realtime control-plane events defined below.
"""

from ..messages import (
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
)
from ._session import RealtimeEvent, RealtimeSession, TranscriptUpdate
from .codec import RealtimeSessionInput
from .model import (
    KnownRealtimeModelName,
    RealtimeClientSecret,
    RealtimeError,
    RealtimeModel,
    RealtimeProviderSession,
    WebRTCAnswer,
    WebRTCSession,
    infer_realtime_model,
)
from .profiles import RealtimeModelProfile, RealtimeModelProfileSpec
from .settings import (
    AudioRetention,
    KnownRealtimeTranscriptionModelName,
    RealtimeModelSettings,
    ReconnectPolicy,
    TurnDetection,
)

__all__ = (
    # Realtime session ABCs, models, settings, and the control-plane events a session yields.
    # Session input reuses the shared message vocabulary (`str`, `BinaryContent`, `BinaryImage`,
    # `BinaryAudio`), and the shared message/part events a session also yields (`SpeechPart`,
    # `PartStartEvent`, `FunctionToolCallEvent`, ...) likewise live in `pydantic_ai.messages` and the
    # root `pydantic_ai`.
    # The lower-level codec vocabulary (`RealtimeConnection`, codec events, turn-control verbs, and the
    # profile helpers) lives in [`pydantic_ai.realtime.codec`][pydantic_ai.realtime.codec].
    'AudioRetention',
    'RealtimeInputSpeechStartEvent',
    'RealtimeInputSpeechEndEvent',
    'RealtimeOutputSpeechStartEvent',
    'RealtimeOutputSpeechEndEvent',
    'RealtimeInputTranscriptionErrorEvent',
    'KnownRealtimeTranscriptionModelName',
    'KnownRealtimeModelName',
    'RealtimeEvent',
    'RealtimeClientSecret',
    'RealtimeError',
    'RealtimeModel',
    'RealtimeModelProfile',
    'RealtimeModelProfileSpec',
    'RealtimeModelSettings',
    'RealtimeSession',
    'RealtimeSessionInput',
    'ReconnectPolicy',
    'RealtimeProviderSession',
    'RealtimeSessionErrorEvent',
    'RealtimeSessionReconnectEvent',
    'TranscriptUpdate',
    'RealtimeTurnCompleteEvent',
    'TurnDetection',
    'RealtimeResponseInterruptedEvent',
    'WebRTCAnswer',
    'WebRTCSession',
    'infer_realtime_model',
)
