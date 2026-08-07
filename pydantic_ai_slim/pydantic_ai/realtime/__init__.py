"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models, which use a persistent bidirectional
connection rather than the request-response pattern of the standard
[`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules. The
high-level entry point is [`Agent.realtime`][pydantic_ai.agent.Agent.realtime], followed by
[`AgentRealtime.session`][pydantic_ai.agent.AgentRealtime.session].

A session translates the low-level codec events (the connection-facing `RealtimeCodecEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), re-exported here for convenience, plus the realtime control-plane events defined below.
"""

from ..exceptions import UserError
from ..messages import (
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
)
from ._base import (
    AudioInput,
    AudioRetention,
    ImageInput,
    KnownRealtimeTranscriptionModelName,
    RealtimeError,
    RealtimeEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelProfileSpec,
    RealtimeModelSettings,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionInput,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
    ReconnectPolicy,
    TextInput,
    TranscriptUpdate,
    TurnDetection,
)
from ._session import RealtimeSession


def infer_realtime_model(model: str) -> RealtimeModel:
    """Infer a realtime model from a `provider:model` identifier.

    No realtime providers ship in this build yet, so every identifier is rejected; pass a
    [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] instance instead.
    """
    raise UserError(
        f'Cannot infer a realtime model from {model!r}: no realtime providers are available in this build. '
        'Pass a `RealtimeModel` instance instead.'
    )


__all__ = (
    # Shared message/part events (re-exported from `pydantic_ai.messages`) that a session yields.
    'SpeechPart',
    'SpeechPartDelta',
    'DeferredToolRequestsEvent',
    'DeferredToolResultsEvent',
    'FunctionToolCallEvent',
    'FunctionToolResultEvent',
    'PartDeltaEvent',
    'PartEndEvent',
    'PartStartEvent',
    # Realtime session ABCs, models, settings, inputs, and the control-plane events a session yields.
    # The lower-level codec vocabulary (`RealtimeConnection`, codec events, turn-control verbs, and the
    # profile helpers) lives in [`pydantic_ai.realtime.codec`][pydantic_ai.realtime.codec].
    'AudioInput',
    'AudioRetention',
    'ImageInput',
    'RealtimeInputSpeechStartEvent',
    'RealtimeInputSpeechEndEvent',
    'RealtimeInputTranscriptionErrorEvent',
    'KnownRealtimeTranscriptionModelName',
    'RealtimeEvent',
    'RealtimeError',
    'RealtimeModel',
    'RealtimeModelProfile',
    'RealtimeModelProfileSpec',
    'RealtimeModelSettings',
    'RealtimeSession',
    'RealtimeSessionInput',
    'ReconnectPolicy',
    'RealtimeSessionErrorEvent',
    'RealtimeSessionReconnectEvent',
    'TextInput',
    'TranscriptUpdate',
    'RealtimeTurnCompleteEvent',
    'TurnDetection',
    'RealtimeResponseInterruptedEvent',
    'infer_realtime_model',
)
