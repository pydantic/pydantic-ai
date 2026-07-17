"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Gemini Live,
Amazon Nova Sonic, ...) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules
(e.g. `pydantic_ai.realtime.openai`). The high-level entry point is
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session].

A session translates the low-level codec events (the connection-facing `RealtimeCodecEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), re-exported here for convenience, plus the realtime control-plane events defined below.
"""

from typing import Literal

from typing_extensions import TypeAliasType

from ..exceptions import UserError
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
    DEFAULT_REALTIME_PROFILE,
    AudioDelta,
    AudioInput,
    AudioRetention,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscript,
    KnownRealtimeTranscriptionModelName,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeError,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSessionInput,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionUsageEvent,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    merge_realtime_profile,
)
from ._session import RealtimeSession

KnownRealtimeModelName = TypeAliasType(
    'KnownRealtimeModelName',
    Literal[
        'openai:gpt-realtime',
        'openai:gpt-realtime-2.1',
        'openai:gpt-realtime-2.1-mini',
        'xai:grok-voice-latest',
        'google:gemini-2.5-flash-native-audio-latest',
        'google:gemini-3.1-flash-live-preview',
    ],
)
"""Known realtime model identifiers, surfaced for autocomplete."""


def infer_realtime_model(model: KnownRealtimeModelName | str) -> RealtimeModel:
    """Infer a realtime model from a `provider:model` identifier."""
    provider, separator, model_name = model.partition(':')
    if not separator or not model_name:
        provider = ''
    if provider == 'openai':
        from .openai import OpenAIRealtimeModel

        return OpenAIRealtimeModel(model_name)
    if provider == 'xai':
        from .xai import XaiRealtimeModel

        return XaiRealtimeModel(model_name)
    if provider == 'google':
        from .google import GoogleRealtimeModel

        return GoogleRealtimeModel(model_name)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `xai`, and `google`.'
    )


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
    'InputTranscript',
    'KnownRealtimeTranscriptionModelName',
    'KnownRealtimeModelName',
    'RealtimeConnection',
    'RealtimeCodecEvent',
    'RealtimeEvent',
    'RealtimeError',
    'RealtimeInput',
    'RealtimeModel',
    'RealtimeModelProfile',
    'RealtimeModelSettings',
    'RealtimeSession',
    'RealtimeSessionInput',
    'ReconnectPolicy',
    'ReconnectedEvent',
    'SessionUsageEvent',
    'InputSpeechStartEvent',
    'InputSpeechEndEvent',
    'ToolCall',
    'ToolResult',
    'Transcript',
    'TruncateOutput',
    'TurnCompleteEvent',
    'infer_realtime_model',
    'ImageInput',
    'TextInput',
    'DEFAULT_REALTIME_PROFILE',
    'merge_realtime_profile',
)
