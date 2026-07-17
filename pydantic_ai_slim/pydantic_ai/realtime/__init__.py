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
    AudioDelta,
    AudioInput,
    AudioRetention,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    KnownRealtimeTranscriptionModelName,
    NativeToolParts,
    RateLimit,
    RateLimitsEvent,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSessionEvent,
    RealtimeSessionInput,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    SourcesEvent,
    SpeechStartedEvent,
    SpeechStoppedEvent,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    WebSource,
)
from ._session import RealtimeSession, ToolRunner

KnownRealtimeModelName = TypeAliasType(
    'KnownRealtimeModelName',
    Literal[
        'openai:gpt-realtime',
        'openai:gpt-realtime-2.1',
        'openai:gpt-realtime-2.1-mini',
        'xai:grok-voice-latest',
        'google:gemini-2.5-flash-native-audio-latest',
        'google:gemini-3.1-flash-live-preview',
        'bedrock:amazon.nova-2-sonic-v1:0',
        'bedrock:amazon.nova-sonic-v1:0',
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
    if provider == 'bedrock':
        from .bedrock import BedrockRealtimeModel

        return BedrockRealtimeModel(model_name)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `xai`, `google`, and `bedrock`.'
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
    'ImageInput',
    'InputTranscript',
    'KnownRealtimeTranscriptionModelName',
    'KnownRealtimeModelName',
    'NativeToolParts',
    'RateLimit',
    'RateLimitsEvent',
    'RealtimeConnection',
    'RealtimeEvent',
    'RealtimeInput',
    'RealtimeModel',
    'RealtimeModelProfile',
    'RealtimeModelSettings',
    'RealtimeSession',
    'RealtimeSessionEvent',
    'RealtimeSessionInput',
    'ReconnectPolicy',
    'ReconnectedEvent',
    'SessionErrorEvent',
    'SessionUsageEvent',
    'SourcesEvent',
    'SpeechStartedEvent',
    'SpeechStoppedEvent',
    'TextInput',
    'ToolCall',
    'ToolResult',
    'ToolRunner',
    'Transcript',
    'TruncateOutput',
    'TurnCompleteEvent',
    'WebSource',
    'infer_realtime_model',
)
