"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Azure OpenAI,
Gemini Live, and xAI Grok Voice) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules
(e.g. `pydantic_ai.realtime.openai`). The high-level entry point is
[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session].

A session translates the low-level codec events (the connection-facing `RealtimeCodecEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), re-exported here for convenience, plus the realtime control-plane events defined below.
"""

from typing import TYPE_CHECKING, Literal

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
    AudioInput,
    AudioRetention,
    ImageInput,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscriptionFailedEvent,
    KnownRealtimeTranscriptionModelName,
    RealtimeClientSecret,
    RealtimeError,
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSessionInput,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    TurnCompleteEvent,
    TurnDetection,
    WebRTCAnswer,
    WebRTCCall,
)
from ._session import RealtimeSession

if TYPE_CHECKING:
    from .azure import AzureRealtimeModel

KnownRealtimeModelName = TypeAliasType(
    'KnownRealtimeModelName',
    Literal[
        'openai:gpt-realtime',
        'openai:gpt-realtime-2.1',
        'openai:gpt-realtime-2.1-mini',
        'azure:gpt-realtime',
        'xai:grok-voice-latest',
        'google:gemini-2.5-flash-native-audio-latest',
        'google:gemini-3.1-flash-live-preview',
    ],
)
"""Known realtime model identifiers, surfaced for autocomplete."""


def infer_realtime_model(model: KnownRealtimeModelName | str) -> RealtimeModel:
    """Infer a realtime model from a `provider:model` identifier.

    Accepts a bare provider (`openai`, `azure`, `xai`, `google`) or a
    [Pydantic AI Gateway](../gateway.md) route (`gateway/openai:gpt-realtime`,
    `gateway/google:gemini-live-2.5-flash`), which connects through the gateway's built-in provider —
    the provider string is passed to the realtime model as its `provider`, so authentication and the
    base URL come from [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider].
    """
    provider, separator, model_name = model.partition(':')
    if not separator or not model_name:
        provider = ''
    # `gateway/openai` routes the OpenAI realtime protocol through the Pydantic AI Gateway: the
    # provider string is passed straight to `OpenAIRealtimeModel`, whose handshake reads the gateway
    # base URL and bearer key from `gateway_provider` and already carries the same trace context the
    # gateway's HTTP request hook would add. xAI isn't a gateway upstream, so it has no route.
    if provider in ('openai', 'gateway/openai'):
        from .openai import OpenAIRealtimeModel

        return OpenAIRealtimeModel(model_name, provider=provider)
    if provider == 'azure':
        from .azure import AzureRealtimeModel

        return AzureRealtimeModel(model_name)
    if provider == 'xai':
        from .xai import XaiRealtimeModel

        return XaiRealtimeModel(model_name)
    # `gateway/google` (and its `gateway/google-cloud` alias) route Gemini Live through the gateway's
    # Vertex upstream: the provider string flows into `GoogleRealtimeModel`, which resolves it via
    # `gateway_provider` and adds the gateway's bearer auth to the `google-genai` WebSocket handshake.
    if provider in ('google', 'gateway/google', 'gateway/google-cloud'):
        from .google import GoogleRealtimeModel

        return GoogleRealtimeModel(model_name, provider=provider)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `azure`, '
        '`xai`, and `google`, or `gateway/openai` / `gateway/google` to route OpenAI or Gemini Live '
        'realtime through the Pydantic AI Gateway.'
    )


def __getattr__(name: str) -> object:
    if name == 'AzureRealtimeModel':
        from .azure import AzureRealtimeModel

        return AzureRealtimeModel
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = (
    # Shared message/part events (re-exported from `pydantic_ai.messages`) that a session yields.
    'SpeechPart',
    'SpeechPartDelta',
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
    'AzureRealtimeModel',
    'ImageInput',
    'InputSpeechStartEvent',
    'InputSpeechEndEvent',
    'InputTranscriptionFailedEvent',
    'KnownRealtimeTranscriptionModelName',
    'KnownRealtimeModelName',
    'RealtimeClientSecret',
    'RealtimeEvent',
    'RealtimeError',
    'RealtimeModel',
    'RealtimeModelProfile',
    'RealtimeModelSettings',
    'RealtimeSession',
    'RealtimeSessionInput',
    'ReconnectPolicy',
    'ReconnectedEvent',
    'SessionErrorEvent',
    'SessionUsageEvent',
    'TextInput',
    'TurnDetection',
    'TurnCompleteEvent',
    'WebRTCAnswer',
    'WebRTCCall',
    'infer_realtime_model',
)
