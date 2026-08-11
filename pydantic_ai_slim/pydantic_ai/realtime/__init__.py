"""Realtime multimodal session support for bidirectional streaming models.

This package adds support for native speech-to-speech models (OpenAI Realtime, Azure OpenAI,
Gemini Live, and xAI Grok Voice) which use a persistent bidirectional connection rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

The provider-agnostic ABCs and event types live here; concrete providers live in submodules
(e.g. `pydantic_ai.realtime.openai`). The high-level entry point is
[`Agent.realtime`][pydantic_ai.agent.Agent.realtime], followed by
[`AgentRealtime.session`][pydantic_ai.agent.AgentRealtime.session].

A session translates the low-level codec events (the connection-facing `RealtimeCodecEvent` vocabulary)
into the shared message/part event vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages]
([`PartStartEvent`][pydantic_ai.messages.PartStartEvent], [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent],
...), plus the realtime control-plane events defined below.
"""

from typing import Literal

from typing_extensions import TypeAliasType

from ..exceptions import UserError
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

KnownRealtimeModelName = TypeAliasType(
    'KnownRealtimeModelName',
    Literal[
        'openai:gpt-realtime',
        'openai:gpt-realtime-2.1',
        'openai:gpt-realtime-2.1-mini',
        'azure:gpt-realtime',
        'xai:grok-voice-latest',
        'xai:grok-voice-think-fast-2.0',
        'google:gemini-2.5-flash-native-audio-latest',
        'google:gemini-3.1-flash-live-preview',
    ],
)
"""Known realtime model identifiers, surfaced for autocomplete."""


def infer_realtime_model(model: KnownRealtimeModelName | str) -> RealtimeModel:
    """Infer a realtime model from a `provider:model` identifier.

    The provider is one of `openai`, `azure`, `xai`, `google` (the Gemini Developer API), or
    `google-cloud` (Vertex AI) — e.g. `openai:gpt-realtime` — or a
    [Pydantic AI Gateway](../gateway.md) route (`gateway/openai:gpt-realtime`,
    `gateway/google:gemini-live-2.5-flash`), which connects through the gateway's built-in provider —
    the provider string is passed to the realtime model as its `provider`, so authentication and the
    base URL come from [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider].
    """
    provider, separator, model_name = model.partition(':')
    if not separator or not model_name:
        raise UserError(
            f'Realtime model identifiers use the `provider:model` format (e.g. `openai:gpt-realtime`); got {model!r}.'
        )
    model_kind = provider
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        # Same alias resolution as `infer_model`: the gateway's Google upstream is the Vertex route,
        # so `gateway/google` collapses onto `google-cloud`. The un-normalized string stays the
        # model's `provider`, whose handshake reads the gateway base URL and bearer key from
        # `gateway_provider` (the OpenAI protocol already carries the same trace context the
        # gateway's HTTP request hook would add).
        model_kind = normalize_gateway_provider(model_kind)
        if model_kind not in ('openai', 'google-cloud'):
            raise UserError(
                f'Realtime model provider {provider!r} cannot be routed through the Pydantic AI Gateway. '
                'Supported gateway routes are `gateway/openai` and `gateway/google`.'
            )

    if model_kind == 'openai':
        from .openai import OpenAIRealtimeModel

        return OpenAIRealtimeModel(model_name, provider=provider)
    if model_kind == 'azure':
        from .azure import AzureRealtimeModel

        return AzureRealtimeModel(model_name)
    if model_kind == 'xai':
        from .xai import XaiRealtimeModel

        return XaiRealtimeModel(model_name)
    # `google` is the Gemini Developer API and `google-cloud` is Vertex AI, exactly as in `infer_model`.
    if model_kind in ('google', 'google-cloud'):
        from .google import GoogleRealtimeModel

        return GoogleRealtimeModel(model_name, provider=provider)
    raise UserError(
        f'Unknown realtime model provider {provider!r}. Supported providers are `openai`, `azure`, '
        '`xai`, `google`, and `google-cloud`, or `gateway/openai` / `gateway/google` to route OpenAI '
        'or Gemini Live realtime through the Pydantic AI Gateway.'
    )


__all__ = (
    # Realtime session ABCs, models, settings, inputs, and the control-plane events a session yields.
    # The shared message/part events a session also yields (`SpeechPart`, `PartStartEvent`,
    # `FunctionToolCallEvent`, ...) live in `pydantic_ai.messages` and the root `pydantic_ai`.
    # The lower-level codec vocabulary (`RealtimeConnection`, codec events, turn-control verbs, and the
    # profile helpers) lives in [`pydantic_ai.realtime.codec`][pydantic_ai.realtime.codec].
    'AudioInput',
    'AudioRetention',
    'ImageInput',
    'RealtimeInputSpeechStartEvent',
    'RealtimeInputSpeechEndEvent',
    'RealtimeInputTranscriptionErrorEvent',
    'KnownRealtimeTranscriptionModelName',
    'KnownRealtimeModelName',
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
