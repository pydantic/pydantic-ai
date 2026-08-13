"""Realtime model profiles, mirroring [`pydantic_ai.profiles`][pydantic_ai.profiles] for the standard models."""

from __future__ import annotations as _annotations

from collections.abc import Callable

from typing_extensions import TypeAliasType, TypedDict

from ..native_tools import AbstractNativeTool


class RealtimeModelProfile(TypedDict, total=False):
    """Describes what a [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] supports, so a session can tailor its behavior to the model.

    Mirrors the shape and `supports_`-prefixed naming of
    [`ModelProfile`][pydantic_ai.profiles.ModelProfile] for the standard request-response
    [`Model`][pydantic_ai.models.Model], which realtime models don't share a hierarchy with.

    A [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] reads these flags to reject unsupported
    operations with a clear error *before* sending them, rather than letting the provider fail
    mid-session. Read a model's via [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile];
    each flag maps to the session methods a provider may not support.

    All fields are optional. Consumers treat absent boolean flags as `False` — except the handful
    documented below as defaulting to `True`, which describe a capability every provider has unless it
    says otherwise — absent `supported_native_tools` as empty, and absent sample rates as the values in
    [`DEFAULT_REALTIME_PROFILE`][pydantic_ai.realtime.codec.DEFAULT_REALTIME_PROFILE].
    """

    supports_image_input: bool
    """Whether the model accepts discrete image/video frames via
    image [`BinaryContent`][pydantic_ai.messages.BinaryContent] passed to
    [`send`][pydantic_ai.realtime.RealtimeSession.send]."""
    supports_manual_turn_control: bool
    """Whether the model supports manual turn-taking — [`commit_audio`][pydantic_ai.realtime.RealtimeSession.commit_audio],
    [`clear_audio`][pydantic_ai.realtime.RealtimeSession.clear_audio], and
    [`create_response`][pydantic_ai.realtime.RealtimeSession.create_response] (push-to-talk). When `False`
    the model drives turn-taking itself via automatic voice activity detection."""
    supports_interruption: bool
    """Whether the model supports server-side interruption — cancelling the model's in-progress response
    via [`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt]."""
    supports_output_truncation: bool
    """Whether the model can truncate its in-progress audio output to the point the user actually heard,
    via the `played_ms` argument of [`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt].

    Distinct from [`supports_interruption`][pydantic_ai.realtime.RealtimeModelProfile.supports_interruption]:
    a provider may support cancelling a response (barge-in) without supporting output truncation. OpenAI
    supports both; xAI Grok Voice supports cancellation but not truncation."""
    supports_text_output: bool
    """Whether the model can generate text instead of speech, via
    [`output_modality='text'`][pydantic_ai.realtime.RealtimeModelSettings.output_modality].

    Defaults to `True`: a realtime model that only speaks is the exception, not the rule. When `False`,
    [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] rejects `output_modality='text'` with a
    [`UserError`][pydantic_ai.exceptions.UserError] before connecting, rather than letting the provider
    fail the handshake (Gemini Live answers `1007 The requested combination of response modalities
    (TEXT) is not supported by the model`) or, worse, silently produce speech anyway (xAI)."""
    supports_session_seeding: bool
    """Whether the model can seed a session with prior conversation (`message_history`)."""
    supports_webrtc: bool
    """Whether the model supports browser WebRTC signaling, ephemeral client secrets, and a server-side
    control-plane sideband via [`answer_webrtc_offer`][pydantic_ai.realtime.RealtimeModel.answer_webrtc_offer],
    [`create_client_secret`][pydantic_ai.realtime.RealtimeModel.create_client_secret], and
    [`connect_webrtc`][pydantic_ai.realtime.RealtimeModel.connect_webrtc].

    Supported by OpenAI and Azure OpenAI. Gemini Live and xAI Grok Voice are WebSocket-only."""
    supports_seeding_images: bool
    """Whether prior images can be included when seeding a session with `message_history`."""
    supports_seeding_audio: bool
    """Whether retained user audio can be included when seeding a session with `message_history`."""
    supports_thinking: bool
    """Whether the model supports reasoning/thinking configuration via the
    [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting — OpenAI's `gpt-realtime-2*`
    reasoning models, Gemini's native-audio models, and xAI's `grok-voice-latest` and
    `grok-voice-think-*` models. When `False` (the default), a `thinking` setting is silently ignored
    rather than sent to a model that would reject it."""
    supports_async_tool_calls: bool
    """Whether the model runs tool calls asynchronously without blocking generation.

    Gemini Live maps this to `Behavior.NON_BLOCKING` on function declarations and
    `FunctionResponseScheduling.INTERRUPT` on function responses."""
    supports_tool_return_schema: bool
    """Whether the model natively renders a tool's [`return_schema`][pydantic_ai.tools.ToolDefinition.return_schema]
    (Gemini Live's function-declaration `response` schema). Where it can't, a tool that opted in via
    `include_return_schema` gets the schema injected into its description instead, exactly as on a
    standard [`Model`][pydantic_ai.models.Model]."""
    supported_native_tools: frozenset[type[AbstractNativeTool]]
    """The [native tools][pydantic_ai.native_tools.AbstractNativeTool] the model runs server-side, e.g.
    [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool].

    [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] validates the session's native
    tools against this set before connecting, raising a [`UserError`][pydantic_ai.exceptions.UserError]
    that names any the model doesn't support — mirroring the classic
    [`Model.supported_native_tools`][pydantic_ai.models.Model.supported_native_tools] check."""
    emits_input_speech_events: bool
    """Whether the provider reports when the user starts and stops speaking, as
    [`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] and
    [`RealtimeInputSpeechEndEvent`][pydantic_ai.realtime.RealtimeInputSpeechEndEvent].

    `emits_` rather than `supports_` because this describes events that appear in the stream, not an
    operation the session can invoke. The OpenAI-protocol providers (OpenAI, Azure OpenAI, xAI) emit
    them; Gemini Live does not — a UI that shows a "listening" indicator should read this flag rather
    than wait for events that will never arrive."""
    audio_input_sample_rate: int
    """The sample rate, in Hz, expected for raw PCM audio input.

    Read it via [`RealtimeSession.audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
    (or [`RealtimeModel.audio_input_sample_rate`][pydantic_ai.realtime.RealtimeModel.audio_input_sample_rate]
    before a session exists), which fall back to the default when a profile omits it."""
    audio_output_sample_rate: int
    """The sample rate, in Hz, produced in raw PCM audio output deltas.

    Read it via [`RealtimeSession.audio_output_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_output_sample_rate]
    (or [`RealtimeModel.audio_output_sample_rate`][pydantic_ai.realtime.RealtimeModel.audio_output_sample_rate]
    before a session exists), which fall back to the default when a profile omits it."""


DEFAULT_AUDIO_SAMPLE_RATE = 24000
"""The sample rate, in Hz, assumed for PCM audio when a realtime model profile doesn't specify one."""

DEFAULT_REALTIME_PROFILE: RealtimeModelProfile = {
    'supports_image_input': False,
    'supports_manual_turn_control': False,
    'supports_interruption': False,
    'supports_output_truncation': False,
    'supports_text_output': True,
    'supports_session_seeding': False,
    'supports_webrtc': False,
    'supports_seeding_images': False,
    'supports_seeding_audio': False,
    'supports_async_tool_calls': False,
    'supports_tool_return_schema': False,
    'supported_native_tools': frozenset(),
    'emits_input_speech_events': False,
    'audio_input_sample_rate': DEFAULT_AUDIO_SAMPLE_RATE,
    'audio_output_sample_rate': DEFAULT_AUDIO_SAMPLE_RATE,
}
"""Default realtime model profile values."""


RealtimeModelProfileSpec = TypeAliasType(
    'RealtimeModelProfileSpec', 'RealtimeModelProfile | Callable[[RealtimeModelProfile], RealtimeModelProfile]'
)
"""What a user may pass as a realtime model's `profile=`, mirroring [`ModelProfileSpec`][pydantic_ai.profiles.ModelProfileSpec].

Either a partial [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] merged over the
resolved profile, or a callable taking the resolved profile and returning the one to use, for full
control.
"""


def merge_realtime_profile(
    base: RealtimeModelProfile | None, *overrides: RealtimeModelProfile | None
) -> RealtimeModelProfile:
    """Merge realtime profiles, with later layers overriding earlier ones."""
    resolved: RealtimeModelProfile = {}
    if base:
        resolved.update(base)
    for override in overrides:
        if override:
            resolved.update(override)
    return resolved
