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
    says otherwise — absent `supported_native_tools` as empty, absent sample rates as the values in
    [`DEFAULT_REALTIME_PROFILE`][pydantic_ai.realtime.codec.DEFAULT_REALTIME_PROFILE], and an absent
    `context_window` as unknown.
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
    thinking_always_enabled: bool
    """Whether the model always reasons, so thinking cannot be turned off. Default: `False`.

    Mirrors [`ModelProfile.thinking_always_enabled`][pydantic_ai.profiles.ModelProfile.thinking_always_enabled].
    When `True`, a [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting of `False` is
    ignored rather than sent, and a model whose API *requires* a thinking configuration gets one even when the
    session didn't ask for one — `gemini-3.8-live-extended-thinking` closes the handshake with `1007 Thinking
    level must be specified for this model` otherwise."""
    supports_async_tool_calls: bool
    """Whether the model runs tool calls asynchronously without blocking generation.

    Gemini Live maps this to `Behavior.NON_BLOCKING` on function declarations and
    `FunctionResponseScheduling.INTERRUPT` on function responses."""
    requires_async_tool_calls: bool
    """Whether the model *only* runs tool calls asynchronously, having no blocking mode. Default: `False`.

    Stronger than [`supports_async_tool_calls`][pydantic_ai.realtime.RealtimeModelProfile.supports_async_tool_calls],
    which describes a mode a session opts into: when this is `True` async tool calls are the only mode the model
    has, so the provider sends them whether or not the session asked. `gemini-3.8-live-extended-thinking` reasons
    and speaks at the same time, and closes the session with `1007 BLOCKING function calls are not supported for
    this model` on anything else."""
    supports_async_tool_call_scheduling: bool
    """Whether the model lets the session say *when* an async tool's result is delivered. Default: `False`.

    Only meaningful alongside [`supports_async_tool_calls`][pydantic_ai.realtime.RealtimeModelProfile.supports_async_tool_calls].
    Gemini Live maps it to `FunctionResponseScheduling.INTERRUPT` on function responses, so the result cuts
    into the speech the model is producing while the tool runs. `gemini-3.8-live-extended-thinking` schedules
    its own results around its reasoning and closes the session with `1007 Function response scheduling is not
    supported for this model` if the field is sent at all, so it reports `False` and the result goes back
    unscheduled."""
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
    context_window: int | None
    """The maximum number of tokens the model can hold in a session, input and output combined. Default: `None` (unknown).

    When no profile layer sets this, [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile]
    fills it in from [genai-prices](https://github.com/pydantic/genai-prices) data if the model is known
    there. Set it explicitly for models it doesn't know, e.g. `profile={'context_window': 128_000}`.
    Read it via [`RealtimeModel.context_window`][pydantic_ai.realtime.RealtimeModel.context_window]."""


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
    'thinking_always_enabled': False,
    'supports_async_tool_calls': False,
    'requires_async_tool_calls': False,
    'supports_async_tool_call_scheduling': False,
    'supports_tool_return_schema': False,
    'supported_native_tools': frozenset(),
    'emits_input_speech_events': False,
    'audio_input_sample_rate': DEFAULT_AUDIO_SAMPLE_RATE,
    'audio_output_sample_rate': DEFAULT_AUDIO_SAMPLE_RATE,
    'context_window': None,
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
