"""Abstractions for realtime multimodal models.

Providers like OpenAI Realtime, Gemini Live, and Amazon Nova Sonic offer bidirectional
streaming APIs over a persistent connection (WebSocket or HTTP/2) rather than the
request-response pattern of the standard [`Model`][pydantic_ai.models.Model] interface.

This module defines the provider-facing ABCs ([`RealtimeModel`][pydantic_ai.realtime.RealtimeModel],
[`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]) and the event/input types
exchanged over a realtime session.
"""

from __future__ import annotations as _annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypeAliasType, TypedDict

from ..exceptions import UnexpectedModelBehavior
from ..messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    UserPromptPart,
)
from ..models import AbstractModel, ModelRequestParameters
from ..native_tools import AbstractNativeTool
from ..settings import ThinkingLevel, ToolChoice
from ..usage import RequestUsage

if TYPE_CHECKING:
    from ..providers import Provider

AudioRetention = TypeAliasType('AudioRetention', Literal['transcript_only', 'input', 'output', 'both'])
"""How much audio a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] retains in its history.

- `'transcript_only'` (default): keep only transcripts; drop all audio bytes.
- `'input'`: also retain the user's spoken audio.
- `'output'`: also retain the model's spoken audio.
- `'both'`: retain both sides' audio.

Retained audio is stored on the [`SpeechPart`][pydantic_ai.messages.SpeechPart]'s
`audio` as raw PCM [`BinaryContent`][pydantic_ai.messages.BinaryContent]. Alignment between retained
audio and its transcript is approximate.
"""


class RealtimeError(UnexpectedModelBehavior):
    """A fatal realtime connection or protocol error."""


@dataclass
class TurnDetection:
    """Cross-provider automatic voice-activity detection (VAD) knobs.

    Set as [`RealtimeModelSettings.turn_detection`][pydantic_ai.realtime.RealtimeModelSettings] to turn
    automatic detection on with these settings. (Pass `True` for the provider defaults, or `False` to
    disable it entirely for push-to-talk.) Each field maps to the closest knob on each provider; for
    finer, provider-specific control use the provider-prefixed escape hatch (`openai_turn_detection`,
    `xai_turn_detection`, `google_vad`), which fully overrides this when set.
    """

    sensitivity: Literal['low', 'medium', 'high'] | None = None
    """How readily the provider detects turn boundaries (speech start/end). Higher is snappier but more
    prone to false triggers. Maps per provider: **OpenAI / xAI** → server-VAD `threshold`
    (`low`≈0.7, `medium`≈0.5, `high`≈0.3); **Gemini** → both start and end sensitivity (`low`→`low`,
    `high`→`high`, `medium` leaves the provider default); **Amazon Nova 2 Sonic** → endpointing
    sensitivity (`LOW`/`MEDIUM`/`HIGH`; Nova Sonic v1 ignores it). `None` uses the provider default."""

    prefix_padding_ms: int | None = None
    """Audio retained before detected speech onset, in milliseconds. Honored by OpenAI, xAI, and
    Gemini; Amazon Nova Sonic does not expose this and ignores it."""

    silence_duration_ms: int | None = None
    """Silence required to mark the end of speech, in milliseconds. Honored by OpenAI, xAI, and
    Gemini; Amazon Nova Sonic does not expose this and ignores it."""


class RealtimeModelSettings(TypedDict, total=False):
    """Settings to configure a realtime model session.

    Includes only settings which are supported by all realtime model providers; providers with
    additional generation parameters extend it, e.g.
    [`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings].
    """

    max_tokens: int
    """The maximum number of tokens to generate per response before stopping."""

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls."""

    tool_choice: ToolChoice
    """Control which function tools the model can use.

    See the [Tool Choice guide](../tools-advanced.md#tool-choice) for detailed documentation.
    Restrictions that realtime providers can't express are dropped: OpenAI and xAI support
    `'auto'`/`'none'`/`'required'` and a single-tool list, but not multi-tool lists or
    [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput].
    """

    voice: str
    """Voice used for audio output, e.g. `alloy` (OpenAI), `Puck` (Gemini), or `eve` (xAI)."""

    input_transcription_model: KnownRealtimeTranscriptionModelName | str | None
    """Model used to transcribe the user's audio input, so their turns are captured into history.

    `'auto'` (the default) uses the provider's recommended realtime transcription model; pass a
    specific id (e.g. `'gpt-4o-transcribe'`) to pin one, or `None` to disable transcription (see
    `audio_retention` to retain the raw audio instead). Gemini transcribes natively and ignores
    this; use `google_input_transcription`.
    """

    output_modality: Literal['audio', 'text']
    """The single modality generated by the model. Defaults to `'audio'`."""

    thinking: ThinkingLevel
    """Enable or configure reasoning/thinking, mirroring the unified
    [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting on the request-response models.

    `True` enables it at the provider default, `False` disables it, and `'minimal'`/`'low'`/`'medium'`/
    `'high'`/`'xhigh'` selects an effort level. Only applied to models whose profile reports
    [`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] (OpenAI's
    `gpt-realtime-2*` reasoning models and Gemini's native-audio models); it is ignored with a warning
    on models without reasoning support. Providers with a richer native config expose it separately
    (e.g. Gemini's `google_thinking_config`), which takes precedence."""

    turn_detection: bool | TurnDetection
    """Automatic voice-activity detection (VAD) / turn-taking. Modeled on
    [`thinking`][pydantic_ai.settings.ModelSettings.thinking]:

    - Absent (the default) or `True`: automatic turn detection on, at the provider's defaults.
    - `False`: disable it — push-to-talk, drive turns manually with `commit_audio()` /
      `create_response()` (only on providers whose [model profile](#model-profile) reports
      `supports_manual_turn_control`).
    - [`TurnDetection`][pydantic_ai.realtime.TurnDetection]: on, with specific cross-provider knobs.

    For finer, provider-specific control use the provider-prefixed setting documented on each provider's
    settings type (`openai_turn_detection`, `xai_turn_detection`, `google_vad`); when present it fully
    overrides this field.
    """

    handshake_timeout: float
    """Seconds to wait for a realtime protocol handshake event. Defaults to `30.0`."""


# Input content types (fed into the connection via `send`).


@dataclass
class AudioInput:
    """A chunk of audio data to stream to the model."""

    data: bytes
    """Raw PCM audio bytes. The expected sample rate is provider-specific."""


@dataclass
class ImageInput:
    """An image frame to send to the model (e.g. for Gemini Live video).

    Not every provider accepts image input; sending this to one that doesn't raises
    `NotImplementedError`.
    """

    data: bytes
    """Raw image bytes."""
    mime_type: str = 'image/jpeg'
    """The image media type."""


@dataclass
class TextInput:
    """A text message to send to the model as a complete turn.

    Useful for text-only conversations or for injecting text context alongside audio.
    """

    text: str
    """The message text."""


@dataclass
class ToolResult:
    """The result of a tool call, to send back to the model."""

    tool_call_id: str
    """Identifier of the `ToolCall` this result answers."""
    output: str
    """The tool's output, rendered as a string."""


@dataclass
class CommitAudio:
    """Commit the buffered input audio as a user turn (manual turn-taking / push-to-talk).

    Only needed when automatic voice activity detection is disabled; with server-side VAD the
    provider commits audio and triggers a response automatically.
    """


@dataclass
class ClearAudio:
    """Discard any buffered, uncommitted input audio."""


@dataclass
class CreateResponse:
    """Ask the model to generate a response now (manual turn-taking, after `CommitAudio`)."""


@dataclass
class CancelResponse:
    """Cancel the model's in-progress response (maps to the provider's response-cancel)."""


@dataclass
class TruncateOutput:
    """Truncate the model's current audio output at `audio_end_ms`.

    After a barge-in the user only heard part of the model's audio. Truncating tells the provider how
    much was actually played, so its stored transcript matches and the conversation context stays
    consistent. The provider resolves which output item to truncate from its own state.
    """

    audio_end_ms: int
    """Milliseconds of the current output audio that were actually played before the interruption."""


RealtimeSessionInput = TypeAliasType(
    'RealtimeSessionInput',
    'AudioInput | ImageInput | TextInput | CommitAudio | ClearAudio | CreateResponse | CancelResponse | TruncateOutput',
)
"""The content types a caller feeds into [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send].

This is [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] minus [`ToolResult`][pydantic_ai.realtime.ToolResult]:
the session sends tool results itself (see `RealtimeSession`), so a caller never sends one.
"""

RealtimeInput = TypeAliasType('RealtimeInput', 'RealtimeSessionInput | ToolResult')
"""Union of content types accepted by [`RealtimeConnection.send`][pydantic_ai.realtime.RealtimeConnection.send].

A superset of [`RealtimeSessionInput`][pydantic_ai.realtime.RealtimeSessionInput]: the low-level
connection additionally accepts [`ToolResult`][pydantic_ai.realtime.ToolResult], which the session sends
on the caller's behalf when a tool completes.
"""

KnownRealtimeTranscriptionModelName = TypeAliasType(
    'KnownRealtimeTranscriptionModelName',
    Literal[
        'auto',
        'whisper-1',
        'gpt-4o-transcribe',
        'gpt-4o-mini-transcribe',
        'gpt-realtime-whisper',
        'grok-transcribe',
    ],
)
"""Known values for the OpenAI-protocol models' `input_transcription_model`, surfaced for autocomplete.

`'auto'` is the sentinel that resolves to the provider's recommended transcription model; the rest are
concrete model ids. The values span providers, so an id valid for one provider (e.g. `'grok-transcribe'`
for xAI) is rejected by another at connect time. The field also accepts any other `str`, so a newer id
not listed here still works — this is just an autocomplete aid, like
[`KnownModelName`][pydantic_ai.models.KnownModelName].
"""


# Connection-level events (yielded by `RealtimeConnection.__aiter__`).


@dataclass
class AudioDelta:
    """A chunk of audio output from the model."""

    data: bytes
    """Raw PCM audio bytes. The sample rate is provider-specific."""


@dataclass
class Transcript:
    """The model's textual output (partial or final): an audio transcript, or plain text output."""

    text: str
    """Transcript text. A partial event carries the incremental delta; a final event the full turn."""
    is_final: bool = False
    """Whether this is the final transcript for the turn."""
    output_text: bool = False
    """Whether this is the model's plain text output (`output_modalities=('text',)`) rather than a
    transcription of spoken audio. Text output becomes a [`TextPart`][pydantic_ai.messages.TextPart];
    an audio transcript becomes a [`SpeechPart`][pydantic_ai.messages.SpeechPart]."""


@dataclass
class InputTranscript:
    """A transcription of the user's audio input (partial or final).

    The session attributes these to the current user turn by arrival order; it does not carry the
    provider's per-item id. OpenAI documents input-transcription ordering as not guaranteed between
    turns, so a fast back-to-back turn could in principle misattribute a late transcript. This is
    accepted for now (single-turn-at-a-time voice is the norm); thread an item id through if
    interleaved-turn attribution becomes necessary.
    """

    text: str
    """Transcript text."""
    is_final: bool = False
    """Whether this is the final transcript for the user's turn."""


@dataclass
class ToolCall:
    """The model is requesting a tool call."""

    tool_call_id: str
    """Provider-assigned identifier for this call."""
    tool_name: str
    """Name of the tool to invoke."""
    args: str
    """Raw JSON-encoded arguments. May be an empty string if the model sent no arguments."""


@dataclass
class ToolCallCancelled:
    """The model cancelled in-flight tool calls (e.g. the user barged in before they finished).

    Gemini Live sends this as `toolCallCancellation`; the session cancels the matching running tool
    tasks so their now-unwanted results are never sent back to the model.
    """

    tool_call_ids: list[str]
    """Identifiers of the [`ToolCall`][pydantic_ai.realtime.ToolCall]s that were cancelled."""


@dataclass
class TurnCompleteEvent:
    """The model finished (or was interrupted during) its turn."""

    interrupted: bool = False
    """Whether the turn ended because it was cancelled (e.g. the user barged in)."""

    event_kind: Literal['turn_complete'] = 'turn_complete'
    """Event type identifier, used as a discriminator."""


@dataclass
class InputSpeechStartEvent:
    """The provider detected that the user started speaking.

    Useful for barge-in: stop playing any buffered model audio when this arrives, since the model's
    in-progress turn is being interrupted.
    """

    event_kind: Literal['input_speech_start'] = 'input_speech_start'
    """Event type identifier, used as a discriminator."""


@dataclass
class InputSpeechEndEvent:
    """The provider detected that the user stopped speaking.

    Useful as a 'processing' indicator: the user's turn has ended and the model is about to respond.
    """

    event_kind: Literal['input_speech_end'] = 'input_speech_end'
    """Event type identifier, used as a discriminator."""


@dataclass
class SessionUsageEvent:
    """Token usage reported by the provider for a completed model response."""

    usage: RequestUsage
    """Normalized token usage for the response, ready to accumulate into a `RunUsage`."""

    event_kind: Literal['session_usage'] = 'session_usage'
    """Event type identifier, used as a discriminator."""


@dataclass
class ReconnectedEvent:
    """The connection dropped and was automatically re-established.

    Session configuration (instructions, tools, voice, ...) is restored, but server-side conversation
    state (the audio buffer and prior turns) is not — treat it as the start of a fresh turn.
    """

    event_kind: Literal['reconnected'] = 'reconnected'
    """Event type identifier, used as a discriminator."""


@dataclass
class SessionErrorEvent:
    """A provider-reported error occurred in the session."""

    message: str
    """Human-readable error message."""
    type: str | None = None
    """Provider error category, e.g. `invalid_request_error` or `server_error`."""
    code: str | None = None
    """Provider error code, if any."""
    recoverable: bool = True
    """Whether the session can continue. A protocol `error` is recoverable; a dropped connection is not."""

    event_kind: Literal['session_error'] = 'session_error'
    """Event type identifier, used as a discriminator."""


RealtimeCodecEvent = TypeAliasType(
    'RealtimeCodecEvent',
    AudioDelta
    | Transcript
    | InputTranscript
    | ToolCall
    | ToolCallCancelled
    | TurnCompleteEvent
    | InputSpeechStartEvent
    | InputSpeechEndEvent
    | SessionUsageEvent
    | ReconnectedEvent
    | PartStartEvent
    | PartEndEvent
    | SessionErrorEvent,
)
"""Union of the low-level codec events yielded by [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection].

This is the provider-facing vocabulary: providers translate their wire protocol into these events, and
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] translates them again into the shared
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] vocabulary while building
[`ModelMessage`][pydantic_ai.messages.ModelMessage] history.
"""


# Session-level events (yielded by `RealtimeSession.__aiter__`).
#
# A session translates the low-level codec events into the shared message/part event vocabulary from
# `pydantic_ai.messages`: `AudioDelta`/`Transcript`/`InputTranscript` become `PartStartEvent` /
# `PartDeltaEvent` / `PartEndEvent` for `SpeechPart`s, and `ToolCall` becomes a
# `ToolCallPart` part (start/end) plus `FunctionToolCallEvent` / `FunctionToolResultEvent` around its
# execution. The remaining control-plane events pass through unchanged.


RealtimeEvent = TypeAliasType(
    'RealtimeEvent',
    PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | FunctionToolCallEvent
    | FunctionToolResultEvent
    | TurnCompleteEvent
    | InputSpeechStartEvent
    | InputSpeechEndEvent
    | ReconnectedEvent
    | SessionErrorEvent,
)
"""Union of events yielded by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession].

Content is streamed as the shared [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
events (carrying [`SpeechPart`][pydantic_ai.messages.SpeechPart]s and
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s), tool execution as
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] /
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent], and the rest as realtime
control-plane events.
"""


class RealtimeModelProfile(TypedDict, total=False):
    """Describes what a [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] supports, so a session can tailor its behavior to the model.

    Mirrors the shape and `supports_`-prefixed naming of
    [`ModelProfile`][pydantic_ai.profiles.ModelProfile] for the standard request-response
    [`Model`][pydantic_ai.models.Model], which realtime models don't share a hierarchy with.

    A [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] reads these flags to reject unsupported
    operations with a clear error *before* sending them, rather than letting the provider fail
    mid-session. Read a model's via [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile];
    each flag maps to the session methods a provider may not support.

    All fields are optional; absent keys use
    [`DEFAULT_REALTIME_PROFILE`][pydantic_ai.realtime.DEFAULT_REALTIME_PROFILE].
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
    via the `audio_end_ms` argument of [`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt].

    Distinct from [`supports_interruption`][pydantic_ai.realtime.RealtimeModelProfile.supports_interruption]:
    a provider may support cancelling a response (barge-in) without supporting output truncation. OpenAI
    supports both; xAI Grok Voice supports cancellation but not truncation."""
    supports_session_seeding: bool
    """Whether the model can seed a session with prior conversation (`message_history`)."""
    supports_thinking: bool
    """Whether the model supports reasoning/thinking configuration via the
    [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting — OpenAI's `gpt-realtime-2*`
    reasoning models and Gemini's native-audio models. When `False` (the default), a `thinking` setting
    is ignored with a warning rather than sent to a model that would reject it."""
    supported_native_tools: frozenset[type[AbstractNativeTool]]
    """The [native tools][pydantic_ai.native_tools.AbstractNativeTool] the model runs server-side, e.g.
    [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool].

    [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] validates the session's native
    tools against this set before connecting, raising a [`UserError`][pydantic_ai.exceptions.UserError]
    that names any the model doesn't support — mirroring the classic
    [`Model.supported_native_tools`][pydantic_ai.models.Model.supported_native_tools] check."""


DEFAULT_REALTIME_PROFILE: RealtimeModelProfile = {
    'supports_image_input': False,
    'supports_manual_turn_control': False,
    'supports_interruption': False,
    'supports_output_truncation': False,
    'supports_session_seeding': False,
    'supported_native_tools': frozenset(),
}
"""Fully populated default realtime model profile."""


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


class RealtimeConnection(ABC):
    """A live connection to a realtime model.

    Providers implement this to handle protocol-specific framing (WebSocket frames,
    HTTP/2 messages, etc.). Content is fed in via [`send`][pydantic_ai.realtime.RealtimeConnection.send]
    and events are consumed by iterating the connection.
    """

    @abstractmethod
    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the session.

        The accepted input types depend on the provider: OpenAI accepts `AudioInput`,
        `TextInput`, and `ToolResult`, while Gemini Live additionally accepts `ImageInput`.
        Providers raise `NotImplementedError` for input types they don't support.
        """
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        """Iterate over events received from the model."""
        raise NotImplementedError

    @property
    def input_transcription_enabled(self) -> bool:
        """Whether this connection will emit [`InputTranscript`][pydantic_ai.realtime.InputTranscript] events for the user's audio.

        Providers that transcribe the user's input (the default) leave this `True`. When it is `False`,
        no transcript arrives, so [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] finalizes a
        user turn from retained input audio instead (see `audio_retention`). Defaults to `True` so a
        connection that doesn't override it never triggers the audio-only path (which would risk a
        duplicate turn if transcripts did arrive).
        """
        return True


class RealtimeModel(AbstractModel):
    """Abstract base class for realtime model providers.

    [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] and the request-response
    [`Model`][pydantic_ai.models.Model] share [`AbstractModel`][pydantic_ai.models.AbstractModel].
    A realtime model opens a persistent bidirectional connection for streaming content in and out.

    Like [`Model`][pydantic_ai.models.Model], the `settings` attribute and the `model_settings`
    passed to `connect` are typed as the shared [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings];
    each provider narrows to its own `TypedDict` subclass internally with a `cast` (as the
    request-response models do for `ModelSettings`), rather than the base class being generic over the
    settings type.
    """

    settings: RealtimeModelSettings | None = None
    """Model settings used as defaults for realtime sessions."""

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the native tool types implemented by this realtime model class."""
        return frozenset()

    def _merge_model_settings(self, model_settings: RealtimeModelSettings | None) -> RealtimeModelSettings | None:
        """Merge model-level defaults with connection-level overrides."""
        settings = self.settings.copy() if self.settings else None
        if model_settings:
            if settings is None:
                settings = model_settings.copy()
            else:
                settings.update(model_settings)
        return settings

    @abstractmethod
    def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        """Open a connection to the realtime model.

        Args:
            messages: Prior conversation and the current request carrying session instructions,
                projected to the provider's
                initial conversation items. Only text/transcript content is seeded (v1): audio is not
                replayed. Providers degrade gracefully, dropping content they can't project.
            model_settings: Optional provider-specific settings.
            model_request_parameters: Function and native tools available to the session.

        Returns:
            An async context manager yielding a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection].
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name, e.g. `gpt-realtime`."""
        raise NotImplementedError

    @property
    def profile(self) -> RealtimeModelProfile:
        """Resolve provider facts and intersect native tools with this model's implementation."""
        provider: Provider[object] | None = getattr(self, '_provider', None)
        provider_profile = provider.realtime_model_profile(self.model_name) if provider is not None else None
        resolved = merge_realtime_profile(DEFAULT_REALTIME_PROFILE, provider_profile)
        profile_supported = resolved.get('supported_native_tools', frozenset())
        effective_tools = profile_supported & self.__class__.supported_native_tools()
        if effective_tools != profile_supported:
            resolved = merge_realtime_profile(resolved, RealtimeModelProfile(supported_native_tools=effective_tools))
        return resolved


@dataclass
class ReconnectPolicy:
    """How to recover when a realtime connection drops mid-session.

    On a dropped connection the session is re-dialed and its configuration (instructions, tools,
    voice, ...) re-applied, emitting a [`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] event. What
    server-side state survives depends on the provider: OpenAI Realtime starts a fresh turn (the audio
    buffer and prior turns are lost), while Gemini Live restores conversation state when
    `google_enable_session_resumption=True` (a prerequisite for reconnecting there).
    """

    max_attempts: int = 3
    """Number of re-dial attempts before giving up and raising [`RealtimeError`][pydantic_ai.realtime.RealtimeError]."""
    base_delay: float = 0.5
    """Base backoff delay in seconds; doubles each attempt up to `max_delay`."""
    max_delay: float = 30.0
    """Maximum backoff delay in seconds."""
    jitter: bool = True
    """Whether to apply random jitter to each backoff delay to avoid thundering herds."""


async def reconnect_with_backoff(policy: ReconnectPolicy, attempt: Callable[[], Awaitable[bool]]) -> bool:
    """Retry `attempt` with exponential backoff (and optional jitter) until it succeeds or attempts run out.

    `attempt` performs one provider-specific re-dial and returns whether it succeeded.
    """
    for i in range(policy.max_attempts):
        delay = min(policy.max_delay, policy.base_delay * (2**i))
        if policy.jitter:
            delay *= 0.5 + random.random() * 0.5
        await asyncio.sleep(delay)
        if await attempt():
            return True
    return False


def inject_trace_context(headers: MutableMapping[str, str]) -> None:
    """Add the current W3C trace context (`traceparent`) to a realtime WebSocket's handshake headers.

    A realtime connection is a raw WebSocket that bypasses the provider's `httpx` client, so the
    trace-context injection that client's event hooks would normally perform (see the gateway
    provider's request hook in `providers/gateway.py`) doesn't happen automatically. Calling this when
    building the handshake headers propagates trace context to the server, so a proxy like the
    Pydantic AI Gateway can nest its own realtime spans under the client's trace.

    It is a no-op when no span is active (the default propagator writes nothing without a valid span
    context) and harmless against providers that ignore the header, so it is safe to call
    unconditionally.
    """
    from opentelemetry.propagate import inject

    inject(headers)


def user_prompt_text(part: UserPromptPart) -> str:
    """Extract the plain text from a `UserPromptPart` (dropping multimodal content for text seeding)."""
    if isinstance(part.content, str):
        return part.content
    return ''.join(item for item in part.content if isinstance(item, str))
