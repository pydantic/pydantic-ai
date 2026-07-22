"""Abstractions for realtime multimodal models.

Providers like OpenAI Realtime and Gemini Live offer bidirectional streaming APIs over a persistent
connection rather than the request-response pattern of the standard
[`Model`][pydantic_ai.models.Model] interface.

This module defines the provider-facing ABCs ([`RealtimeModel`][pydantic_ai.realtime.RealtimeModel],
[`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]) and the event/input types
exchanged over a realtime session.
"""

from __future__ import annotations as _annotations

import asyncio
import io
import random
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypeAliasType, TypedDict, assert_never

from ..exceptions import UnexpectedModelBehavior, UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    DocumentUrl,
    FinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    TextContent,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from ..models import AbstractModel, ModelRequestParameters, download_item
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

Retained audio is stored on the [`SpeechPart`][pydantic_ai.messages.SpeechPart]'s `audio` as WAV
[`BinaryContent`][pydantic_ai.messages.BinaryContent]. Live audio deltas remain raw PCM. Retained
audio is attached to its own user turn (by provider item id where the provider reports one, so
overlapping turns stay correct); only the exact split at a turn boundary is approximate.
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
    prone to false triggers. Maps per provider: **OpenAI / Azure / xAI** → server-VAD `threshold`
    (`low`≈0.7, `medium`≈0.5, `high`≈0.3); **Gemini** → both start and end sensitivity (`low`→`low`,
    `high`→`high`, `medium` leaves the provider default). `None` uses the provider default."""

    prefix_padding_ms: int | None = None
    """Audio retained before detected speech onset, in milliseconds. Honored by OpenAI, xAI, and
    Gemini."""

    silence_duration_ms: int | None = None
    """Silence required to mark the end of speech, in milliseconds. Honored by OpenAI, xAI, and
    Gemini."""


class RealtimeModelSettings(TypedDict, total=False):
    """Settings to configure a realtime model session.

    Defines the common settings vocabulary used across realtime model providers. Unsupported settings
    are ignored by a provider; in particular, Gemini ignores `parallel_tool_calls`, `tool_choice`, and
    `handshake_timeout`, while xAI ignores `output_modality` and `thinking`. Providers with additional
    generation parameters extend it, e.g.
    [`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings].
    """

    max_tokens: int
    """The maximum number of tokens to generate per response before stopping."""

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls. Gemini ignores this setting."""

    tool_choice: ToolChoice
    """Control which function tools the model can use.

    Gemini ignores this setting.

    See the [Tool Choice guide](../tools-advanced.md#tool-choice) for detailed documentation.
    Restrictions that realtime providers can't express are dropped: OpenAI, Azure OpenAI, and xAI support
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
    """The single modality generated by the model. Defaults to `'audio'`. xAI ignores this setting and
    always generates audio."""

    thinking: ThinkingLevel
    """Enable or configure reasoning/thinking, mirroring the unified
    [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting on the request-response models.

    `True` enables it at the provider default, and `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`
    selects an effort level. `False` disables thinking on Gemini. OpenAI realtime does not accept a
    disabled effort, so `False` omits `reasoning` and leaves the model's default behavior unchanged.
    xAI ignores this setting. OpenAI and Gemini apply it only to models whose profile reports
    [`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] (OpenAI's
    `gpt-realtime-2*` reasoning models and Gemini's native-audio models) and warn when those providers'
    selected model lacks reasoning support. Providers with a richer native config expose it separately
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
    """Seconds to wait for a realtime protocol handshake event. Defaults to `30.0`. Gemini ignores
    this setting."""


# Input content types (fed into the connection via `send`).


@dataclass
class AudioInput:
    """A chunk of audio data to stream to the model."""

    data: bytes
    """Raw PCM audio bytes. The expected sample rate is provider-specific."""


@dataclass
class ImageInput:
    """An image frame to send to the model (e.g. for Gemini Live video).

    Not every provider accepts image input. [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send]
    checks the model profile and raises [`UserError`][pydantic_ai.exceptions.UserError] when images are
    unsupported; a direct low-level connection may raise `NotImplementedError`.
    """

    data: bytes
    """Raw image bytes."""
    media_type: str = 'image/jpeg'
    """The image media type. Named to match [`BinaryContent.media_type`][pydantic_ai.messages.BinaryContent]."""


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
    content: Sequence[UserContent] | None = None
    """Additional user content to send after the tool output when the provider supports it."""


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
        'azure-speech',
        'mai-transcribe',
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
    item_id: str | None = None
    """Provider item ID for the spoken output this chunk belongs to, when available."""


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
    item_id: str | None = None
    """Provider item ID for the spoken output, when available."""


@dataclass
class InputTranscript:
    """A transcription of the user's audio input (partial or final).

    Providers with per-item IDs use `item_id` to associate interleaved transcripts with the correct
    user turn. Providers without them retain arrival-order association.
    """

    text: str
    """Transcript text."""
    is_final: bool = False
    """Whether this is the final transcript for the user's turn."""
    item_id: str | None = None
    """Provider item ID for the user's turn, when available."""


@dataclass
class ToolCall:
    """The model is requesting a tool call."""

    tool_call_id: str
    """Provider-assigned identifier for this call."""
    tool_name: str
    """Name of the tool to invoke."""
    args: str
    """Raw JSON-encoded arguments. May be an empty string if the model sent no arguments."""
    response_usage_follows: bool = False
    """Whether a per-response [`SessionUsageEvent`][pydantic_ai.realtime.SessionUsageEvent] will follow
    this call before the provider's response is complete.

    OpenAI-protocol providers report calls before `response.done`, which carries usage; the session
    uses this signal to keep all calls and their usage on the same `ModelResponse`."""
    item_id: str | None = None
    """Provider conversation-item ID for this call, when available."""


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

    provider_response_id: str | None = None
    """Provider-assigned ID for the completed response, when available."""

    finish_reason: FinishReason | None = None
    """Normalized reason the provider finished the response, when available."""

    provider_details: dict[str, Any] | None = None
    """Raw provider terminal status details retained on the finalized response, when available."""

    event_kind: Literal['turn_complete'] = 'turn_complete'
    """Event type identifier, used as a discriminator."""


@dataclass
class InputSpeechStartEvent:
    """The provider detected that the user started speaking.

    Useful for barge-in: stop playing any buffered model audio when this arrives, since the model's
    in-progress turn is being interrupted.
    """

    item_id: str | None = None
    """Provider id of the user input item this speech segment belongs to, when reported."""

    event_kind: Literal['input_speech_start'] = 'input_speech_start'
    """Event type identifier, used as a discriminator."""


@dataclass
class InputSpeechEndEvent:
    """The provider detected that the user stopped speaking.

    Useful as a 'processing' indicator: the user's turn has ended and the model is about to respond.
    """

    item_id: str | None = None
    """Provider id of the user input item this speech segment belongs to, when reported.

    Used to attach retained input audio (`audio_retention='input'`/`'both'`) to the right user turn
    when turns overlap, since transcripts for different items can finalize out of order.
    """

    event_kind: Literal['input_speech_end'] = 'input_speech_end'
    """Event type identifier, used as a discriminator."""


@dataclass
class InputTranscriptionFailedEvent:
    """The provider failed to transcribe a user audio input turn, but the session continues.

    This is recoverable; `item_id` and `content_index` locate the affected user turn.
    """

    message: str
    """Human-readable error message."""
    type: str | None = None
    """Provider error category, if any."""
    code: str | None = None
    """Provider error code, if any."""
    item_id: str | None = None
    """Provider conversation-item ID for the affected user turn, when available."""
    content_index: int | None = None
    """Content index within the affected user turn, when available."""

    event_kind: Literal['input_transcription_failed'] = 'input_transcription_failed'
    """Event type identifier, used as a discriminator."""


@dataclass
class SessionUsageEvent:
    """Usage reported by the provider for a model response or another run-level operation."""

    usage: RequestUsage
    """Normalized usage ready to accumulate into a `RunUsage`."""

    provider_response_id: str | None = None
    """Provider-assigned ID for the response this usage belongs to, when available."""

    finish_reason: FinishReason | None = None
    """Normalized completion reason for the response this usage belongs to, when available."""

    response_scoped: bool = True
    """Whether this usage belongs to a specific model response.

    `True`, the default, accumulates it both into the run total and the response's
    `ModelResponse.usage`. `False` is run-level only, e.g. input audio transcription usage,
    which is billed on a separate model/meter and is accumulated into the run's `RunUsage`
    but attributed to no `ModelResponse`.
    """

    event_kind: Literal['session_usage'] = 'session_usage'
    """Event type identifier, used as a discriminator."""


@dataclass
class ReconnectedEvent:
    """The connection dropped and was automatically re-established; inspect `state_restored` for continuity.

    Session configuration (instructions, tools, voice, ...) is restored on every reconnect. OpenAI
    starts with fresh server-side conversation state; Gemini and xAI restore conversation state when
    their session-resumption support is enabled (xAI enables it automatically when `reconnect` is set).
    """

    state_restored: bool = False
    """Whether the provider restored prior conversation state on reconnect.

    `False` (OpenAI/Azure OpenAI — the server starts a fresh conversation) means the consumer should
    treat the session as having lost prior turns; `True` (xAI Grok Voice and Gemini Live, when their native
    session resumption is active) means prior turns were restored.
    """

    event_kind: Literal['reconnected'] = 'reconnected'
    """Event type identifier, used as a discriminator."""


@dataclass
class ConversationCreated:
    """An OpenAI-protocol server assigned a conversation ID.

    This is a codec-level control event. Providers consume it during their handshake when possible;
    the session silently consumes any instance that reaches the live stream.
    """

    conversation_id: str
    """Provider-assigned conversation ID."""


@dataclass
class ConversationItemCreated:
    """An OpenAI-protocol server reported a conversation item.

    xAI uses `replayed=True` for item events emitted during the resume handshake. The session consumes
    those items and remembers their newly assigned IDs so any follow-on content or tool events aren't
    appended or executed again.
    """

    item_id: str | None = None
    """Provider-assigned conversation-item ID, when present."""
    tool_call_id: str | None = None
    """Provider-assigned tool-call ID, for function call and result items."""
    replayed: bool = False
    """Whether the provider identified this item as part of a resumption replay."""


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
    | InputTranscriptionFailedEvent
    | SessionUsageEvent
    | ReconnectedEvent
    | ConversationCreated
    | ConversationItemCreated
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
    | DeferredToolRequestsEvent
    | DeferredToolResultsEvent
    | TurnCompleteEvent
    | InputSpeechStartEvent
    | InputSpeechEndEvent
    | InputTranscriptionFailedEvent
    | ReconnectedEvent
    | SessionErrorEvent,
)
"""Union of events yielded by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession].

Content is streamed as the shared [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
events (carrying [`SpeechPart`][pydantic_ai.messages.SpeechPart]s and
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s), tool execution as
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] /
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent], inline deferred resolution
as [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] /
[`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent], and the rest as realtime
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

    All fields are optional. Consumers treat absent boolean flags as `False`, absent
    `supported_native_tools` as empty, and absent sample rates as the values in
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
    supports_seeding_images: bool
    """Whether prior images can be included when seeding a session with `message_history`."""
    supports_seeding_audio: bool
    """Whether retained user audio can be included when seeding a session with `message_history`."""
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
    audio_input_sample_rate: int
    """The sample rate, in Hz, expected for raw PCM audio input."""
    audio_output_sample_rate: int
    """The sample rate, in Hz, produced in raw PCM audio output deltas."""


DEFAULT_REALTIME_PROFILE: RealtimeModelProfile = {
    'supports_image_input': False,
    'supports_manual_turn_control': False,
    'supports_interruption': False,
    'supports_output_truncation': False,
    'supports_session_seeding': False,
    'supports_seeding_images': False,
    'supports_seeding_audio': False,
    'supported_native_tools': frozenset(),
    'audio_input_sample_rate': 24000,
    'audio_output_sample_rate': 24000,
}
"""Default realtime model profile values."""


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

        Concrete connections accept provider-specific data and control inputs. OpenAI accepts audio,
        text, images, tool results, manual turn controls, cancellation, and truncation; Gemini accepts
        audio, text, images, and tool results. A high-level
        [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] checks profile-gated operations and
        raises [`UserError`][pydantic_ai.exceptions.UserError]; direct low-level connections may raise
        `NotImplementedError`.
        """
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        """Iterate over events received from the model."""
        raise NotImplementedError

    @property
    def model_name(self) -> str | None:
        """The model id the server reported serving this session, when the provider reports one.

        Captured from the connect handshake (e.g. the OpenAI protocol's `session.created`). It can
        differ from the requested model id: xAI accepts any model slug and silently substitutes its
        current default, reporting the actually-served model only here. `None` when the provider
        doesn't report one (e.g. Gemini Live). The session stamps this on each
        [`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name], mirroring how
        request-response models record the response's reported model rather than the requested one.
        """
        return None

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
                projected to the provider's initial conversation items. Replayable text, transcripts,
                thinking, tool rounds, images, and retained user audio are seeded according to the
                model profile; content the provider cannot represent raises `UserError`.
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
    def base_url(self) -> str | None:
        """The provider API base URL, when this model is backed by a provider."""
        provider: Provider[object] | None = getattr(self, '_provider', None)
        return provider.base_url if provider is not None else None

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
    server-side state survives depends on the provider: OpenAI Realtime and Azure OpenAI start a
    fresh turn (the audio buffer and prior turns are lost), while Gemini Live and xAI restore prior
    turns. Gemini requires `google_enable_session_resumption=True`; xAI enables native resumption
    automatically whenever a reconnect policy is set.
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


SeedContent = TypeAliasType('SeedContent', 'str | BinaryContent')
"""Provider-neutral text or image/audio bytes ready for realtime history seeding."""


async def seed_user_content(part: UserPromptPart, *, provider_name: str, supports_images: bool) -> list[SeedContent]:
    """Normalize a `UserPromptPart` to replayable text and image content.

    Image URLs are downloaded because realtime history APIs accept inline image bytes, not arbitrary
    HTTPS URLs. `CachePoint`s are deliberately ignored, matching request-response model adapters.
    Other file kinds cannot be represented faithfully and raise [`UserError`][pydantic_ai.exceptions.UserError].
    """
    content: Sequence[UserContent] = [part.content] if isinstance(part.content, str) else part.content
    result: list[SeedContent] = []
    for item in content:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, TextContent):
            result.append(item.content)
        elif isinstance(item, CachePoint):
            continue
        elif isinstance(item, ImageUrl):
            if not supports_images:
                raise UserError(
                    f'{provider_name} realtime history seeding does not support images. '
                    'Remove the image from `message_history` or use a realtime provider that supports image seeding.'
                )
            downloaded = await download_item(item, data_format='bytes')
            image = BinaryContent(data=downloaded['data'], media_type=downloaded['data_type'])
            if not image.is_image:
                raise UserError(
                    f'`ImageUrl` resolved to unsupported media type {image.media_type!r} while seeding '
                    f'{provider_name} realtime history. Use a URL that returns an image or filter it from '
                    '`message_history`.'
                )
            result.append(image)
        elif isinstance(item, BinaryContent):
            if not item.is_image:
                raise UserError(
                    f'`BinaryContent` with media type {item.media_type!r} cannot be seeded into '
                    f'{provider_name} realtime history. Convert it to text or an image, or filter it from '
                    '`message_history`.'
                )
            if not supports_images:
                raise UserError(
                    f'{provider_name} realtime history seeding does not support images. '
                    'Remove the image from `message_history` or use a realtime provider that supports image seeding.'
                )
            result.append(item)
        elif isinstance(item, (AudioUrl, VideoUrl, DocumentUrl, UploadedFile)):
            content_type = item.__class__.__name__
            raise UserError(
                f'`{content_type}` cannot be seeded into {provider_name} realtime history. '
                'Convert it to text or an inline image, or filter it from `message_history`.'
            )
        else:
            assert_never(item)
    return result


def seed_speech_content(
    part: SpeechPart,
    *,
    provider_name: str,
    supports_audio: bool,
) -> SeedContent:
    """Return replayable content for a `SpeechPart`, preferring its transcript.

    Only retained user audio can be replayed, and only when the provider profile explicitly supports
    it. Assistant audio cannot be inserted into any supported realtime history API.
    """
    if part.transcript is not None:
        return part.transcript
    if part.speaker == 'assistant':
        raise UserError(
            f'An assistant `SpeechPart` without a transcript cannot be seeded into {provider_name} realtime history. '
            'Enable output transcription or filter the part from `message_history` before connecting.'
        )
    if part.audio is None:
        raise UserError(
            'A user `SpeechPart` without a transcript or retained audio cannot be seeded into realtime history. '
            'Enable `input_transcription_model` or `audio_retention`, or filter the part from `message_history` '
            'before connecting.'
        )
    if not part.audio.is_audio:
        raise UserError(
            f'`SpeechPart.audio` with media type {part.audio.media_type!r} cannot be seeded into realtime history. '
            'Use retained audio bytes or filter the part from `message_history` before connecting.'
        )
    if not supports_audio:
        raise UserError(
            f'{provider_name} realtime history seeding does not support retained user audio. '
            'Enable input transcription so the turn has a transcript, or filter the part from `message_history`.'
        )
    return part.audio


def seed_pcm_audio(audio: BinaryContent, *, provider_name: str, sample_rate: int) -> bytes:
    """Extract mono PCM16 bytes from retained WAV audio for a realtime input stream.

    Realtime wire protocols accept raw PCM rather than a container. The WAV's rate must match the
    target session because this path deliberately does not resample audio.
    """
    if audio.media_type != 'audio/wav':
        raise UserError(
            f'`SpeechPart.audio` with media type {audio.media_type!r} cannot be seeded into '
            f'{provider_name} realtime history. Use WAV audio matching the target session input format.'
        )
    try:
        with wave.open(io.BytesIO(audio.data), 'rb') as wav:
            source_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
            if source_rate != sample_rate:
                raise UserError(
                    f'Cannot seed retained audio recorded at {source_rate} Hz into a {provider_name} realtime session '
                    f'expecting {sample_rate} Hz. Resample it before passing `message_history`.'
                )
            if channels != 1 or sample_width != 2 or compression != 'NONE':
                raise UserError(
                    f'Cannot seed retained audio into {provider_name} realtime history: expected mono 16-bit PCM WAV, '
                    f'got {channels} channel(s), {sample_width * 8}-bit samples, compression {compression!r}.'
                )
            pcm = wav.readframes(wav.getnframes())
    except (EOFError, wave.Error) as e:
        raise UserError(
            f'`SpeechPart.audio` cannot be seeded into {provider_name} realtime history because it is not valid WAV audio.'
        ) from e
    return pcm
