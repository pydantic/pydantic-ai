"""Settings for realtime model sessions, mirroring [`pydantic_ai.settings`][pydantic_ai.settings]."""

from __future__ import annotations as _annotations

from typing import Literal

from typing_extensions import TypeAliasType, TypedDict

from ..settings import ThinkingLevel, ToolChoice

AudioRetention = TypeAliasType('AudioRetention', Literal['transcript_only', 'input_audio', 'output_audio', 'all'])
"""How much audio a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] retains in its history.

Values other than `'transcript_only'` are additive: transcripts are always kept, and the named audio is
retained alongside them.

- `'transcript_only'` (default): keep only transcripts; drop all audio bytes.
- `'input_audio'`: also retain the user's spoken audio.
- `'output_audio'`: also retain the model's spoken audio.
- `'all'`: retain both sides' audio.

Retained audio is stored on the [`SpeechPart`][pydantic_ai.messages.SpeechPart]'s `audio` as WAV
[`BinaryContent`][pydantic_ai.messages.BinaryContent]. Live audio deltas remain raw PCM. Retained
input audio is split only at boundaries the provider reports. OpenAI, Azure, and xAI report the end
of each detected speech segment, so each user part normally contains audio sent since the preceding
speech-end boundary through the current one. This can include inter-turn microphone input. Gemini
does not report speech-end boundaries, so its user part contains everything sent since the preceding
response completed through the current response completion, including silence sent while the model
is responding. Retention records the microphone stream only; it does not mix the model's output audio
into the user's part unless that output is present in the microphone input itself.
"""


class TurnDetection(TypedDict, total=False):
    """Cross-provider automatic voice-activity detection (VAD) knobs.

    Set as [`RealtimeModelSettings.turn_detection`][pydantic_ai.realtime.RealtimeModelSettings] to turn
    automatic detection on with these settings. (Pass `True` for the provider defaults, or `False` to
    disable it entirely for push-to-talk.) Each field maps to the closest knob on each provider; for
    finer, provider-specific control use the provider-prefixed escape hatch (`openai_turn_detection`,
    `xai_turn_detection`, `google_vad`), which fully overrides this when set.
    """

    sensitivity: Literal['low', 'medium', 'high']
    """How readily the provider detects turn boundaries (speech start/end). Higher is snappier but more
    prone to false triggers. Defaults to the provider default. Maps per provider: **OpenAI / Azure / xAI** → server-VAD `threshold`
    (`low`≈0.7, `medium`≈0.5, `high`≈0.3); **Gemini** → both start and end sensitivity (`low`→`low`,
    `high`→`high`, `medium` leaves the provider default)."""

    prefix_padding_ms: int
    """Audio retained before detected speech onset, in milliseconds. Honored by OpenAI, xAI, and
    Gemini. Defaults to the provider default."""

    silence_duration_ms: int
    """Silence required to mark the end of speech, in milliseconds. Honored by OpenAI, xAI, and
    Gemini. Defaults to the provider default."""


class RealtimeModelSettings(TypedDict, total=False):
    """Settings to configure a realtime model session.

    Defines the common settings vocabulary used across realtime model providers. Unsupported settings
    are silently ignored. Providers with additional
    generation parameters extend it, e.g.
    [`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings].
    """

    max_tokens: int
    """The maximum number of tokens to generate per response before stopping.

    Supported by: OpenAI, Azure OpenAI, Gemini, and xAI.
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by: OpenAI, Azure OpenAI, and xAI.
    """

    tool_choice: ToolChoice
    """Control which function tools the model can use.

    See the [Tool Choice guide](../tools-advanced.md#tool-choice) for detailed documentation. Every
    form is resolved exactly as it is for a standard run, including the error a name that matches no
    tool raises; a session has no output tools, so
    [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput] restricts the function tools while leaving the
    model free to just speak.

    `'none'` and function-tool allow-lists are enforced on every provider by restricting the tools
    advertised when the session is created. OpenAI, Azure OpenAI, and xAI additionally support
    declarative `'auto'` and `'required'` choices. Gemini has no declarative tool-choice configuration,
    so `'required'` is ignored and allow-lists restrict availability without requiring a tool call.

    Supported by: OpenAI, Azure OpenAI, Gemini (`'none'` and function-tool allow-lists only), and xAI.
    """

    input_transcription_model: KnownRealtimeTranscriptionModelName | str | None
    """Model used to transcribe the user's audio input, so their turns are captured into history.

    `'auto'` (the default) uses the provider's recommended realtime transcription model; pass a
    specific id (e.g. `'gpt-4o-transcribe'`) to pin one, or `None` to disable transcription (see
    `audio_retention` to retain the raw audio instead).

    `None` turns transcription off on every provider. A *pinned* id applies only to the providers that
    transcribe with a separate model — Gemini transcribes natively, with no model to point at, and
    ignores it (`google_input_transcription` configures Gemini's own transcription).

    Supported by: OpenAI, Azure OpenAI, Gemini (`None` only), and xAI.
    """

    output_modality: Literal['audio', 'text']
    """The single modality generated by the model. Defaults to `'audio'`.

    Unlike the other settings here, an unsupported value is *not* silently ignored: a model whose
    profile reports [`supports_text_output=False`][pydantic_ai.realtime.RealtimeModelProfile.supports_text_output]
    raises [`UserError`][pydantic_ai.exceptions.UserError] before connecting, because a session that
    quietly spoke instead of writing would be worse than one that didn't start.

    Supported by: OpenAI and Azure OpenAI. Gemini Live and xAI always generate audio — read the spoken
    answer from the transcript on the [`SpeechPart`][pydantic_ai.messages.SpeechPart] instead.
    """

    thinking: ThinkingLevel
    """Enable or configure reasoning/thinking, mirroring the unified
    [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting on the request-response models.

    `True` enables it at the provider default, and `'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`
    selects an effort level. `False` disables thinking on Gemini. OpenAI realtime does not accept a
    disabled effort, so `False` omits `reasoning` and leaves the model's default behavior unchanged.
    OpenAI and Gemini apply it only to models whose profile reports
    [`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking]. Other models
    silently ignore it. Providers with a richer native config expose it separately
    (e.g. Gemini's `google_thinking_config`), which takes precedence.

    Supported by: OpenAI `gpt-realtime-2*` models, Gemini native-audio models, and xAI's reasoning
    Grok Voice models (`grok-voice-latest` and the `grok-voice-think-*` family).
    """

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
    """Seconds to wait for a realtime protocol handshake event. Defaults to `30.0`.

    Supported by: OpenAI, Azure OpenAI, and xAI.
    """

    reconnect: ReconnectPolicy
    """[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently recover from a
    dropped connection. Without a policy, an unexpectedly closed connection is fatal: the low-level
    connection reports a non-recoverable session error and `RealtimeSession` raises
    [`RealtimeError`][pydantic_ai.realtime.RealtimeError] from iteration.

    What server-side state survives a reconnect depends on the provider (see
    [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy]). Setting a policy enables native
    session resumption on the providers that offer it: xAI always, and Gemini unless
    `google_enable_session_resumption=False` is set explicitly — that combination raises
    [`UserError`][pydantic_ai.exceptions.UserError] at connect time, since a re-dial without
    resumption would lose the conversation.

    Supported by: OpenAI, Azure OpenAI, Gemini, and xAI.
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
"""Known values for the OpenAI-protocol models' `input_transcription_model`, pinned by a provider sync test.

`'auto'` is the sentinel that resolves to the provider's recommended transcription model; the rest are
concrete model ids. The values span providers, so an id valid for one provider (e.g. `'grok-transcribe'`
for xAI) is rejected by another at connect time. The field also accepts any other `str`, so a newer id
not listed here still works — this is just an autocomplete aid, like
[`KnownModelName`][pydantic_ai.models.KnownModelName].
"""


class ReconnectPolicy(TypedDict, total=False):
    """How to recover when a realtime connection drops mid-session.

    Set as the `reconnect` key of [`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings],
    either as a model-level default (`settings=`) or per session (`model_settings=`).

    On a dropped connection the session is re-dialed and its configuration (instructions, tools,
    voice, ...) re-applied, emitting a
    [`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent] event. What server-side state
    survives depends on the provider: OpenAI Realtime and Azure OpenAI start a fresh turn (the audio
    buffer and prior turns are lost), while Gemini Live and xAI restore prior turns through native
    session resumption, enabled automatically whenever a reconnect policy is set (Gemini honors an
    explicit `google_enable_session_resumption=False` opt-out by refusing the combination with a
    [`UserError`][pydantic_ai.exceptions.UserError]).
    """

    max_attempts: int
    """Number of re-dial attempts per drop before giving up and raising
    [`RealtimeError`][pydantic_ai.realtime.RealtimeError]. Defaults to `3`."""
    max_reconnects: int
    """Total successful reconnects allowed for the life of the session.

    `max_attempts` bounds the retries for a single drop, and resets once a dial succeeds, so on its
    own it cannot stop a session that reconnects, drops, and reconnects forever. This bounds the whole
    session instead.

    The default is generous for the case this exists to serve: providers end sessions at a duration
    cap (OpenAI at 60 minutes) and a long-running session legitimately renews at that boundary, so 50
    covers days of continuous conversation. It only bites a server that hangs up as fast as we dial.
    Defaults to `50`.
    """
    base_delay: float
    """Base backoff delay in seconds; doubles each attempt up to `max_delay`. Defaults to `0.5`."""
    max_delay: float
    """Maximum backoff delay in seconds. Defaults to `30.0`."""
    jitter: bool
    """Whether to apply random jitter to each backoff delay to avoid thundering herds. Defaults to `True`."""
