"""Gemini Live API provider for realtime speech-to-speech (and live video) sessions.

Built on the `google-genai` SDK, which manages the WebSocket transport for you. Available via the
`google` optional group:

    pip install "pydantic-ai-slim[google]"

Unlike the OpenAI provider, Gemini wants **16 kHz** PCM input audio (output is 24 kHz), produces a
single response modality per session (audio *or* text), and natively accepts a stream of video
frames sent as `ImageInput`.

Use `provider='google'` for the Gemini Developer API, or `provider='google-cloud'` /
[`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] for Google Cloud with
Application Default Credentials.
"""

from __future__ import annotations as _annotations

import json
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, cast
from weakref import WeakKeyDictionary

from anyio import Lock
from typing_extensions import assert_never

try:
    from google.genai import Client, errors as genai_errors, types as genai_types
    from google.genai.live import AsyncSession, ConnectionClosed
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Gemini realtime model, '
        'you can use the `google` optional group - `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from .._utils import generate_tool_call_id
from ..exceptions import UserError
from ..messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    CompactionPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from ..models import ModelRequestParameters

# Reuse the classic `GoogleModel`'s native tool mappers so a realtime turn's grounding / code-execution
# native tool parts are byte-identical in shape to a classic request's, rather than duplicating the
# mapping and risking drift.
from ..models.google import (
    _map_code_execution_result,  # pyright: ignore[reportPrivateUsage]
    _map_executable_code,  # pyright: ignore[reportPrivateUsage]
    _map_grounding_metadata,  # pyright: ignore[reportPrivateUsage]
    _map_url_context_metadata,  # pyright: ignore[reportPrivateUsage]
)
from ..native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool
from ..profiles import DEFAULT_THINKING_TAGS
from ..providers import Provider, infer_provider
from ..settings import ThinkingEffort, ThinkingLevel
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._base import (
    AudioDelta,
    AudioInput,
    ImageInput,
    InputSpeechStartEvent,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    Transcript,
    TurnCompleteEvent,
    TurnDetection,
    inject_trace_context,
    reconnect_with_backoff,
    seed_speech_content,
    seed_user_content,
)


class GoogleRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings used for a Gemini Live session."""

    temperature: float
    """Amount of randomness injected into the response."""

    top_p: float
    """Nucleus sampling probability mass."""

    top_k: int
    """Only sample from the top K options for each subsequent token."""

    seed: int
    """The random seed to use for the session."""

    google_thinking_config: genai_types.ThinkingConfigDict
    """The thinking configuration to use for the model."""

    google_video_resolution: genai_types.MediaResolution
    """The video resolution to use for the model."""

    google_language_code: str
    """BCP-47 language code for audio output."""
    google_multi_speaker: MultiSpeaker
    """Per-speaker voice assignments; takes precedence over `voice`."""
    google_affective_dialog: bool
    """Whether to enable emotion-aware delivery (native-audio models only)."""
    google_proactive_audio: bool
    """Whether the model may decide *when* to respond, including staying silent on input not
    addressed to it (native-audio models only). Useful for "react to the camera" experiences."""
    google_input_transcription: bool
    """Whether to transcribe input audio. Defaults to `True`.

    When `False`, the session's `audio_retention` must be `'input_audio'` or `'all'` so user turns can be
    recorded.
    """
    google_output_transcription: bool
    """Whether to transcribe output audio. Defaults to `True`.

    When `False`, retain output audio if assistant audio turns need to appear in history. Assistant
    audio without a transcript cannot be handed off or seeded.
    """
    google_transcription_language_codes: list[str]
    """Language hints applied to input and output transcription."""
    google_vad: AutomaticVAD
    """Gemini-specific server-side voice activity detection settings.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    Do not use `AutomaticVAD(disabled=True)` through `RealtimeSession`: Pydantic AI does not expose
    Gemini activity markers or manual turn controls, so the resulting session cannot drive turns.
    """
    google_activity_handling: Literal['interrupts', 'no_interruption']
    """Whether detected user activity interrupts the model."""
    google_turn_coverage: Literal['activity_only', 'all_input', 'all_video']
    """Which realtime input is attached to a turn — `'activity_only'`, `'all_input'` (everything
    between turns too), or `'all_video'` (all video frames plus audio during activity; ideal for
    live-camera use). Absent uses the provider default."""
    google_context_compression: ContextCompression
    """Sliding-window context compression for long-running sessions."""
    google_config_overrides: dict[str, Any]
    """Raw values merged last into the Google `LiveConnectConfig`."""

    google_enable_session_resumption: bool
    """Whether to request session-resumption handles. Defaults to `False`."""


INPUT_SAMPLE_RATE = 16000
"""Sample rate (Hz) Gemini expects for PCM16 input audio."""


@dataclass
class AutomaticVAD:
    """Server-side voice activity detection — the default turn-taking mode for Gemini Live."""

    disabled: bool = False
    """Turn off automatic VAD entirely.

    Do not set this through `RealtimeSession`: Pydantic AI does not expose Gemini activity markers or
    manual turn controls. Use automatic VAD instead; the shared `turn_detection=False` setting is
    rejected for the same reason.
    """
    start_sensitivity: Literal['high', 'low'] | None = None
    """How readily speech onset is detected. `high` triggers on quieter audio; `low` is stricter."""
    end_sensitivity: Literal['high', 'low'] | None = None
    """How readily the end of speech is detected. `high` ends turns sooner; `low` waits longer."""
    prefix_padding_ms: int | None = None
    """Audio to include before detected speech, in milliseconds."""
    silence_duration_ms: int | None = None
    """Silence required to detect the end of speech, in milliseconds."""


@dataclass
class MultiSpeaker:
    """Assign prebuilt voices to named speakers for multi-speaker audio output."""

    voices: dict[str, str]
    """Mapping of speaker label to prebuilt voice name, e.g. `{'Joe': 'Puck', 'Jane': 'Kore'}`."""


@dataclass
class ContextCompression:
    """Sliding-window context compression so long sessions don't exceed the context window."""

    trigger_tokens: int | None = None
    """Compress once the context passes this many tokens; `None` uses the provider default."""
    target_tokens: int | None = None
    """Target size (in tokens) of the retained sliding window after compression."""


# Literal -> SDK enum mappings, kept as small tables so the public API stays string-friendly.
_START_SENSITIVITY = {
    'high': genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
    'low': genai_types.StartSensitivity.START_SENSITIVITY_LOW,
}
_END_SENSITIVITY = {
    'high': genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
    'low': genai_types.EndSensitivity.END_SENSITIVITY_LOW,
}
_ACTIVITY_HANDLING = {
    'interrupts': genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
    'no_interruption': genai_types.ActivityHandling.NO_INTERRUPTION,
}
_TURN_COVERAGE = {
    'activity_only': genai_types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
    'all_input': genai_types.TurnCoverage.TURN_INCLUDES_ALL_INPUT,
    'all_video': genai_types.TurnCoverage.TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO,
}

_WS_CONNECT_LOCKS: WeakKeyDictionary[Client, Lock] = WeakKeyDictionary()


def _ws_connect_lock(client: Client) -> Lock:
    """Return the lock serializing temporary handshake-header mutations for one SDK client."""
    lock = _WS_CONNECT_LOCKS.get(client)
    if lock is None:
        lock = Lock()
        _WS_CONNECT_LOCKS[client] = lock
    return lock


_GEMINI_THINKING_LEVEL: dict[ThinkingEffort, genai_types.ThinkingLevel] = {
    'minimal': genai_types.ThinkingLevel.MINIMAL,
    'low': genai_types.ThinkingLevel.LOW,
    'medium': genai_types.ThinkingLevel.MEDIUM,
    'high': genai_types.ThinkingLevel.HIGH,
    'xhigh': genai_types.ThinkingLevel.HIGH,  # Gemini has no `xhigh`; map it to the highest level.
}


def _thinking_to_config(thinking: ThinkingLevel) -> genai_types.ThinkingConfig:
    """Map the unified `thinking` setting to a Gemini `ThinkingConfig`."""
    if thinking is False:
        return genai_types.ThinkingConfig(thinking_budget=0)  # disable thinking
    level = genai_types.ThinkingLevel.MEDIUM if thinking is True else _GEMINI_THINKING_LEVEL[thinking]
    return genai_types.ThinkingConfig(thinking_level=level)


def _automatic_vad_from_turn_detection(turn_detection: TurnDetection) -> AutomaticVAD:
    """Map cross-provider turn detection to Gemini's automatic-VAD shape."""
    sensitivity = turn_detection.sensitivity if turn_detection.sensitivity != 'medium' else None
    return AutomaticVAD(
        start_sensitivity=sensitivity,
        end_sensitivity=sensitivity,
        prefix_padding_ms=turn_detection.prefix_padding_ms,
        silence_duration_ms=turn_detection.silence_duration_ms,
    )


async def _seed_turns(
    messages: Sequence[ModelMessage], *, profile: RealtimeModelProfile, provider_name: str
) -> list[genai_types.Content | genai_types.ContentDict]:
    """Map prior history to Gemini `clientContent.turns`.

    Text, transcripts, inline images, and tag-wrapped thinking are replayed in part order. Gemini Live
    rejects function parts in `clientContent.turns`, so function calls and results are projected as
    structured text: `[Tool call: name(args)]`, `[Tool "name" returned: result]`, and
    `[Tool "name" error: error]`. Native-tool parts are skipped because they describe provider-executed
    work whose answer text is already retained.

    Thinking signatures and `provider_details` are provider-session-bound and are not replayed.
    `SystemPromptPart`s are routed through `system_instruction`, and `CachePoint`s are ignored. Gemini
    does not accept audio in seeded turns, so speech requires a transcript. Other unrepresentable
    content raises [`UserError`][pydantic_ai.exceptions.UserError].
    """
    turns: list[genai_types.Content | genai_types.ContentDict] = []
    supports_images = profile.get('supports_seeding_images', False)
    supports_audio = profile.get('supports_seeding_audio', False)
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = await _seed_request_parts(
                message.parts,
                provider_name=provider_name,
                supports_images=supports_images,
                supports_audio=supports_audio,
            )
            role = 'user'
        else:
            parts = _seed_response_parts(message.parts, provider_name=provider_name, supports_audio=supports_audio)
            role = 'model'
        if parts:
            turns.append(genai_types.Content(role=role, parts=parts))
    return turns


async def _seed_request_parts(
    message_parts: Sequence[ModelRequestPart],
    *,
    provider_name: str,
    supports_images: bool,
    supports_audio: bool,
) -> list[genai_types.Part]:
    parts: list[genai_types.Part] = []
    for part in message_parts:
        if isinstance(part, SystemPromptPart):
            continue
        elif isinstance(part, UserPromptPart):
            parts.extend(
                _genai_user_parts(
                    await seed_user_content(part, provider_name=provider_name, supports_images=supports_images)
                )
            )
        elif isinstance(part, SpeechPart):
            content = seed_speech_content(part, provider_name=provider_name, supports_audio=supports_audio)
            assert isinstance(content, str)
            if content:
                parts.append(genai_types.Part(text=content))
        elif isinstance(part, ToolReturnPart):
            output, user_content = part.model_response_str_and_user_content()
            parts.append(genai_types.Part(text=f'[Tool "{part.tool_name}" returned: {output}]'))
            if user_content:
                parts.extend(
                    _genai_user_parts(
                        await seed_user_content(
                            UserPromptPart(content=user_content),
                            provider_name=provider_name,
                            supports_images=supports_images,
                        )
                    )
                )
        elif isinstance(part, RetryPromptPart):
            output = part.model_response()
            text = output if part.tool_name is None else f'[Tool "{part.tool_name}" error: {output}]'
            parts.append(genai_types.Part(text=text))
        else:
            assert_never(part)
    return parts


def _seed_response_parts(
    message_parts: Sequence[ModelResponsePart], *, provider_name: str, supports_audio: bool
) -> list[genai_types.Part]:
    parts: list[genai_types.Part] = []
    for part in message_parts:
        if isinstance(part, TextPart):
            if part.content:
                parts.append(genai_types.Part(text=part.content))
        elif isinstance(part, ThinkingPart):
            if part.content:
                start_tag, end_tag = DEFAULT_THINKING_TAGS
                parts.append(genai_types.Part(text='\n'.join([start_tag, part.content, end_tag])))
        elif isinstance(part, ToolCallPart):
            parts.append(genai_types.Part(text=f'[Tool call: {part.tool_name}({part.args_as_json_str()})]'))
        elif isinstance(part, (NativeToolCallPart, NativeToolReturnPart)):
            continue
        elif isinstance(part, SpeechPart):
            content = seed_speech_content(part, provider_name=provider_name, supports_audio=supports_audio)
            if content:
                assert isinstance(content, str)
                parts.append(genai_types.Part(text=content))
        elif isinstance(part, CompactionPart):
            # Provider-session-bound compaction state can't round-trip into another session; classic
            # model adapters skip it when crossing APIs (e.g. Chat Completions), and seeding matches.
            continue
        elif isinstance(part, FilePart):
            raise UserError(
                f'`FilePart` cannot be seeded into {provider_name} realtime history. '
                'Convert it to text or filter it from `message_history` before connecting.'
            )
        else:
            assert_never(part)
    return parts


def _genai_user_parts(content: Sequence[str | BinaryContent]) -> list[genai_types.Part]:
    return [
        genai_types.Part(text=item)
        if isinstance(item, str)
        else genai_types.Part(inline_data=genai_types.Blob(data=item.data, mime_type=item.media_type))
        for item in content
        if not isinstance(item, str) or item
    ]


def _tool_def_to_genai(tool: ToolDefinition) -> genai_types.FunctionDeclaration:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to a Gemini function declaration."""
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or None,
        parameters_json_schema=tool.parameters_json_schema,
    )


def _native_tool_to_genai(tool: AbstractNativeTool) -> genai_types.Tool:
    """Map a supported Gemini built-in native tool to a genai `Tool`.

    [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] maps to Grounding with Google Search,
    [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] to URL context, and
    [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] to Gemini's code execution tool.
    [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] validates native tools against
    the model's [`supported_native_tools`][pydantic_ai.realtime.RealtimeModelProfile.supported_native_tools]
    profile before connecting, so only these three reach this mapping.

    Note: some Gemini native-audio Live models reject certain built-in tools at connect time. That's a
    request-time concern surfaced by the API; capturing whatever code-execution parts the model emits
    (see `_map_server_content`) is always safe and needs no tool to have been requested.
    """
    if isinstance(tool, WebSearchTool):
        return genai_types.Tool(google_search=genai_types.GoogleSearch())
    if isinstance(tool, WebFetchTool):
        return genai_types.Tool(url_context=genai_types.UrlContext())
    # Only `CodeExecutionTool` remains, per the profile's `supported_native_tools`.
    return genai_types.Tool(code_execution=genai_types.ToolCodeExecution())


def _map_grounding_parts(content: genai_types.LiveServerContent, provider_name: str) -> list[ModelResponsePart]:
    """Reconstruct the native tool call/return parts for a grounded turn, for history.

    Reuses [`GoogleModel`][pydantic_ai.models.google.GoogleModel]'s grounding mappers so a grounded
    realtime turn's history is byte-identical in shape to a classic request's — a
    [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] /
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] pair for Google Search grounding
    and another for URL context. The session folds these parts into the turn's `ModelResponse`.
    """
    parts: list[ModelResponsePart] = []
    search_call, search_return = _map_grounding_metadata(content.grounding_metadata, provider_name)
    if search_call and search_return:
        parts += [search_call, search_return]
    fetch_call, fetch_return = _map_url_context_metadata(content.url_context_metadata, provider_name)
    if fetch_call and fetch_return:
        parts += [fetch_call, fetch_return]
    return parts


def _modality_tokens(
    details: Sequence[genai_types.ModalityTokenCount] | None, modality: genai_types.MediaModality
) -> int:
    """Sum the token counts for `modality` in Gemini's per-modality usage breakdown."""
    return sum(entry.token_count or 0 for entry in details or [] if entry.modality is modality)


def _map_usage(usage: genai_types.UsageMetadata) -> RequestUsage:
    """Map Gemini `usage_metadata` to a [`RequestUsage`][pydantic_ai.usage.RequestUsage].

    Realtime audio bills at a much higher rate than text, so the per-modality split (Gemini's
    `*_tokens_details`) is mapped into the audio/text token fields rather than dropped — otherwise
    every audio session would be mispriced from the totals alone.
    """
    audio, text = genai_types.MediaModality.AUDIO, genai_types.MediaModality.TEXT
    details: dict[str, int] = {}
    for key, count in (
        ('input_text_tokens', _modality_tokens(usage.prompt_tokens_details, text)),
        ('output_text_tokens', _modality_tokens(usage.response_tokens_details, text)),
        # Reasoning tokens are billed but Gemini leaves them out of `responseTokenCount`/`totalTokenCount`,
        # so they'd be invisible if dropped (mirrors the classic `GoogleModel` mapping).
        ('thoughts_tokens', usage.thoughts_token_count or 0),
    ):
        if count:
            details[key] = count
    return RequestUsage(
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.response_token_count or 0,
        cache_read_tokens=usage.cached_content_token_count or 0,
        input_audio_tokens=_modality_tokens(usage.prompt_tokens_details, audio),
        output_audio_tokens=_modality_tokens(usage.response_tokens_details, audio),
        cache_audio_read_tokens=_modality_tokens(usage.cache_tokens_details, audio),
        details=details,
    )


@contextmanager
def _single_ws_user_agent(client: Client) -> Generator[None]:
    """Drop a duplicate `User-Agent` header for the duration of a Gemini Live WebSocket handshake.

    `google-genai` forwards the client's HTTP headers verbatim as the Live WebSocket's
    `additional_headers`. The `GoogleProvider` adds a capitalized `User-Agent` (for HTTP, where `httpx`
    folds it together with the SDK's own lowercase `user-agent`), but the `websockets` library stores
    headers case-insensitively and rejects the two as a duplicate, failing the handshake. We remove our
    capitalized variant just for the connect and restore it after, so a single user-agent reaches the
    socket while HTTP requests keep pydantic-ai's user-agent.
    """
    # Reach into the SDK's private HTTP options; guarded with `getattr` so custom / fake clients that
    # don't expose them (e.g. in tests) simply skip the reconciliation.
    raw_headers = getattr(getattr(getattr(client, '_api_client', None), '_http_options', None), 'headers', None)
    if not isinstance(raw_headers, dict):
        yield
        return
    headers = cast('dict[str, str]', raw_headers)
    duplicates = [key for key in headers if key.lower() == 'user-agent']
    if len(duplicates) < 2:
        yield
        return
    # Keep the SDK's own lowercase `user-agent` if present, otherwise the first seen; drop the rest.
    keep = 'user-agent' if 'user-agent' in duplicates else duplicates[0]
    removed = {key: headers.pop(key) for key in duplicates if key != keep}
    try:
        yield
    finally:
        headers.update(removed)


@contextmanager
def _ws_trace_context(client: Client) -> Generator[None]:
    """Add the current trace context to the Gemini Live handshake headers for the connect only.

    `google-genai` forwards the client's HTTP headers as the Live WebSocket's `additional_headers`, so
    injecting `traceparent` here propagates trace context to the server (e.g. a gateway) over the
    handshake — see `inject_trace_context` for the rationale. The keys it added are removed afterwards
    so the shared client's later HTTP requests don't carry a stale trace context. Guarded like
    `_single_ws_user_agent`: custom/fake clients without the private HTTP options simply skip injection.
    """
    raw_headers = getattr(getattr(getattr(client, '_api_client', None), '_http_options', None), 'headers', None)
    if not isinstance(raw_headers, dict):
        yield
        return
    headers = cast('dict[str, str]', raw_headers)
    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    added = {key: value for key, value in carrier.items() if key not in headers}
    headers.update(added)
    try:
        yield
    finally:
        for key in added:
            headers.pop(key, None)


@dataclass
class GoogleRealtimeModel(RealtimeModel):
    """Gemini Live API model.

    Session and generation configuration is read from
    [`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings], passed
    through `settings` as model-level defaults or as `model_settings` when opening a session.

    Authentication and the underlying `google-genai` client come from a
    [`Provider`][pydantic_ai.providers.Provider], mirroring [`GoogleModel`][pydantic_ai.models.google.GoogleModel].
    Pass `provider='google'` (the default) for the Gemini Developer API (reads `GOOGLE_API_KEY` /
    `GEMINI_API_KEY`), `provider='google-cloud'` for Vertex AI (Application Default Credentials, useful
    where org policy disallows API keys), or a [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] /
    [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instance for a custom
    key, client, or region. Gemini Live is available on both surfaces.

    Args:
        model: The model name, e.g. `gemini-2.5-flash-native-audio-latest` (an alias that tracks the
            newest native-audio Live model) or `gemini-3.1-flash-live-preview`.
        provider: The provider to use for authentication and API access — `'google'` (Gemini Developer
            API, the default) or `'google-cloud'` (Vertex AI), or a `Provider` instance.
        settings: Model-level defaults for session and generation configuration.
        reconnect: Backoff policy for transparently re-dialing a dropped session; requires
            `google_enable_session_resumption=True`. With no policy, the low-level connection reports
            a non-recoverable session error; `RealtimeSession` raises
            [`RealtimeError`][pydantic_ai.realtime.RealtimeError] from iteration.
    """

    model: str = 'gemini-2.5-flash-native-audio-latest'
    provider: InitVar[Provider[Client] | str] = 'google'
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
    reconnect: ReconnectPolicy | None = None
    _provider: Provider[Client] = field(init=False, repr=False)

    def __post_init__(self, provider: Provider[Client] | str) -> None:
        if isinstance(provider, str):
            provider = cast('Provider[Client]', infer_provider(provider))
        self._provider = provider

    @property
    def client(self) -> Client:
        """The underlying `google.genai.Client` from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset({WebSearchTool, WebFetchTool, CodeExecutionTool})

    def _speech_config(self, model_settings: GoogleRealtimeModelSettings) -> genai_types.SpeechConfig | None:
        """Build the speech/voice config from `voice`, `multi_speaker`, and `language_code`.

        `multi_speaker` takes precedence over `voice` (they are mutually exclusive in the API).
        """
        voice_config: genai_types.VoiceConfig | None = None
        multi_speaker_config: genai_types.MultiSpeakerVoiceConfig | None = None
        multi_speaker = model_settings.get('google_multi_speaker')
        voice = model_settings.get('voice')
        language_code = model_settings.get('google_language_code')
        if multi_speaker is not None:
            multi_speaker_config = genai_types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    genai_types.SpeakerVoiceConfig(
                        speaker=speaker,
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)
                        ),
                    )
                    for speaker, voice in multi_speaker.voices.items()
                ]
            )
        elif voice:
            voice_config = genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=voice)
            )
        if voice_config is None and multi_speaker_config is None and language_code is None:
            return None
        return genai_types.SpeechConfig(
            voice_config=voice_config,
            multi_speaker_voice_config=multi_speaker_config,
            language_code=language_code,
        )

    def _realtime_input_config(
        self, model_settings: GoogleRealtimeModelSettings
    ) -> genai_types.RealtimeInputConfig | None:
        """Build the turn-taking config from `vad`, `activity_handling`, and `turn_coverage`."""
        detection: genai_types.AutomaticActivityDetection | None = None
        if 'google_vad' in model_settings:
            vad = model_settings['google_vad']
        elif 'turn_detection' in model_settings:
            turn_detection = model_settings['turn_detection']
            if turn_detection is False:
                # Disabling VAD is push-to-talk, which needs manual turn control Gemini Live doesn't
                # expose through this session API yet (no `commit_audio()`/`create_response()`), so a
                # disabled session would be unusable. Fail loudly instead.
                raise UserError(
                    'Gemini Live does not support disabling automatic turn detection (push-to-talk) '
                    'through the realtime session API yet, as it has no manual turn controls. Use '
                    'automatic turn detection (the default) or configure `google_vad`.'
                )
            # `True` means the provider default (on), same as an absent setting.
            vad = None if turn_detection is True else _automatic_vad_from_turn_detection(turn_detection)
        else:
            vad = None
        if vad is not None:
            detection = genai_types.AutomaticActivityDetection(
                disabled=vad.disabled or None,
                start_of_speech_sensitivity=_START_SENSITIVITY[vad.start_sensitivity]
                if vad.start_sensitivity
                else None,
                end_of_speech_sensitivity=_END_SENSITIVITY[vad.end_sensitivity] if vad.end_sensitivity else None,
                prefix_padding_ms=vad.prefix_padding_ms,
                silence_duration_ms=vad.silence_duration_ms,
            )
        activity_handling = model_settings.get('google_activity_handling')
        turn_coverage = model_settings.get('google_turn_coverage')
        activity = _ACTIVITY_HANDLING[activity_handling] if activity_handling else None
        coverage = _TURN_COVERAGE[turn_coverage] if turn_coverage else None
        if detection is None and activity is None and coverage is None:
            return None
        return genai_types.RealtimeInputConfig(
            automatic_activity_detection=detection, activity_handling=activity, turn_coverage=coverage
        )

    def _apply_generation(
        self, config: genai_types.LiveConnectConfig, model_settings: GoogleRealtimeModelSettings | None
    ) -> None:
        """Apply generation params from `model_settings` (base keys + Google-specific ones)."""
        if not model_settings:
            return
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config.max_output_tokens = max_tokens
        if (temperature := model_settings.get('temperature')) is not None:
            config.temperature = temperature
        if (top_p := model_settings.get('top_p')) is not None:
            config.top_p = top_p
        if (top_k := model_settings.get('top_k')) is not None:
            config.top_k = top_k
        if (seed := model_settings.get('seed')) is not None:
            config.seed = seed
        if (google_thinking := model_settings.get('google_thinking_config')) is not None:
            # The Gemini-native config takes precedence over the cross-provider `thinking` setting.
            config.thinking_config = genai_types.ThinkingConfig(**google_thinking)
        elif (thinking := model_settings.get('thinking')) is not None:
            if self.profile.get('supports_thinking', False):
                config.thinking_config = _thinking_to_config(thinking)
            else:
                warnings.warn(
                    f'The {self.model!r} realtime model does not support the `thinking` setting; ignoring it.',
                    UserWarning,
                )
        if (resolution := model_settings.get('google_video_resolution')) is not None:
            config.media_resolution = resolution

    def _config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: GoogleRealtimeModelSettings | None,
        *,
        native_tools: list[AbstractNativeTool] | None = None,
        resumption_handle: str | None = None,
    ) -> genai_types.LiveConnectConfig:
        settings = cast('GoogleRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        modality = (
            genai_types.Modality.AUDIO
            if settings.get('output_modality', 'audio') == 'audio'
            else genai_types.Modality.TEXT
        )
        config = genai_types.LiveConnectConfig(response_modalities=[modality])
        if instructions:
            config.system_instruction = instructions
        config.speech_config = self._speech_config(settings)
        transcription_language_codes = settings.get('google_transcription_language_codes')
        if settings.get('google_input_transcription', True):
            config.input_audio_transcription = genai_types.AudioTranscriptionConfig(
                language_codes=transcription_language_codes
            )
        if settings.get('google_output_transcription', True):
            config.output_audio_transcription = genai_types.AudioTranscriptionConfig(
                language_codes=transcription_language_codes
            )
        config.realtime_input_config = self._realtime_input_config(settings)
        if settings.get('google_affective_dialog', False):
            config.enable_affective_dialog = True
        if settings.get('google_proactive_audio', False):
            config.proactivity = genai_types.ProactivityConfig(proactive_audio=True)
        if (context_compression := settings.get('google_context_compression')) is not None:
            config.context_window_compression = genai_types.ContextWindowCompressionConfig(
                trigger_tokens=context_compression.trigger_tokens,
                sliding_window=genai_types.SlidingWindow(target_tokens=context_compression.target_tokens),
            )
        if settings.get('google_enable_session_resumption', False):
            config.session_resumption = genai_types.SessionResumptionConfig(handle=resumption_handle)
        # Typed as `list[Any]` because `LiveConnectConfig.tools` is a broad union (Tool | Callable |
        # MCP types); a precisely-typed `list[Tool]` isn't assignable to it (list invariance).
        genai_tools: list[Any] = []
        if tools:
            genai_tools.append(genai_types.Tool(function_declarations=[_tool_def_to_genai(t) for t in tools]))
        genai_tools.extend(_native_tool_to_genai(t) for t in native_tools or [])
        if genai_tools:
            config.tools = genai_tools
        self._apply_generation(config, settings)
        if config_overrides := settings.get('google_config_overrides'):
            for key, value in config_overrides.items():
                setattr(config, key, value)
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[GoogleRealtimeConnection]:
        client = self._provider.client
        settings = cast('GoogleRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        instructions = get_instructions(messages) or ''
        # Transparent reconnect needs both a backoff policy and session resumption (so the server
        # restores state on re-dial). Without resumption a re-dial would lose the conversation.
        reconnectable = self.reconnect is not None and settings.get('google_enable_session_resumption', False)
        # The live connection's context manager. A reconnect closes the previous one before opening
        # the next (so they don't accumulate), and teardown closes whatever is current.
        cm: AbstractAsyncContextManager[AsyncSession] | None = None

        async def dial(handle: str | None) -> AsyncSession:
            nonlocal cm
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            config = self._config(
                instructions,
                model_request_parameters.function_tools,
                settings,
                native_tools=model_request_parameters.native_tools,
                resumption_handle=handle,
            )
            opening = client.aio.live.connect(model=self.model, config=config)
            async with _ws_connect_lock(client):
                with _single_ws_user_agent(client), _ws_trace_context(client):
                    session = await opening.__aenter__()
            cm = opening
            return session

        try:
            session = await dial(None)
            # Seed prior conversation once, after the initial connect, as inactive context turns (no
            # `turn_complete`, so the model doesn't respond yet). Reconnects don't re-seed: session
            # resumption restores server state, and a `ReconnectedEvent` starts a fresh turn.
            if turns := await _seed_turns(messages, profile=self.profile, provider_name=self.system):
                await session.send_client_content(turns=turns, turn_complete=False)
            yield GoogleRealtimeConnection(
                session,
                provider_name=self._provider.name,
                dial=dial if reconnectable else None,
                reconnect=self.reconnect if reconnectable else None,
                input_transcription_enabled=settings.get('google_input_transcription', True),
            )
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)


class GoogleRealtimeConnection(RealtimeConnection):
    """A live connection to the Gemini Live API, backed by a `google-genai` session."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        provider_name: str = 'google',
        dial: Callable[[str | None], Awaitable[AsyncSession]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
    ) -> None:
        self._session = session
        self._input_transcription_enabled = input_transcription_enabled
        # Provider name stamped onto native-tool history parts (grounding / code execution), matching the
        # classic `GoogleModel` (`NativeToolCallPart.provider_name`), so a turn's history is provider-tagged
        # identically whether it came from a realtime session or a classic run.
        self._provider_name = provider_name
        # internal call id -> (tool name, Gemini call id), so a `ToolResult` can echo the name and id
        # Gemini requires. Calls Gemini sends without an id get a unique internal id so parallel
        # id-less calls don't collide.
        self._tool_calls: dict[str, tuple[str, str | None]] = {}
        self._call_index = 0
        # The `tool_call_id` generated for the most recent `executable_code` part, reused to pair the
        # following `code_execution_result` return with its call — mirroring the classic `GoogleModel`
        # streaming path, which threads a single id from the code part to its result.
        self._code_execution_tool_call_id: str | None = None
        # `dial` re-establishes a configured session from the latest resumption handle; with a
        # `reconnect` policy it recovers a dropped connection.
        self._dial = dial
        self._reconnect = reconnect
        self._resumption_handle: str | None = None
        self._turn_interrupted = False

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the Gemini Live API.

        Accepts `AudioInput` (PCM16, 16kHz, mono), `TextInput`, `ImageInput` (a live video frame),
        and `ToolResult`. The manual turn-taking verbs are not supported (Gemini uses automatic VAD).
        """
        # `send_realtime_input` is typed against a PIL.Image union the SDK leaves partially untyped.
        if isinstance(content, AudioInput):
            await self._session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                audio=genai_types.Blob(data=content.data, mime_type=f'audio/pcm;rate={INPUT_SAMPLE_RATE}')
            )
        elif isinstance(content, TextInput):
            # A typed message is a discrete turn: commit it with `send_client_content(turn_complete=True)`
            # so the model replies, rather than buffering it as streaming realtime input.
            await self._session.send_client_content(
                turns=genai_types.Content(role='user', parts=[genai_types.Part(text=content.text)]),
                turn_complete=True,
            )
        elif isinstance(content, ImageInput):
            await self._session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                video=genai_types.Blob(data=content.data, mime_type=content.media_type)
            )
        elif isinstance(content, ToolResult):
            name, gemini_id = self._tool_calls.pop(content.tool_call_id, ('', None))
            output = content.output
            if content.content:
                text_content: list[str] = []
                for item in content.content:
                    if isinstance(item, str):
                        text_content.append(item)
                    elif isinstance(item, TextContent):
                        text_content.append(item.content)
                    elif isinstance(item, CachePoint):
                        continue
                    elif isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent, UploadedFile)):
                        text_content.append(f'[{type(item).__name__}: {item.identifier}]')
                    else:
                        assert_never(item)
                output = '\n\n'.join(part for part in (output, *text_content) if part)
            await self._session.send_tool_response(
                function_responses=genai_types.FunctionResponse(
                    id=gemini_id,
                    name=name,
                    response={'output': output},
                )
            )
        else:
            raise NotImplementedError(f'Gemini Live does not support {type(content).__name__} input')

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        # `session.receive()` yields a single model turn and then returns, so loop to keep serving
        # subsequent turns. When the server closes the WebSocket — its connection-time limit, or on
        # teardown — `receive()` raises (the SDK surfaces a closed socket as an `APIError`). Without a
        # reconnect policy that ends the stream; with one, re-dial from the latest resumption handle.
        while True:
            try:
                # Coverage cannot attribute the normal async-generator exhaustion back to this outer
                # loop; `test_connect_continues_after_empty_server_turn` exercises that continuation.
                async for message in self._session.receive():  # pragma: no branch
                    for event in self._map_message(message):
                        yield event
            except (ConnectionClosed, genai_errors.APIError) as e:
                if self._dial is None or self._reconnect is None:
                    # No reconnect policy: a dropped connection is fatal. Surface it as a
                    # non-recoverable error and end the stream cleanly, rather than returning silently
                    # (mirroring the OpenAI provider), so callers don't treat a truncated turn as complete.
                    yield SessionErrorEvent(message=f'Gemini realtime connection closed: {e}', recoverable=False)
                    return
                if await self._try_reconnect():
                    yield ReconnectedEvent(state_restored=True)
                    continue
                yield SessionErrorEvent(
                    message=f'Gemini realtime connection closed; reconnect failed: {e}', recoverable=False
                )
                return
            # `receive()` returned normally → the turn ended; loop for the next one.

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff, resuming from the latest handle; return whether it worked."""
        assert self._dial is not None and self._reconnect is not None
        return await reconnect_with_backoff(self._reconnect, self._attempt_reconnect)

    async def _attempt_reconnect(self) -> bool:
        assert self._dial is not None
        try:
            self._session = await self._dial(self._resumption_handle)
        except (genai_errors.APIError, ConnectionClosed, OSError, TimeoutError):
            # Expected dial failures: SDK-reported API errors, a closed socket, and network/timeout
            # errors. A retry may still succeed. Anything else is a bug in `dial()` and propagates
            # rather than masquerading as a failed reconnect.
            return False
        return True

    def _map_server_content(self, content: genai_types.LiveServerContent) -> list[RealtimeCodecEvent]:
        """Translate a `server_content` message (audio/transcripts/native tools/turn boundary) to events."""
        events: list[RealtimeCodecEvent] = []
        # Native tool call/return parts reconstructed for history (code execution here, web grounding
        # below), folded into the turn's `ModelResponse` by the session rather than yielded live.
        native_tool_parts: list[ModelResponsePart] = []
        if content.model_turn is not None:
            for part in content.model_turn.parts or []:
                if part.inline_data is not None and part.inline_data.data:
                    events.append(AudioDelta(data=part.inline_data.data))
                elif part.executable_code is not None:
                    # Reuse the classic `GoogleModel` mapper so the code-execution call part is
                    # byte-identical; generate and stash the id to pair the following result with it.
                    self._code_execution_tool_call_id = generate_tool_call_id()
                    native_tool_parts.append(
                        _map_executable_code(
                            part.executable_code, self._provider_name, self._code_execution_tool_call_id
                        )
                    )
                elif part.code_execution_result is not None:
                    # The result always follows its `executable_code` part, so the id is set (mirrors the
                    # classic streaming path's assertion).
                    assert self._code_execution_tool_call_id is not None
                    native_tool_parts.append(
                        _map_code_execution_result(
                            part.code_execution_result, self._provider_name, self._code_execution_tool_call_id
                        )
                    )
                elif part.text and not part.thought:
                    # Skip thinking parts: native-audio models stream their reasoning as `thought`
                    # text alongside the spoken answer, and it must not leak into the transcript. A
                    # model-turn text part is the model's plain text output (`response_modality='text'`),
                    # distinct from the spoken-audio transcription in `output_transcription` below, so it
                    # becomes a `TextPart` rather than a `SpeechPart`.
                    events.append(Transcript(text=part.text, is_final=False, output_text=True))
        if content.input_transcription is not None and content.input_transcription.text:
            events.append(
                InputTranscript(
                    text=content.input_transcription.text, is_final=bool(content.input_transcription.finished)
                )
            )
        if content.output_transcription is not None and content.output_transcription.text:
            events.append(
                Transcript(text=content.output_transcription.text, is_final=bool(content.output_transcription.finished))
            )
        if content.interrupted:
            self._turn_interrupted = True
            events.append(InputSpeechStartEvent())
        native_tool_parts += _map_grounding_parts(content, self._provider_name)
        for index, part in enumerate(native_tool_parts):
            events.extend((PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)))
        # `turn_complete` is emitted by `_map_message` *after* the message's `usage_metadata`, not here:
        # Gemini packs `turnComplete` and `usageMetadata` into the same message, and the session
        # finalizes the response's usage on `TurnCompleteEvent`, so the usage must be accounted first
        # (matching OpenAI's codec, which emits usage before the turn boundary).
        return events

    def _map_message(self, message: genai_types.LiveServerMessage) -> list[RealtimeCodecEvent]:
        events: list[RealtimeCodecEvent] = []
        if message.server_content is not None:
            events.extend(self._map_server_content(message.server_content))
        if message.tool_call is not None:
            for call in message.tool_call.function_calls or []:
                name = call.name or ''
                # Gemini usually assigns an id, but fall back to a unique internal one so parallel
                # id-less calls don't collide on the same key.
                call_id = call.id or f'__call_{self._call_index}'
                self._call_index += 1
                self._tool_calls[call_id] = (name, call.id)
                events.append(ToolCall(tool_call_id=call_id, tool_name=name, args=json.dumps(call.args or {})))
        if message.tool_call_cancellation is not None and (cancelled_ids := message.tool_call_cancellation.ids):
            # The cancellation carries Gemini's own call ids, which match the `tool_call_id`s emitted
            # above whenever Gemini assigned them (id-less calls can't be cancelled by id anyway).
            events.append(ToolCallCancelled(tool_call_ids=list(cancelled_ids)))
        if message.usage_metadata is not None:
            events.append(SessionUsageEvent(usage=_map_usage(message.usage_metadata)))
        # Emit the turn boundary last — after this message's usage — so the session folds the turn's
        # tokens into the finalized `ModelResponse` / `chat` span before `TurnCompleteEvent` closes it.
        if message.server_content is not None and message.server_content.turn_complete:
            interrupted = self._turn_interrupted
            events.append(TurnCompleteEvent(interrupted=interrupted))
            self._turn_interrupted = False
        # Track the resumption handle (internal state, not an event) so a reconnect can resume state.
        update = message.session_resumption_update
        if update is not None and update.new_handle:
            self._resumption_handle = update.new_handle
        return events
