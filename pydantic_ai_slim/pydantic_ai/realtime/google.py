"""Gemini Live API provider for realtime speech-to-speech (and live video) sessions.

Built on the `google-genai` SDK, which manages the WebSocket transport for you. Available via the
`google` optional group:

    pip install "pydantic-ai-slim[google]"

Unlike the OpenAI provider, Gemini wants **16 kHz** PCM input audio (output is 24 kHz), produces a
single response modality per session (audio *or* text), and natively accepts a stream of video
frames sent as `ImageInput`.

Authenticates against the Gemini Developer API with an API key by default, or against Vertex AI with
Application Default Credentials when `vertexai=True` (useful where org policy disallows API keys).
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, cast

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
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponsePart,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    TextPart,
    UserPromptPart,
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
from ..providers import Provider, infer_provider
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
    RealtimeModelSettings,
    ReconnectedEvent,
    ReconnectPolicy,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCompleteEvent,
    inject_trace_context,
    reconnect_with_backoff,
    user_prompt_text,
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
    """Whether to transcribe input audio. Defaults to `True`."""
    google_output_transcription: bool
    """Whether to transcribe output audio. Defaults to `True`."""
    google_transcription_language_codes: list[str]
    """Language hints applied to input and output transcription."""
    google_vad: AutomaticVAD
    """Server-side voice activity detection settings."""
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
    """Turn off automatic VAD entirely; the client then drives turns with activity markers."""
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


def _seed_turns(messages: Sequence[ModelMessage]) -> list[genai_types.Content | genai_types.ContentDict]:
    """Project prior conversation to Gemini `Content` turns (text/transcript only, v1).

    User prompts and user-spoken transcripts become `user` turns; assistant text and assistant-spoken
    transcripts become `model` turns. `SystemPromptPart`s are skipped (`system_instruction` covers
    system-level guidance) and tool calls/results are skipped (full tool-round replay is out of scope
    for v1). Content that can't be projected is dropped rather than erroring.
    """
    turns: list[genai_types.Content | genai_types.ContentDict] = []
    for message in messages:
        texts: list[str] = []
        if isinstance(message, ModelRequest):
            for req_part in message.parts:
                if isinstance(req_part, UserPromptPart) and (text := user_prompt_text(req_part)):
                    texts.append(text)
                elif isinstance(req_part, SpeechPart) and req_part.transcript:
                    texts.append(req_part.transcript)
            role = 'user'
        else:
            for resp_part in message.parts:
                if isinstance(resp_part, TextPart) and resp_part.content:
                    texts.append(resp_part.content)
                elif isinstance(resp_part, SpeechPart) and resp_part.transcript:
                    texts.append(resp_part.transcript)
            role = 'model'
        if texts:
            turns.append(genai_types.Content(role=role, parts=[genai_types.Part(text=t) for t in texts]))
    return turns


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


def _map_usage(usage: genai_types.UsageMetadata) -> RequestUsage:
    """Map Gemini `usage_metadata` to a [`RequestUsage`][pydantic_ai.usage.RequestUsage]."""
    return RequestUsage(
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.response_token_count or 0,
        cache_read_tokens=usage.cached_content_token_count or 0,
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
            `google_enable_session_resumption=True`. `None` (default) makes a drop end the stream.
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
        """The underlying `google-genai` [`Client`][google.genai.Client] from the provider."""
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
        vad = model_settings.get('google_vad')
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
        if (thinking := model_settings.get('google_thinking_config')) is not None:
            config.thinking_config = genai_types.ThinkingConfig(**thinking)
        if (resolution := model_settings.get('google_video_resolution')) is not None:
            config.media_resolution = resolution

    def _config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: RealtimeModelSettings | None,
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
            with _single_ws_user_agent(client), _ws_trace_context(client):
                session = await opening.__aenter__()
            cm = opening
            return session

        try:
            session = await dial(None)
            # Seed prior conversation once, after the initial connect, as inactive context turns (no
            # `turn_complete`, so the model doesn't respond yet). Reconnects don't re-seed: session
            # resumption restores server state, and a `ReconnectedEvent` starts a fresh turn.
            if turns := _seed_turns(messages):
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
                video=genai_types.Blob(data=content.data, mime_type=content.mime_type)
            )
        elif isinstance(content, ToolResult):
            name, gemini_id = self._tool_calls.pop(content.tool_call_id, ('', None))
            await self._session.send_tool_response(
                function_responses=genai_types.FunctionResponse(
                    id=gemini_id,
                    name=name,
                    response={'output': content.output},
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
                async for message in self._session.receive():
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
                    yield ReconnectedEvent()
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
            events.append(InputSpeechStartEvent())
        native_tool_parts += _map_grounding_parts(content, self._provider_name)
        for index, part in enumerate(native_tool_parts):
            events.extend((PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)))
        if content.turn_complete:
            events.append(TurnCompleteEvent(interrupted=False))
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
        if message.usage_metadata is not None:
            events.append(SessionUsageEvent(usage=_map_usage(message.usage_metadata)))
        # Track the resumption handle (internal state, not an event) so a reconnect can resume state.
        update = message.session_resumption_update
        if update is not None and update.new_handle:
            self._resumption_handle = update.new_handle
        return events
