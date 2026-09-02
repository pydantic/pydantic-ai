"""Gemini Live API provider for realtime speech-to-speech (and live video) sessions.

Built on the `google-genai` SDK, which manages the WebSocket transport for you. Available via the
`google` optional group:

    pip install "pydantic-ai-slim[google-realtime]"

Unlike the OpenAI provider, Gemini wants **16 kHz** PCM input audio (output is 24 kHz), produces a
single response modality per session (audio *or* text), and natively accepts a stream of video
frames sent as [`BinaryImage`][pydantic_ai.messages.BinaryImage].

Use `provider='google'` for the Gemini Developer API, or `provider='google-cloud'` /
[`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] for Google Cloud with
Application Default Credentials.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, ExitStack, asynccontextmanager, contextmanager
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, cast

from anyio import Lock
from anyio.lowlevel import RunVar
from pydantic_core import to_json
from typing_extensions import TypedDict, assert_never

try:
    import websockets
    from google.genai import Client, errors as genai_errors, types as genai_types
    from google.genai.live import AsyncSession, ConnectionClosed
except ImportError as _import_error:
    raise ImportError(
        'Please install the `google-genai` package to use the Gemini realtime model, '
        'you can use the `google-realtime` optional group - `pip install "pydantic-ai-slim[google-realtime]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from .._utils import generate_tool_call_id
from ..exceptions import ModelHTTPError, UserError
from ..messages import (
    AudioUrl,
    BinaryAudio,
    BinaryContent,
    BinaryImage,
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
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
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
    _map_api_error,  # pyright: ignore[reportPrivateUsage]
    _map_code_execution_result,  # pyright: ignore[reportPrivateUsage]
    _map_executable_code,  # pyright: ignore[reportPrivateUsage]
    _map_grounding_metadata,  # pyright: ignore[reportPrivateUsage]
    _map_url_context_metadata,  # pyright: ignore[reportPrivateUsage]
    _thinking_effort_to_level,  # pyright: ignore[reportPrivateUsage]
    _usage_metadata_as_usage,  # pyright: ignore[reportPrivateUsage]
)
from ..native_tools import AbstractNativeTool, CodeExecutionTool, WebFetchTool, WebSearchTool
from ..profiles import DEFAULT_THINKING_TAGS
from ..profiles.google import (
    GoogleOpenAPISchemaTransformer,
    _drop_unsupported_schema_keywords,  # pyright: ignore[reportPrivateUsage]
)
from ..providers import Provider, infer_provider
from ..settings import ThinkingLevel
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._utils import (
    inject_trace_context,
    reconnect_with_backoff,
    require_pcm_audio,
    resolve_advertised_tools,
    seed_speech_content,
    seed_user_content,
)
from .codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)
from .model import RealtimeError, RealtimeModel
from .profiles import DEFAULT_REALTIME_PROFILE, RealtimeModelProfile, RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy, TurnDetection

LatestGoogleRealtimeModelNames = Literal[
    'gemini-2.5-flash-native-audio-latest',
    'gemini-3.1-flash-live-preview',
]
GoogleRealtimeModelName = str | LatestGoogleRealtimeModelNames

__all__ = (
    'GoogleRealtimeModel',
    'GoogleRealtimeModelSettings',
    'GoogleRealtimeConnection',
    'AutomaticVAD',
    'MultiSpeaker',
    'ContextCompression',
)


class AutomaticVAD(TypedDict, total=False):
    """Server-side voice activity detection — the default turn-taking mode for Gemini Live."""

    disabled: bool
    """Turn off automatic VAD entirely. Defaults to `False`.

    Do not set this through `RealtimeSession`: Pydantic AI does not expose Gemini activity markers or
    manual turn controls. Use automatic VAD instead; the shared `turn_detection=False` setting is
    rejected for the same reason.
    """
    start_sensitivity: Literal['high', 'low']
    """How readily speech onset is detected. `high` triggers on quieter audio; `low` is stricter.
    Defaults to the provider default."""
    end_sensitivity: Literal['high', 'low']
    """How readily the end of speech is detected. `high` ends turns sooner; `low` waits longer.
    Defaults to the provider default."""
    prefix_padding_ms: int
    """Audio to include before detected speech, in milliseconds. Defaults to the provider default."""
    silence_duration_ms: int
    """Silence required to detect the end of speech, in milliseconds. Defaults to the provider default."""


class MultiSpeaker(TypedDict, total=False):
    """Assign prebuilt voices to named speakers for multi-speaker audio output."""

    voices: dict[str, str]
    """Mapping of speaker label to prebuilt voice name, e.g. `{'Joe': 'Puck', 'Jane': 'Kore'}`.
    Defaults to an empty mapping."""


class ContextCompression(TypedDict, total=False):
    """Sliding-window context compression so long sessions don't exceed the context window."""

    trigger_tokens: int
    """Compress once the context passes this many tokens. Defaults to the provider default."""
    target_tokens: int
    """Target size (in tokens) of the retained sliding window after compression.
    Defaults to the provider default."""


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
    google_voice: str
    """Prebuilt voice used for audio output, e.g. `Puck`."""
    google_multi_speaker: MultiSpeaker
    """Per-speaker voice assignments; takes precedence over `google_voice`."""
    google_affective_dialog: bool
    """Whether to enable emotion-aware delivery (native-audio models only)."""
    google_proactive_audio: bool
    """Whether the model may decide *when* to respond, including staying silent on input not
    addressed to it (native-audio models only). Useful for "react to the camera" experiences."""
    google_input_transcription: bool
    """Whether to transcribe input audio. Defaults to `True`.

    When `False`, user turns are recorded as retained audio when available, or as content-less
    placeholders otherwise. Takes precedence over the shared
    [`input_transcription_model`][pydantic_ai.realtime.RealtimeModelSettings.input_transcription_model],
    whose `None` also turns transcription off here.
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
    `google_vad={'disabled': True}` raises a `UserError`, like `turn_detection=False`: Pydantic AI does
    not expose Gemini activity markers or manual turn controls, so the resulting session could not
    drive turns.
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
    """Whether to request session-resumption handles, which let a re-dial restore the server-side
    conversation.

    When absent, handles are requested exactly when a
    [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy is set. An explicit
    `False` cannot be combined with a `reconnect` policy: a re-dial without resumption would lose the
    conversation, so `connect` raises [`UserError`][pydantic_ai.exceptions.UserError] rather than
    silently reconnecting into a model that remembers nothing.
    """

    google_async_tool_calls: bool
    """Whether tool calls may run without pausing the model's speech. Defaults to `False`.

    By default Gemini stops generating while a tool call is outstanding, so the caller hears silence
    for as long as the tool takes. Enabling this declares tools `NON_BLOCKING` and returns their
    results with `INTERRUPT` scheduling, so the model keeps talking (typically narrating what it's
    doing) and the result cuts into that speech when it arrives.

    This pays off for tools that take a noticeable moment. It is a poor trade for fast tools: the
    result interrupts a reply the model has barely started, leaving an extra interrupted turn in
    history with nothing in it. Verified live against `gemini-2.5-flash-native-audio-latest`.

    Supported by Gemini native-audio models (see
    [`supports_async_tool_calls`][pydantic_ai.realtime.RealtimeModelProfile.supports_async_tool_calls]).
    Other models silently ignore it.
    """


INPUT_SAMPLE_RATE = 16000
"""Sample rate (Hz) Gemini expects for PCM16 input audio."""


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

_WS_CONNECT_LOCK: RunVar[Lock] = RunVar('gemini_live_ws_connect_lock')


def _ws_connect_lock() -> Lock:
    """Return the lock serializing the temporary mutations a Gemini Live handshake needs.

    Process-wide, not per-client, because one of those mutations — the gateway URL rewrite — replaces
    `google.genai.live.ws_connect`, a module global. Two sessions on *different* clients would each
    take their own lock, and whichever restored second would put the other's replacement back as the
    "original", leaving every later Vertex session pointed at the gateway path. A handshake is short,
    so serializing them costs little next to that.

    One lock per running event loop rather than a single module-level one: an `anyio.Lock` binds to
    the loop (and async backend) it is first used on, so a shared instance breaks an app that opens
    sessions from more than one runtime — the same hazard `Provider._enter_lock` defers construction
    to avoid. Sessions racing this rewrite from different loops is the far-fetched case; sessions
    racing it from one is the case that has to hold. A `RunVar` holds them because its storage is
    weak-keyed on the running loop, so an app that starts and tears down loops repeatedly doesn't
    accumulate one lock per loop it ever ran.
    """
    lock = _WS_CONNECT_LOCK.get(None)
    if lock is None:
        lock = Lock()
        _WS_CONNECT_LOCK.set(lock)
    return lock


def _thinking_to_config(thinking: ThinkingLevel) -> genai_types.ThinkingConfig:
    """Map the unified `thinking` setting to a Gemini `ThinkingConfig`."""
    if thinking is False:
        return genai_types.ThinkingConfig(thinking_budget=0)  # disable thinking
    level = (
        genai_types.ThinkingLevel.MEDIUM
        if thinking is True
        else genai_types.ThinkingLevel(_thinking_effort_to_level(thinking))
    )
    return genai_types.ThinkingConfig(thinking_level=level)


def _automatic_vad_from_turn_detection(turn_detection: TurnDetection) -> AutomaticVAD:
    """Map cross-provider turn detection to Gemini's automatic-VAD shape."""
    sensitivity = turn_detection.get('sensitivity')
    result: AutomaticVAD = {}
    if sensitivity is not None and sensitivity != 'medium':
        result['start_sensitivity'] = sensitivity
        result['end_sensitivity'] = sensitivity
    if (prefix_padding_ms := turn_detection.get('prefix_padding_ms')) is not None:
        result['prefix_padding_ms'] = prefix_padding_ms
    if (silence_duration_ms := turn_detection.get('silence_duration_ms')) is not None:
        result['silence_duration_ms'] = silence_duration_ms
    return result


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
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = await _seed_request_parts(
                message.parts,
                provider_name=provider_name,
                supports_images=supports_images,
            )
            role = 'user'
        else:
            parts = _seed_response_parts(message.parts, provider_name=provider_name)
            role = 'model'
        if parts:
            turns.append(genai_types.Content(role=role, parts=parts))
    return turns


async def _seed_request_parts(
    message_parts: Sequence[ModelRequestPart],
    *,
    provider_name: str,
    supports_images: bool,
) -> list[genai_types.Part]:
    parts: list[genai_types.Part] = []
    for part in message_parts:
        if isinstance(part, (SystemPromptPart, ToolAvailabilityDeltaPart)):
            # System prompts are seeded through session instructions, and tool-availability news
            # from a prior standard run is stale here: the session advertises its own tools.
            continue
        elif isinstance(part, UserPromptPart):
            parts.extend(
                _genai_user_parts(
                    await seed_user_content(part=part, provider_name=provider_name, supports_images=supports_images)
                )
            )
        elif isinstance(part, SpeechPart):
            # Gemini has no client-content channel for raw audio, so seeding never replays retained
            # audio regardless of profile flags — the typed result is always a transcript string.
            content = seed_speech_content(part=part, provider_name=provider_name, supports_audio=False)
            if content:
                parts.append(genai_types.Part(text=content))
        elif isinstance(part, ToolReturnPart):
            output, user_content = part.model_response_str_and_user_content()
            parts.append(genai_types.Part(text=f'[Tool {part.tool_call_id}: {part.tool_name} returned: {output}]'))
            if user_content:
                parts.extend(
                    _genai_user_parts(
                        await seed_user_content(
                            part=UserPromptPart(content=user_content),
                            provider_name=provider_name,
                            supports_images=supports_images,
                        )
                    )
                )
        elif isinstance(part, RetryPromptPart):
            output = part.model_response()
            text = output if part.tool_name is None else f'[Tool {part.tool_call_id}: {part.tool_name} error: {output}]'
            parts.append(genai_types.Part(text=text))
        else:
            assert_never(part)
    return parts


def _seed_response_parts(message_parts: Sequence[ModelResponsePart], *, provider_name: str) -> list[genai_types.Part]:
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
            parts.append(
                genai_types.Part(text=f'[Tool {part.tool_call_id}: {part.tool_name}({part.args_as_json_str()})]')
            )
        elif isinstance(part, (NativeToolCallPart, NativeToolReturnPart)):
            continue
        elif isinstance(part, SpeechPart):
            # Assistant audio can't be replayed on any provider; the typed result is a transcript string.
            content = seed_speech_content(part=part, provider_name=provider_name, supports_audio=False)
            if content:
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


def _schema_from_json_schema(json_schema: dict[str, Any]) -> genai_types.Schema:
    """Convert a JSON schema to the `Schema` a Gemini Live function declaration carries.

    A declaration's parameters are *either* `parametersJsonSchema` (full JSON Schema, which
    [`GoogleModel`][pydantic_ai.models.google.GoogleModel] sends) *or* `parameters` — and Live only
    implements the latter, silently ignoring the former, which leaves the model guessing at argument
    names. So the schema goes through
    [`GoogleOpenAPISchemaTransformer`][pydantic_ai.profiles.google.GoogleOpenAPISchemaTransformer]
    into the OpenAPI subset instead, exactly as a standard request did before it moved to JSON Schema.

    `Schema.from_json_schema` would be the obvious builder, but it routes through
    `genai_types.JSONSchema`, which has no `nullable` — every optional argument would arrive
    advertised as non-nullable.
    """
    transformed = GoogleOpenAPISchemaTransformer(json_schema, strict=None).walk()
    accepted_keywords = frozenset(field.alias or name for name, field in genai_types.Schema.model_fields.items())
    return genai_types.Schema.model_validate(
        _drop_unsupported_schema_keywords(transformed, accepted_keywords=accepted_keywords)
    )


def _tool_def_to_genai(tool: ToolDefinition, *, async_tool_calls: bool = False) -> genai_types.FunctionDeclaration:
    """Convert a [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] to a Gemini function declaration."""
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or '',
        parameters=_schema_from_json_schema(tool.parameters_json_schema),
        response=_schema_from_json_schema(tool.return_schema) if tool.return_schema else None,
        behavior=genai_types.Behavior.NON_BLOCKING if async_tool_calls else None,
    )


def _native_tool_to_genai(tool: AbstractNativeTool) -> genai_types.Tool:
    """Map a supported Gemini built-in native tool to a genai `Tool`.

    Today's Live profile enables Google Search only. URL context and code execution remain class-level
    capabilities so a future model profile can enable them through the standard capability/profile
    intersection without another adapter change.
    """
    if isinstance(tool, WebSearchTool):
        return genai_types.Tool(google_search=genai_types.GoogleSearch())
    if isinstance(tool, WebFetchTool):
        return genai_types.Tool(url_context=genai_types.UrlContext())
    if isinstance(tool, CodeExecutionTool):
        return genai_types.Tool(code_execution=genai_types.ToolCodeExecution())
    raise UserError(f'Google realtime does not support the native tool {type(tool).__name__!r}.')


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


def _map_usage(usage: genai_types.UsageMetadata, *, provider_name: str, provider_url: str) -> RequestUsage:
    """Map Gemini Live `usage_metadata` through the standard Gemini usage mapper.

    Live's metadata is the generate-content shape with its output fields renamed from `candidates*`
    to `response*`, so the counts pass straight through and only the extraction payload — which
    genai-prices reads by the generate-content names — is translated back.

    `provider_url` is the provider's HTTP base URL, not the WebSocket the session actually dialed:
    it's what genai-prices matches providers on, and it's the same URL a standard request would
    have reported, so Vertex and the Gemini API resolve exactly as they do off a realtime session.
    """
    extract_data = usage.model_dump(by_alias=True, exclude={'response_token_count', 'response_tokens_details'})
    extract_data['candidatesTokenCount'] = usage.response_token_count
    extract_data['candidatesTokensDetails'] = [
        item.model_dump(by_alias=True) for item in usage.response_tokens_details or ()
    ]
    return _usage_metadata_as_usage(
        prompt_token_count=usage.prompt_token_count,
        output_token_count=usage.response_token_count,
        cached_content_token_count=usage.cached_content_token_count,
        thoughts_token_count=usage.thoughts_token_count,
        tool_use_prompt_token_count=usage.tool_use_prompt_token_count,
        prompt_tokens_details=usage.prompt_tokens_details,
        cache_tokens_details=usage.cache_tokens_details,
        output_tokens_details=usage.response_tokens_details,
        tool_use_prompt_tokens_details=usage.tool_use_prompt_tokens_details,
        output_details_prefix='response',
        extract_data={'usageMetadata': extract_data},
        provider=provider_name,
        provider_url=provider_url,
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
    headers = client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert headers is not None
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
    so the shared client's later HTTP requests don't carry a stale trace context.
    """
    headers = client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert headers is not None
    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    # Compared case-insensitively: header names are, and `websockets` stores them that way, so adding a
    # lowercase `traceparent` next to a `Traceparent` the client already carries is a duplicate header
    # the handshake can be rejected for (the same hazard `_single_ws_user_agent` above reconciles).
    existing = {key.lower() for key in headers}
    added = {key: value for key, value in carrier.items() if key.lower() not in existing}
    headers.update(added)
    try:
        yield
    finally:
        for key in added:
            headers.pop(key, None)


@dataclass(init=False)
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
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use. Mirrors `profile=` on a standard
            [`Model`][pydantic_ai.models.Model], and is the escape hatch when a model name doesn't
            identify the model (e.g. an Azure deployment named something other than its model).
    """

    model: GoogleRealtimeModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: Provider[Client] = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument while
    # `RealtimeModel.profile` stays the *resolved* profile, exactly as on a standard `Model` — a
    # dataclass field of that name would shadow the property.
    def __init__(
        self,
        model: GoogleRealtimeModelName,
        *,
        provider: Literal['google', 'google-cloud', 'gateway'] | Provider[Client] = 'google',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.model = model
        if isinstance(provider, str):
            provider_name = 'gateway/google-cloud' if provider == 'gateway' else provider
            provider = cast('Provider[Client]', infer_provider(provider_name))
        self._provider = provider

    @property
    def client(self) -> Client:
        """The underlying `google.genai.Client` from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> GoogleRealtimeModelName:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset({WebSearchTool, WebFetchTool, CodeExecutionTool})

    def _speech_config(self, model_settings: GoogleRealtimeModelSettings) -> genai_types.SpeechConfig | None:
        """Build the speech/voice config from `google_voice`, `google_multi_speaker`, and `google_language_code`.

        `google_multi_speaker` takes precedence over `google_voice` (they are mutually exclusive in the API).
        """
        voice_config: genai_types.VoiceConfig | None = None
        multi_speaker_config: genai_types.MultiSpeakerVoiceConfig | None = None
        multi_speaker = model_settings.get('google_multi_speaker')
        voice = model_settings.get('google_voice')
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
                    for speaker, voice in multi_speaker.get('voices', {}).items()
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

    def _async_tool_calls(self, model_settings: GoogleRealtimeModelSettings | None) -> bool:
        """Whether to run this session's tool calls without pausing the model's speech.

        Opt-in, and only where the model actually honors it — the other Live families accept
        `NON_BLOCKING` and then block anyway, so enabling it there would promise something the
        provider doesn't deliver.
        """
        if not model_settings or not model_settings.get('google_async_tool_calls', False):
            return False
        if not self.profile.get('supports_async_tool_calls', False):
            return False
        return True

    def _input_transcription(self, settings: GoogleRealtimeModelSettings) -> bool:
        """Whether to transcribe the user's audio.

        Gemini has no separate transcription model to point at, so a *pinned* `input_transcription_model`
        can't be honored — but `None` ("don't transcribe") can be, and must be: it's the setting someone
        reaches for to keep the user's words out of history, and silently transcribing anyway would defeat
        the one thing it exists to do. The provider-specific `google_input_transcription` still wins where
        both are given.
        """
        if (enabled := settings.get('google_input_transcription')) is not None:
            return enabled
        # Absent and `None` are different here: only an explicit `None` asks for transcription off.
        if 'input_transcription_model' in settings and settings['input_transcription_model'] is None:
            return False
        return True

    def _session_resumption_enabled(self, settings: GoogleRealtimeModelSettings) -> bool:
        """Whether to request session-resumption handles.

        An explicit `google_enable_session_resumption` wins; when absent, a `reconnect` policy implies
        resumption, since a re-dial without a handle would lose the conversation.
        """
        if (enabled := settings.get('google_enable_session_resumption')) is not None:
            return enabled
        return settings.get('reconnect') is not None

    def _realtime_input_config(
        self, model_settings: GoogleRealtimeModelSettings
    ) -> genai_types.RealtimeInputConfig | None:
        """Build the turn-taking config from `vad`, `activity_handling`, and `turn_coverage`."""
        detection: genai_types.AutomaticActivityDetection | None = None
        vad: AutomaticVAD | None
        if 'google_vad' in model_settings:
            vad = model_settings['google_vad']
        elif 'turn_detection' in model_settings:
            turn_detection = model_settings['turn_detection']
            # `True` means the provider default (on), same as an absent setting. `False` asks for the
            # same thing as `google_vad={'disabled': True}`, so both land on the check below.
            if turn_detection is False:
                vad = {'disabled': True}
            else:
                vad = None if turn_detection is True else _automatic_vad_from_turn_detection(turn_detection)
        else:
            vad = None
        if vad is not None:
            if vad.get('disabled', False):
                # Disabling VAD is push-to-talk, which needs manual turn control Gemini Live doesn't
                # expose through this session API yet (no `commit_audio()`/`create_response()`), so a
                # disabled session would connect but never take a turn. Fail loudly instead.
                raise UserError(
                    'Gemini Live does not support disabling automatic turn detection (push-to-talk) '
                    'through the realtime session API yet, as it has no manual turn controls. Use '
                    'automatic turn detection (the default) instead.'
                )
            detection = genai_types.AutomaticActivityDetection(
                start_of_speech_sensitivity=_START_SENSITIVITY[start_sensitivity]
                if (start_sensitivity := vad.get('start_sensitivity'))
                else None,
                end_of_speech_sensitivity=_END_SENSITIVITY[end_sensitivity]
                if (end_sensitivity := vad.get('end_sensitivity'))
                else None,
                prefix_padding_ms=vad.get('prefix_padding_ms'),
                silence_duration_ms=vad.get('silence_duration_ms'),
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
        if (resolution := model_settings.get('google_video_resolution')) is not None:
            config.media_resolution = resolution

    def _config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: GoogleRealtimeModelSettings | None,
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
        if self._input_transcription(settings):
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
                trigger_tokens=context_compression.get('trigger_tokens'),
                sliding_window=genai_types.SlidingWindow(target_tokens=context_compression.get('target_tokens')),
            )
        if self._session_resumption_enabled(settings):
            config.session_resumption = genai_types.SessionResumptionConfig(handle=resumption_handle)
        # Typed as `list[Any]` because `LiveConnectConfig.tools` is a broad union (Tool | Callable |
        # MCP types); a precisely-typed `list[Tool]` isn't assignable to it (list invariance).
        genai_tools: list[Any] = []
        # Gemini's live config has no `tool_config`, so the only expressible restriction is which
        # functions are advertised; the mode the resolution asks for is dropped.
        advertised_tools, _ = resolve_advertised_tools(tools, settings.get('tool_choice'))
        if advertised_tools:
            genai_tools.append(
                genai_types.Tool(
                    function_declarations=[
                        _tool_def_to_genai(t, async_tool_calls=self._async_tool_calls(settings))
                        for t in advertised_tools
                    ]
                )
            )
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
        instructions = get_instructions(messages, model_request_parameters) or ''
        # Transparent reconnect needs session resumption, so the server restores state on re-dial;
        # a `reconnect` policy requests it automatically (see `_session_resumption_enabled`). An
        # explicit opt-out alongside a policy would silently reconnect into a model that remembers
        # nothing, so it fails loudly instead.
        reconnect = settings.get('reconnect')
        if reconnect is not None and settings.get('google_enable_session_resumption') is False:
            raise UserError(
                'A `reconnect` policy requires Gemini session resumption, but '
                '`google_enable_session_resumption=False` explicitly disables it. Remove the '
                '`reconnect` policy, or leave `google_enable_session_resumption` unset so the '
                'policy enables resumption.'
            )
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
                model_settings=settings,
                native_tools=model_request_parameters.native_tools,
                resumption_handle=handle,
            )
            opening = client.aio.live.connect(model=self.model, config=config)
            async with _ws_connect_lock():
                with ExitStack() as stack:
                    stack.enter_context(_single_ws_user_agent(client))
                    stack.enter_context(_ws_trace_context(client))
                    # A gateway route needs nothing extra here: the relay routes the SDK's native
                    # Vertex Bidi path, and the gateway bearer auth reaches the handshake via a
                    # static header set on the client at build time (see `_set_google_ws_gateway_auth`).
                    session = await opening.__aenter__()
            cm = opening
            return session

        try:
            # A rejected config (unsupported `voice`, unknown model) closes the WebSocket, which the SDK
            # surfaces as an `APIError`. Map it to the same typed exceptions a regular request raises,
            # mirroring `GoogleModel`. Reconnects dial from the receive loop, which keeps handling the
            # `APIError` as a retryable drop.
            try:
                session = await dial(None)
            except genai_errors.APIError as e:
                mapped_error = _map_api_error(e, self.model)
                if isinstance(mapped_error, ModelHTTPError):
                    raise mapped_error from e
                raise RealtimeError(model_name=self.model, message=str(e)) from e  # pragma: no cover
            except websockets.InvalidStatus as e:
                # A rejected WebSocket upgrade (e.g. bad key → 401) surfaces from `google-genai` as a raw
                # `websockets` error rather than an `APIError`; the WebSocket is the API here, so map its
                # HTTP status to `ModelHTTPError` like a regular request.
                response = e.response
                body = response.body.decode(errors='replace') if response.body else response.reason_phrase
                raise ModelHTTPError(
                    status_code=response.status_code,
                    model_name=self.model,
                    body=body,
                    headers=dict(response.headers),
                ) from e
            except websockets.WebSocketException as e:
                # Any other raw `websockets` handshake failure the SDK didn't wrap as an `APIError`; no HTTP
                # status, so surface it as a `RealtimeError` rather than letting it escape untyped.
                raise RealtimeError(model_name=self.model, message=f'WebSocket error during connect: {e}') from e
            except OSError as e:
                # The connection never came up: DNS failure, refused, reset, or the dial timing out
                # (`TimeoutError` is an `OSError`). No HTTP status exists, so this is a `RealtimeError`
                # too, rather than a bare built-in from what looks like an ordinary model call.
                raise RealtimeError(model_name=self.model, message=f'Could not reach the realtime API: {e}') from e
            # Seed prior conversation once, after the initial connect, as inactive context turns (no
            # `turn_complete`, so the model doesn't respond yet). Reconnects don't re-seed: session
            # resumption restores server state, and a `RealtimeSessionReconnectEvent` starts a fresh turn.
            if turns := await _seed_turns(messages, profile=self.profile, provider_name=self.system):
                await session.send_client_content(turns=turns, turn_complete=False)
            yield GoogleRealtimeConnection(
                session,
                profile=self.profile,
                provider_name=self._provider.name,
                provider_url=self._provider.base_url,
                dial=dial if reconnect is not None else None,
                reconnect=reconnect,
                input_transcription_enabled=self._input_transcription(settings),
                async_tool_calls=self._async_tool_calls(settings),
            )
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)


class GoogleRealtimeConnection(RealtimeConnection):
    """A live connection to the Gemini Live API, backed by a `google-genai` session."""

    # The SDK surfaces a closed socket as `ConnectionClosed` or an `APIError`; `OSError` covers the
    # socket-level failures underneath both.
    transport_errors = (ConnectionClosed, genai_errors.APIError, OSError)
    # How this provider names itself in error messages.
    _provider_label = 'Gemini Live'

    def __init__(
        self,
        session: AsyncSession,
        *,
        profile: RealtimeModelProfile | None = None,
        provider_name: str = 'google',
        dial: Callable[[str | None], Awaitable[AsyncSession]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        async_tool_calls: bool = False,
        provider_url: str = '',
    ) -> None:
        self._session = session
        self._profile = profile if profile is not None else DEFAULT_REALTIME_PROFILE
        self._input_transcription_enabled = input_transcription_enabled
        self._reconnects_used = 0
        self._async_tool_calls_enabled = async_tool_calls
        # Provider name stamped onto native-tool history parts (grounding / code execution), matching the
        # classic `GoogleModel` (`NativeToolCallPart.provider_name`), so a turn's history is provider-tagged
        # identically whether it came from a realtime session or a classic run.
        self._provider_name = provider_name
        # The provider's HTTP base URL, which is how genai-prices identifies a provider for pricing.
        self._provider_url = provider_url
        # internal call id -> (tool name, Gemini call id), so a `ToolResult` can echo the name and id
        # Gemini requires. Calls Gemini sends without an id get a synthetic one so parallel id-less
        # calls don't collide.
        self._tool_calls: dict[str, tuple[str, str | None]] = {}
        self._native_part_index = 0
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
        # Whether the model has streamed response output (audio, transcript, text, or native-tool
        # parts) since the last `turn_complete`. A dropped-and-redialed connection never continues an
        # in-flight turn (session resumption restores conversation state, not the generation;
        # verified live), so when this is set at reconnect time the turn's boundary would otherwise
        # never arrive — see `__aiter__`, which closes the orphaned turn before the reconnect event.
        self._turn_open = False

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the Gemini Live API.

        Accepts `BinaryAudio` (raw PCM16, 16kHz, mono), a `str` text turn, `BinaryImage` (a live
        video frame), and `ToolResult`. The manual turn-taking verbs are not supported (Gemini uses
        automatic VAD).
        """
        # `send_realtime_input` is typed against a PIL.Image union the SDK leaves partially untyped.
        if isinstance(content, BinaryAudio):
            require_pcm_audio(content, provider_name=self._provider_name)
            await self._session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                audio=genai_types.Blob(data=content.data, mime_type=f'audio/pcm;rate={INPUT_SAMPLE_RATE}')
            )
        elif isinstance(content, str):
            # A typed message is a discrete turn: commit it with `send_client_content(turn_complete=True)`
            # so the model replies, rather than buffering it as streaming realtime input.
            await self._session.send_client_content(
                turns=genai_types.Content(role='user', parts=[genai_types.Part(text=content)]),
                turn_complete=True,
            )
        elif isinstance(content, BinaryImage):
            await self._session.send_realtime_input(  # pyright: ignore[reportUnknownMemberType]
                video=genai_types.Blob(data=content.data, mime_type=content.media_type)
            )
        elif isinstance(content, ToolResult):
            name, gemini_id = self._tool_calls.pop(content.tool_call_id, ('', None))
            # `FunctionResponse.response` is JSON-only, so text attachments are folded into the
            # output and binary attachments raise — loudly, with the tool result unsent, never a
            # silent placeholder. Every live delivery channel was probed and fails: content in a
            # `send_client_content(turn_complete=False)` turn or a `send_realtime_input` frame is
            # invisible to the generation `send_tool_response` triggers (the model guesses), a
            # `turn_complete=True` turn is seen but first triggers a spurious extra spoken response,
            # and `FunctionResponse.parts` — the true analog of the classic Gemini 3 multimodal
            # function-response path — doesn't serialize in the SDK's live path yet. Tracked in
            # https://github.com/pydantic/pydantic-ai/issues/7362.
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
                        raise UserError(
                            f'{self._provider_label} tool results are JSON-only, so `{type(item).__name__}` '
                            'content attached to a tool return cannot be delivered. Return text instead, or '
                            'use a realtime provider that supports tool-result media. '
                            'See https://github.com/pydantic/pydantic-ai/issues/7362.'
                        )
                    else:
                        assert_never(item)
                output = '\n\n'.join(part for part in (output, *text_content) if part)
            await self._session.send_tool_response(
                function_responses=genai_types.FunctionResponse(
                    id=gemini_id,
                    name=name,
                    response={'output': output},
                    # `INTERRUPT`, not `WHEN_IDLE`: a non-blocking model keeps talking while the
                    # tool runs, and `WHEN_IDLE` holds the result until it stops — by which point it
                    # has usually answered from its own knowledge, so the tool's answer contradicts
                    # what was already said. (Recorded live: a tool returning "foggy and 12 degrees"
                    # while the model said "15 degrees with clouds".) A model calls a tool because it
                    # needs the result, so cut in with it.
                    scheduling=genai_types.FunctionResponseScheduling.INTERRUPT
                    if self._async_tool_calls_enabled
                    else None,
                )
            )
        else:
            raise UserError(f'{self._provider_label} does not support {type(content).__name__} input.')

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
            except self.transport_errors as e:
                if self._dial is None or self._reconnect is None:
                    # No reconnect policy: a dropped connection is fatal. Surface it as a
                    # non-recoverable error and end the stream cleanly, rather than returning silently
                    # (mirroring the OpenAI provider), so callers don't treat a truncated turn as complete.
                    yield RealtimeSessionErrorEvent(
                        message=f'{self._provider_label} connection closed: {e}', recoverable=False
                    )
                    return
                state_restored = self._resumption_handle is not None
                if await self._try_reconnect():
                    if not state_restored and self._tool_calls:
                        # Without a resumption handle the re-dialed session is a fresh one that never
                        # issued these calls, so a tool task still running for the lost session would
                        # send its result back against an id Gemini doesn't know. Abandon them the way
                        # Gemini's own `tool_call_cancellation` does: the tasks are cancelled and each
                        # call still gets a matching return in history.
                        yield ToolCallCancelled(tool_call_ids=list(self._tool_calls))
                        self._tool_calls.clear()
                    if self._turn_open:
                        # The dropped connection was mid-turn. Gemini never continues an in-flight
                        # generation on the re-dialed connection (resumption restores conversation
                        # state only), so its `turn_complete` will never arrive — without a synthetic
                        # boundary the session would keep the partial response open forever, never
                        # ending the turn or delivering messages queued behind it.
                        self._turn_open = False
                        self._turn_interrupted = False
                        self._native_part_index = 0
                        yield ResponseDone(interrupted=True)
                    yield RealtimeSessionReconnectEvent(state_restored=state_restored)
                    continue
                yield RealtimeSessionErrorEvent(
                    message=f'{self._provider_label} connection closed; reconnect failed: {e}', recoverable=False
                )
                return
            # `receive()` returned normally → the turn ended; loop for the next one.

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff, resuming from the latest handle; return whether it worked."""
        assert self._dial is not None and self._reconnect is not None
        if not await reconnect_with_backoff(
            self._reconnect, self._attempt_reconnect, reconnects_used=self._reconnects_used
        ):
            return False
        self._reconnects_used += 1
        return True

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
                    events.append(OutputTranscript(text=part.text, is_final=False, output_text=True))
        if content.input_transcription is not None and content.input_transcription.text:
            events.append(
                InputTranscript(
                    text=content.input_transcription.text, is_final=bool(content.input_transcription.finished)
                )
            )
        if content.output_transcription is not None and content.output_transcription.text:
            events.append(
                OutputTranscript(
                    text=content.output_transcription.text, is_final=bool(content.output_transcription.finished)
                )
            )
        if content.interrupted:
            self._turn_interrupted = True
            events.append(RealtimeResponseInterruptedEvent())
        native_tool_parts += _map_grounding_parts(content, self._provider_name)
        for part in native_tool_parts:
            index = self._native_part_index
            self._native_part_index += 1
            events.extend((PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)))
        # Only response output opens a turn — input transcripts stream between turns too, and a turn
        # "opened" by one would close as an empty interrupted response if the connection then dropped.
        if native_tool_parts or any(isinstance(event, (AudioDelta, OutputTranscript)) for event in events):
            self._turn_open = True
        # `turn_complete` is emitted by `_map_message` *after* the message's `usage_metadata`, not here:
        # Gemini packs `turnComplete` and `usageMetadata` into the same message, and the session
        # finalizes the response's usage on `ResponseDone`, so the usage must be accounted first
        # (matching OpenAI's codec, which emits usage before the turn boundary).
        return events

    def _map_message(self, message: genai_types.LiveServerMessage) -> list[RealtimeCodecEvent]:
        events: list[RealtimeCodecEvent] = []
        if message.server_content is not None:
            events.extend(self._map_server_content(message.server_content))
        if message.tool_call is not None:
            for call in message.tool_call.function_calls or []:
                name = call.name or ''
                # Gemini usually assigns an id, but fall back to the same synthetic id a standard
                # request builds for an id-less call, so parallel calls don't collide on one key and
                # the `pyd_ai_` prefix still marks the id as ours after a handoff to a standard run.
                # The provider's own id (`None` here) is what goes back on the wire — echoing one it
                # never issued is what "Gemini rejects unknown ids" is about.
                call_id = call.id or generate_tool_call_id()
                self._tool_calls[call_id] = (name, call.id)
                # A tool call opens the turn like audio output does: the session holds a partial
                # response for it, so a drop before `turn_complete` needs the same synthetic boundary.
                self._turn_open = True
                events.append(ToolCall(tool_call_id=call_id, tool_name=name, args=to_json(call.args or {}).decode()))
        if message.tool_call_cancellation is not None and (cancelled_ids := message.tool_call_cancellation.ids):
            # The cancellation carries Gemini's own call ids, which match the `tool_call_id`s emitted
            # above whenever Gemini assigned them (id-less calls can't be cancelled by id anyway).
            # A cancelled call never sends a result, so nothing else will ever pop it: forget it here
            # or every barge-in leaks an entry for the life of the connection.
            for call_id in cancelled_ids:
                self._tool_calls.pop(call_id, None)
            events.append(ToolCallCancelled(tool_call_ids=list(cancelled_ids)))
        if message.usage_metadata is not None:
            events.append(
                SessionUsage(
                    usage=_map_usage(
                        message.usage_metadata,
                        provider_name=self._provider_name,
                        provider_url=self._provider_url,
                    )
                )
            )
        # Emit the turn boundary last — after this message's usage — so the session folds the turn's
        # tokens into the finalized `ModelResponse` / `chat` span before `ResponseDone` closes it.
        if message.server_content is not None and message.server_content.turn_complete:
            interrupted = self._turn_interrupted
            events.append(ResponseDone(interrupted=interrupted))
            self._turn_interrupted = False
            self._turn_open = False
            self._native_part_index = 0
        # Track the resumption handle (internal state, not an event) so a reconnect can resume state.
        update = message.session_resumption_update
        if update is not None and update.new_handle:
            self._resumption_handle = update.new_handle
        return events
