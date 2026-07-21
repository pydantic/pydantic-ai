"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` package, available via the `realtime` optional group:

    pip install "pydantic-ai-slim[realtime]"
"""

from __future__ import annotations as _annotations

import base64
import json
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the OpenAI Realtime model, '
        'you can use the `realtime` and `openai` optional groups - `pip install "pydantic-ai-slim[realtime,openai]"`'
    ) from _import_error

if TYPE_CHECKING:
    # Only needed for typing: the provider supplies the concrete client at runtime, so importing the
    # protocol helpers below (e.g. from the xAI realtime provider) doesn't require the `openai` package.
    from openai import AsyncOpenAI
    from openai.types.realtime.realtime_truncation_param import RealtimeTruncationParam

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..profiles.openai import OPENAI_REASONING_EFFORT_MAP
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._base import (
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
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
    ToolResult,
    TruncateOutput,
    inject_trace_context,
    reconnect_with_backoff,
)
from ._openai_protocol import (
    AUDIO_DELTA_TYPES,
    SemanticVAD,
    ServerVAD,
    expect_event,
    loads_obj,
    map_event,
    obj,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)

# `input_transcription_model='auto'` resolves to this — OpenAI's recommended realtime transcription model
# ("For the lowest-latency streaming transcription path, use gpt-realtime-whisper"; it's natively streaming
# and designed for realtime sessions, unlike the legacy `whisper-1`). Kept behind the `'auto'` sentinel
# (see `resolve_transcription_model`) so it can be bumped without changing the behavior of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'gpt-realtime-whisper'

__all__ = (
    'OpenAIRealtimeModel',
    'OpenAIRealtimeModelSettings',
    'OpenAIRealtimeConnection',
    'ServerVAD',
    'SemanticVAD',
    'map_event',
)


class OpenAIRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to OpenAI realtime models."""

    openai_input_noise_reduction: Literal['near_field', 'far_field']
    """Noise reduction tuned for `near_field` (headset) or `far_field` (laptop/conference) microphones.

    Absent disables it.
    """
    openai_output_speed: float
    """Playback speed multiplier for generated audio (0.25-1.5)."""
    openai_turn_detection: ServerVAD | SemanticVAD
    """OpenAI-specific server or semantic VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """
    openai_truncation: RealtimeTruncationParam
    """How the session truncates conversation context once it exceeds the model's window.

    `'auto'` (the server default) drops the oldest turns; `'disabled'` keeps everything (and errors
    when the window is full); a `retention_ratio` truncation (`{'type': 'retention_ratio',
    'retention_ratio': 0.8}`) keeps a fixed fraction, holding the prompt-cached prefix stable across
    turns (cached audio is far cheaper). This is the OpenAI SDK's `truncation` shape, forwarded as-is.
    """


def _int(value: Any) -> int:
    """Return `value` if it is an int (but not a bool), otherwise `0`."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _map_usage(response: dict[str, Any]) -> RequestUsage | None:
    """Map a `response.done` `usage` payload to a [`RequestUsage`][pydantic_ai.usage.RequestUsage]."""
    usage = obj(response.get('usage'))
    if not usage:
        return None
    inp = obj(usage.get('input_token_details'))
    out = obj(usage.get('output_token_details'))
    cached = obj(inp.get('cached_tokens_details'))
    details: dict[str, int] = {}
    for key, raw in (
        ('input_text_tokens', inp.get('text_tokens')),
        ('input_image_tokens', inp.get('image_tokens')),
        ('output_text_tokens', out.get('text_tokens')),
    ):
        if isinstance(raw, int) and not isinstance(raw, bool):
            details[key] = raw
    # xAI-specific usage (absent on OpenAI): the `grok_tokens` buckets, and second-based audio billing —
    # xAI bills Grok Voice by audio second, so `billable_audio_seconds` is the authoritative cost and is
    # not reconstructable from token counts. Included only when non-zero, so OpenAI's `details` is unchanged.
    for key, raw in (
        ('input_grok_tokens', inp.get('grok_tokens')),
        ('output_grok_tokens', out.get('grok_tokens')),
        ('billable_audio_seconds', usage.get('billable_audio_seconds')),
    ):
        if isinstance(raw, int) and not isinstance(raw, bool) and raw:
            details[key] = raw
    return RequestUsage(
        input_tokens=_int(usage.get('input_tokens')),
        output_tokens=_int(usage.get('output_tokens')),
        input_audio_tokens=_int(inp.get('audio_tokens')),
        cache_read_tokens=_int(inp.get('cached_tokens')),
        cache_audio_read_tokens=_int(cached.get('audio_tokens')),
        output_audio_tokens=_int(out.get('audio_tokens')),
        details=details,
    )


class OpenAIRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI Realtime API."""

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
    ) -> None:
        self._ws = ws
        self._model_name = model_name
        # `dial` re-establishes a fully configured connection; with a `reconnect` policy it is used to
        # recover from a dropped WebSocket.
        self._dial = dial
        self._reconnect = reconnect
        self._input_transcription_enabled = input_transcription_enabled
        # The Realtime API rejects `response.create` while a response is already being generated.
        # We track that window and defer requests (e.g. a background tool result that lands while the
        # model is mid-answer) until the active response finishes, so the model still announces it.
        self._response_active = False
        self._active_response_id: str | None = None
        self._pending_response = False
        # The current output audio item, tracked from output-audio deltas so a `TruncateOutput` can
        # name it. These are mutated by `__aiter__` and read by `send` from a separate task; under a
        # single cooperative event loop the plain reads/writes are safe and eventually consistent.
        self._current_item_id: str | None = None
        self._current_content_index = 0

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `AudioInput` (PCM16, 24kHz, mono), `TextInput`, `ImageInput`, `ToolResult`, and the
        control verbs `CommitAudio`, `ClearAudio`, `CreateResponse`, and `CancelResponse`.
        """
        if isinstance(content, AudioInput):
            await self._send_event(
                {
                    'type': 'input_audio_buffer.append',
                    'audio': base64.b64encode(content.data).decode('ascii'),
                }
            )
        elif isinstance(content, TextInput):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': content.text}],
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ToolResult):
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'function_call_output',
                        'call_id': content.tool_call_id,
                        'output': content.output,
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ImageInput):
            # An image is added as conversation context (like a video frame), not a turn of its own,
            # so it doesn't trigger a response — drive that with audio (VAD) or `CreateResponse`.
            data_uri = f'data:{content.mime_type};base64,{base64.b64encode(content.data).decode("ascii")}'
            await self._send_event(
                {
                    'type': 'conversation.item.create',
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': data_uri}],
                    },
                }
            )
        elif isinstance(content, CommitAudio):
            await self._send_event({'type': 'input_audio_buffer.commit'})
        elif isinstance(content, ClearAudio):
            await self._send_event({'type': 'input_audio_buffer.clear'})
        elif isinstance(content, CreateResponse):
            await self._request_response()
        elif isinstance(content, CancelResponse):
            # Only cancel when a response is actually active: with server VAD the provider may have
            # already cancelled on the user's barge-in, and a redundant cancel raises a session error.
            if self._response_active:
                await self._send_event({'type': 'response.cancel'})
            self._response_active = False
            self._active_response_id = None
            self._pending_response = False
        elif isinstance(content, TruncateOutput):
            # No current output item (e.g. the model wasn't speaking) → nothing to truncate.
            if self._current_item_id is not None:
                await self._send_event(
                    {
                        'type': 'conversation.item.truncate',
                        'item_id': self._current_item_id,
                        'content_index': self._current_content_index,
                        'audio_end_ms': content.audio_end_ms,
                    }
                )
        else:
            raise NotImplementedError(f'OpenAI Realtime does not support {type(content).__name__} input')

    async def _request_response(self) -> None:
        """Ask the model to respond now, or defer until the active response completes."""
        if self._response_active:
            self._pending_response = True
        else:
            self._response_active = True
            self._active_response_id = None
            await self._send_event({'type': 'response.create'})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(json.dumps(event))

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        """Map a raw provider frame to a codec event.

        A hook so protocol clones (e.g. the xAI Grok Voice provider) can reuse the whole connection while
        overriding only how frames map to events. Defaults to the OpenAI [`map_event`][pydantic_ai.realtime.openai.map_event].
        """
        return map_event(data)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            try:
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        events = await self._decode_frame(raw)
                    except ValueError as e:
                        # A malformed frame (bad JSON or audio payload) shouldn't tear down the whole
                        # session; surface it as a recoverable error and keep reading.
                        yield SessionErrorEvent(message=f'Failed to parse OpenAI realtime event: {e}', recoverable=True)
                        continue
                    for event in events:
                        yield event
                return  # the upstream iterator ended without dropping
            except websockets.ConnectionClosed as e:
                if self._reconnect is None or self._dial is None:
                    # No reconnect policy: a dropped connection is fatal. Surface it as a
                    # non-recoverable error and end the stream cleanly, rather than raising.
                    yield SessionErrorEvent(message=f'OpenAI realtime connection closed: {e}', recoverable=False)
                    return
                if await self._try_reconnect():
                    yield ReconnectedEvent()
                    continue
                yield SessionErrorEvent(
                    message=f'OpenAI realtime connection closed; reconnect failed: {e}', recoverable=False
                )
                return

    async def _decode_frame(self, raw: str) -> list[RealtimeCodecEvent]:
        """Parse one text frame into events, updating tracked response state.

        Raises `ValueError` (incl. `json.JSONDecodeError` / `binascii.Error`) on a malformed payload.
        """
        data = loads_obj(raw)
        event_type = data.get('type')
        events: list[RealtimeCodecEvent] = []
        if event_type == 'response.created':
            self._response_active = True
            response_id = obj(data.get('response')).get('id')
            self._active_response_id = response_id if isinstance(response_id, str) else None
        elif event_type in AUDIO_DELTA_TYPES:
            # Track the speaking item so a later `TruncateOutput` can name it.
            item_id = data.get('item_id')
            if isinstance(item_id, str):
                self._current_item_id = item_id
                content_index = data.get('content_index')
                self._current_content_index = content_index if isinstance(content_index, int) else 0
        elif event_type == 'response.done':
            response = obj(data.get('response'))
            response_id = response.get('id')
            # OpenAI response events always carry an ID. Keep the ID-less fallback for compatible
            # protocol implementations and defensive unit inputs that predate response tracking.
            matches_active_response = not isinstance(response_id, str) or (
                self._response_active and response_id == self._active_response_id
            )
            if matches_active_response:
                self._response_active = False
                self._active_response_id = None
                self._current_item_id = None
            # Emit usage for every response (including intermediate function-call-only ones)
            # so the session accounts for all tokens. Only the active response may replay a pending
            # request; a late completion for a superseded response must not change current state.
            # OpenAI nests usage under `response.usage`; xAI Grok Voice reports the same shape at the
            # top level of the `response.done` frame (its `response.usage` is empty), so fall back to it.
            usage = _map_usage(response) or _map_usage(data)
            if usage is not None:
                events.append(SessionUsageEvent(usage=usage))
            if matches_active_response and self._pending_response:
                self._pending_response = False
                # A cancelled response means the user barged in: a new turn is starting, so
                # don't replay the deferred response over it.
                if response.get('status') != 'cancelled':
                    self._response_active = True
                    self._active_response_id = None
                    await self._send_event({'type': 'response.create'})
        if (event := self._map_event(data)) is not None:
            events.append(event)
        return events

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff; return whether a new connection was established."""
        assert self._reconnect is not None and self._dial is not None
        return await reconnect_with_backoff(self._reconnect, self._attempt_reconnect)

    async def _attempt_reconnect(self) -> bool:
        assert self._dial is not None
        try:
            self._ws = await self._dial()
        except (websockets.WebSocketException, OSError, TimeoutError):
            # Expected dial/handshake failures: protocol/connection errors, network failures (DNS,
            # refused, reset), and the handshake timeout. A retry may still succeed. Anything else is a
            # bug in `dial()` and propagates rather than masquerading as a failed reconnect.
            return False
        self._response_active = False
        self._active_response_id = None
        self._pending_response = False
        self._current_item_id = None
        return True


@dataclass
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Authentication, base URL, and the HTTP/WebSocket client come from a
    [`Provider`][pydantic_ai.providers.Provider], mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
    Pass `provider='openai'` (the default) to read `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the
    environment, or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] instance for a
    custom key, base URL, or `httpx` client. The realtime WebSocket URL is derived from the provider's
    base URL (e.g. `https://api.openai.com/v1/` → `wss://api.openai.com/v1/realtime`), so
    OpenAI-compatible endpoints that expose a realtime API work too.

    Args:
        model: The model name, e.g. `gpt-realtime` or `gpt-realtime-2.1-mini`.
        provider: The provider to use for authentication and the base URL. Defaults to `'openai'`.
            Azure OpenAI is not supported (its realtime endpoint uses a different URL and auth scheme).
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently
            recover from a dropped connection. `None` (the default) surfaces a drop as a non-recoverable
            `SessionErrorEvent` instead.
    """

    model: str = 'gpt-realtime'
    provider: InitVar[Provider[AsyncOpenAI] | str] = 'openai'
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
    reconnect: ReconnectPolicy | None = None
    _provider: Provider[AsyncOpenAI] = field(init=False, repr=False)

    def __post_init__(self, provider: Provider[AsyncOpenAI] | str) -> None:
        if isinstance(provider, str):
            provider = cast('Provider[AsyncOpenAI]', infer_provider(provider))
        if provider.name == 'azure':
            raise UserError(
                'Azure OpenAI is not supported for realtime sessions: its realtime endpoint uses a '
                'different URL and authentication scheme. Use the `openai` provider instead.'
            )
        self._provider = provider

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying [`AsyncOpenAI`](https://github.com/openai/openai-python) client from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if 'openai_turn_detection' in model_settings:
            turn_detection = model_settings['openai_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection = ServerVAD()
        # `turn_detection` is always set: a dict enables VAD, `None` (explicit null) disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': 24000},
            'turn_detection': turn_detection_config(turn_detection),
        }
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        if (noise_reduction := model_settings.get('openai_input_noise_reduction')) is not None:
            audio_input['noise_reduction'] = {'type': noise_reduction}
        audio_output: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        if voice := model_settings.get('voice'):
            audio_output['voice'] = voice
        if (output_speed := model_settings.get('openai_output_speed')) is not None:
            audio_output['speed'] = output_speed
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': [model_settings.get('output_modality', 'audio')],
            'audio': {'input': audio_input, 'output': audio_output},
        }
        if tools:
            config['tools'] = [tool_def_to_openai(t) for t in tools]
        # Note: GA realtime sessions have no `temperature` field, so it is intentionally not forwarded.
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if (tool_choice := tool_choice_config(model_settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        if (truncation := model_settings.get('openai_truncation')) is not None:
            # Already the OpenAI `truncation` wire shape (`'auto'`/`'disabled'`/retention-ratio dict).
            config['truncation'] = truncation
        if (thinking := model_settings.get('thinking')) is not None:
            if self.profile.get('supports_thinking', False):
                # `False` maps to `'none'`, which the realtime `reasoning.effort` doesn't accept — omit
                # it so a reasoning model falls back to its default rather than erroring.
                if (effort := OPENAI_REASONING_EFFORT_MAP[thinking]) != 'none':
                    config['reasoning'] = {'effort': effort}
            else:
                warnings.warn(
                    f'The {self.model!r} realtime model does not support the `thinking` setting '
                    '(only the `gpt-realtime-2*` reasoning models do); ignoring it.',
                    UserWarning,
                )
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        url = f'{realtime_websocket_url(self._provider.base_url)}?model={self.model}'
        # `AsyncOpenAI` accepts an async `api_key` provider, in which case `client.api_key` is empty
        # until resolved. The raw WebSocket handshake bypasses the SDK's request path (which resolves
        # it per request), so resolve it here via the SDK's own refresh — a no-op returning the static
        # key when no provider is configured, so the handshake stays byte-identical in that case.
        api_key = await self._provider.client._refresh_api_key()  # pyright: ignore[reportPrivateUsage]
        headers = {'Authorization': f'Bearer {api_key}'}
        # Propagate trace context over the handshake so a proxy (e.g. the Pydantic AI Gateway) can nest
        # its realtime spans under this session's trace; the raw WebSocket bypasses the provider's
        # `httpx` client, which would otherwise inject it.
        inject_trace_context(headers)
        settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages) or ''
        session_config = self._session_config(instructions, model_request_parameters.function_tools, settings)
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        # `dial` opens and configures a fresh connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake) before opening the next, so sockets
        # don't accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        # The model the server reports actually serving, from the `session.created` handshake; it can
        # differ from the requested id (see `RealtimeConnection.model_name`).
        server_model: str | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, server_model
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            created = await expect_event(ws, 'session.created', timeout=handshake_timeout)
            model = obj(created.get('session')).get('model')
            if isinstance(model, str) and model:
                server_model = model
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            await expect_event(ws, 'session.updated', timeout=handshake_timeout)
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the initial handshake. Reconnects deliberately don't
            # re-seed: server state is lost on drop and a `ReconnectedEvent` starts a fresh turn.
            for item in await seed_items(messages, profile=self.profile, provider_name=self.system):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield OpenAIRealtimeConnection(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
            )
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)
