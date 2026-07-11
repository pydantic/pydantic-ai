"""xAI Grok Voice realtime API provider for speech-to-speech sessions.

Connects to `wss://api.x.ai/v1/realtime` over a WebSocket. xAI's realtime API is a deliberate clone of
the OpenAI Realtime protocol, so this provider reuses the OpenAI codec from
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] — event mapping, session seeding, tool
conversion, server-VAD config, and the WebSocket connection itself — and diverges only where xAI does:

- the `session.update` shape (`voice`/`turn_detection` sit at the session top level, not nested under
  `audio` as on OpenAI's GA surface);
- input audio transcription, delivered as cumulative
  `conversation.item.input_audio_transcription.updated` snapshots plus a final `.completed`, rather
  than OpenAI's incremental `.delta` events (see [`map_event`][pydantic_ai.realtime.xai.map_event]);
- no output truncation (`conversation.item.truncate` is unsupported), so
  [`RealtimeModelProfile.supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation]
  is `False` while cancellation-based interruption still works.

Requires the `websockets` package (the `realtime` optional group) and `xai-sdk` (the `xai` group, for
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider]):

    pip install "pydantic-ai-slim[realtime,xai]"
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any, cast

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the xAI Grok Voice realtime model, '
        'you can use the `realtime` and `xai` optional groups - `pip install "pydantic-ai-slim[realtime,xai]"`'
    ) from _import_error

from ..exceptions import UserError
from ..messages import ModelMessage
from ..native_tools import AbstractNativeTool
from ..providers import infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ._base import (
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelProfile,
    ReconnectPolicy,
    inject_trace_context,
)
from ._openai_protocol import (
    expect_event,
    map_event as _map_openai_event,
    realtime_websocket_url,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from .openai import OpenAIRealtimeConnection, ServerVAD

if TYPE_CHECKING:
    from ..providers.xai import XaiProvider


def map_event(data: dict[str, Any]) -> RealtimeEvent | None:
    """Map a raw xAI Grok Voice realtime event to a [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent].

    xAI clones the OpenAI Realtime protocol, so all but one event map identically via the OpenAI codec.
    The exception is input audio transcription: xAI emits cumulative
    `conversation.item.input_audio_transcription.updated` snapshots (which may retroactively *correct*
    earlier text) plus a final `.completed`, rather than OpenAI's incremental `.delta`. Only the final
    `.completed` is surfaced — its event name is identical to OpenAI's, so it maps through the shared
    codec — because the session's transcript accumulator reconciles incremental and prefix-extending
    updates but can't undo a retroactive correction from a cumulative snapshot; dropping the `.updated`
    partials keeps the finalized user transcript correct at the cost of live partial input transcripts.
    """
    if data.get('type') == 'conversation.item.input_audio_transcription.updated':
        return None
    return _map_openai_event(data)


class XaiRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the xAI Grok Voice realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] wholesale —
    the wire protocol is identical — overriding only event mapping to handle xAI's cumulative
    input-transcription events (see [`map_event`][pydantic_ai.realtime.xai.map_event]).
    """

    def _map_event(self, data: dict[str, Any]) -> RealtimeEvent | None:
        return map_event(data)


@dataclass
class XaiRealtimeModel(RealtimeModel):
    """xAI Grok Voice realtime API model.

    Authentication and the base URL come from an [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider],
    mirroring [`XaiModel`][pydantic_ai.models.xai.XaiModel]. Pass `provider='xai'` (the default, reads
    `XAI_API_KEY`) or an [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] instance for a custom key.
    The realtime WebSocket URL is derived from the provider's base URL
    (`https://api.x.ai/v1` → `wss://api.x.ai/v1/realtime`).

    Args:
        model: The model name, e.g. `grok-voice-latest` (the default, tracks the current model) or a
            pinned version like `grok-voice-think-fast-1.0`. The `model` query parameter is required by
            the server, which otherwise falls back to a default silently.
        provider: The provider to use for authentication and the base URL. Defaults to `'xai'`.
        voice: Voice for audio output — one of `eve` (default), `ara`, `rex`, `sal`, `leo`, or a custom
            voice ID. `None` uses the server default (`eve`).
        input_transcription_model: Model used to transcribe the user's audio input, `grok-transcribe`
            by default so the user's turns are captured into history (pass `None` to disable). Transcripts
            are surfaced at the end of each user turn (see [`map_event`][pydantic_ai.realtime.xai.map_event]).
        handshake_timeout: Seconds to wait for each handshake event (`session.created` / `session.updated`)
            before failing, so `connect()` doesn't hang if the server never responds.
        turn_detection: How the server decides when the user's turn ends. A
            [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] (the default) configures automatic
            detection; `None` disables it for manual turn-taking (push-to-talk), where you drive the turn
            with `commit_audio()` + `create_response()`.
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently
            recover from a dropped connection. `None` (the default) surfaces a drop as a non-recoverable
            `SessionErrorEvent` instead.
    """

    model: str = 'grok-voice-latest'
    provider: InitVar[XaiProvider | str] = 'xai'
    voice: str | None = None
    input_transcription_model: str | None = 'grok-transcribe'
    handshake_timeout: float = 30.0
    turn_detection: ServerVAD | None = field(default_factory=ServerVAD)
    reconnect: ReconnectPolicy | None = None
    _provider: XaiProvider = field(init=False, repr=False)
    _api_key: str = field(init=False, repr=False)

    def __post_init__(self, provider: XaiProvider | str) -> None:
        if isinstance(provider, str):
            provider = cast('XaiProvider', infer_provider(provider))
        api_key = provider.api_key
        if not api_key:
            raise UserError(
                'The xAI realtime provider needs an API key for the WebSocket connection, but the '
                '`XaiProvider` was built from a pre-configured `xai_client` whose key is not exposed. '
                'Pass `provider=XaiProvider(api_key=...)` (or set `XAI_API_KEY`) instead.'
            )
        self._provider = provider
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def profile(self) -> RealtimeModelProfile:
        # `supports_output_truncation=False`: xAI has no `conversation.item.truncate`, so barge-in cancels the
        # response (`supports_interruption`) but can't sync the transcript to a mid-audio playback point.
        return RealtimeModelProfile(
            supports_image_input=False,
            supports_manual_turn_control=True,
            supports_interruption=True,
            supports_output_truncation=False,
            supports_session_seeding=True,
            supported_native_tools=frozenset(),
        )

    def _session_config(
        self, instructions: str, tools: list[ToolDefinition] | None, model_settings: ModelSettings | None
    ) -> dict[str, Any]:
        # xAI puts `voice` and `turn_detection` at the session top level, unlike OpenAI's GA surface which
        # nests them under `audio`. `turn_detection` is always set: a dict enables VAD, `None` disables it.
        audio_input: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        if self.input_transcription_model:
            audio_input['transcription'] = {'model': self.input_transcription_model}
        config: dict[str, Any] = {
            'instructions': instructions,
            'turn_detection': turn_detection_config(self.turn_detection),
            'audio': {'input': audio_input, 'output': {'format': {'type': 'audio/pcm', 'rate': 24000}}},
        }
        if self.voice:
            config['voice'] = self.voice
        if tools:
            config['tools'] = [tool_def_to_openai(t) for t in tools]
        if model_settings:
            if (max_tokens := model_settings.get('max_tokens')) is not None:
                config['max_output_tokens'] = max_tokens
            if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
                config['parallel_tool_calls'] = parallel_tool_calls
            if (tool_choice := tool_choice_config(model_settings.get('tool_choice'))) is not None:
                config['tool_choice'] = tool_choice
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: ModelSettings | None = None,
        messages: Sequence[ModelMessage] | None = None,
    ) -> AsyncGenerator[XaiRealtimeConnection]:
        # The `model` query parameter is required: without it the server silently falls back to a default.
        url = f'{realtime_websocket_url(self._provider.base_url)}?model={self.model}'
        headers = {'Authorization': f'Bearer {self._api_key}'}
        # Propagate trace context over the handshake (see the OpenAI provider for the rationale).
        inject_trace_context(headers)
        session_config = self._session_config(instructions, tools, model_settings)

        # `dial` opens and configures a fresh connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake) before opening the next, so sockets don't
        # accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        async def dial() -> ClientConnection:
            nonlocal cm
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            await expect_event(ws, 'session.created', timeout=self.handshake_timeout)
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))
            await expect_event(ws, 'session.updated', timeout=self.handshake_timeout)
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the initial handshake. Reconnects deliberately don't
            # re-seed: server state is lost on drop and a `ReconnectedEvent` starts a fresh turn.
            for item in seed_items(messages or ()):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            yield XaiRealtimeConnection(ws, dial=dial, reconnect=self.reconnect)
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)
