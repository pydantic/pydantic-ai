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
- native conversation resumption when a reconnect policy is configured: the provider-assigned
  `conversation.id` is reused and its replay burst is suppressed from local history;
- no output truncation (`conversation.item.truncate` is unsupported), so
  [`RealtimeModelProfile.supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation]
  is `False` while cancellation-based interruption still works.

Requires the `websockets` package (the `realtime` optional group) and `xai-sdk` (the `xai` group, for
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider]):

    pip install "pydantic-ai-slim[realtime,xai]"
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

try:
    import websockets
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the xAI Grok Voice realtime model, '
        'you can use the `realtime` and `xai` optional groups - `pip install "pydantic-ai-slim[realtime,xai]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..providers import infer_provider
from ..tools import ToolDefinition
from ._base import (
    ConversationCreated,
    ConversationItemCreated,
    RealtimeCodecEvent,
    RealtimeModel,
    RealtimeModelSettings,
    ReconnectedEvent,
    ReconnectPolicy,
    ToolCall,
    inject_trace_context,
)
from ._openai_protocol import (
    expect_event,
    map_conversation_event,
    map_event as _map_openai_event,
    obj,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from .openai import OpenAIRealtimeConnection, ServerVAD

if TYPE_CHECKING:
    from ..providers.xai import XaiProvider

# `input_transcription_model='auto'` resolves to this — xAI's realtime transcription model. Kept behind
# the `'auto'` sentinel (see `resolve_transcription_model`) so it can change without altering the behavior
# of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'grok-transcribe'


class XaiRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to xAI realtime models.

    xAI ignores the inherited `output_modality` and `thinking` settings and always produces audio
    output.
    """

    xai_turn_detection: ServerVAD
    """xAI-specific server-VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw xAI Grok Voice realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.RealtimeCodecEvent].

    xAI clones the OpenAI Realtime protocol, so most events map identically via the OpenAI codec.
    The first exception is input audio transcription: xAI emits cumulative
    `conversation.item.input_audio_transcription.updated` snapshots (which may retroactively *correct*
    earlier text) plus a final `.completed`, rather than OpenAI's incremental `.delta`. Only the final
    `.completed` is surfaced — its event name is identical to OpenAI's, so it maps through the shared
    codec — because the session's transcript accumulator reconciles incremental and prefix-extending
    updates but can't undo a retroactive correction from a cumulative snapshot; dropping the `.updated`
    partials keeps the finalized user transcript correct at the cost of live partial input transcripts.
    The other exception is xAI's conversation lifecycle events, which are surfaced as codec control
    events so the connection can capture `conversation.id` and the session can suppress resume replay.
    """
    if data.get('type') == 'conversation.item.input_audio_transcription.updated':
        return None
    if data.get('type') in ('conversation.created', 'conversation.item.added', 'conversation.item.created'):
        return map_conversation_event(data)
    event = _map_openai_event(data)
    if isinstance(event, ToolCall):
        item_id = data.get('item_id')
        if isinstance(item_id, str) and item_id:
            event = replace(event, item_id=item_id)
    return event


class XaiRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the xAI Grok Voice realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the
    shared wire protocol, while mapping xAI's cumulative input transcription and conversation lifecycle
    events and emitting the resumption replay controls captured during reconnect handshakes.
    """

    _provider_name = 'xai'
    _supports_tool_result_images = False

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
        conversation_id: str | None = None,
        replayed_items: list[ConversationItemCreated] | None = None,
    ) -> None:
        super().__init__(
            ws,
            dial=dial,
            reconnect=reconnect,
            input_transcription_enabled=input_transcription_enabled,
            model_name=model_name,
        )
        self._conversation_id = conversation_id
        self._replayed_items = replayed_items if replayed_items is not None else []

    @property
    def conversation_id(self) -> str | None:
        """The xAI conversation ID used for native session resumption."""
        return self._conversation_id

    @conversation_id.setter
    def conversation_id(self, conversation_id: str | None) -> None:
        self._conversation_id = conversation_id

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return map_event(data)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        async for event in super().__aiter__():
            yield event
            if isinstance(event, ReconnectedEvent):
                replayed_items = self._replayed_items[:]
                self._replayed_items.clear()
                for replayed_item in replayed_items:
                    yield replayed_item


@dataclass
class XaiRealtimeModel(RealtimeModel):
    """xAI Grok Voice realtime API model.

    Pass `provider='xai'` (the default, which reads `XAI_API_KEY`) or an
    [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] constructed with `api_key=`. A custom
    `api_host` is not supported, and a provider constructed only with `xai_client=` cannot be used
    because the WebSocket connection needs access to the API key. The realtime WebSocket URL is
    `wss://api.x.ai/v1/realtime`.

    Args:
        model: The model name, e.g. `grok-voice-latest` (the default, tracks the current model) or a
            pinned version like `grok-voice-think-fast-1.0`. The `model` query parameter is required by
            the server, which otherwise falls back to a default silently.
        provider: The provider to use for authentication and the base URL. Defaults to `'xai'`.
        reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently
            recover from a dropped connection. Setting a policy enables xAI's native session resumption;
            prior turns are restored when reconnecting within xAI's 30-minute inactivity window. With no
            policy, the low-level connection reports a non-recoverable session error; `RealtimeSession` raises
            [`RealtimeError`][pydantic_ai.realtime.RealtimeError] from iteration.
    """

    model: str = 'grok-voice-latest'
    provider: InitVar[XaiProvider | str] = 'xai'
    settings: RealtimeModelSettings | None = field(default=None, kw_only=True)
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
        if provider.api_host is not None:
            # The realtime WebSocket URL is derived from `base_url` (the canonical xAI host), not the
            # gRPC channel target set by `api_host`. Rather than silently connect to the canonical host
            # with the key while the user expects their custom host, fail loudly.
            raise UserError(
                'The xAI realtime provider does not support a custom `api_host`: the realtime WebSocket '
                'connects to the canonical xAI realtime endpoint, not the gRPC channel target that '
                '`api_host` sets. Remove `api_host` from the `XaiProvider` to use realtime.'
            )
        self._provider = provider
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def system(self) -> str:
        return 'xai'

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: XaiRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        # xAI puts `voice` and `turn_detection` at the session top level, unlike OpenAI's GA surface which
        # nests them under `audio`. `turn_detection` is always set: a dict enables VAD, `None` disables it.
        audio_input: dict[str, Any] = {'format': {'type': 'audio/pcm', 'rate': 24000}}
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        if 'xai_turn_detection' in model_settings:
            turn_detection = model_settings['xai_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection = ServerVAD()
        config: dict[str, Any] = {
            'instructions': instructions,
            'turn_detection': turn_detection_config(turn_detection),
            'audio': {'input': audio_input, 'output': {'format': {'type': 'audio/pcm', 'rate': 24000}}},
        }
        if voice := model_settings.get('voice'):
            config['voice'] = voice
        if tools:
            config['tools'] = [tool_def_to_openai(t) for t in tools]
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if (tool_choice := tool_choice_config(model_settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        if self.reconnect is not None:
            config['resumption'] = {'enabled': True}
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[XaiRealtimeConnection]:
        # The `model` query parameter is required: without it the server silently falls back to a default.
        url = f'{realtime_websocket_url(self._provider.base_url)}?model={self.model}'
        headers = {'Authorization': f'Bearer {self._api_key}'}
        # Propagate trace context over the handshake (see the OpenAI provider for the rationale).
        inject_trace_context(headers)
        settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages) or ''
        session_config = self._session_config(instructions, model_request_parameters.function_tools, settings)
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        # `dial` opens and configures a connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake), then resumes the captured conversation,
        # so sockets don't accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        # The model the server reports actually serving, from the `session.created` handshake. xAI
        # accepts any model slug and silently substitutes its current default, so this is the only
        # record of what actually served the session (see `RealtimeConnection.model_name`).
        server_model: str | None = None
        conversation_id: str | None = None
        replayed_items: list[ConversationItemCreated] = []
        connection: XaiRealtimeConnection | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, conversation_id, server_model
            if cm is not None:
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            resume_id = connection.conversation_id if connection is not None else None
            dial_url = f'{url}&conversation_id={quote(resume_id, safe="")}' if resume_id else url
            opening = websockets.connect(dial_url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            created = await expect_event(ws, 'session.created', timeout=handshake_timeout)
            model = obj(created.get('session')).get('model')
            if isinstance(model, str) and model:
                server_model = model
            if self.reconnect is not None:
                conversation = map_conversation_event(
                    await expect_event(ws, 'conversation.created', timeout=handshake_timeout)
                )
                if not isinstance(conversation, ConversationCreated):
                    raise RuntimeError('xAI realtime `conversation.created` event did not include `conversation.id`')
                conversation_id = conversation.conversation_id
                if connection is not None:
                    connection.conversation_id = conversation_id
            await ws.send(json.dumps({'type': 'session.update', 'session': session_config}))

            def capture_replayed_item(data: dict[str, Any]) -> None:
                event = map_conversation_event(data, replayed=True)
                if isinstance(event, ConversationItemCreated):
                    replayed_items.append(event)

            await expect_event(
                ws,
                'session.updated',
                timeout=handshake_timeout,
                on_unexpected=capture_replayed_item if resume_id is not None else None,
            )
            return ws

        try:
            ws = await dial()
            # Seed prior conversation once, after the initial handshake. Reconnects don't re-seed:
            # xAI restores the server-side conversation and replays its item lifecycle events instead.
            for item in await seed_items(messages, profile=self.profile, provider_name=self.system):
                await ws.send(json.dumps({'type': 'conversation.item.create', 'item': item}))
            connection = XaiRealtimeConnection(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
                conversation_id=conversation_id,
                replayed_items=replayed_items,
            )
            yield connection
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)
