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
  is `False` while cancellation-based interruption still works;
- no text output — the API has no response-modality control and always speaks — so
  [`RealtimeModelProfile.supports_text_output`][pydantic_ai.realtime.RealtimeModelProfile.supports_text_output]
  is `False` and `output_modality='text'` raises rather than silently coming back as audio.

Requires the `websockets` package (the `realtime` optional group), `xai-sdk` (the `xai` group, for
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider]), and `openai` (the `openai` group, whose SDK
supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[xai-realtime]"
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

try:
    import websockets as websockets
    from openai.types.realtime import (
        ConversationCreatedEvent,
        ConversationItem,
        ConversationItemAdded,
        ConversationItemCreatedEvent,
        RealtimeConversationItemFunctionCall,
        RealtimeConversationItemFunctionCallOutput,
        RealtimeResponseUsage,
        ResponseFunctionCallArgumentsDoneEvent,
    )
    from pydantic import BaseModel, ConfigDict, TypeAdapter
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` and `openai` packages to use the xAI Grok Voice realtime model '
        '(`openai` supplies the event types of the OpenAI codec this provider reuses), you can use the '
        '`realtime`, `xai`, and `openai` optional groups - '
        '`pip install "pydantic-ai-slim[xai-realtime]"`'
    ) from _import_error

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage, RealtimeSessionReconnectEvent
from ..models import ModelRequestParameters
from ..providers import infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import (
    RealtimeHandshakeError,
    connect_openai_protocol,
    expect_event,
    map_event as _map_openai_event,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from ._utils import inject_trace_context, resolve_advertised_tools
from .codec import (
    ConversationCreated,
    ConversationItemCreated,
    InputTranscript,
    RealtimeCodecEvent,
    ToolCall,
)
from .model import RealtimeModel
from .openai import OpenAIRealtimeConnection, ServerVAD
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

if TYPE_CHECKING:
    from ..providers.xai import XaiProvider

# `input_transcription_model='auto'` resolves to this — xAI's realtime transcription model. Kept behind
# the `'auto'` sentinel (see `resolve_transcription_model`) so it can change without altering the behavior
# of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'grok-transcribe'
_CONVERSATION_CREATED_EVENT = 'conversation.created'

LatestXaiRealtimeModelNames = Literal['grok-voice-latest', 'grok-voice-think-fast-2.0']
XaiRealtimeModelName = str | LatestXaiRealtimeModelNames

LatestXaiRealtimeTranscriptionModelNames = Literal['grok-transcribe']
XaiRealtimeTranscriptionModelName = str | LatestXaiRealtimeTranscriptionModelNames


class _ProtocolConversationItem(BaseModel):
    """Minimal typed item for xAI conversation lifecycle frames."""

    model_config = ConfigDict(extra='allow')

    type: str
    id: str | None = None
    call_id: str | None = None


class _ProtocolConversationItemAdded(BaseModel):
    event_id: str
    item: ConversationItem | _ProtocolConversationItem
    type: Literal['conversation.item.added']


_ConversationItemAddedEvent = ConversationItemAdded | _ProtocolConversationItemAdded
_CONVERSATION_ITEM_ADDED_ADAPTER: TypeAdapter[_ConversationItemAddedEvent] = TypeAdapter(_ConversationItemAddedEvent)


def map_conversation_event(
    data: dict[str, Any], *, replayed: bool | None = None
) -> ConversationCreated | ConversationItemCreated | None:
    """Map xAI's conversation handshake and item lifecycle events to codec control events."""
    event_type = data.get('type')
    if event_type == _CONVERSATION_CREATED_EVENT:
        event = ConversationCreatedEvent.model_validate(data)
        conversation_id = event.conversation.id
        return ConversationCreated(conversation_id) if conversation_id else None
    if event_type == 'conversation.item.added':
        event = _CONVERSATION_ITEM_ADDED_ADAPTER.validate_python(data)
    elif event_type == 'conversation.item.created':
        event = ConversationItemCreatedEvent.model_validate(data)
    else:
        return None
    item_id = event.item.id or data.get('item_id')
    tool_call_id = (
        event.item.call_id
        if isinstance(event.item, (RealtimeConversationItemFunctionCall, RealtimeConversationItemFunctionCallOutput))
        else data.get('call_id')
    )
    item_id = item_id if isinstance(item_id, str) and item_id else None
    tool_call_id = tool_call_id if isinstance(tool_call_id, str) and tool_call_id else None
    if item_id is not None or tool_call_id is not None:
        return ConversationItemCreated(item_id=item_id, tool_call_id=tool_call_id, replayed=bool(replayed))
    return None


class _InputAudioTranscriptionUpdatedEvent(BaseModel):
    """Typed xAI-only cumulative input-transcription event."""

    type: Literal['conversation.item.input_audio_transcription.updated']
    transcript: str | None = None
    item_id: str | None = None


class _XaiSession(BaseModel):
    """Fields read from xAI's SDK-incompatible session-created payload."""

    model: str | None = None


class _XaiSessionCreatedEvent(BaseModel):
    type: Literal['session.created']
    event_id: str
    session: _XaiSession


__all__ = (
    'XaiRealtimeModel',
    'XaiRealtimeModelSettings',
    'XaiRealtimeConnection',
    'map_event',
)


class XaiRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to xAI realtime models.

    Grok Voice always produces audio, so its profile reports
    [`supports_text_output=False`][pydantic_ai.realtime.RealtimeModelProfile.supports_text_output] and
    the inherited `output_modality='text'` is rejected up front rather than quietly ignored.
    """

    xai_voice: str
    """Voice used for audio output, e.g. `eve`, or a custom voice ID."""

    xai_turn_detection: ServerVAD
    """xAI-specific server-VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw xAI Grok Voice realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent].

    xAI clones the OpenAI Realtime protocol, so most events map identically via the OpenAI codec.
    The first exception is input audio transcription: xAI emits cumulative
    `conversation.item.input_audio_transcription.updated` snapshots (which may retroactively *correct*
    earlier text — `'Hello?'` becomes `'Hello, my name is'`) plus cumulative `.completed` snapshots,
    rather than OpenAI's incremental `.delta`. The partials are surfaced as cumulative
    [`InputTranscript`][pydantic_ai.realtime.codec.InputTranscript]s so a live transcript can render
    the user's words as they are spoken; the session adopts each snapshot wholesale, appending when it
    merely extends and replacing when xAI revises itself. The shared codec still drops interim
    `.completed` snapshots.
    The other exception is xAI's conversation lifecycle events, which are surfaced as codec control
    events so the connection can capture `conversation.id` and the session can suppress resume replay.
    """
    event_type = data.get('type')
    if event_type == 'conversation.item.input_audio_transcription.updated':
        event = _InputAudioTranscriptionUpdatedEvent.model_validate(data)
        return InputTranscript(
            text=event.transcript or '',
            cumulative=True,
            item_id=event.item_id,
        )
    if event_type in ('conversation.created', 'conversation.item.added', 'conversation.item.created'):
        return map_conversation_event(data)
    event = _map_openai_event(data)
    if isinstance(event, ToolCall):
        item_id = ResponseFunctionCallArgumentsDoneEvent.model_validate(data).item_id
        if item_id:
            event = replace(event, item_id=item_id)
    elif isinstance(event, InputTranscript):
        # xAI's final `.completed` is a whole snapshot like its `.updated` partials, so it too must
        # replace the accumulated text. Read as an increment it would be *appended* to the snapshots it
        # supersedes, and a revised turn would end up saying everything twice.
        event = replace(event, cumulative=True)
    return event


class XaiRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the xAI Grok Voice realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the
    shared wire protocol, while mapping xAI's cumulative input transcription and conversation lifecycle
    events and emitting the resumption replay controls captured during reconnect handshakes.
    """

    _provider_name = 'xai'
    _provider_label = 'xAI Grok Voice'
    _supports_tool_result_images = False

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
        model_name_getter: Callable[[], str | None] | None = None,
        conversation_id: str | None = None,
        replayed_items: list[ConversationItemCreated] | None = None,
    ) -> None:
        super().__init__(
            ws,
            dial=dial,
            reconnect=reconnect,
            input_transcription_enabled=input_transcription_enabled,
            model_name=model_name,
            model_name_getter=model_name_getter,
        )
        self._restores_state_on_reconnect = True
        self._conversation_id = conversation_id
        self._replayed_items = replayed_items if replayed_items is not None else []

    def _map_response_usage(self, usage: RealtimeResponseUsage | None) -> RequestUsage | None:
        mapped = super()._map_response_usage(usage)
        if mapped is None:
            return None
        assert usage is not None
        inp = usage.input_token_details or None
        out = usage.output_token_details or None
        # xAI bills Grok Voice by audio second, so this provider-owned bucket cannot be reconstructed
        # from the OpenAI-protocol token counts.
        for key, raw in (
            ('input_grok_tokens', (inp.model_extra or {}).get('grok_tokens') if inp is not None else None),
            ('output_grok_tokens', (out.model_extra or {}).get('grok_tokens') if out is not None else None),
            ('billable_audio_seconds', (usage.model_extra or {}).get('billable_audio_seconds')),
        ):
            if isinstance(raw, int) and not isinstance(raw, bool) and raw:
                mapped.details[key] = raw
        return mapped

    def set_message_history(self, message_history: Callable[[], Sequence[ModelMessage]]) -> None:
        """Ignored: xAI restores the conversation itself, so replaying it would say everything twice."""

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        # xAI resumes the conversation server-side, so the in-flight turn is not the session's to settle
        # (unlike the OpenAI base, which reconnects by replaying finalized history only).
        return True

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
            if isinstance(event, RealtimeSessionReconnectEvent):
                replayed_items = self._replayed_items[:]
                self._replayed_items.clear()
                for replayed_item in replayed_items:
                    yield replayed_item


@dataclass(init=False)
class XaiRealtimeModel(RealtimeModel):
    """xAI Grok Voice realtime API model.

    Pass `provider='xai'` (the default, which reads `XAI_API_KEY`) or an
    [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] constructed with `api_key=`. A custom
    `api_host` is not supported, and a provider constructed only with `xai_client=` cannot be used
    because the WebSocket connection needs access to the API key. The realtime WebSocket URL is
    `wss://api.x.ai/v1/realtime`.

    Args:
        model: The model name, e.g. `grok-voice-latest` (which tracks the current model) or a
            pinned version like `grok-voice-think-fast-1.0`. The `model` query parameter is required by
            the server, which otherwise falls back to a default silently.
        provider: The provider to use for authentication and the base URL. Defaults to `'xai'`.
        settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults for
            realtime sessions. A [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy enables xAI's native session resumption: prior turns are restored when reconnecting
            within xAI's resumption window (reportedly ~30 minutes).
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use. Mirrors `profile=` on a standard
            [`Model`][pydantic_ai.models.Model], and is the escape hatch when a model name doesn't
            identify the model (e.g. an Azure deployment named something other than its model).
    """

    model: XaiRealtimeModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: XaiProvider = field(init=False, repr=False)
    _api_key: str = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument while
    # `RealtimeModel.profile` stays the *resolved* profile, exactly as on a standard `Model` — a
    # dataclass field of that name would shadow the property.
    def __init__(
        self,
        model: XaiRealtimeModelName,
        *,
        provider: XaiProvider | str = 'xai',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self._profile = profile
        if isinstance(provider, str):
            provider = cast('XaiProvider', infer_provider(provider))
        if provider.name != 'xai':
            # Reading the xAI-specific `api_key`/`api_host` off a foreign provider below would fail with
            # an `AttributeError` naming a field the user never heard of, instead of the real mistake.
            raise UserError(f"`XaiRealtimeModel` requires an `XaiProvider` or `provider='xai'`; got {provider.name!r}.")
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
    def model_name(self) -> XaiRealtimeModelName:
        return self.model

    @property
    def system(self) -> str:
        return 'xai'

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: XaiRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        # xAI puts `voice` and `turn_detection` at the session top level, unlike OpenAI's GA surface which
        # nests them under `audio`. `turn_detection` is always set: a dict enables VAD, `None` disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_input_sample_rate', 24000)}
        }
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
            turn_detection: ServerVAD | None = {'type': 'server_vad'}
        config: dict[str, Any] = {
            'instructions': instructions,
            'turn_detection': turn_detection_config(turn_detection),
            'audio': {
                'input': audio_input,
                'output': {
                    'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_output_sample_rate', 24000)}
                },
            },
        }
        if voice := model_settings.get('xai_voice'):
            config['voice'] = voice
        advertised_tools, tool_choice = resolve_advertised_tools(tools, model_settings.get('tool_choice'))
        if advertised_tools:
            config['tools'] = [tool_def_to_openai(t) for t in advertised_tools]
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if tool_choice is not None:
            config['tool_choice'] = tool_choice_config(tool_choice)
        if (thinking := model_settings.get('thinking')) is not None and self.profile.get('supports_thinking', False):
            # Grok Voice exposes only enabled-at-high and disabled, so every enabled unified effort
            # maps to its sole enabled value.
            config['reasoning'] = {'effort': 'high' if thinking is not False else 'none'}
        if model_settings.get('reconnect') is not None:
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
        url = realtime_websocket_url(self._provider.base_url, model=self.model)
        headers = {'Authorization': f'Bearer {self._api_key}'}
        # Propagate trace context over the handshake (see the OpenAI provider for the rationale).
        inject_trace_context(headers)
        settings = cast('XaiRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        reconnect = settings.get('reconnect')
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions=instructions, tools=model_request_parameters.function_tools, model_settings=settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None
        conversation_id: str | None = None
        replayed_items: list[ConversationItemCreated] = []
        connection: XaiRealtimeConnection | None = None

        async def dial_headers() -> dict[str, str]:
            return headers

        def dial_url() -> str:
            resume_id = connection.conversation_id if connection is not None else None
            dial_url = f'{url}&conversation_id={quote(resume_id, safe="")}' if resume_id else url
            return dial_url

        def session_model(created: dict[str, Any]) -> str | None:
            return _XaiSessionCreatedEvent.model_validate(created).session.model

        async def after_session_created(ws: ClientConnection, _: dict[str, Any]) -> None:
            nonlocal conversation_id
            if reconnect is not None:
                conversation = map_conversation_event(
                    await expect_event(ws, _CONVERSATION_CREATED_EVENT, timeout=handshake_timeout)
                )
                if not isinstance(conversation, ConversationCreated):
                    raise RealtimeHandshakeError(
                        '`conversation.created` did not include a `conversation.id`, so the session '
                        'cannot be resumed after a drop'
                    )
                conversation_id = conversation.conversation_id
                if connection is not None:
                    connection.conversation_id = conversation_id

        def on_unexpected_during_update() -> Callable[[dict[str, Any]], None] | None:
            if connection is None:
                return None

            def capture_replayed_item(data: dict[str, Any]) -> None:
                event = map_conversation_event(data, replayed=True)
                if isinstance(event, ConversationItemCreated):
                    replayed_items.append(event)

            return capture_replayed_item

        def build_connection(
            ws: ClientConnection,
            dial: Callable[[], Awaitable[ClientConnection]],
            server_model: str | None,
            model_name_getter: Callable[[], str | None],
        ) -> XaiRealtimeConnection:
            nonlocal connection
            connection = XaiRealtimeConnection(
                ws,
                dial=dial,
                reconnect=reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
                model_name_getter=model_name_getter,
                conversation_id=conversation_id,
                replayed_items=replayed_items,
            )
            return connection

        async with connect_openai_protocol(
            model_name=self.model,
            messages=messages,
            profile=self.profile,
            provider_name=self.system,
            session_config=session_config,
            handshake_timeout=handshake_timeout,
            dial_headers=dial_headers,
            dial_url=dial_url,
            session_model=session_model,
            build_connection=build_connection,
            after_session_created=after_session_created,
            on_unexpected_during_update=on_unexpected_during_update,
        ) as connected:
            yield connected
