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

from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..providers import infer_provider
from ..tools import ToolDefinition
from ._base import (
    RealtimeCodecEvent,
    RealtimeModel,
    RealtimeModelSettings,
    ReconnectPolicy,
    inject_trace_context,
)
from ._openai_protocol import (
    expect_event,
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

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return map_event(data)


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
            recover from a dropped connection. With no policy, the low-level connection reports a
            non-recoverable session error; `RealtimeSession` raises
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

        # `dial` opens and configures a fresh connection. A reconnect closes the previous connection
        # (including one left half-open by a failed handshake) before opening the next, so sockets don't
        # accumulate; teardown closes whatever is current.
        cm: AbstractAsyncContextManager[ClientConnection] | None = None

        # The model the server reports actually serving, from the `session.created` handshake. xAI
        # accepts any model slug and silently substitutes its current default, so this is the only
        # record of what actually served the session (see `RealtimeConnection.model_name`).
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
            yield XaiRealtimeConnection(
                ws,
                dial=dial,
                reconnect=self.reconnect,
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
            )
        finally:
            if cm is not None:
                await cm.__aexit__(None, None, None)
