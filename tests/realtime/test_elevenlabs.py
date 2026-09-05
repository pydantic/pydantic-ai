"""Tests for the ElevenLabs Agents realtime provider, all network-free.

ElevenLabs has no direct realtime conversational model: the adapter wraps a hosted agent, so unlike
the sibling provider tests these cover the REST preflight (override-permission checks, tool
sync-or-error) against a mocked HTTP transport, plus the adapter's own codec (event mapping, send
framing, ping/pong, interruption bookkeeping) against hand-built frames mirroring live captures.
The real protocol end-to-end is covered by the cassette-backed `test_elevenlabs_ws.py`.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
from collections.abc import Generator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from typing import Any
from unittest import mock

import httpx2
import pytest

from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    TextContent,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import RealtimeError
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CommitAudio,
    InputTranscript,
    OutputTranscript,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolResult,
)
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import
from .ws_helpers import collect_codec_events

with try_import() as imports_successful:
    import websockets
    from websockets.frames import Close

    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.realtime import elevenlabs as rt_elevenlabs
    from pydantic_ai.realtime.elevenlabs import (
        ElevenLabsRealtimeConnection,
        ElevenLabsRealtimeModel,
        ElevenLabsRealtimeModelSettings,
    )

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='websockets not installed')]

AGENT_ID = 'agent_0101test'


def test_elevenlabs_public_exports_are_curated() -> None:
    assert rt_elevenlabs.__all__ == (
        'ElevenLabsRealtimeModel',
        'ElevenLabsRealtimeModelSettings',
        'ElevenLabsRealtimeConnection',
        'ElevenLabsTTSSettings',
    )


# --- fakes ---------------------------------------------------------------------------------------


class FakeWebSocket:
    """A minimal stand-in for a `websockets` client connection.

    Running out of scripted frames stands in for the server closing the connection normally, raised
    as `ConnectionClosedOK` exactly as `websockets.recv()` reports a 1000 close.
    """

    url: str | None = None

    def __init__(self, incoming: Sequence[dict[str, Any] | str]) -> None:
        self._incoming = [frame if isinstance(frame, str) else json.dumps(frame) for frame in incoming]
        self.sent: list[str] = []
        self.closed = False

    async def recv(self) -> str:
        if not self._incoming:
            raise websockets.exceptions.ConnectionClosedOK(Close(1000, ''), None)
        return self._incoming.pop(0)

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def sent_frames(self) -> list[dict[str, Any]]:
        return [json.loads(frame) for frame in self.sent]


def agent_json(
    *,
    overrides: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_ids: list[str] | None = None,
    custom_llm_extra_body: bool = False,
) -> dict[str, Any]:
    """A `GET /v1/convai/agents/{id}` response holding just what the preflight reads."""
    return {
        'agent_id': AGENT_ID,
        'conversation_config': {
            'agent': {
                'language': 'en',
                'prompt': {'prompt': 'Existing', 'tools': tools or [], 'tool_ids': tool_ids or []},
            }
        },
        'platform_settings': {
            'auth': {'enable_auth': False},
            'overrides': {
                'conversation_config_override': overrides or {},
                'custom_llm_extra_body': custom_llm_extra_body,
            },
        },
    }


def tools_listing(*entries: tuple[str, str, str], next_cursor: str | None = None) -> dict[str, Any]:
    """One `GET /v1/convai/tools` page: `(id, type, name)` per workspace tool.

    A `next_cursor` marks the page as partial (`has_more`), the way the cursor-paginated listing
    reports it.
    """
    listing: dict[str, Any] = {
        'tools': [{'id': id, 'tool_config': {'type': type, 'name': name}} for id, type, name in entries]
    }
    if next_cursor is not None:
        listing['has_more'] = True
        listing['next_cursor'] = next_cursor
    return listing


ALL_OVERRIDES_ENABLED: dict[str, Any] = {
    'agent': {'first_message': True, 'language': True, 'prompt': {'prompt': True, 'llm': True}},
    'tts': {'voice_id': True, 'model_id': True, 'stability': True, 'speed': True, 'similarity_boost': True},
    'conversation': {'text_only': True},
}


class RestRecorder:
    """Serves canned REST responses through `httpx2.MockTransport`, recording every request.

    An `extra` value may be a list of responses, served in order across repeated calls to the same
    method and path (e.g. the pages of the cursor-paginated tool listing).
    """

    def __init__(
        self,
        agent: dict[str, Any],
        *,
        extra: dict[tuple[str, str], httpx2.Response | list[httpx2.Response]] | None = None,
    ) -> None:
        self.agent = agent
        self.extra = extra or {}
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        canned = self.extra.get((request.method, request.url.path))
        if isinstance(canned, list):
            return canned.pop(0)
        if canned is not None:
            return canned
        if request.method == 'GET' and request.url.path == f'/v1/convai/agents/{AGENT_ID}':
            return httpx2.Response(200, json=self.agent)
        if request.method == 'GET' and request.url.path == '/v1/convai/conversation/get-signed-url':
            return httpx2.Response(200, json={'signed_url': f'wss://api.elevenlabs.io/signed?agent_id={AGENT_ID}'})
        raise AssertionError(f'unexpected request: {request.method} {request.url}')  # pragma: no cover

    def request_summaries(self) -> list[str]:
        return [f'{request.method} {request.url.path}' for request in self.requests]


def _model(
    recorder: RestRecorder,
    *,
    settings: ElevenLabsRealtimeModelSettings | None = None,
    **kwargs: Any,
) -> ElevenLabsRealtimeModel:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(recorder))
    provider = ElevenLabsProvider(api_key='test-api-key', http_client=client)
    return ElevenLabsRealtimeModel(AGENT_ID, provider=provider, settings=settings, **kwargs)


HANDSHAKE_FRAME: dict[str, Any] = {
    'type': 'conversation_initiation_metadata',
    'conversation_initiation_metadata_event': {
        'conversation_id': 'conv_1',
        'agent_output_audio_format': 'pcm_16000',
        'user_input_audio_format': 'pcm_16000',
    },
}


def _connect(
    model: ElevenLabsRealtimeModel,
    *,
    instructions: str | None = None,
    tools: list[ToolDefinition] | None = None,
    model_settings: ElevenLabsRealtimeModelSettings | None = None,
) -> AbstractAsyncContextManager[ElevenLabsRealtimeConnection]:
    messages: list[ModelMessage] = [ModelRequest(parts=[], instructions=instructions)]
    return model.connect(
        messages=messages,
        model_settings=model_settings,
        model_request_parameters=ModelRequestParameters(function_tools=tools or []),
    )


class FakeOpening:
    """Stands in for the object `websockets.connect(...)` returns: an async context manager."""

    def __init__(self, ws: FakeWebSocket | None = None, exc: Exception | None = None) -> None:
        self._ws, self._exc = ws, exc

    async def __aenter__(self) -> FakeWebSocket:
        if self._exc is not None:
            raise self._exc
        assert self._ws is not None
        return self._ws

    async def __aexit__(self, *exc_info: Any) -> None:
        assert self._ws is not None
        await self._ws.close()


@contextmanager
def patched_connect(ws: FakeWebSocket) -> Generator[None]:
    """Patch the adapter's `websockets.connect` to hand back `ws`, recording the dialed URL on it."""

    def connect(url: str, **kwargs: Any) -> FakeOpening:
        ws.url = url
        return FakeOpening(ws)

    with mock.patch.object(rt_elevenlabs.websockets, 'connect', connect):
        yield


@contextmanager
def failing_connect(exc: Exception) -> Generator[None]:
    """Patch the adapter's `websockets.connect` so the dial (`__aenter__`) raises `exc`."""

    def connect(url: str, **kwargs: Any) -> FakeOpening:
        return FakeOpening(exc=exc)

    with mock.patch.object(rt_elevenlabs.websockets, 'connect', connect):
        yield


# --- model basics --------------------------------------------------------------------------------


def test_model_name_is_the_agent_id() -> None:
    model = _model(RestRecorder(agent_json()))
    assert model.model_name == AGENT_ID
    assert model.system == 'elevenlabs'
    assert model.base_url == 'https://api.elevenlabs.io'
    # Nothing about the LLM can be inferred from an agent id, so the profile pins the window as unknown.
    assert model.context_window is None


def test_rejects_non_elevenlabs_provider() -> None:
    from pydantic_ai.providers.openai import OpenAIProvider

    with pytest.raises(UserError, match='requires an `ElevenLabsProvider`'):
        ElevenLabsRealtimeModel(AGENT_ID, provider=OpenAIProvider(api_key='k'))  # type: ignore[arg-type]


# --- settings the WebSocket cannot honor ---------------------------------------------------------


async def test_reconnect_policy_raises() -> None:
    model = _model(RestRecorder(agent_json()), settings=ElevenLabsRealtimeModelSettings(reconnect={}))
    with pytest.raises(UserError, match='cannot be resumed after a drop'):
        async with _connect(model):
            pass  # pragma: no cover


async def test_turn_detection_config_raises() -> None:
    model = _model(RestRecorder(agent_json()))
    with pytest.raises(UserError, match='own turn-taking server-side'):
        async with _connect(model, model_settings=ElevenLabsRealtimeModelSettings(turn_detection=False)):
            pass  # pragma: no cover


async def test_turn_detection_true_is_accepted() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    with patched_connect(ws):
        async with _connect(model, model_settings=ElevenLabsRealtimeModelSettings(turn_detection=True)):
            pass


async def test_disabling_input_transcription_raises() -> None:
    model = _model(RestRecorder(agent_json()))
    with pytest.raises(UserError, match='cannot disable input transcription'):
        async with _connect(model, model_settings=ElevenLabsRealtimeModelSettings(input_transcription_model=None)):
            pass  # pragma: no cover


# --- REST preflight: override permissions --------------------------------------------------------


async def test_instructions_require_the_prompt_override_toggle() -> None:
    # A Pydantic AI agent with instructions must fail loudly when the ElevenLabs agent
    # does not permit the prompt override, naming the exact toggle to flip.
    model = _model(RestRecorder(agent_json(overrides={'agent': {'prompt': {'prompt': False}}})))
    with pytest.raises(
        UserError,
        match=r'platform_settings\.overrides\.conversation_config_override\.agent\.prompt\.prompt',
    ):
        async with _connect(model, instructions='Be terse.'):
            pass  # pragma: no cover


async def test_no_instructions_inherits_the_agent_prompt_silently() -> None:
    # Without local instructions the ElevenLabs-side prompt is inherited: no override is sent, and no
    # override permission is required.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder(agent_json())
    model = _model(recorder)
    with patched_connect(ws):
        async with _connect(model) as connection:
            assert connection.conversation_id == 'conv_1'
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert ws.url == f'wss://api.elevenlabs.io/signed?agent_id={AGENT_ID}'
    assert ws.sent_frames() == [{'type': 'conversation_initiation_client_data'}]
    assert ws.closed


async def test_initiation_payload_builds_every_permitted_override() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    settings = ElevenLabsRealtimeModelSettings(
        output_modality='text',
        elevenlabs_voice_id='voice_1',
        elevenlabs_language='de',
        elevenlabs_first_message='Hallo!',
        elevenlabs_llm='gpt-5.2',
        elevenlabs_tts={'stability': 0.4, 'speed': 1.1},
        elevenlabs_dynamic_variables={'plan': 'starter'},
        elevenlabs_user_id='user-42',
        elevenlabs_custom_llm_extra_body={'temperature': 0.1},
        elevenlabs_config_override={'conversation': {'max_duration_seconds': 120}},
    )
    model = _model(RestRecorder(agent_json(overrides=ALL_OVERRIDES_ENABLED, custom_llm_extra_body=True)))
    with patched_connect(ws):
        async with _connect(model, instructions='Be helpful.', model_settings=settings):
            pass
    assert ws.sent_frames() == [
        {
            'type': 'conversation_initiation_client_data',
            'conversation_config_override': {
                'agent': {
                    'prompt': {'prompt': 'Be helpful.', 'llm': 'gpt-5.2'},
                    'first_message': 'Hallo!',
                    'language': 'de',
                },
                'tts': {'voice_id': 'voice_1', 'stability': 0.4, 'speed': 1.1},
                'conversation': {'text_only': True, 'max_duration_seconds': 120},
            },
            'custom_llm_extra_body': {'temperature': 0.1},
            'dynamic_variables': {'plan': 'starter'},
            'user_id': 'user-42',
        }
    ]


async def test_each_settings_override_requires_its_toggle() -> None:
    cases: list[tuple[ElevenLabsRealtimeModelSettings, str]] = [
        (ElevenLabsRealtimeModelSettings(elevenlabs_voice_id='v'), r'tts\.voice_id'),
        (ElevenLabsRealtimeModelSettings(elevenlabs_language='de'), r'agent\.language'),
        (ElevenLabsRealtimeModelSettings(elevenlabs_first_message='Hi'), r'agent\.first_message'),
        (ElevenLabsRealtimeModelSettings(elevenlabs_llm='gpt-5.2'), r'agent\.prompt\.llm'),
        (ElevenLabsRealtimeModelSettings(elevenlabs_tts={'stability': 0.4}), r'tts\.stability'),
        (ElevenLabsRealtimeModelSettings(output_modality='text'), r'conversation\.text_only'),
    ]
    for settings, toggle_pattern in cases:
        model = _model(RestRecorder(agent_json()))
        with pytest.raises(UserError, match=toggle_pattern):
            async with _connect(model, model_settings=settings):
                pass  # pragma: no cover


async def test_custom_llm_extra_body_requires_its_sibling_toggle() -> None:
    model = _model(RestRecorder(agent_json()))
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_custom_llm_extra_body={'temperature': 0.1})
    with pytest.raises(UserError, match=r'platform_settings\.overrides\.custom_llm_extra_body'):
        async with _connect(model, model_settings=settings):
            pass  # pragma: no cover


async def test_bare_agent_response_without_platform_settings_is_tolerated() -> None:
    # `GET agent` responses are mirrored just far enough for the preflight: a response missing the
    # `conversation_config`/`platform_settings` sections means no overrides permitted and no tools.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder({'agent_id': AGENT_ID}))
    with patched_connect(ws):
        async with _connect(model):
            pass
    assert ws.sent_frames() == [{'type': 'conversation_initiation_client_data'}]


async def test_bare_agent_response_denies_custom_llm_extra_body() -> None:
    model = _model(RestRecorder({'agent_id': AGENT_ID}))
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_custom_llm_extra_body={'temperature': 0.1})
    with pytest.raises(UserError, match='custom_llm_extra_body'):
        async with _connect(model, model_settings=settings):
            pass  # pragma: no cover


async def test_preflight_http_error_maps_to_model_http_error() -> None:
    recorder = RestRecorder(agent_json())
    recorder.extra[('GET', f'/v1/convai/agents/{AGENT_ID}')] = httpx2.Response(404, text='agent not found')
    model = _model(recorder)
    with pytest.raises(ModelHTTPError) as exc_info:
        async with _connect(model):
            pass  # pragma: no cover
    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == AGENT_ID


async def test_preflight_transport_error_maps_to_realtime_error() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('connection refused')

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(refuse))
    provider = ElevenLabsProvider(api_key='test-api-key', http_client=client)
    model = ElevenLabsRealtimeModel(AGENT_ID, provider=provider)
    with pytest.raises(RealtimeError, match='Could not reach the ElevenLabs API'):
        async with _connect(model):
            pass  # pragma: no cover


# --- REST preflight: tools -----------------------------------------------------------------------


WEATHER_TOOL = ToolDefinition(
    name='get_weather',
    description='Get the weather.',
    parameters_json_schema={
        'additionalProperties': False,
        'type': 'object',
        'properties': {'city': {'type': 'string', 'description': 'City name'}},
        'required': ['city'],
    },
)

# The stored form mirrors how the server actually normalizes a schema (verified live): bookkeeping
# fields injected on every node, `enum: null` on properties, empty descriptions added, and keywords
# it cannot express (`additionalProperties`) gone.
WEATHER_TOOL_REMOTE: dict[str, Any] = {
    'type': 'client',
    'name': 'get_weather',
    'description': 'Get the weather.',
    'expects_response': True,
    'parameters': {
        'description': '',
        'dynamic_variable': '',
        'is_omitted': False,
        'type': 'object',
        'required': ['city'],
        'properties': {
            'city': {
                'type': 'string',
                'description': 'City name',
                'enum': None,
                'is_system_provided': False,
                'dynamic_variable': '',
                'constant_value': '',
                'is_omitted': False,
            }
        },
    },
}


async def test_tool_mismatch_errors_by_default() -> None:
    # Default `elevenlabs_tool_sync='error'`: every difference between the session's tools and the
    # agent's client tools is reported in one error.
    remote = dict(
        WEATHER_TOOL_REMOTE, description='Different description.', parameters={'type': 'object', 'properties': {}}
    )
    extra_remote: dict[str, Any] = {'type': 'client', 'name': 'close_widget', 'expects_response': False}
    local_only = ToolDefinition(name='local_only')
    model = _model(RestRecorder(agent_json(tools=[remote, extra_remote])))
    with pytest.raises(UserError) as exc_info:
        async with _connect(model, tools=[WEATHER_TOOL, local_only]):
            pass  # pragma: no cover
    message = str(exc_info.value)
    assert "tool 'local_only' is not configured on the agent" in message
    assert "agent client tool 'close_widget' is not defined by this agent run" in message
    assert "tool 'get_weather' differs in description" in message
    assert "tool 'get_weather' differs in parameters schema" in message


async def test_matching_tools_pass_the_default_check() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE]))
    model = _model(recorder)
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass
    # No sync calls were made: the agent's tools already match.
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_error_mode_matches_nested_and_optional_parameters() -> None:
    # The `'error'` comparison sees the local schema through the same preparation as sync mode:
    # `$defs` inlined and nullable unions collapsed, so a tool with a nested-model or optional
    # parameter matches its faithfully configured server-side counterpart instead of tripping a
    # false mismatch.
    local = ToolDefinition(
        name='find_place',
        description='Find a place.',
        parameters_json_schema={
            '$defs': {
                'Location': {
                    'type': 'object',
                    'properties': {'lat': {'type': 'number', 'description': 'Latitude'}},
                    'required': ['lat'],
                }
            },
            'type': 'object',
            'required': ['where'],
            'properties': {
                'where': {'$ref': '#/$defs/Location', 'description': 'The location'},
                'when': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'description': 'When, if known'},
            },
        },
    )
    remote: dict[str, Any] = {
        'type': 'client',
        'name': 'find_place',
        'description': 'Find a place.',
        'expects_response': True,
        'parameters': {
            'description': '',
            'is_omitted': False,
            'type': 'object',
            'required': ['where'],
            'properties': {
                'where': {
                    'type': 'object',
                    'description': 'The location',
                    'is_omitted': False,
                    'required': ['lat'],
                    'properties': {'lat': {'type': 'number', 'description': 'Latitude', 'enum': None}},
                },
                'when': {'type': 'string', 'description': 'When, if known', 'enum': None, 'is_omitted': False},
            },
        },
    }
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json(tools=[remote])))
    with patched_connect(ws):
        async with _connect(model, tools=[local]):
            pass


async def test_expects_response_disabled_on_the_agent_is_a_mismatch() -> None:
    remote = dict(WEATHER_TOOL_REMOTE, expects_response=False)
    model = _model(RestRecorder(agent_json(tools=[remote])))
    with pytest.raises(UserError, match='has `expects_response` disabled'):
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass  # pragma: no cover


async def test_server_side_tools_do_not_participate_in_the_check() -> None:
    # Webhook/MCP/system tools are ElevenLabs-owned; only client tools are compared.
    webhook: dict[str, Any] = {'id': 'tool_9', 'type': 'webhook', 'name': 'crm_lookup'}
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE, webhook])))
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass


async def test_tool_choice_none_restricts_the_advertised_set() -> None:
    # `tool_choice='none'` advertises no tools, so an agent with none configured matches.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    settings = ElevenLabsRealtimeModelSettings(tool_choice='none')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass


async def test_tool_sync_off_trusts_the_agent() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='off')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass


async def test_tool_sync_creates_updates_and_repoints() -> None:
    # `'sync'` mode: create the missing tool, update the differing one (its id resolved through the
    # workspace tool listing, since resolved agent tools don't carry ids), and re-point the agent's
    # `tool_ids` at the synced set behind the untouched server-side tools.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    webhook: dict[str, Any] = {'type': 'webhook', 'name': 'crm_lookup'}
    unchanged: dict[str, Any] = {
        'type': 'client',
        'name': 'get_time',
        'description': 'Get the time.',
        'expects_response': True,
        'parameters': {'type': 'object', 'properties': {}},
    }
    unchanged_tool = ToolDefinition(name='get_time', description='Get the time.')
    new_tool = ToolDefinition(name='book_table', description='Book a table.')
    recorder = RestRecorder(
        agent_json(tools=[differing, webhook, unchanged], tool_ids=['tool_1', 'tool_9', 'tool_5']),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200,
                json=tools_listing(
                    ('tool_1', 'client', 'get_weather'),
                    ('tool_9', 'webhook', 'crm_lookup'),
                    ('tool_5', 'client', 'get_time'),
                ),
            ),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', '/v1/convai/tools/tool_1'): httpx2.Response(200, json={'id': 'tool_1'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[new_tool, WEATHER_TOOL, unchanged_tool], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'POST /v1/convai/tools',
        'PATCH /v1/convai/tools/tool_1',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    created = json.loads(recorder.requests[2].content)
    assert created == {
        'tool_config': {
            'type': 'client',
            'name': 'book_table',
            'description': 'Book a table.',
            'expects_response': True,
            'parameters': {'type': 'object', 'properties': {}},
        }
    }
    # The PATCHed tool config is converted to the server's parameters dialect: `additionalProperties`
    # stripped, only supported keys sent.
    patched = json.loads(recorder.requests[3].content)
    assert patched == {
        'tool_config': {
            'type': 'client',
            'name': 'get_weather',
            'description': 'Get the weather.',
            'expects_response': True,
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string', 'description': 'City name'}},
                'required': ['city'],
            },
        }
    }
    repointed = json.loads(recorder.requests[4].content)
    assert repointed == {
        'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_9', 'tool_new', 'tool_1', 'tool_5']}}}
    }


async def test_tool_sync_patch_only_skips_the_repoint() -> None:
    # When updating a differing tool leaves the attached ids unchanged, no agent PATCH is needed.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    recorder = RestRecorder(
        agent_json(tools=[differing], tool_ids=['tool_1']),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing(('tool_1', 'client', 'get_weather'))),
            ('PATCH', '/v1/convai/tools/tool_1'): httpx2.Response(200, json={'id': 'tool_1'}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'PATCH /v1/convai/tools/tool_1',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_adopts_a_matching_unattached_workspace_tool() -> None:
    # A sync that failed between its create and the final re-point leaves the created tool in the
    # workspace but unattached; the retry adopts it by exact config match instead of POSTing a
    # duplicate.
    recorder = RestRecorder(
        agent_json(),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json={'tools': [{'id': 'tool_orphan', 'tool_config': WEATHER_TOOL_REMOTE}]}
            ),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    repointed = json.loads(recorder.requests[2].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_orphan']}}}}


async def test_tool_sync_leaves_a_differing_unattached_tool_alone() -> None:
    # An unattached same-named tool whose config differs may belong to another agent: it is neither
    # adopted nor PATCHed, and a fresh tool is created instead.
    recorder = RestRecorder(
        agent_json(),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200,
                json={'tools': [{'id': 'tool_orphan', 'tool_config': dict(WEATHER_TOOL_REMOTE, description='Stale.')}]},
            ),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'POST /v1/convai/tools',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    repointed = json.loads(recorder.requests[3].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_new']}}}}


async def test_tool_sync_follows_the_paginated_tool_listing() -> None:
    # `GET /v1/convai/tools` is cursor-paginated; every page is fetched so an attached tool on a
    # later page is updated in place rather than misclassified and duplicated by a fresh create.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    recorder = RestRecorder(
        agent_json(tools=[differing], tool_ids=['tool_1']),
        extra={
            ('GET', '/v1/convai/tools'): [
                httpx2.Response(200, json=tools_listing(('tool_7', 'client', 'other_tool'), next_cursor='cursor_2')),
                httpx2.Response(200, json=tools_listing(('tool_1', 'client', 'get_weather'))),
            ],
            ('PATCH', '/v1/convai/tools/tool_1'): httpx2.Response(200, json={'id': 'tool_1'}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools',
        'PATCH /v1/convai/tools/tool_1',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert recorder.requests[1].url.params.get('cursor') is None
    assert recorder.requests[2].url.params.get('cursor') == 'cursor_2'


async def test_tool_sync_stops_on_a_repeated_pagination_cursor() -> None:
    # A misbehaving server that keeps returning an already-seen cursor with `has_more` set must not
    # be refetched forever: the listing ends there and the sync proceeds with the pages it has
    # (attached ids left unresolved are preserved, never dropped).
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    recorder = RestRecorder(
        agent_json(tools=[differing], tool_ids=['tool_1']),
        extra={
            ('GET', '/v1/convai/tools'): [
                httpx2.Response(200, json=tools_listing(('tool_1', 'client', 'get_weather'), next_cursor='cursor_2')),
                httpx2.Response(200, json=tools_listing(next_cursor='cursor_2')),
            ],
            ('PATCH', '/v1/convai/tools/tool_1'): httpx2.Response(200, json={'id': 'tool_1'}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools',
        'PATCH /v1/convai/tools/tool_1',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_preserves_attached_ids_the_listing_does_not_report() -> None:
    # An attached tool id the workspace cannot see (the per-id fetch 404s, the listing omits it) is
    # never dropped.
    recorder = RestRecorder(
        agent_json(tool_ids=['tool_x']),
        extra={
            ('GET', '/v1/convai/tools/tool_x'): httpx2.Response(404, json={'detail': 'Tool not found'}),
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing()),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[ToolDefinition(name='book_table')], model_settings=settings):
            pass
    repointed = json.loads(recorder.requests[4].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_x', 'tool_new']}}}}


async def test_tool_sync_replaces_a_client_tool_missing_from_the_listing() -> None:
    # A differing client tool whose id the listing cannot resolve is replaced by a fresh create
    # rather than failing the sync.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    recorder = RestRecorder(
        agent_json(tools=[differing], tool_ids=[]),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing()),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'POST /v1/convai/tools',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_with_matching_tools_makes_no_extra_requests() -> None:
    # Nothing differs, so sync doesn't even fetch the workspace tool listing.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE], tool_ids=['tool_1']))
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_requires_parameter_descriptions() -> None:
    # The server rejects a client-tool property without a description (verified live), which
    # Pydantic AI cannot invent: sync fails loudly naming the parameter, before any workspace
    # request beyond the preflight (not even the tool listing is fetched for an invalid set).
    bare = ToolDefinition(
        name='get_weather',
        description='Get the weather.',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
    )
    recorder = RestRecorder(agent_json())
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with pytest.raises(UserError, match=r'requires a description for every tool parameter, and `city` has none'):
        async with _connect(model, tools=[bare], model_settings=settings):
            pass  # pragma: no cover
    assert recorder.request_summaries() == [f'GET /v1/convai/agents/{AGENT_ID}']


async def test_tool_sync_validation_reports_every_offender_and_writes_nothing() -> None:
    # A validation failure must cost zero workspace mutations, and must name every offending tool
    # and parameter in one error. Validating inside the create loop did neither: a production
    # deployment syncing seven tools was left with the valid tools already created but never
    # attached (the re-point at the end never ran), and learned about its two invalid tools one
    # connect at a time.
    bare = ToolDefinition(
        name='set_language',
        description='Switch the language.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'language': {'type': 'string'}, 'region': {'type': 'string'}},
        },
    )
    unexpressable = ToolDefinition(
        name='add_items',
        description='Add items.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}], 'description': 'Items.'}},
        },
    )
    recorder = RestRecorder(agent_json())
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with pytest.raises(UserError) as exc_info:
        async with _connect(model, tools=[WEATHER_TOOL, bare, unexpressable], model_settings=settings):
            pass  # pragma: no cover
    message = str(exc_info.value)
    assert "Cannot sync tool 'set_language'" in message
    assert '`language` has none' in message
    assert '`region` has none' in message
    assert "Cannot sync tool 'add_items'" in message
    assert '`items` uses a JSON Schema feature' in message
    assert 'get_weather' not in message
    # Only the preflight ran: no tool listing, no creates, no agent re-point.
    assert recorder.request_summaries() == [f'GET /v1/convai/agents/{AGENT_ID}']


def test_elevenlabs_parameters_dialect_conversion() -> None:
    """Unit-pins the JSON-schema-to-dialect conversion (not reachable in one shape via the public API).

    Verified live: `enum`, `items`, and nested objects are supported; `additionalProperties` is
    rejected by the server and numeric bounds are silently dropped, so neither is sent; array items
    and nested properties need descriptions like top-level properties do.
    """
    tool = ToolDefinition(
        name='rich',
        description='Rich tool.',
        parameters_json_schema={
            'additionalProperties': False,
            'type': 'object',
            'required': ['city'],
            'properties': {
                'city': {'type': 'string', 'description': 'City'},
                'days': {'type': 'integer', 'description': 'Days ahead', 'minimum': 0, 'maximum': 7},
                'units': {'type': 'string', 'description': 'Units', 'enum': ['metric', 'imperial']},
                'tags': {'type': 'array', 'description': 'Tags', 'items': {'type': 'string', 'description': 'Tag'}},
                'loc': {
                    'type': 'object',
                    'description': 'Location',
                    'additionalProperties': False,
                    'properties': {'lat': {'type': 'number', 'description': 'Latitude'}},
                },
            },
        },
    )
    config = rt_elevenlabs._tool_config(tool)  # pyright: ignore[reportPrivateUsage]
    assert config['parameters'] == {
        'type': 'object',
        'required': ['city'],
        'properties': {
            'city': {'type': 'string', 'description': 'City'},
            'days': {'type': 'integer', 'description': 'Days ahead'},
            'units': {'type': 'string', 'description': 'Units', 'enum': ['metric', 'imperial']},
            'tags': {'type': 'array', 'description': 'Tags', 'items': {'type': 'string', 'description': 'Tag'}},
            'loc': {
                'type': 'object',
                'description': 'Location',
                'properties': {'lat': {'type': 'number', 'description': 'Latitude'}},
            },
        },
    }

    array_without_item_description = ToolDefinition(
        name='bad',
        parameters_json_schema={
            'type': 'object',
            'properties': {'tags': {'type': 'array', 'description': 'Tags', 'items': {'type': 'string'}}},
        },
    )
    with pytest.raises(UserError, match=r'`tags\[\]` has none'):
        rt_elevenlabs._tool_config(array_without_item_description)  # pyright: ignore[reportPrivateUsage]


def test_elevenlabs_parameters_inline_defs_and_collapse_nullable_unions() -> None:
    """Unit-pins the schema preparation for the shapes pydantic generates routinely.

    A nested model arrives as `$defs` plus a `$ref` property and an optional parameter as
    `anyOf: [X, null]`; neither is expressible in the ElevenLabs dialect directly, so the
    definitions are inlined and the nullable union collapses onto its non-null member (optionality
    is carried by `required` alone). A unit test for the same reason as the conversion test above.
    """
    tool = ToolDefinition(
        name='nested',
        description='Nested tool.',
        parameters_json_schema={
            '$defs': {
                'Location': {
                    'type': 'object',
                    'properties': {'lat': {'type': 'number', 'description': 'Latitude'}},
                    'required': ['lat'],
                }
            },
            'type': 'object',
            'required': ['where'],
            'properties': {
                'where': {'$ref': '#/$defs/Location', 'description': 'The location'},
                'when': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'description': 'When, if known'},
            },
        },
    )
    config = rt_elevenlabs._tool_config(tool)  # pyright: ignore[reportPrivateUsage]
    assert config['parameters'] == {
        'type': 'object',
        'required': ['where'],
        'properties': {
            'where': {
                'type': 'object',
                'description': 'The location',
                'required': ['lat'],
                'properties': {'lat': {'type': 'number', 'description': 'Latitude'}},
            },
            'when': {'type': 'string', 'description': 'When, if known'},
        },
    }


def test_elevenlabs_parameters_reject_inexpressible_shapes() -> None:
    """A true union of types, or a recursive reference, has no `type` the dialect could carry.

    Sending such a node would silently degrade the tool (the server stores an untyped parameter), so
    the sync fails loudly instead. A unit test for the same reason as the conversion test above.
    """
    union_tool = ToolDefinition(
        name='bad_union',
        parameters_json_schema={
            'type': 'object',
            'properties': {'value': {'anyOf': [{'type': 'string'}, {'type': 'integer'}], 'description': 'V'}},
        },
    )
    with pytest.raises(UserError, match=r'parameter `value` uses a JSON Schema feature'):
        rt_elevenlabs._tool_config(union_tool)  # pyright: ignore[reportPrivateUsage]

    recursive_tool = ToolDefinition(
        name='bad_recursive',
        parameters_json_schema={
            '$defs': {
                'Node': {
                    'type': 'object',
                    'properties': {'child': {'$ref': '#/$defs/Node', 'description': 'Child node'}},
                }
            },
            '$ref': '#/$defs/Node',
        },
    )
    with pytest.raises(UserError, match=r'the parameters schema uses a JSON Schema feature'):
        rt_elevenlabs._tool_config(recursive_tool)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    'local,remote,expected',
    [
        # The server-normalized remote form (bookkeeping fields, enum null, injected descriptions)
        # matches a local schema that lacks property descriptions.
        (
            {'type': 'object', 'additionalProperties': False, 'properties': {'city': {'type': 'string'}}},
            {
                'type': 'object',
                'is_omitted': False,
                'required': [],
                'properties': {'city': {'type': 'string', 'description': 'City', 'enum': None, 'is_omitted': False}},
            },
            True,
        ),
        # A local description participates in the comparison.
        (
            {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'Town'}}},
            {'type': 'object', 'properties': {'city': {'type': 'string', 'description': 'City'}}},
            False,
        ),
        ({'type': 'object'}, {'type': 'string'}, False),
        ({'type': 'object', 'required': ['a']}, {'type': 'object', 'required': []}, False),
        (
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'string'}}},
            False,
        ),
        (
            {'type': 'object', 'properties': {'a': {'type': 'string', 'enum': ['x']}}},
            {'type': 'object', 'properties': {'a': {'type': 'string', 'enum': None}}},
            False,
        ),
        (
            {'type': 'array', 'items': {'type': 'string'}},
            {'type': 'array', 'items': {'type': 'integer'}},
            False,
        ),
        ({'type': 'array', 'items': {'type': 'string'}}, {'type': 'array'}, False),
        ({'type': 'array'}, {'type': 'array', 'items': {'type': 'string'}}, False),
    ],
)
def test_schemas_match_against_server_normalization(
    local: dict[str, Any], remote: dict[str, Any], expected: bool
) -> None:
    """Unit-pins the lenient schema comparison against the server's stored-schema normalization.

    A unit test because the public-API route (a full preflight per variant) adds nothing: the
    comparison is definitory behavior derived from live captures of how the server rewrites schemas.
    """
    assert rt_elevenlabs._schemas_match(local, remote) is expected  # pyright: ignore[reportPrivateUsage]


async def test_tool_sync_with_a_bare_agent_response_makes_no_requests() -> None:
    # A `GET agent` response without a `conversation_config` reports no tools at all.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder({'agent_id': AGENT_ID})
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_with_a_promptless_agent_makes_no_requests() -> None:
    # An agent config without a prompt section has no tools to reconcile; with no local tools either,
    # sync has nothing to do.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder({'agent_id': AGENT_ID, 'conversation_config': {'agent': {}}})
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


@pytest.mark.parametrize(
    'agent',
    [
        pytest.param({'agent_id': AGENT_ID}, id='bare'),
        pytest.param({'agent_id': AGENT_ID, 'conversation_config': {'agent': {}}}, id='promptless'),
    ],
)
async def test_tool_sync_creates_local_tools_on_a_toolless_agent(agent: dict[str, Any]) -> None:
    # A local tool against an agent config reporting no tools (bare or promptless) still syncs:
    # the tool is created and the agent's `tool_ids` pointed at it.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder(
        agent,
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing()),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    model = _model(recorder)
    settings = ElevenLabsRealtimeModelSettings(elevenlabs_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[ToolDefinition(name='book_table')], model_settings=settings):
            pass
    repointed = json.loads(recorder.requests[3].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_new']}}}}


async def test_preflight_resolves_attached_tools_when_the_agent_omits_them_inline() -> None:
    # ElevenLabs deprecated inline `prompt.tools`; once `GET` stops returning them, each attached id is
    # fetched from the workspace so `'error'` mode still compares against the real tools. Server-side
    # tools are filtered out like inline ones, and a dangling id (404) is skipped rather than fatal.
    agent = agent_json(tool_ids=['tool_1', 'tool_2', 'tool_gone'])
    del agent['conversation_config']['agent']['prompt']['tools']
    recorder = RestRecorder(
        agent,
        extra={
            ('GET', '/v1/convai/tools/tool_1'): httpx2.Response(
                200, json={'id': 'tool_1', 'tool_config': WEATHER_TOOL_REMOTE}
            ),
            ('GET', '/v1/convai/tools/tool_2'): httpx2.Response(
                200, json={'id': 'tool_2', 'tool_config': {'type': 'webhook', 'name': 'crm_lookup'}}
            ),
            ('GET', '/v1/convai/tools/tool_gone'): httpx2.Response(404, json={'detail': 'Tool not found'}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools/tool_1',
        'GET /v1/convai/tools/tool_2',
        'GET /v1/convai/tools/tool_gone',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_preflight_resolves_attached_tools_when_the_inline_list_is_empty() -> None:
    # An empty inline list next to attached ids is the same missing resolution, and the resolved
    # tools take part in the mismatch check exactly like inline ones.
    recorder = RestRecorder(
        agent_json(tools=[], tool_ids=['tool_1']),
        extra={
            ('GET', '/v1/convai/tools/tool_1'): httpx2.Response(
                200, json={'id': 'tool_1', 'tool_config': dict(WEATHER_TOOL_REMOTE, name='other_tool')}
            ),
        },
    )
    model = _model(recorder)
    with pytest.raises(UserError) as exc_info:
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass  # pragma: no cover
    # Both directions of the mismatch must be reported: the `other_tool` line can only appear if the
    # per-id fetch's config was retained into the check, not merely requested and discarded.
    assert "tool 'get_weather' is not configured on the agent" in str(exc_info.value)
    assert "agent client tool 'other_tool' is not defined by this agent run" in str(exc_info.value)
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools/tool_1',
    ]


async def test_preflight_surfaces_a_failed_attached_tool_fetch() -> None:
    # Anything but a 404 on the per-id fetch is a real API failure and is reported as such.
    recorder = RestRecorder(
        agent_json(tools=[], tool_ids=['tool_1']),
        extra={('GET', '/v1/convai/tools/tool_1'): httpx2.Response(500, text='upstream error')},
    )
    model = _model(recorder)
    with pytest.raises(ModelHTTPError) as exc_info:
        async with _connect(model, tools=[WEATHER_TOOL]):
            pass  # pragma: no cover
    assert exc_info.value.status_code == 500


# --- handshake -----------------------------------------------------------------------------------


async def test_dial_websocket_error_maps_to_realtime_error() -> None:
    model = _model(RestRecorder(agent_json()))
    with failing_connect(websockets.exceptions.InvalidHandshake('bad handshake')):
        with pytest.raises(RealtimeError, match='WebSocket error during realtime handshake'):
            async with _connect(model):
                pass  # pragma: no cover


async def test_dial_os_error_maps_to_realtime_error() -> None:
    model = _model(RestRecorder(agent_json()))
    with failing_connect(OSError('connection refused')):
        with pytest.raises(RealtimeError, match='Could not reach the realtime API'):
            async with _connect(model):
                pass  # pragma: no cover


async def test_handshake_rejects_binary_and_malformed_frames() -> None:
    class BinaryFrame(FakeWebSocket):
        async def recv(self) -> Any:
            return b'\x00'

    model = _model(RestRecorder(agent_json()))
    with patched_connect(BinaryFrame([])):
        with pytest.raises(RealtimeError, match='expected a text frame'):
            async with _connect(model):
                pass  # pragma: no cover

    model = _model(RestRecorder(agent_json()))
    with patched_connect(FakeWebSocket(['not json'])):
        with pytest.raises(RealtimeError, match='received a malformed frame'):
            async with _connect(model):
                pass  # pragma: no cover


async def test_handshake_malformed_known_frames_raise_realtime_error() -> None:
    # A known handshake frame type whose payload fails validation is as malformed as bad JSON and
    # must surface as the typed connect error, not leak a pydantic `ValidationError` (a `ValueError`
    # the connect-error mapping does not catch).
    model = _model(RestRecorder(agent_json()))
    with patched_connect(FakeWebSocket([{'type': 'conversation_initiation_metadata'}])):
        with pytest.raises(RealtimeError, match='received a malformed frame'):
            async with _connect(model):
                pass  # pragma: no cover

    model = _model(RestRecorder(agent_json()))
    ws = FakeWebSocket([{'type': 'ping'}, HANDSHAKE_FRAME])
    with patched_connect(ws):
        with pytest.raises(RealtimeError, match='received a malformed frame'):
            async with _connect(model):
                pass  # pragma: no cover
    # The malformed ping failed the handshake before anything was answered.
    assert len(ws.sent) == 1  # only the initiation payload


async def test_handshake_metadata_without_formats_is_tolerated() -> None:
    # Live handshakes always carried both formats; an absent one skips the format validation
    # defensively rather than failing the connect.
    frame = {
        'type': 'conversation_initiation_metadata',
        'conversation_initiation_metadata_event': {'conversation_id': 'conv_1'},
    }
    model = _model(RestRecorder(agent_json()))
    with patched_connect(FakeWebSocket([frame])):
        async with _connect(model) as connection:
            assert connection.conversation_id == 'conv_1'


async def test_handshake_answers_pings_and_surfaces_error_frames() -> None:
    ws = FakeWebSocket([{'type': 'ping', 'ping_event': {'event_id': 7}}, {'type': 'client_error', 'message': 'nope'}])
    model = _model(RestRecorder(agent_json()))
    with patched_connect(ws):
        with pytest.raises(RealtimeError, match='rejected the conversation'):
            async with _connect(model):
                pass  # pragma: no cover
    assert ws.sent_frames()[1] == {'type': 'pong', 'event_id': 7}
    assert ws.closed


async def test_handshake_timeout_raises_realtime_error() -> None:
    class NeverAnswers(FakeWebSocket):
        async def recv(self) -> str:
            await asyncio.sleep(10)
            raise AssertionError('unreachable')  # pragma: no cover

    ws = NeverAnswers([])
    model = _model(RestRecorder(agent_json()))
    with patched_connect(ws):
        with pytest.raises(RealtimeError, match="timed out waiting for a 'conversation_initiation_metadata'"):
            async with _connect(model, model_settings=ElevenLabsRealtimeModelSettings(handshake_timeout=0.01)):
                pass  # pragma: no cover


async def test_handshake_close_maps_invalid_status_to_model_http_error() -> None:
    headers = websockets.datastructures.Headers()
    headers['Retry-After'] = '7'
    response = websockets.http11.Response(403, 'Forbidden', headers, b'denied')
    model = _model(RestRecorder(agent_json()))
    with failing_connect(websockets.exceptions.InvalidStatus(response)):
        with pytest.raises(ModelHTTPError) as exc_info:
            async with _connect(model):
                pass  # pragma: no cover
    assert exc_info.value.status_code == 403
    assert exc_info.value.body == 'denied'
    # The rejected upgrade's headers ride along, exactly like a REST error's: `retry_after` on a
    # 429 depends on them.
    assert exc_info.value.headers is not None
    assert exc_info.value.headers.get('retry-after') == '7'


async def test_audio_format_mismatch_raises_and_profile_override_fixes_it() -> None:
    # Formats are per-agent; the handshake echoes the actual ones and the adapter validates them
    # against the profile's sample rates, pointing at `profile=` as the escape hatch.
    frame = {
        'type': 'conversation_initiation_metadata',
        'conversation_initiation_metadata_event': {
            'conversation_id': 'conv_1',
            'agent_output_audio_format': 'pcm_24000',
            'user_input_audio_format': 'pcm_24000',
        },
    }
    model = _model(RestRecorder(agent_json()))
    with patched_connect(FakeWebSocket([frame])):
        with pytest.raises(RealtimeError, match='expects mono PCM16 at 16000 Hz'):
            async with _connect(model):
                pass  # pragma: no cover

    adjusted = _model(
        RestRecorder(agent_json()),
        profile={'audio_input_sample_rate': 24000, 'audio_output_sample_rate': 24000},
    )
    with patched_connect(FakeWebSocket([frame])):
        async with _connect(adjusted):
            pass


async def test_ulaw_telephony_format_is_rejected() -> None:
    frame = {
        'type': 'conversation_initiation_metadata',
        'conversation_initiation_metadata_event': {'conversation_id': 'c', 'user_input_audio_format': 'ulaw_8000'},
    }
    model = _model(RestRecorder(agent_json()))
    with patched_connect(FakeWebSocket([frame])):
        with pytest.raises(RealtimeError, match='ulaw_8000'):
            async with _connect(model):
                pass  # pragma: no cover


# --- send ----------------------------------------------------------------------------------------


def _connection(incoming: Sequence[dict[str, Any] | str] = ()) -> tuple[ElevenLabsRealtimeConnection, FakeWebSocket]:
    ws = FakeWebSocket(list(incoming))
    return ElevenLabsRealtimeConnection(ws, conversation_id='conv_1'), ws  # type: ignore[arg-type]


# Every finalized response of a connection that knows its conversation carries the id in
# `provider_details`, the key for the post-hoc cost lookup on the conversations API.
_CONVERSATION = {'conversation_id': 'conv_1'}


async def test_send_audio_frames_user_audio_chunk() -> None:
    connection, ws = _connection()
    await connection.send(BinaryAudio(data=b'\x01\x02', media_type='audio/pcm'))
    assert ws.sent_frames() == [{'user_audio_chunk': base64.b64encode(b'\x01\x02').decode()}]


async def test_send_audio_rejects_non_pcm_media_type() -> None:
    connection, ws = _connection()
    with pytest.raises(UserError, match='require raw PCM audio'):
        await connection.send(BinaryAudio(data=b'RIFF', media_type='audio/wav'))
    assert ws.sent == []


async def test_send_text_frames_user_message() -> None:
    connection, ws = _connection()
    await connection.send('What is the weather?')
    assert ws.sent_frames() == [{'type': 'user_message', 'text': 'What is the weather?'}]


async def test_send_tool_result_frames_client_tool_result() -> None:
    connection, ws = _connection()
    await connection.send(ToolResult(tool_call_id='call_1', output='sunny'))
    assert ws.sent_frames() == [
        {'type': 'client_tool_result', 'tool_call_id': 'call_1', 'result': 'sunny', 'is_error': False}
    ]


async def test_send_tool_result_folds_text_content_and_rejects_media() -> None:
    connection, ws = _connection()
    await connection.send(
        ToolResult(
            tool_call_id='call_1',
            output='sunny',
            content=['and warm', TextContent(content='UV index 5'), CachePoint()],
        )
    )
    assert ws.sent_frames()[-1]['result'] == 'sunny\n\nand warm\n\nUV index 5'

    with pytest.raises(UserError, match='tool results are text-only'):
        await connection.send(
            ToolResult(
                tool_call_id='call_2',
                output='chart',
                content=[BinaryContent(data=b'...', media_type='image/png')],
            )
        )


async def test_manual_turn_verbs_are_rejected() -> None:
    connection, ws = _connection()
    with pytest.raises(UserError, match='does not support CommitAudio input'):
        await connection.send(CommitAudio())
    assert ws.sent == []


# --- event mapping -------------------------------------------------------------------------------


async def _events(incoming: Sequence[dict[str, Any] | str]) -> tuple[list[Any], FakeWebSocket]:
    connection, ws = _connection(incoming)
    return await collect_codec_events(connection), ws


async def test_audio_event_maps_to_audio_delta() -> None:
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events([{'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 1}}])
    assert events == [AudioDelta(data=b'\x00\x01')]


async def test_agent_response_is_the_turn_boundary() -> None:
    # `agent_response` carries the whole response text and closes the turn (verified live: it is the
    # only end-of-turn signal that arrives in every mode with default `client_events`).
    events, _ = await _events(
        [{'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!', 'response_id': 'resp_1'}}]
    )
    assert events == [
        OutputTranscript(text='Hello!', is_final=True),
        ResponseDone(provider_response_id='resp_1', provider_details=_CONVERSATION),
    ]


async def test_a_connection_without_a_conversation_id_leaves_provider_details_unset() -> None:
    # The id is only attached when the handshake reported one; nothing is invented otherwise.
    ws = FakeWebSocket([{'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!'}}])
    connection = ElevenLabsRealtimeConnection(ws)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [OutputTranscript(text='Hello!', is_final=True), ResponseDone()]


async def test_text_output_mode_streams_chat_parts_then_finalizes() -> None:
    # Text-only conversations stream `agent_chat_response_part` deltas before the whole-text
    # `agent_response`; the empty start/stop markers yield nothing.
    ws = FakeWebSocket(
        [
            {'type': 'agent_chat_response_part', 'text_response_part': {'text': '', 'type': 'start'}},
            {'type': 'agent_chat_response_part', 'text_response_part': {'text': 'Hel', 'type': 'delta'}},
            {'type': 'agent_chat_response_part', 'text_response_part': {'text': 'lo!', 'type': 'delta'}},
            {'type': 'agent_chat_response_part', 'text_response_part': {'text': '', 'type': 'stop'}},
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!'}},
        ]
    )
    connection = ElevenLabsRealtimeConnection(ws, text_output=True)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [
        OutputTranscript(text='Hel', output_text=True),
        OutputTranscript(text='lo!', output_text=True),
        OutputTranscript(text='Hello!', is_final=True, output_text=True),
        ResponseDone(),
    ]


async def test_user_transcript_maps_to_final_input_transcript() -> None:
    events, _ = await _events(
        [{'type': 'user_transcript', 'user_transcription_event': {'user_transcript': 'Hi there'}}]
    )
    assert events == [InputTranscript(text='Hi there', is_final=True)]


async def test_response_complete_after_the_boundary_is_dropped() -> None:
    # `agent_response_complete` (text mode only, gated behind `client_events`) trails the
    # `agent_response` that already closed the turn, so it must not emit a second boundary.
    events, _ = await _events(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!'}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}},
        ]
    )
    assert events == [OutputTranscript(text='Hello!', is_final=True), ResponseDone(provider_details=_CONVERSATION)]


async def test_response_complete_closes_an_audio_only_turn() -> None:
    # Defensive boundary for agents whose `client_events` omit `agent_response` itself.
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 1}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}},
        ]
    )
    assert events == [AudioDelta(data=b'\x00\x01'), ResponseDone(provider_details=_CONVERSATION)]


async def test_a_boundary_with_nothing_streamed_is_dropped() -> None:
    events, _ = await _events([{'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}}])
    assert events == []


async def test_interruption_during_generation_closes_the_turn_as_interrupted() -> None:
    # The user barged in while the response was still streaming: the correction closes the turn,
    # keeping the truncated transcript in `provider_details` (the codec cannot shrink the transcript
    # already emitted).
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 1}},
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {
                'type': 'agent_response_correction',
                'agent_response_correction_event': {
                    'original_agent_response': 'Long answer...',
                    'corrected_agent_response': 'Long an',
                },
            },
        ]
    )
    assert events == [
        AudioDelta(data=b'\x00\x01'),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(interrupted=True, provider_details={'corrected_agent_response': 'Long an', **_CONVERSATION}),
    ]


async def test_interruption_during_playback_does_not_taint_the_next_turn() -> None:
    # The common audio-mode case (verified live): synthesis outruns playback, so `agent_response`
    # has already closed the turn when the user barges in. The interruption is surfaced, the
    # correction has nothing left to finalize, and the *next* turn must not be marked interrupted.
    events, _ = await _events(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Long answer...'}},
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {
                'type': 'agent_response_correction',
                'agent_response_correction_event': {
                    'original_agent_response': 'Long answer...',
                    'corrected_agent_response': 'Long an',
                },
            },
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Next turn.'}},
        ]
    )
    assert events == [
        OutputTranscript(text='Long answer...', is_final=True),
        ResponseDone(provider_details=_CONVERSATION),
        RealtimeResponseInterruptedEvent(),
        OutputTranscript(text='Next turn.', is_final=True),
        ResponseDone(provider_details=_CONVERSATION),
    ]


async def test_client_tool_call_maps_to_tool_call_with_serialized_args() -> None:
    # `parameters` arrives as a parsed object; the codec's `args` is a string, so it is re-serialized.
    events, _ = await _events(
        [
            {
                'type': 'client_tool_call',
                'client_tool_call': {
                    'tool_name': 'get_weather',
                    'tool_call_id': 'call_1',
                    'parameters': {'city': 'Berlin'},
                    'expects_response': True,
                },
            }
        ]
    )
    assert events == [ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city":"Berlin"}')]


async def test_fire_and_forget_tool_calls_never_send_a_result() -> None:
    frame: dict[str, Any] = {
        'type': 'client_tool_call',
        'client_tool_call': {
            'tool_name': 'log_event',
            'tool_call_id': 'call_2',
            'parameters': {},
            'expects_response': False,
        },
    }
    connection, ws = _connection()
    [call] = await connection._map_event(frame)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(call, ToolCall)
    # The session settles the call locally; its result must not go back on the wire.
    await connection.send(ToolResult(tool_call_id='call_2', output='done'))
    assert ws.sent == []
    # A later, regular call's result still goes out.
    await connection.send(ToolResult(tool_call_id='call_3', output='done'))
    assert [frame['tool_call_id'] for frame in ws.sent_frames()] == ['call_3']


async def test_client_tool_call_string_parameters_pass_through() -> None:
    # Tolerated defensively: parameters already serialized as a JSON string are forwarded as-is.
    connection, _ = _connection()
    frame = {
        'type': 'client_tool_call',
        'client_tool_call': {
            'tool_name': 'get_weather',
            'tool_call_id': 'call_1',
            'parameters': '{"city": "Berlin"}',
            'expects_response': True,
        },
    }
    events = await connection._map_event(frame)  # pyright: ignore[reportPrivateUsage]
    assert events == [ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city": "Berlin"}')]


async def test_context_usage_maps_to_session_usage_and_model_name() -> None:
    connection, _ = _connection()
    frame = {
        'type': 'context_usage',
        'context_usage_event': {'model': 'gpt-5.2', 'context_tokens': 321, 'context_limit_tokens': 128000},
    }
    events = await connection._map_event(frame)  # pyright: ignore[reportPrivateUsage]
    [usage_event] = events
    assert isinstance(usage_event, SessionUsage)
    assert usage_event.usage.input_tokens == 321
    assert usage_event.usage.details == {'context_limit_tokens': 128000}
    # Verified live: `context_usage` arrives once per user turn, *after* the turn boundary, so it
    # cannot be attributed to a specific model response and stays run-level.
    assert usage_event.response_scoped is False
    assert connection.model_name == 'gpt-5.2'

    # Without a `model` in the payload, the last reported LLM stands.
    bare_frame = {'type': 'context_usage', 'context_usage_event': {'context_tokens': 400}}
    [bare_usage] = await connection._map_event(bare_frame)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(bare_usage, SessionUsage)
    assert bare_usage.usage.input_tokens == 400
    assert bare_usage.usage.details == {}
    assert connection.model_name == 'gpt-5.2'


async def test_ping_is_answered_with_pong_and_yields_nothing() -> None:
    events, ws = await _events([{'type': 'ping', 'ping_event': {'event_id': 42, 'ping_ms': 25}}])
    assert events == []
    assert ws.sent_frames() == [{'type': 'pong', 'event_id': 42}]


async def test_client_error_maps_to_recoverable_session_error() -> None:
    # The AsyncAPI docs wrap the payload as `error_event`; a bare top-level payload is tolerated as
    # a fallback (the frame was never observed live: rejections arrive as WebSocket closes).
    events, _ = await _events([{'type': 'client_error', 'error_event': {'message': 'tool timeout', 'code': 1008}}])
    assert events == [RealtimeSessionErrorEvent(message='ElevenLabs Agents error: tool timeout', recoverable=True)]

    events, _ = await _events([{'type': 'client_error', 'error_name': 'tool_error'}])
    assert events == [RealtimeSessionErrorEvent(message='ElevenLabs Agents error: tool_error', recoverable=True)]


async def test_unknown_and_informational_events_are_ignored() -> None:
    events, _ = await _events(
        [
            {'type': 'vad_score', 'vad_score_event': {'vad_score': 0.9}},
            {'type': 'agent_tool_response', 'agent_tool_response': {'tool_name': 'crm_lookup', 'status': 'success'}},
            {
                'type': 'internal_tentative_agent_response',
                'tentative_agent_response_internal_event': {'tentative_agent_response': 'Hello '},
            },
            {'type': 'brand_new_event', 'brand_new_event_event': {}},
        ]
    )
    assert events == []


async def test_malformed_frame_is_a_recoverable_error() -> None:
    events, _ = await _events(['[1, 2, 3]'])
    assert events == [
        RealtimeSessionErrorEvent(
            message='Failed to parse ElevenLabs Agents event: expected a JSON object frame, got list',
            recoverable=True,
        )
    ]


async def test_malformed_event_payload_is_recoverable_and_leaves_turn_state_alone() -> None:
    # A known frame type with a bad payload must not tear down the session (mirroring the OpenAI
    # provider) and must not mark the response open: the trailing boundary after the bad frame is
    # dropped, while a later well-formed frame streams and finalizes normally.
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 1}},
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 2}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 3}},
        ]
    )
    error, *rest = events
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable
    assert error.message.startswith('Failed to parse ElevenLabs Agents event')
    assert rest == [AudioDelta(data=b'\x00\x01'), ResponseDone(provider_details=_CONVERSATION)]


async def test_corrupted_audio_base64_is_a_recoverable_error() -> None:
    # Without strict decoding, '!!!!' silently decodes to b'' and marks the response open; corrupted
    # provider audio must surface like any other malformed payload and leave turn state alone.
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {'audio_base_64': '!!!!', 'event_id': 1}},
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 2}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 3}},
        ]
    )
    error, *rest = events
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable
    assert error.message.startswith('Failed to parse ElevenLabs Agents event')
    assert rest == [AudioDelta(data=b'\x00\x01'), ResponseDone(provider_details=_CONVERSATION)]


def test_audio_format_validation_rejects_non_ascii_digits() -> None:
    # `str.isdigit` accepts Unicode digits that `int()` rejects; a handshake reporting one must
    # produce the typed error, not leak a raw ValueError.
    with pytest.raises(RealtimeError, match='expects mono PCM16'):
        rt_elevenlabs._validate_audio_format(  # pyright: ignore[reportPrivateUsage]
            'pcm_\u00b2', expected_rate=16000, direction='input', model_name=AGENT_ID
        )


async def test_pong_send_failure_is_fatal() -> None:
    # Answering a ping can hit an already-dropped socket; that is the same fatal condition as a
    # failed receive, so the stream ends on one non-recoverable error instead of leaking a
    # `websockets` exception out of iteration.
    class FailingSend(FakeWebSocket):
        async def send(self, data: str) -> None:
            raise websockets.exceptions.ConnectionClosedError(None, None)

    ws = FailingSend([{'type': 'ping', 'ping_event': {'event_id': 1}}, HANDSHAKE_FRAME])
    connection = ElevenLabsRealtimeConnection(ws)  # type: ignore[arg-type]
    # `collect_codec_events` asserts and strips the trailing non-recoverable close event itself, so
    # nothing else was yielded, and the frame scripted after the ping was never consumed.
    assert await collect_codec_events(connection) == []
    assert len(ws._incoming) == 1  # pyright: ignore[reportPrivateUsage]


async def test_binary_frames_are_skipped() -> None:
    class BinaryThenClose(FakeWebSocket):
        def __init__(self) -> None:
            super().__init__([])
            self._served = False

        async def recv(self) -> Any:
            if not self._served:
                self._served = True
                return b'\x00\x01'
            return await super().recv()

    connection = ElevenLabsRealtimeConnection(BinaryThenClose())  # type: ignore[arg-type]
    assert await collect_codec_events(connection) == []


async def test_server_hangup_is_a_fatal_error_event() -> None:
    connection, _ = _connection()
    events = [event async for event in connection]
    [closed] = events
    assert isinstance(closed, RealtimeSessionErrorEvent)
    assert not closed.recoverable
    assert 'connection closed' in closed.message
