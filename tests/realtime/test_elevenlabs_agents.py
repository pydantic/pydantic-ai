"""Tests for the ElevenLabs Agents realtime provider, all network-free.

The adapter wraps a hosted ElevenLabs agent, so unlike
the sibling provider tests these cover the REST preflight (override-permission checks, tool
sync-or-error) against a mocked HTTP transport, plus the adapter's own codec (event mapping, send
framing, ping/pong, interruption bookkeeping) against hand-built frames mirroring live captures.
The real protocol end-to-end is covered by the cassette-backed `test_elevenlabs_agents_ws.py`.
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import json
import re
import warnings
from collections.abc import AsyncIterator, Generator, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from typing import Any
from unittest import mock

import anyio
import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, NativeToolCallPart, NativeToolReturnPart, PartStartEvent, RequestUsage
from pydantic_ai.exceptions import ModelHTTPError, UnsupportedSettingWarning, UserError
from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    CachePoint,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RetryInfo,
    SpeechPart,
    TextContent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters, get_user_agent
from pydantic_ai.realtime import RealtimeError, RealtimeModelProfile
from pydantic_ai.realtime.codec import (
    AudioDelta,
    CommitAudio,
    InputTranscript,
    OutputTranscript,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
)
from pydantic_ai.realtime.lifecycle import (
    AnswersRevised,
    InputAdded,
    InputLost,
    RealtimeFrame,
    ResponseEnded,
    ResponseStarted,
    StaleEvent,
    UserTurnEnded,
    UserTurnStarted,
)
from pydantic_ai.settings import ToolOrOutput
from pydantic_ai.tools import ToolDefinition

from ..conftest import IsDatetime, try_import
from .ws_helpers import collect_codec_events

with try_import() as imports_successful:
    import websockets
    from websockets.frames import Close

    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.realtime import elevenlabs_agents as rt_elevenlabs
    from pydantic_ai.realtime.elevenlabs_agents import (
        ElevenLabsAgentsRealtimeConnection,
        ElevenLabsAgentsRealtimeModel,
        ElevenLabsAgentsRealtimeModelSettings,
    )

with try_import() as openai_imports_successful:
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='websockets not installed')

AGENT_ID = 'agent_0101test'


def test_elevenlabs_public_exports_are_curated() -> None:
    assert rt_elevenlabs.__all__ == (
        'ElevenLabsAgentsRealtimeModel',
        'ElevenLabsAgentsRealtimeModelSettings',
        'ElevenLabsAgentsRealtimeConnection',
        'ElevenLabsAgentsTTSSettings',
    )


# --- fakes ---------------------------------------------------------------------------------------


class FakeWebSocket:
    """A minimal stand-in for a `websockets` client connection.

    Running out of scripted frames stands in for the server closing the connection normally, raised
    as `ConnectionClosedOK` exactly as `websockets.recv()` reports a 1000 close.
    """

    url: str | None = None

    hold_open = False
    """Whether running out of scripted frames leaves the connection quiet rather than closing it."""

    def __init__(self, incoming: Sequence[dict[str, Any] | str]) -> None:
        self._incoming = [frame if isinstance(frame, str) else json.dumps(frame) for frame in incoming]
        self.sent: list[str] = []
        self.closed = False
        self.drained = asyncio.Event()
        """Set once a connection held open asks for a frame past the scripted ones.

        The connection only asks for its next frame once the session has taken in the previous one, so by then
        every scripted frame has been applied."""

    async def recv(self) -> str:
        if not self._incoming:
            if self.hold_open:
                self.drained.set()
                await asyncio.Event().wait()
            raise websockets.exceptions.ConnectionClosedOK(Close(1000, ''), None)
        return self._incoming.pop(0)

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def sent_frames(self) -> list[dict[str, Any]]:
        return [json.loads(frame) for frame in self.sent]


SOURCE_INFO = {'source': 'unknown', 'version': get_user_agent()}


_NO_CONTEXT_USAGE = pytest.mark.filterwarnings(
    "ignore:The ElevenLabs agent .* doesn't send `context_usage`:UserWarning"
)
"""For a test of an agent configuration that leaves out its `client_events`."""


def agent_json(
    *,
    overrides: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_ids: list[str] | None = None,
    custom_llm_extra_body: bool = False,
    realtime_model: str | None = None,
) -> dict[str, Any]:
    """A `GET /v1/convai/agents/{id}` response holding just what the preflight reads."""
    return {
        'agent_id': AGENT_ID,
        'conversation_config': {
            'agent': {
                'language': 'en',
                'prompt': {'prompt': 'Existing', 'tools': tools or [], 'tool_ids': tool_ids or []},
            },
            'conversation': {'client_events': ['audio', 'agent_response', 'user_transcript', 'context_usage']},
            'realtime_model': realtime_model,
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
        if request.method == 'GET' and request.url.path.endswith('/dependent-agents'):
            # By default, a tool is this agent's alone.
            return httpx2.Response(200, json={'agents': [{'type': 'available', 'id': AGENT_ID}], 'has_more': False})
        if request.method == 'GET' and request.url.path == '/v1/convai/conversation/get-signed-url':
            return httpx2.Response(200, json={'signed_url': f'wss://api.elevenlabs.io/signed?agent_id={AGENT_ID}'})
        raise AssertionError(f'unexpected request: {request.method} {request.url}')  # pragma: no cover

    def request_summaries(self) -> list[str]:
        return [f'{request.method} {request.url.path}' for request in self.requests]


def _model(
    recorder: RestRecorder,
    *,
    settings: ElevenLabsAgentsRealtimeModelSettings | None = None,
    **kwargs: Any,
) -> ElevenLabsAgentsRealtimeModel:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(recorder))
    provider = ElevenLabsProvider(api_key='test-api-key', http_client=client)
    return ElevenLabsAgentsRealtimeModel(AGENT_ID, provider=provider, settings=settings, **kwargs)


HANDSHAKE_FRAME: dict[str, Any] = {
    'type': 'conversation_initiation_metadata',
    'conversation_initiation_metadata_event': {
        'conversation_id': 'conv_1',
        'agent_output_audio_format': 'pcm_16000',
        'user_input_audio_format': 'pcm_16000',
    },
}


def _connect(
    model: ElevenLabsAgentsRealtimeModel,
    *,
    instructions: str | None = None,
    tools: list[ToolDefinition] | None = None,
    model_settings: ElevenLabsAgentsRealtimeModelSettings | None = None,
) -> AbstractAsyncContextManager[ElevenLabsAgentsRealtimeConnection]:
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


def test_profile() -> None:
    """The whole resolved profile, spelled out so a new default cannot be absorbed unnoticed.

    Every fact describes the Agents platform, not the agent: server-side turn-taking (no manual
    turns, no client interruption, no truncation), no seeding channel, 16 kHz audio both ways, and
    an unknown context window because the LLM is chosen on the agent. No genai-prices pinning is
    needed: the profile sets `context_window` explicitly, which skips the lookup.
    """
    assert _model(RestRecorder(agent_json())).profile == RealtimeModelProfile(
        supports_image_input=False,
        image_input_requires_response=False,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_output_truncation=False,
        supports_text_output=True,  # the toggle-gated `text_only` override
        supports_session_seeding=False,
        supports_system_messages=False,
        supports_webrtc=False,
        supports_seeding_images=False,
        supports_seeding_audio=False,
        supports_thinking=False,
        async_tool_call_mode='never',
        supports_async_tool_calls=False,
        supports_tool_return_schema=False,
        emits_input_speech_events=False,
        synthesizes_turn_boundary=False,
        responses_are_requests=True,
        response_usage_covers_context=False,
        audio_input_sample_rate=16000,
        audio_output_sample_rate=16000,
        supported_native_tools=frozenset(),
        context_window=None,
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
def test_rejects_non_elevenlabs_provider() -> None:
    with pytest.raises(UserError, match='requires an `ElevenLabsProvider`'):
        ElevenLabsAgentsRealtimeModel(AGENT_ID, provider=OpenAIProvider(api_key='k'))  # type: ignore[arg-type]


# --- settings the WebSocket cannot honor ---------------------------------------------------------


async def test_reconnect_policy_is_ignored_with_a_warning() -> None:
    """A conversation isn't resumed after a drop, so a `reconnect` policy warns, as an ignored guarantee does."""
    settings = ElevenLabsAgentsRealtimeModelSettings(reconnect={})
    model = _model(RestRecorder(agent_json()), settings=settings)
    with pytest.warns(
        UnsupportedSettingWarning, match='`reconnect` setting is not supported.*A dropped connection ends'
    ):
        model._warn_unsupported_settings(settings, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    # The session still opens; a drop ends it, as without a policy.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    with patched_connect(ws):
        async with _connect(model):
            pass


async def test_turn_detection_config_raises() -> None:
    model = _model(RestRecorder(agent_json()))
    with pytest.raises(UserError, match='own turn-taking server-side'):
        async with _connect(model, model_settings=ElevenLabsAgentsRealtimeModelSettings(turn_detection=False)):
            pass  # pragma: no cover


@pytest.mark.parametrize(
    'settings',
    [
        ElevenLabsAgentsRealtimeModelSettings(turn_detection=True),
        ElevenLabsAgentsRealtimeModelSettings(turn_detection={'sensitivity': 'high'}),
        ElevenLabsAgentsRealtimeModelSettings(input_transcription_model=None),
    ],
    ids=['turn_detection=True', 'turn_detection_knobs', 'transcription=None'],
)
async def test_settings_the_agent_cannot_tune_are_ignored(settings: ElevenLabsAgentsRealtimeModelSettings) -> None:
    """As on a standard model, a knob the agent WebSocket has no surface for is ignored rather than refused."""
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass


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


async def test_an_agent_without_platform_settings_permits_no_override() -> None:
    # An agent configuration that leaves out `platform_settings` has every override toggle off.
    agent = agent_json()
    del agent['platform_settings']
    model = _model(RestRecorder(agent))
    with pytest.raises(UserError, match=r'does not permit the `agent\.prompt\.prompt` conversation override'):
        async with _connect(model, instructions='Be terse.'):
            pass  # pragma: no cover


async def test_an_agent_without_context_usage_reports_warns_once() -> None:
    # An agent left at ElevenLabs' default `client_events` reports no context usage, and its spoken replies end on a
    # quiet stretch: the session works, but says once what turning the reports on gets.
    agent = agent_json()
    del agent['conversation_config']['conversation']
    model = _model(RestRecorder(agent))
    with pytest.warns(UserWarning, match=r"doesn't send `context_usage` reports"):
        with patched_connect(FakeWebSocket([HANDSHAKE_FRAME])):
            async with _connect(model):
                pass
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with patched_connect(FakeWebSocket([HANDSHAKE_FRAME])):
            async with _connect(model):
                pass


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
    assert ws.sent_frames() == [{'type': 'conversation_initiation_client_data', 'source_info': SOURCE_INFO}]
    assert ws.closed


async def test_initiation_payload_builds_every_permitted_override() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    settings = ElevenLabsAgentsRealtimeModelSettings(
        output_modality='text',
        elevenlabs_agents_voice_id='voice_1',
        elevenlabs_agents_language='de',
        elevenlabs_agents_first_message='Hallo!',
        elevenlabs_agents_llm='gpt-5.2',
        elevenlabs_agents_tts={'stability': 0.4, 'speed': 1.1},
        elevenlabs_agents_dynamic_variables={'plan': 'starter'},
        elevenlabs_agents_user_id='user-42',
        elevenlabs_agents_custom_llm_extra_body={'temperature': 0.1},
        elevenlabs_agents_config_override={'conversation': {'max_duration_seconds': 120}},
    )
    model = _model(RestRecorder(agent_json(overrides=ALL_OVERRIDES_ENABLED, custom_llm_extra_body=True)))
    with patched_connect(ws):
        async with _connect(model, instructions='Be helpful.', model_settings=settings):
            pass
    assert ws.sent_frames() == [
        {
            'type': 'conversation_initiation_client_data',
            'source_info': SOURCE_INFO,
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
    cases: list[tuple[ElevenLabsAgentsRealtimeModelSettings, str]] = [
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_voice_id='v'), r'tts\.voice_id'),
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_language='de'), r'agent\.language'),
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_first_message='Hi'), r'agent\.first_message'),
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_llm='gpt-5.2'), r'agent\.prompt\.llm'),
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tts={'stability': 0.4}), r'tts\.stability'),
        (ElevenLabsAgentsRealtimeModelSettings(output_modality='text'), r'conversation\.text_only'),
    ]
    for settings, toggle_pattern in cases:
        model = _model(RestRecorder(agent_json()))
        with pytest.raises(UserError, match=toggle_pattern):
            async with _connect(model, model_settings=settings):
                pass  # pragma: no cover


@pytest.mark.parametrize(
    ('settings', 'setting'),
    [
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_llm='gpt-5.2'), r'agent\.prompt\.llm'),
        (
            ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tts={'model_id': 'eleven_turbo_v2'}),
            r'tts\.model_id',
        ),
    ],
    ids=['llm', 'tts-model'],
)
async def test_an_agent_in_realtime_voice_mode_refuses_the_overrides_it_ignores(
    settings: ElevenLabsAgentsRealtimeModelSettings, setting: str
) -> None:
    """In Realtime voice mode, ElevenLabs picks the LLM and TTS model itself.

    Verified live: the server accepts both overrides and answers with the agent's Realtime model and TTS model all the
    same, so the preflight refuses them rather than let them be ignored. A unit test rather than a recording, because
    the preflight raises before anything is sent.
    """
    model = _model(RestRecorder(agent_json(overrides=ALL_OVERRIDES_ENABLED, realtime_model='eleven_realtime_v1_mini')))
    with pytest.raises(UserError, match=rf"Realtime voice mode \('eleven_realtime_v1_mini'\).*`{setting}`"):
        async with _connect(model, model_settings=settings):
            pass  # pragma: no cover


async def test_an_agent_in_realtime_voice_mode_takes_the_overrides_it_honors() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    settings = ElevenLabsAgentsRealtimeModelSettings(
        elevenlabs_agents_voice_id='voice_1', elevenlabs_agents_tts={'stability': 0.4}
    )
    model = _model(RestRecorder(agent_json(overrides=ALL_OVERRIDES_ENABLED, realtime_model='eleven_realtime_v1_mini')))
    with patched_connect(ws):
        async with _connect(model, instructions='Be helpful.', model_settings=settings):
            pass
    assert ws.sent_frames()[0]['conversation_config_override'] == {
        'agent': {'prompt': {'prompt': 'Be helpful.'}},
        'tts': {'voice_id': 'voice_1', 'stability': 0.4},
    }


async def test_custom_llm_extra_body_requires_its_sibling_toggle() -> None:
    model = _model(RestRecorder(agent_json()))
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_custom_llm_extra_body={'temperature': 0.1})
    with pytest.raises(UserError, match=r'platform_settings\.overrides\.custom_llm_extra_body'):
        async with _connect(model, model_settings=settings):
            pass  # pragma: no cover


@_NO_CONTEXT_USAGE
async def test_bare_agent_response_without_platform_settings_is_tolerated() -> None:
    # `GET agent` responses are mirrored just far enough for the preflight: a response missing the
    # `conversation_config`/`platform_settings` sections means no overrides permitted and no tools.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder({'agent_id': AGENT_ID}))
    with patched_connect(ws):
        async with _connect(model):
            pass
    assert ws.sent_frames() == [{'type': 'conversation_initiation_client_data', 'source_info': SOURCE_INFO}]


@_NO_CONTEXT_USAGE
async def test_bare_agent_response_denies_custom_llm_extra_body() -> None:
    model = _model(RestRecorder({'agent_id': AGENT_ID}))
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_custom_llm_extra_body={'temperature': 0.1})
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
    model = ElevenLabsAgentsRealtimeModel(AGENT_ID, provider=provider)
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
    # Default `elevenlabs_agents_tool_sync='error'`: every difference between the session's tools and the
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


@pytest.mark.parametrize('tool_choice', ['required', ['get_weather']], ids=['required', 'tool-names'])
async def test_forcing_tool_choice_raises_before_any_rest_call(tool_choice: Any) -> None:
    # ElevenLabs has no per-response tool choice, so a forced call would hold for every response, the one
    # after the tool result included; that is refused up front, before the preflight spends a request.
    recorder = RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE]))
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice=tool_choice)
    with pytest.raises(UserError, match="can't force a tool call"):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass  # pragma: no cover
    assert recorder.requests == []


TIME_TOOL = ToolDefinition(
    name='get_time',
    description='Get the time.',
    parameters_json_schema={'type': 'object', 'properties': {}},
)
TIME_TOOL_REMOTE: dict[str, Any] = {
    'type': 'client',
    'name': 'get_time',
    'description': 'Get the time.',
    'expects_response': True,
    'parameters': {'type': 'object', 'properties': {}},
}


@pytest.mark.parametrize(
    'tool_choice,tool_ids',
    [
        ('none', ['tool_mcp']),
        (ToolOrOutput(function_tools=['get_time']), ['tool_time', 'tool_mcp']),
    ],
    ids=['none', 'allow-list'],
)
async def test_tool_choice_restricts_the_attached_tools_per_conversation(tool_choice: Any, tool_ids: list[str]) -> None:
    # The agent keeps every tool of the run attached (it is shared with every conversation), and the
    # conversation is narrowed through the `tool_ids` override: the client tools the choice leaves out are
    # dropped, while ElevenLabs' own server-side tools stay.
    recorder = RestRecorder(
        agent_json(
            overrides={'agent': {'prompt': {'tool_ids': True}}},
            tools=[WEATHER_TOOL_REMOTE, TIME_TOOL_REMOTE],
            tool_ids=['tool_weather', 'tool_time', 'tool_mcp'],
        ),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200,
                json=tools_listing(
                    ('tool_weather', 'client', 'get_weather'),
                    ('tool_time', 'client', 'get_time'),
                    ('tool_mcp', 'mcp', 'crm'),
                ),
            )
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice=tool_choice)
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL, TIME_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert ws.sent_frames() == [
        {
            'type': 'conversation_initiation_client_data',
            'source_info': SOURCE_INFO,
            'conversation_config_override': {'agent': {'prompt': {'tool_ids': tool_ids}}},
        }
    ]


async def test_a_tool_choice_that_leaves_out_nothing_sends_no_override() -> None:
    # `'none'` on an agent whose only tools are ElevenLabs' own narrows nothing: no `tool_ids` override is sent, so
    # the agent needn't permit one.
    recorder = RestRecorder(
        agent_json(tools=[{'type': 'mcp', 'name': 'crm'}], tool_ids=['tool_mcp']),
        extra={('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing(('tool_mcp', 'mcp', 'crm')))},
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice='none', elevenlabs_agents_tool_sync='off')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert ws.sent_frames() == [{'type': 'conversation_initiation_client_data', 'source_info': SOURCE_INFO}]


async def test_tool_choice_restriction_requires_the_tool_ids_override() -> None:
    recorder = RestRecorder(
        agent_json(tools=[WEATHER_TOOL_REMOTE], tool_ids=['tool_weather']),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json=tools_listing(('tool_weather', 'client', 'get_weather'))
            )
        },
    )
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice='none')
    with pytest.raises(
        UserError,
        match=r'does not permit the `agent\.prompt\.tool_ids` conversation override required by the `tool_choice` setting',
    ):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass  # pragma: no cover


async def test_tool_choice_auto_is_the_default_and_accepted() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE])))
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice='auto')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass


async def test_tool_sync_off_trusts_the_agent() -> None:
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='off')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass


async def test_tool_sync_off_hands_undefined_tool_calls_to_the_tool_manager() -> None:
    # `'off'` skips the preflight, so the agent keeps `get_weather` attached while the run defines no
    # tool at all. The call is not filtered in the codec: it reaches the session's tool manager,
    # which answers with the standard unknown-tool error, and both the call and its outcome are
    # visible in history and to consumers, exactly as on every other provider.
    call_frame: dict[str, Any] = {
        'type': 'client_tool_call',
        'client_tool_call': {
            'tool_name': 'get_weather',
            'tool_call_id': 'call_1',
            'parameters': {'city': 'Berlin'},
            'expects_response': True,
        },
    }
    # The agent recovers in speech; `agent_response` closes the turn so the call is finalized into
    # history before the consumer stops.
    reply_frame: dict[str, Any] = {
        'type': 'agent_response',
        'agent_response_event': {'agent_response': 'I cannot look that up right now.'},
    }
    ws = FakeWebSocket([HANDSHAKE_FRAME, call_frame, reply_frame])
    # The session settles tool calls in the background: the socket stays quiet rather than closing at the last frame.
    ws.hold_open = True
    model = _model(RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE])))
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='off')
    agent: Agent[None, str] = Agent()
    events: list[Any] = []
    with patched_connect(ws):
        async with agent.realtime(model, model_settings=settings).session() as session:
            with anyio.fail_after(10):
                async for event in session:  # pragma: no branch
                    events.append(event)
                    if isinstance(event, FunctionToolResultEvent):
                        break

    [result] = [event for event in events if isinstance(event, FunctionToolResultEvent)]
    assert isinstance(result.part, ToolReturnPart)
    assert (result.part.outcome, result.part.retry) == (
        'retried',
        RetryInfo(cause='unknown_tool', attempt=1, max_retries=1),
    )
    assert 'Unknown tool name' in str(result.part.content)
    assert 'get_weather' in str(result.part.content)
    assert [event for event in events if isinstance(event, RealtimeSessionErrorEvent)] == []
    # The call is a `ToolCallPart` in history, and the unknown-tool text went back to the agent as
    # the call's result so it can recover in speech (the codec channel carries no error flag).
    calls = [
        part
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    assert [(call.tool_name, call.tool_call_id) for call in calls] == [('get_weather', 'call_1')]
    [wire_result] = [frame for frame in ws.sent_frames() if frame.get('type') == 'client_tool_result']
    assert wire_result['tool_call_id'] == 'call_1'
    assert 'Unknown tool name' in wire_result['result']


async def test_tool_sync_creates_updates_and_repoints() -> None:
    # `'sync'` mode: create the missing tool, update the differing one (its id resolved through the
    # workspace tool listing, since resolved agent tools don't carry ids), and re-point the agent's
    # `tool_ids` at the synced set behind the untouched server-side tools.
    # A timeout raised on the ElevenLabs side survives the update, which would otherwise reset it.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.', response_timeout_secs=60)
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[new_tool, WEATHER_TOOL, unchanged_tool], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'POST /v1/convai/tools',
        'GET /v1/convai/tools/tool_1/dependent-agents',
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
    patched = json.loads(recorder.requests[4].content)
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
            'response_timeout_secs': 60,
        }
    }
    repointed = json.loads(recorder.requests[5].content)
    assert repointed == {
        'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_9', 'tool_new', 'tool_1', 'tool_5']}}}
    }


@pytest.mark.parametrize(
    ('overrides', 'tool_choice', 'instructions', 'toggle'),
    [
        ({'agent': {'prompt': {'prompt': False}}}, 'auto', 'Be terse.', 'agent.prompt.prompt'),
        ({'agent': {'prompt': {'tool_ids': False}}}, 'none', '', 'agent.prompt.tool_ids'),
    ],
    ids=['instructions', 'tool-choice'],
)
async def test_tool_sync_changes_nothing_when_an_override_is_refused(
    overrides: dict[str, Any], tool_choice: Any, instructions: str, toggle: str
) -> None:
    # The sync changes workspace state, so a conversation refused for a missing override toggle must not get to it.
    recorder = RestRecorder(
        agent_json(overrides=overrides),
        extra={('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing())},
    )
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync', tool_choice=tool_choice)
    with pytest.raises(UserError, match=re.escape(f'`{toggle}`')):
        async with _connect(model, instructions=instructions, tools=[WEATHER_TOOL], model_settings=settings):
            pass  # pragma: no cover
    assert recorder.request_summaries() == [f'GET /v1/convai/agents/{AGENT_ID}']


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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_1/dependent-agents',
        'PATCH /v1/convai/tools/tool_1',
        'GET /v1/convai/conversation/get-signed-url',
    ]


async def test_tool_sync_adopts_a_matching_unattached_workspace_tool() -> None:
    # A sync that failed between its create and the final re-point leaves the created tool in the
    # workspace but unattached; the retry adopts it by exact config match instead of POSTing a
    # duplicate, once the dependent-agents check confirms nothing else uses it.
    recorder = RestRecorder(
        agent_json(),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json={'tools': [{'id': 'tool_orphan', 'tool_config': WEATHER_TOOL_REMOTE}]}
            ),
            ('GET', '/v1/convai/tools/tool_orphan/dependent-agents'): httpx2.Response(
                200, json={'agents': [], 'has_more': False}
            ),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_orphan/dependent-agents',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    repointed = json.loads(recorder.requests[3].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_orphan']}}}}


async def test_tool_sync_does_not_adopt_a_tool_another_agent_depends_on() -> None:
    # "Not attached to this agent" is not "orphaned": a matching tool that another agent depends on
    # is left alone (adopting it would let a later sync's PATCH rewrite it under that agent) and a
    # fresh tool is created instead.
    recorder = RestRecorder(
        agent_json(),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json={'tools': [{'id': 'tool_shared', 'tool_config': WEATHER_TOOL_REMOTE}]}
            ),
            ('GET', '/v1/convai/tools/tool_shared/dependent-agents'): httpx2.Response(
                200, json={'agents': [{'id': 'agent_other', 'name': 'Other agent'}], 'has_more': False}
            ),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_new'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_shared/dependent-agents',
        'POST /v1/convai/tools',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    repointed = json.loads(recorder.requests[4].content)
    assert repointed == {'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_new']}}}}


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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_1/dependent-agents',
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_1/dependent-agents',
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with pytest.raises(UserError, match=r"requires a description for every tool parameter, and 'city' has none"):
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with pytest.raises(UserError) as exc_info:
        async with _connect(model, tools=[WEATHER_TOOL, bare, unexpressable], model_settings=settings):
            pass  # pragma: no cover
    message = str(exc_info.value)
    assert "Cannot sync tool 'set_language'" in message
    assert "'language' has none" in message
    assert "'region' has none" in message
    assert "Cannot sync tool 'add_items'" in message
    assert "'items' uses a JSON Schema feature" in message
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
    with pytest.raises(UserError, match=r"'tags\[\]' has none"):
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
    with pytest.raises(UserError, match=r"parameter 'value' uses a JSON Schema feature"):
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


@_NO_CONTEXT_USAGE
async def test_tool_sync_with_a_bare_agent_response_makes_no_requests() -> None:
    # A `GET agent` response without a `conversation_config` reports no tools at all.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder({'agent_id': AGENT_ID})
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


@_NO_CONTEXT_USAGE
async def test_tool_sync_with_a_promptless_agent_makes_no_requests() -> None:
    # An agent config without a prompt section has no tools to reconcile; with no local tools either,
    # sync has nothing to do.
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    recorder = RestRecorder({'agent_id': AGENT_ID, 'conversation_config': {'agent': {}}})
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]


@_NO_CONTEXT_USAGE
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
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
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
            async with _connect(model, model_settings=ElevenLabsAgentsRealtimeModelSettings(handshake_timeout=0.01)):
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


def _connection(
    incoming: Sequence[dict[str, Any] | str] = (),
) -> tuple[ElevenLabsAgentsRealtimeConnection, FakeWebSocket]:
    ws = FakeWebSocket(list(incoming))
    return ElevenLabsAgentsRealtimeConnection(ws, conversation_id='conv_1'), ws  # type: ignore[arg-type]


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
    # Under an id naming the input, which the reply answering it names in its `in_response_to_ids`.
    assert ws.sent_frames() == [{'type': 'user_message', 'text': 'What is the weather?', 'id': 'pydantic_ai_input_0'}]


async def test_send_text_context_frames_contextual_update() -> None:
    # `session.send(text, respond=False)` reaches the codec as `TextContext`; ElevenLabs' counterpart
    # is `contextual_update`, which the agent reads from its next turn on without interrupting or
    # replying. Its optional `context_id` has no `TextContext` equivalent and is never sent.
    connection, ws = _connection()
    await connection.send(TextContext('The visitor is called Ada.'))
    assert ws.sent_frames() == [{'type': 'contextual_update', 'text': 'The visitor is called Ada.'}]


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


_VOICED = {'type': 'context_usage', 'context_usage_event': {'context_tokens': 10, 'context_limit_tokens': 100}}
"""What arrives with a reply's last audio, on an agent sending `context_usage` (verified live)."""
_VOICED_USAGE = SessionUsage(usage=RequestUsage(input_tokens=10), response_scoped=False, context_window_used=0.1)


async def test_a_reply_ends_with_its_last_audio_not_its_text() -> None:
    # Verified live: `agent_response` carries the whole text once the LLM is done writing, while TTS can still be
    # voicing it; the reply ends with its last audio, which `context_usage` arrives with.
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 1}},
            {
                'type': 'agent_response',
                'agent_response_event': {'agent_response': 'Hello!', 'response_id': 'resp_1', 'event_id': 1},
            },
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 1}},
            _VOICED,
        ]
    )
    assert events == [
        AudioDelta(data=b'\x00\x01'),
        OutputTranscript(text='Hello!', is_final=True, item_id='resp_1'),
        AudioDelta(data=b'\x00\x01'),
        ResponseDone(provider_response_id='resp_1', provider_details=_CONVERSATION),
        _VOICED_USAGE,
    ]


@pytest.mark.parametrize(
    'next_turn',
    [
        {'type': 'user_transcript', 'user_transcription_event': {'user_transcript': 'Next.'}},
        {'type': 'audio', 'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 2}},
    ],
    ids=['user-words', 'next-reply-audio'],
)
async def test_a_reply_ends_when_the_next_turn_begins(next_turn: dict[str, Any]) -> None:
    events, _ = await _events(
        [{'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!', 'event_id': 1}}, next_turn]
    )
    assert events[:2] == [OutputTranscript(text='Hello!', is_final=True), ResponseDone(provider_details=_CONVERSATION)]


async def test_a_reply_ends_after_a_quiet_stretch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without `context_usage`, nothing marks a reply's last audio: a gap where TTS would have sent more is taken
    # as the end.
    monkeypatch.setattr(rt_elevenlabs, '_AUDIO_TAIL_GRACE', 0.01)
    ws = FakeWebSocket([{'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!'}}])
    ws.hold_open = True
    connection = ElevenLabsAgentsRealtimeConnection(ws, conversation_id='conv_1')  # type: ignore[arg-type]
    frames = connection.frames()
    first = await anext(frames)
    second = await anext(frames)
    assert [event for event in [*first, *second] if isinstance(event, (OutputTranscript, ResponseDone))] == [
        OutputTranscript(text='Hello!', is_final=True),
        ResponseDone(provider_details=_CONVERSATION),
    ]


async def test_a_connection_without_a_conversation_id_leaves_provider_details_unset() -> None:
    # The id is only attached when the handshake reported one; nothing is invented otherwise.
    ws = FakeWebSocket([{'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!'}}, _VOICED])
    connection = ElevenLabsAgentsRealtimeConnection(ws)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [OutputTranscript(text='Hello!', is_final=True), ResponseDone(), _VOICED_USAGE]


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
    connection = ElevenLabsAgentsRealtimeConnection(ws, text_output=True)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [
        OutputTranscript(text='Hel', output_text=True),
        OutputTranscript(text='lo!', output_text=True),
        OutputTranscript(text='Hello!', is_final=True, output_text=True),
        ResponseDone(),
    ]


@pytest.mark.parametrize(
    ('text', 'stripped', 'tags'),
    [
        ('[laughs] Hi there! [whispers] Come closer.', 'Hi there! Come closer.', ['laughs', 'whispers']),
        ("[freundlich] Hallo! Wie geht's? [lacht]", "Hallo! Wie geht's?", ['freundlich', 'lacht']),
        ('Oh [strong French accent] bonjour.', 'Oh bonjour.', ['strong French accent']),
        ('Well [sighs], fine.', 'Well, fine.', ['sighs']),
        ('[笑う] こんにちは', 'こんにちは', ['笑う']),
        # Bracketed text that isn't a tag is kept: citations, checkboxes, Markdown links, and placeholders.
        ('See [1] and [the docs](https://example.com).', 'See [1] and [the docs](https://example.com).', []),
        ('- [x] done, signed [Your Name]', '- [x] done, signed [Your Name]', []),
        ('An unclosed [laughs and more', 'An unclosed [laughs and more', []),
        ('No tags at all.', 'No tags at all.', []),
    ],
)
def test_strip_audio_tags(text: str, stripped: str, tags: list[str]) -> None:
    assert rt_elevenlabs._strip_audio_tags(text) == (stripped, tags)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    'text',
    [
        '[laughs] Hi there! [whispers] Come closer.',
        'Hi [laughs]   \n  next [sighs]',
        'See [1] and [the docs](https://example.com) [Your Name] [excited] ok.',
        '  [laughs]  [sighs]  hi [x] [ab](',
    ],
)
def test_streamed_tag_stripping_adds_up_to_the_whole_text_stripped(text: str) -> None:
    # However the text is split into pieces, a tag split across them included, the stripped pieces are the
    # start of the whole text stripped at once, and only held-back text (at most a tag's length) is missing.
    whole, _ = rt_elevenlabs._strip_audio_tags(text)  # pyright: ignore[reportPrivateUsage]
    for first in range(1, len(text)):
        for second in range(first, len(text)):
            stripper = rt_elevenlabs._AudioTagStripper()  # pyright: ignore[reportPrivateUsage]
            streamed = ''.join(stripper.feed(piece) for piece in (text[:first], text[first:second], text[second:]))
            assert whole.startswith(streamed)
            assert len(whole) - len(streamed) <= 52


async def test_audio_tags_are_kept_when_stripping_is_off() -> None:
    ws = FakeWebSocket(
        [
            {'type': 'agent_chat_response_part', 'text_response_part': {'text': '[laughs] Hi', 'type': 'delta'}},
            {'type': 'agent_response', 'agent_response_event': {'agent_response': '[laughs] Hi!'}},
        ]
    )
    connection = ElevenLabsAgentsRealtimeConnection(ws, text_output=True, strip_audio_tags=False)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [
        OutputTranscript(text='[laughs] Hi', output_text=True),
        OutputTranscript(text='[laughs] Hi!', is_final=True, output_text=True),
        ResponseDone(),
    ]

    # Audio mode: the correction of a reply cut off keeps its tags too.
    ws = FakeWebSocket(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': '[laughs] Hi! Listen.'}},
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {
                'type': 'agent_response_correction',
                'agent_response_correction_event': {'corrected_agent_response': '[laughs] Hi! Li'},
            },
        ]
    )
    connection = ElevenLabsAgentsRealtimeConnection(ws, strip_audio_tags=False)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [
        OutputTranscript(text='[laughs] Hi! Listen.', is_final=True),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(interrupted=True, provider_details={'corrected_agent_response': '[laughs] Hi! Li'}),
    ]


async def test_the_strip_audio_tags_setting_reaches_the_connection() -> None:
    model = _model(RestRecorder(agent_json()))
    for settings, strips in [
        (ElevenLabsAgentsRealtimeModelSettings(), True),
        (ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_strip_audio_tags=False), False),
    ]:
        with patched_connect(FakeWebSocket([HANDSHAKE_FRAME])):
            async with _connect(model, model_settings=settings) as connection:
                assert (connection._tag_stripper is not None) is strips  # pyright: ignore[reportPrivateUsage]


async def test_stripped_audio_tags_ride_on_the_response() -> None:
    # On by default. Audio mode: the whole text arrives at once, and the correction of a reply cut off is stripped as well.
    ws = FakeWebSocket(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': '[laughs] Hi! [whispers] Listen.'}},
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {
                'type': 'agent_response_correction',
                'agent_response_correction_event': {'corrected_agent_response': '[laughs] Hi! [whispers] Li'},
            },
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'No tags.'}},
            _VOICED,
        ]
    )
    connection = ElevenLabsAgentsRealtimeConnection(ws)  # type: ignore[arg-type]
    events = await collect_codec_events(connection)
    assert events == [
        OutputTranscript(text='Hi! Listen.', is_final=True),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(
            interrupted=True,
            provider_details={'audio_tags': ['laughs', 'whispers'], 'corrected_agent_response': 'Hi! Li'},
        ),
        OutputTranscript(text='No tags.', is_final=True),
        ResponseDone(),
        _VOICED_USAGE,
    ]


async def test_streamed_text_and_history_agree_on_stripped_audio_tags() -> None:
    # Text-only mode streams the reply in pieces, here with a tag split across two of them: the deltas a
    # consumer renders and the text recorded in history are both stripped, and are the same text.
    def part(text: str, type: str = 'delta') -> dict[str, Any]:
        return {'type': 'agent_chat_response_part', 'text_response_part': {'text': text, 'type': type}}

    ws = FakeWebSocket(
        [
            HANDSHAKE_FRAME,
            part('', 'start'),
            part('[lau'),
            part('ghs] Hi there'),
            part(' [whis'),
            part('pers] see [1].'),
            part('', 'stop'),
            {
                'type': 'agent_response',
                'agent_response_event': {'agent_response': '[laughs] Hi there [whispers] see [1].'},
            },
        ]
    )
    ws.hold_open = True
    model = _model(RestRecorder(agent_json(overrides=ALL_OVERRIDES_ENABLED)))
    settings = ElevenLabsAgentsRealtimeModelSettings(output_modality='text')
    agent: Agent[None, str] = Agent()
    streamed = ''
    with patched_connect(ws):
        async with agent.realtime(model, model_settings=settings).session() as session:
            with anyio.fail_after(10):
                async for event in session:  # pragma: no branch
                    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        streamed += event.delta.content_delta
                    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
                        break
            [response] = [message for message in session.all_messages() if isinstance(message, ModelResponse)]

    assert streamed == 'Hi there see [1].'
    assert response.parts == [TextPart(content='Hi there see [1].')]
    assert response.provider_details == {'audio_tags': ['laughs', 'whispers'], 'conversation_id': 'conv_1'}


async def test_user_transcript_maps_to_final_input_transcript() -> None:
    events, _ = await _events(
        [{'type': 'user_transcript', 'user_transcription_event': {'user_transcript': 'Hi there'}}]
    )
    assert events == [InputTranscript(text='Hi there', is_final=True)]


async def test_response_complete_after_the_boundary_is_dropped() -> None:
    # `agent_response_complete` (gated behind `client_events`) trails the `agent_response` that already closed the
    # turn, so it must not emit a second boundary.
    events, _ = await _events(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Hello!', 'event_id': 2}},
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
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 1}},
        ]
    )
    assert events == [AudioDelta(data=b'\x00\x01'), ResponseDone(provider_details=_CONVERSATION)]


async def test_response_complete_of_an_earlier_turn_leaves_the_reply_under_way_alone() -> None:
    # Verified live: in audio mode it arrives once the turn's audio has played out, long after that turn's reply
    # ended, so a later turn's reply can be under way by then.
    audio = base64.b64encode(b'\x00\x01').decode()
    events, _ = await _events(
        [
            {'type': 'audio', 'audio_event': {'audio_base_64': audio, 'event_id': 2}},
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 1}},
            {
                'type': 'agent_response',
                'agent_response_event': {'agent_response': 'Hi!', 'event_id': 2, 'response_id': 'resp_2'},
            },
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}},
        ]
    )
    assert events == [
        AudioDelta(data=b'\x00\x01'),
        OutputTranscript(text='Hi!', is_final=True, item_id='resp_2'),
        ResponseDone(provider_response_id='resp_2', provider_details=_CONVERSATION),
    ]


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


async def test_interruption_while_the_reply_is_voiced_ends_it_with_what_was_heard() -> None:
    # The reply's text is complete, but its audio is still coming when the user cuts in: the correction ends it.
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
        ]
    )
    assert events == [
        OutputTranscript(text='Long answer...', is_final=True),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(interrupted=True, provider_details={'corrected_agent_response': 'Long an', **_CONVERSATION}),
    ]


async def test_interruption_during_playback_does_not_taint_the_next_turn() -> None:
    # Synthesis outruns playback, so a reply can be voiced in full while the user still hears it. A barge-in
    # then is surfaced, the correction has nothing left to finalize, and the *next* turn must not be marked
    # interrupted.
    events, _ = await _events(
        [
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Long answer...'}},
            _VOICED,
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {
                'type': 'agent_response_correction',
                'agent_response_correction_event': {
                    'original_agent_response': 'Long answer...',
                    'corrected_agent_response': 'Long an',
                },
            },
            {'type': 'agent_response', 'agent_response_event': {'agent_response': 'Next turn.'}},
            _VOICED,
        ]
    )
    assert events == [
        OutputTranscript(text='Long answer...', is_final=True),
        ResponseDone(provider_details=_CONVERSATION),
        _VOICED_USAGE,
        RealtimeResponseInterruptedEvent(),
        OutputTranscript(text='Next turn.', is_final=True),
        ResponseDone(provider_details=_CONVERSATION),
        _VOICED_USAGE,
    ]


# One agent turn with two messages, as recorded live on an agent that speaks before calling a tool: what it says
# before the call, then its answer to the result, each with its own `response_id` under the turn's `event_id`.
_SPOKEN_BEFORE_THE_CALL = [
    {'type': 'audio', 'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 10}},
    {
        'type': 'agent_response',
        'agent_response_event': {
            'agent_response': 'I am now going to check the weather in Berlin for you.',
            'event_id': 10,
            'response_id': 'resp_before_call',
            'in_response_to_ids': ['user_1'],
        },
    },
    {
        'type': 'client_tool_call',
        'client_tool_call': {
            'tool_name': 'get_weather',
            'tool_call_id': 'call_1',
            'parameters': {'city': 'Berlin'},
            'event_id': 10,
            'expects_response': True,
        },
    },
    {
        'type': 'agent_tool_response',
        'agent_tool_response': {'tool_name': 'get_weather', 'tool_call_id': 'call_1', 'event_id': 10},
    },
]


def _correction(response_id: str, original: str, corrected: str) -> dict[str, Any]:
    return {
        'type': 'agent_response_correction',
        'agent_response_correction_event': {
            'original_agent_response': original,
            'corrected_agent_response': corrected,
            'event_id': 10,
            'response_id': response_id,
        },
    }


_CUT_BEFORE_THE_CALL = _correction(
    'resp_before_call', 'I am now going to check the weather in Berlin for you.', 'I am now going to...'
)
"""Recorded live: a barge-in while the message before the call plays cuts it down to what was heard."""


async def test_a_correction_of_an_earlier_message_leaves_the_reply_being_voiced_to_its_own() -> None:
    """A barge-in corrects every message of the turn the user didn't hear in full, each by its `response_id`.

    Recorded live: the message the user was hearing is cut down to what they heard, and the answer after the tool
    call, voiced but never heard, is corrected to `''`. The answer, still being voiced (no `context_usage` marks its
    last audio), ends with its own correction, not with the earlier message's. A unit test rather than a recording:
    the barge-in has to land in the second after the reply's last audio arrives, which a live run can't time.
    """
    answer = 'It is sunny and 21 degrees in Berlin.'
    events, _ = await _events(
        [
            *_SPOKEN_BEFORE_THE_CALL,
            {'type': 'audio', 'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 10}},
            {
                'type': 'agent_response',
                'agent_response_event': {'agent_response': answer, 'event_id': 10, 'response_id': 'resp_answer'},
            },
            {'type': 'interruption', 'interruption_event': {'event_id': 47}},
            _CUT_BEFORE_THE_CALL,
            _correction('resp_answer', answer, ''),
        ]
    )
    assert events == [
        AudioDelta(data=b'\x00\x01'),
        OutputTranscript(
            text='I am now going to check the weather in Berlin for you.', is_final=True, item_id='resp_before_call'
        ),
        ResponseDone(provider_response_id='resp_before_call', provider_details=_CONVERSATION),
        ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city":"Berlin"}', response_usage_follows=True),
        SessionUsage(usage=RequestUsage(), response_scoped=True),
        AudioDelta(data=b'\x00\x01'),
        OutputTranscript(text=answer, is_final=True, item_id='resp_answer'),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(
            interrupted=True,
            provider_response_id='resp_answer',
            provider_details={'corrected_agent_response': '', **_CONVERSATION},
        ),
    ]


async def test_a_correction_of_an_earlier_message_ends_the_streamed_reply_without_its_text() -> None:
    """A barge-in while the answer after a tool call streams (recorded live, with text parts in `client_events`).

    The answer never gets its `agent_response`, and the one correction names the message before the call, so the
    answer ends as interrupted, without that message's text as its own.
    """
    events, _ = await _events(
        [
            *_SPOKEN_BEFORE_THE_CALL,
            {
                'type': 'agent_chat_response_part',
                'text_response_part': {'text': '', 'type': 'start', 'event_id': 10, 'response_id': 'resp_answer'},
            },
            {
                'type': 'agent_chat_response_part',
                'text_response_part': {
                    'text': 'The weather in Berlin is',
                    'type': 'delta',
                    'event_id': 10,
                    'response_id': 'resp_answer',
                },
            },
            {'type': 'interruption', 'interruption_event': {'event_id': 43}},
            _CUT_BEFORE_THE_CALL,
        ]
    )
    assert events[-3:] == [
        OutputTranscript(text='The weather in Berlin is', is_final=False, item_id='resp_answer'),
        RealtimeResponseInterruptedEvent(),
        ResponseDone(interrupted=True, provider_details=_CONVERSATION),
    ]


async def test_corrections_after_the_replies_ended_revise_what_history_recorded() -> None:
    """A barge-in during playback of a turn whose replies have all ended, as recorded live.

    Synthesis outruns playback, so both messages of the turn (what the agent said before calling the tool, and its
    answer to the result) have ended with their last audio by the time the user cuts in. ElevenLabs' corrections say
    what was heard of each: history swaps in copies, in the same positions, of the cut-off message cut down to what
    was heard and of the answer, never heard, with no transcript, each keeping what the agent said. A snapshot of
    history taken before is left as it was.
    """
    answer = 'It is sunny and 21 degrees in Berlin.'
    ws = FakeWebSocket(
        [
            HANDSHAKE_FRAME,
            *_SPOKEN_BEFORE_THE_CALL,
            {'type': 'audio', 'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 10}},
            {
                'type': 'agent_response',
                'agent_response_event': {'agent_response': answer, 'event_id': 10, 'response_id': 'resp_answer'},
            },
            _VOICED,
            {'type': 'interruption', 'interruption_event': {'event_id': 47}},
            _CUT_BEFORE_THE_CALL,
            _correction('resp_answer', answer, ''),
        ]
    )
    ws.hold_open = True
    model = _model(RestRecorder(agent_json(tools=[WEATHER_TOOL_REMOTE])))
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        return f'It is sunny in {city}.'

    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='off')

    before: list[ModelMessage] = []
    with patched_connect(ws):
        async with agent.realtime(model, model_settings=settings).session() as session:
            with anyio.fail_after(10):
                async for event in session:  # pragma: no branch
                    if isinstance(event, RealtimeResponseInterruptedEvent):
                        before = session.all_messages()
                        break
                # The corrections make no event of their own: they only revise history.
                await ws.drained.wait()

    def spoken(messages: list[ModelMessage]) -> list[tuple[str | None, dict[str, Any] | None, str]]:
        return [
            (part.transcript, part.provider_details, message.state)
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, SpeechPart)
        ]

    assert spoken(before) == [
        ('I am now going to check the weather in Berlin for you.', None, 'complete'),
        (answer, None, 'complete'),
    ]
    assert spoken(session.all_messages()) == [
        (
            'I am now going to...',
            {'original_transcript': 'I am now going to check the weather in Berlin for you.'},
            'interrupted',
        ),
        (None, {'original_transcript': answer}, 'interrupted'),
    ]
    assert [type(message) for message in session.all_messages()] == [type(message) for message in before]


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
    # Its response ends once no more calls come (here, the socket closing after it).
    assert events == [
        ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city":"Berlin"}', response_usage_follows=True),
        SessionUsage(usage=RequestUsage()),
    ]


@pytest.mark.parametrize('ended_by', ['quiet', 'next-frame', 'drop'])
async def test_a_replys_parallel_tool_calls_are_one_response(ended_by: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Verified live: a reply's parallel calls come as separate frames, back to back, and nothing follows until their
    # results are in. They are one response, which ends once no more come.
    monkeypatch.setattr(rt_elevenlabs, '_TOOL_CALL_BATCH_WINDOW', 0.01)
    after: list[dict[str, Any]] = {'quiet': [], 'next-frame': [_VOICED], 'drop': []}[ended_by]
    connection, ws = _connection(
        [_tool_call_frame('get_weather'), _tool_call_frame('get_time', tool_call_id='call_2'), *after]
    )
    ws.hold_open = ended_by != 'drop'
    frames = connection.frames()
    seen: list[Any] = []
    while not any(isinstance(event, ResponseEnded) for event in seen):
        seen += await anext(frames)
    started = [event for event in seen if isinstance(event, ResponseStarted)]
    ended = [event for event in seen if isinstance(event, ResponseEnded)]
    calls = [event.tool_call_id for event in seen if isinstance(event, ToolCall)]
    assert (len(started), calls) == (1, ['call_1', 'call_2'])
    assert [(event.status, event.finish_reason) for event in ended] == [('completed', 'tool_call')]


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
    assert events == [
        ToolCall(tool_call_id='call_1', tool_name='get_weather', args='{"city": "Berlin"}', response_usage_follows=True)
    ]


def _tool_call_frame(tool_name: str, *, expects_response: bool = True, tool_call_id: str = 'call_1') -> dict[str, Any]:
    return {
        'type': 'client_tool_call',
        'client_tool_call': {
            'tool_name': tool_name,
            'tool_call_id': tool_call_id,
            'parameters': {},
            'expects_response': expects_response,
        },
    }


async def test_client_tool_calls_are_never_filtered_in_the_codec() -> None:
    # The codec does not know or care which tools the run defines: policy lives in the session's tool
    # manager, so a call to any name maps to a `ToolCall`, with nothing answered on the wire.
    connection, ws = _connection()
    assert await connection._map_event(_tool_call_frame('delete_order')) == [  # pyright: ignore[reportPrivateUsage]
        ToolCall(tool_call_id='call_1', tool_name='delete_order', args='{}', response_usage_follows=True)
    ]
    assert ws.sent == []
    # A fire-and-forget call to an undefined tool is tracked like any other: the tool manager's
    # error result for it stays off the wire too.
    [call] = await connection._map_event(_tool_call_frame('log_event', expects_response=False))  # pyright: ignore[reportPrivateUsage]
    assert isinstance(call, ToolCall)
    await connection.send(ToolResult(tool_call_id='call_1', output='Unknown tool name: log_event'))
    assert ws.sent == []


async def test_context_usage_maps_to_session_usage_and_leaves_model_name_unset() -> None:
    connection, _ = _connection()
    assert connection.model_name is None
    assert connection.context_limit_tokens is None
    frame = {
        'type': 'context_usage',
        'context_usage_event': {'model': 'gpt-5.2', 'context_tokens': 321, 'context_limit_tokens': 128000},
    }
    events = await connection._map_event(frame)  # pyright: ignore[reportPrivateUsage]
    [usage_event] = events
    assert isinstance(usage_event, SessionUsage)
    assert usage_event.usage.input_tokens == 321
    # The context window is a property of the conversation, not a per-turn count: `RunUsage` sums
    # every numeric `details` entry across turns, so it lives on the connection instead.
    assert usage_event.usage.details == {}
    assert connection.context_limit_tokens == 128000
    # Verified live: `context_usage` arrives once per user turn, *after* the turn boundary, so it
    # cannot be attributed to a specific model response and stays run-level.
    assert usage_event.response_scoped is False
    # `context_tokens` is the whole context the LLM holds, so the report is the window fraction too.
    assert usage_event.context_window_used == 321 / 128000
    # The reported pipeline LLM is not adopted as the served model: responses keep the agent id.
    assert connection.model_name is None

    bare_frame = {'type': 'context_usage', 'context_usage_event': {'context_tokens': 400}}
    [bare_usage] = await connection._map_event(bare_frame)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(bare_usage, SessionUsage)
    assert bare_usage.usage.input_tokens == 400
    assert bare_usage.usage.details == {}
    # A report without the limit keeps the last one rather than clearing it, and the fraction is
    # measured against it.
    assert connection.context_limit_tokens == 128000
    assert bare_usage.context_window_used == 400 / 128000


async def test_ping_is_answered_with_pong_and_yields_nothing() -> None:
    events, ws = await _events([{'type': 'ping', 'ping_event': {'event_id': 42, 'ping_ms': 25}}])
    assert events == []
    assert ws.sent_frames() == [{'type': 'pong', 'event_id': 42}]


async def test_client_error_maps_to_recoverable_session_error() -> None:
    # The AsyncAPI docs wrap the payload as `error_event` (the frame was never observed live:
    # rejections arrive as WebSocket closes). Without the wrapper it is a malformed frame like any
    # other, reported as a recoverable parse error rather than guessed at.
    events, _ = await _events([{'type': 'client_error', 'error_event': {'message': 'tool timeout', 'code': 1008}}])
    assert events == [RealtimeSessionErrorEvent(message='ElevenLabs Agents error: tool timeout', recoverable=True)]

    events, _ = await _events([{'type': 'client_error', 'error_event': {'error_name': 'tool_error'}}])
    assert events == [RealtimeSessionErrorEvent(message='ElevenLabs Agents error: tool_error', recoverable=True)]

    events, _ = await _events([{'type': 'client_error', 'error_name': 'tool_error'}])
    [error] = events
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is True
    assert error.message.startswith('Failed to parse ElevenLabs Agents event:')


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
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}},
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
            {'type': 'agent_response_complete', 'agent_response_complete_event': {'event_id': 2}},
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
    connection = ElevenLabsAgentsRealtimeConnection(ws)  # type: ignore[arg-type]
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

    connection = ElevenLabsAgentsRealtimeConnection(BinaryThenClose())  # type: ignore[arg-type]
    assert await collect_codec_events(connection) == []


async def test_server_hangup_is_a_fatal_error_event() -> None:
    connection, _ = _connection()
    events = [event async for event in connection]
    [closed] = events
    assert isinstance(closed, RealtimeSessionErrorEvent)
    assert not closed.recoverable
    assert 'connection closed' in closed.message


# --- Lifecycle ----------------------------------------------------------------------------------------


async def _frames(
    connection: ElevenLabsAgentsRealtimeConnection, ws: FakeWebSocket, incoming: Sequence[dict[str, Any]]
) -> list[RealtimeFrame]:
    """The frames the connection makes of `incoming`, up to (not including) the final hang-up frame."""
    ws._incoming.extend(json.dumps(frame) for frame in incoming)  # pyright: ignore[reportPrivateUsage]
    frames = [frame async for frame in connection.frames()]
    return frames[:-1]


_AUDIO = {'type': 'audio', 'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 1}}


def _agent_response(text: str) -> dict[str, Any]:
    return {
        'type': 'agent_response',
        'agent_response_event': {'agent_response': text, 'event_id': 1, 'response_id': 'resp_1'},
    }


async def test_a_text_turn_is_answered_by_the_response_it_starts() -> None:
    connection, ws = _connection()
    assert await connection.send('Hi') == snapshot([InputAdded(input_id=0)])
    assert await _frames(connection, ws, [_AUDIO, _agent_response('Hello!'), _VOICED]) == snapshot(
        [
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [OutputTranscript(text='Hello!', is_final=True, item_id='resp_1')],
            [
                ResponseDone(provider_response_id='resp_1', provider_details={'conversation_id': 'conv_1'}),
                ResponseEnded(
                    response_id='pydantic_ai_response_1',
                    status='completed',
                    provider_details={'conversation_id': 'conv_1'},
                ),
                SessionUsage(usage=RequestUsage(input_tokens=10), response_scoped=False, context_window_used=0.1),
            ],
        ]
    )


async def test_a_barge_in_leaves_the_truncated_reply_with_the_cut_off_response() -> None:
    # Recorded live: interrupted while the reply is still being generated, the server sends the reply's
    # `agent_response` cut down to what was heard, which closes that response rather than starting one.
    connection, ws = _connection()
    await connection.send('Tell me a story')
    frames = await _frames(
        connection,
        ws,
        [_AUDIO, {'type': 'interruption', 'interruption_event': {'event_id': 3}}, _agent_response('Once...')],
    )
    assert frames == snapshot(
        [
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [RealtimeResponseInterruptedEvent()],
            [
                OutputTranscript(text='Once...', is_final=True, item_id='resp_1'),
                ResponseDone(
                    interrupted=True, provider_response_id='resp_1', provider_details={'conversation_id': 'conv_1'}
                ),
                ResponseEnded(
                    response_id='pydantic_ai_response_1',
                    status='cancelled',
                    provider_details={'conversation_id': 'conv_1'},
                ),
            ],
        ]
    )


async def test_a_barge_in_the_users_words_overtake_still_ends_the_cut_off_response() -> None:
    # Should the user's words overtake the cut-off response's terminal, they are a turn of their own, and the
    # truncated transcript the terminal brings trails the response it ends.
    connection, ws = _connection()
    await connection.send('Tell me a story')
    frames = await _frames(
        connection,
        ws,
        [
            _AUDIO,
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {'type': 'user_transcript', 'user_transcription_event': {'user_transcript': 'Stop.', 'event_id': 3}},
            _agent_response('Once...'),
        ],
    )
    assert frames == snapshot(
        [
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [RealtimeResponseInterruptedEvent()],
            [
                UserTurnStarted(turn_id='pydantic_ai_user_turn_1'),
                UserTurnEnded(turn_id='pydantic_ai_user_turn_1'),
                InputTranscript(text='Stop.', is_final=True),
            ],
            [
                ResponseDone(
                    interrupted=True, provider_response_id='resp_1', provider_details={'conversation_id': 'conv_1'}
                ),
                ResponseEnded(
                    response_id='pydantic_ai_response_1',
                    status='cancelled',
                    provider_details={'conversation_id': 'conv_1'},
                ),
                StaleEvent(event=OutputTranscript(text='Once...', is_final=True, item_id='resp_1')),
            ],
        ]
    )


def _reply(text: str, *, event_id: int, response_id: str, answers: list[str]) -> dict[str, Any]:
    return {
        'type': 'agent_response',
        'agent_response_event': {
            'agent_response': text,
            'event_id': event_id,
            'response_id': response_id,
            'in_response_to_ids': answers,
        },
    }


_GREETING_CUT = _reply('...', event_id=1, response_id='resp_greeting', answers=[])
"""Recorded live: the agent's greeting, cut off by text sent as it started, answers nothing."""


async def test_a_reply_that_names_none_of_the_text_leaves_it_to_the_next() -> None:
    """Text sent just as the greeting starts is taken for what the greeting answers, until the greeting names none.

    Recorded live: the text cuts the greeting off, and the reply after it names the text in its
    `in_response_to_ids`. So the text's reply is the second response, which `wait_for_reply()` waits for.
    """
    connection, ws = _connection()
    assert await connection.send('What is two plus two?') == snapshot([InputAdded(input_id=0)])
    second_audio = {
        'type': 'audio',
        'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 2},
    }
    frames = await _frames(
        connection,
        ws,
        [
            _AUDIO,
            {'type': 'interruption', 'interruption_event': {'event_id': 2}},
            _GREETING_CUT,
            second_audio,
            _reply('Four.', event_id=2, response_id='resp_reply', answers=['pydantic_ai_input_0']),
            _VOICED,
        ],
    )
    assert frames == snapshot(
        [
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [RealtimeResponseInterruptedEvent()],
            [
                AnswersRevised(response_id='pydantic_ai_response_1', released=(0,)),
                OutputTranscript(text='...', is_final=True, item_id='resp_greeting'),
                ResponseDone(
                    interrupted=True,
                    provider_response_id='resp_greeting',
                    provider_details={'conversation_id': 'conv_1'},
                ),
                ResponseEnded(
                    response_id='pydantic_ai_response_1',
                    status='cancelled',
                    provider_details={'conversation_id': 'conv_1'},
                ),
            ],
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_2', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [OutputTranscript(text='Four.', is_final=True, item_id='resp_reply')],
            [
                ResponseDone(provider_response_id='resp_reply', provider_details={'conversation_id': 'conv_1'}),
                ResponseEnded(
                    response_id='pydantic_ai_response_2',
                    status='completed',
                    provider_details={'conversation_id': 'conv_1'},
                ),
                SessionUsage(usage=RequestUsage(input_tokens=10), response_scoped=False, context_window_used=0.1),
            ],
        ]
    )


async def test_an_empty_reply_text_still_revises_the_reply_under_way() -> None:
    """Reply text that says nothing (only audio tags, say) makes no output, but its names revise the reply all the same."""
    connection, ws = _connection()
    await connection.send('Hi')
    frames = await _frames(connection, ws, [_AUDIO, _reply('', event_id=1, response_id='resp_1', answers=[])])
    assert frames == snapshot(
        [
            [
                ResponseStarted(
                    response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False
                ),
                AudioDelta(data=b'\x00\x01'),
            ],
            [
                AnswersRevised(response_id='pydantic_ai_response_1', released=(0,)),
                OutputTranscript(text='', is_final=True, item_id='resp_1'),
            ],
        ]
    )


async def test_a_reply_text_after_another_message_of_the_turn_revises_its_own_response() -> None:
    """A turn's next message ends the one still being voiced before its own text starts a response: the text's names
    are that new response's, and leave the one that ended alone (which still answers the text, as its own text said).
    """
    connection, ws = _connection()
    await connection.send('Hi')
    frames = await _frames(
        connection,
        ws,
        [
            _AUDIO,
            _reply('Let me check.', event_id=1, response_id='resp_a', answers=['pydantic_ai_input_0']),
            _AUDIO,
            _reply('All done.', event_id=1, response_id='resp_b', answers=[]),
        ],
    )
    assert not [event for frame in frames for event in frame if isinstance(event, AnswersRevised)]
    assert frames[-1][:3] == [
        ResponseDone(provider_response_id='resp_a', provider_details={'conversation_id': 'conv_1'}),
        ResponseEnded(
            response_id='pydantic_ai_response_1', status='completed', provider_details={'conversation_id': 'conv_1'}
        ),
        ResponseStarted(response_id='pydantic_ai_response_2', answers=(), basis='inferred', provider_id=False),
    ]


async def test_an_echoed_id_that_does_not_parse_is_ignored() -> None:
    """An id with our prefix that names no input is ignored rather than failing the frame, so the reply still lands."""
    connection, ws = _connection()
    await connection.send('Hi')
    frames = await _frames(
        connection,
        ws,
        [
            _AUDIO,
            _reply(
                'Hello!', event_id=1, response_id='resp_1', answers=['pydantic_ai_input_oops', 'pydantic_ai_input_0']
            ),
        ],
    )
    assert frames[-1] == [OutputTranscript(text='Hello!', is_final=True, item_id='resp_1')]


async def test_a_reply_naming_what_the_user_said_releases_no_text() -> None:
    """A reply that names something besides text (speech, by an id the server made up) may have folded that in after
    the text, naming only it: the text it was taken to answer stays answered, rather than wait for a reply naming it.
    """
    connection, ws = _connection()
    await connection.send('Hi')
    frames = await _frames(
        connection,
        ws,
        [_AUDIO, _reply('Hello!', event_id=1, response_id='resp_1', answers=['3ed477e9-19e0-4d5a-a746-e21315b1c646'])],
    )
    assert frames[-1] == [OutputTranscript(text='Hello!', is_final=True, item_id='resp_1')]


class _PacedWebSocket(FakeWebSocket):
    """A connection held open that hands out its scripted frames a batch at a time, as the test releases them."""

    hold_open = True

    def __init__(self, incoming: Sequence[dict[str, Any] | str]) -> None:
        super().__init__(incoming)
        self._released = asyncio.Event()

    def release(self, frames: Sequence[dict[str, Any]]) -> None:
        self.drained.clear()
        self._incoming.extend(json.dumps(frame) for frame in frames)
        self._released.set()

    async def recv(self) -> str:
        while not self._incoming:
            self._released.clear()
            self.drained.set()
            await self._released.wait()
        return self._incoming.pop(0)


async def test_wait_for_reply_waits_past_a_greeting_the_text_cut_off() -> None:
    """`wait_for_reply()` for text sent as the greeting started ends with the reply that names the text.

    The order of what was sent and received credits the greeting with the text; its `in_response_to_ids` (empty, as
    recorded live) says otherwise, so the wait goes on until the reply that does name it.
    """
    ws = _PacedWebSocket([HANDSHAKE_FRAME])
    model = _model(RestRecorder(agent_json()))
    agent: Agent[None, str] = Agent()
    with patched_connect(ws):
        async with agent.realtime(model).session() as session:
            with anyio.fail_after(10):
                await session.send('What is two plus two?')
                replied = anyio.Event()

                async def wait() -> None:
                    await session.wait_for_reply()
                    replied.set()

                async with anyio.create_task_group() as tg:
                    tg.start_soon(wait)
                    ws.release([_AUDIO, {'type': 'interruption', 'interruption_event': {'event_id': 2}}, _GREETING_CUT])
                    await ws.drained.wait()
                    assert not replied.is_set()
                    ws.release(
                        [
                            {
                                'type': 'audio',
                                'audio_event': {'audio_base_64': base64.b64encode(b'\x00\x01').decode(), 'event_id': 2},
                            },
                            _reply('Four.', event_id=2, response_id='resp_reply', answers=['pydantic_ai_input_0']),
                            _VOICED,
                        ]
                    )
    replies = [message for message in session.all_messages() if isinstance(message, ModelResponse)]
    assert [(reply.provider_response_id, reply.state) for reply in replies] == [
        ('resp_greeting', 'interrupted'),
        ('resp_reply', 'complete'),
    ]


@pytest.mark.parametrize(
    ('answers', 'revision'),
    [
        ([], [AnswersRevised(response_id='pydantic_ai_response_1', released=(0,))]),
        (['pydantic_ai_input_0'], []),
    ],
    ids=['names-none', 'names-the-text'],
)
async def test_a_cut_off_reply_whose_text_trails_its_end_is_revised_before_it_ends(
    answers: list[str], revision: list[AnswersRevised]
) -> None:
    """The user's words overtake the cut-off reply's text, which then trails the end of its response: the revision
    its names make comes ahead of that end, while the reply is still under way."""
    connection, ws = _connection()
    await connection.send('What is two plus two?')
    frames = await _frames(
        connection,
        ws,
        [
            _AUDIO,
            {'type': 'interruption', 'interruption_event': {'event_id': 3}},
            {'type': 'user_transcript', 'user_transcription_event': {'user_transcript': 'Stop.', 'event_id': 3}},
            _reply('...', event_id=1, response_id='resp_1', answers=answers),
        ],
    )
    assert frames[-1] == [
        *revision,
        ResponseDone(interrupted=True, provider_response_id='resp_1', provider_details={'conversation_id': 'conv_1'}),
        ResponseEnded(
            response_id='pydantic_ai_response_1', status='cancelled', provider_details={'conversation_id': 'conv_1'}
        ),
        StaleEvent(event=OutputTranscript(text='...', is_final=True, item_id='resp_1')),
    ]


async def test_a_reply_answers_text_sent_after_it_started_that_it_names() -> None:
    """A reply naming text that reached the agent after it started answers that text too, as do any sent before it.

    Text sent while the agent speaks folds into its next reply, which may name only the latest (verified live).
    """
    ws = FakeWebSocket([_AUDIO])
    ws.hold_open = True
    connection = ElevenLabsAgentsRealtimeConnection(ws, conversation_id='conv_1')  # type: ignore[arg-type]
    await connection.send('Hi')
    frames = connection.frames()
    assert await anext(frames) == snapshot(
        [
            ResponseStarted(response_id='pydantic_ai_response_1', answers=(0,), basis='inferred', provider_id=False),
            AudioDelta(data=b'\x00\x01'),
        ]
    )
    await connection.send('And the weather?')
    await connection.send('In Berlin, please.')
    reply = _reply('Hello! It is sunny in Berlin.', event_id=1, response_id='resp_1', answers=['pydantic_ai_input_2'])
    ws._incoming.append(json.dumps(reply))  # pyright: ignore[reportPrivateUsage]
    assert await anext(frames) == snapshot(
        [
            AnswersRevised(response_id='pydantic_ai_response_1', claimed=(1, 2)),
            OutputTranscript(text='Hello! It is sunny in Berlin.', is_final=True, item_id='resp_1'),
        ]
    )


async def test_a_fire_and_forget_result_is_never_answered() -> None:
    connection, ws = _connection()
    await _frames(
        connection,
        ws,
        [
            {
                'type': 'client_tool_call',
                'client_tool_call': {
                    'tool_name': 'log_event',
                    'tool_call_id': 'call_1',
                    'parameters': {},
                    'expects_response': False,
                },
            }
        ],
    )
    assert await connection.send(ToolResult(tool_call_id='call_1', output='done')) == snapshot(
        [InputAdded(input_id=0), InputLost(input_ids=(0,))]
    )
    assert ws.sent == []


async def test_a_failed_send_is_forgotten() -> None:
    connection, _ = _connection()
    with pytest.raises(UserError):
        await connection.send(CommitAudio())
    # The next input is numbered after the failed one, as the session numbers every `send()`.
    assert await connection.send('Hi') == snapshot([InputAdded(input_id=1)])


async def test_hold_audio_while_queued_is_no_response() -> None:
    # A conversation waiting for a free slot in the workspace's concurrency queue gets hold audio over
    # regular `audio` frames, which is no response of the agent's.
    connection, ws = _connection()
    frames = await _frames(
        connection,
        ws,
        [
            {'type': 'queue_status', 'queue_status_event': {'status': 'waiting'}},
            _AUDIO,
            {'type': 'queue_status', 'queue_status_event': {'status': 'admitted'}},
            _AUDIO,
        ],
    )
    assert frames == snapshot(
        [
            [
                ResponseStarted(response_id='pydantic_ai_response_1', basis='inferred', provider_id=False),
                AudioDelta(data=b'\x00\x01'),
            ]
        ]
    )


async def test_a_queue_timeout_close_says_so() -> None:
    connection, ws = _connection()

    async def recv() -> str:
        raise websockets.exceptions.ConnectionClosedError(Close(4300, 'queue timeout'), None)

    ws.recv = recv
    [frame] = [frame async for frame in connection.frames()]
    [error] = frame
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert not error.recoverable
    assert "timed out in the workspace's concurrency queue" in error.message


async def test_a_dropped_connection_loses_the_response_under_way() -> None:
    # It has no terminal to carry the conversation id, so the end of it does: the id is what a cost lookup needs.
    connection, ws = _connection()
    await connection.send('Hi')
    ws._incoming.append(json.dumps(_AUDIO))  # pyright: ignore[reportPrivateUsage]
    frames = [frame async for frame in connection.frames()]
    assert frames[-1] == snapshot(
        [
            ResponseEnded(
                response_id='pydantic_ai_response_1', status='lost', provider_details={'conversation_id': 'conv_1'}
            ),
            RealtimeSessionErrorEvent(
                message='ElevenLabs Agents connection closed: received 1000 (OK); no close frame sent',
                recoverable=False,
            ),
        ]
    )


async def test_the_greeting_answers_no_microphone_audio() -> None:
    # The agent replies to speech only once it has transcribed it, so the microphone audio streamed before its
    # greeting is no turn the greeting answers.
    connection, ws = _connection()
    await connection.send(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    assert await _frames(connection, ws, [_AUDIO]) == snapshot(
        [
            [
                ResponseStarted(response_id='pydantic_ai_response_1', basis='inferred', provider_id=False),
                AudioDelta(data=b'\x00\x01'),
            ]
        ]
    )


async def test_a_failed_tool_result_is_flagged_as_an_error() -> None:
    connection, ws = _connection()
    await connection.send(ToolResult(tool_call_id='call_1', output='{"error": "service down"}', failed=True))
    assert ws.sent_frames() == [
        {
            'type': 'client_tool_result',
            'tool_call_id': 'call_1',
            'result': '{"error": "service down"}',
            'is_error': True,
        }
    ]


async def test_tool_sync_gives_the_agent_its_own_copy_of_a_shared_tool() -> None:
    # Updating a tool another agent also uses would change that agent's tool too: this agent gets a new one.
    differing = dict(WEATHER_TOOL_REMOTE, description='Stale description.')
    recorder = RestRecorder(
        agent_json(tools=[differing], tool_ids=['tool_1']),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json=tools_listing(('tool_1', 'client', 'get_weather'))),
            ('GET', '/v1/convai/tools/tool_1/dependent-agents'): httpx2.Response(
                200,
                json={'agents': [{'type': 'available', 'id': AGENT_ID}, {'type': 'unknown', 'id': 'agent_other'}]},
            ),
            ('POST', '/v1/convai/tools'): httpx2.Response(200, json={'id': 'tool_own'}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_1/dependent-agents',
        'POST /v1/convai/tools',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert json.loads(recorder.requests[4].content) == {
        'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_own']}}}
    }


async def test_tool_sync_adopts_the_first_unattached_tool_that_matches_and_is_unused() -> None:
    stale: dict[str, Any] = {'id': 'tool_stale', 'tool_config': dict(WEATHER_TOOL_REMOTE, description='Stale.')}
    used: dict[str, Any] = {'id': 'tool_used', 'tool_config': WEATHER_TOOL_REMOTE}
    free: dict[str, Any] = {'id': 'tool_free', 'tool_config': WEATHER_TOOL_REMOTE}
    recorder = RestRecorder(
        agent_json(),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(200, json={'tools': [stale, used, free]}),
            ('GET', '/v1/convai/tools/tool_used/dependent-agents'): httpx2.Response(
                200, json={'agents': [], 'has_more': True}
            ),
            ('GET', '/v1/convai/tools/tool_free/dependent-agents'): httpx2.Response(200, json={'agents': []}),
            ('PATCH', f'/v1/convai/agents/{AGENT_ID}'): httpx2.Response(200, json={'agent_id': AGENT_ID}),
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync')
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert recorder.request_summaries() == [
        f'GET /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/tools',
        'GET /v1/convai/tools/tool_used/dependent-agents',
        'GET /v1/convai/tools/tool_free/dependent-agents',
        f'PATCH /v1/convai/agents/{AGENT_ID}',
        'GET /v1/convai/conversation/get-signed-url',
    ]
    assert json.loads(recorder.requests[4].content) == {
        'conversation_config': {'agent': {'prompt': {'tool_ids': ['tool_free']}}}
    }


async def test_tool_choice_none_leaves_out_client_tools_the_run_does_not_define() -> None:
    # With `'off'`, the agent can have client tools the run doesn't define; `'none'` leaves them out all the same.
    recorder = RestRecorder(
        agent_json(
            overrides={'agent': {'prompt': {'tool_ids': True}}},
            tools=[WEATHER_TOOL_REMOTE],
            tool_ids=['tool_weather'],
        ),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json=tools_listing(('tool_weather', 'client', 'get_weather'))
            )
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice='none', elevenlabs_agents_tool_sync='off')
    with patched_connect(ws):
        async with _connect(model, model_settings=settings):
            pass
    assert ws.sent_frames()[0]['conversation_config_override'] == {'agent': {'prompt': {'tool_ids': []}}}


async def test_an_allow_list_that_leaves_nothing_out_needs_no_override() -> None:
    recorder = RestRecorder(
        agent_json(tools=[WEATHER_TOOL_REMOTE], tool_ids=['tool_weather']),
        extra={
            ('GET', '/v1/convai/tools'): httpx2.Response(
                200, json=tools_listing(('tool_weather', 'client', 'get_weather'))
            )
        },
    )
    ws = FakeWebSocket([HANDSHAKE_FRAME])
    model = _model(recorder)
    settings = ElevenLabsAgentsRealtimeModelSettings(tool_choice=ToolOrOutput(function_tools=['get_weather']))
    with patched_connect(ws):
        async with _connect(model, tools=[WEATHER_TOOL], model_settings=settings):
            pass
    assert 'conversation_config_override' not in ws.sent_frames()[0]


# --- MCP tools ElevenLabs runs itself ------------------------------------------------------------------


def _mcp_call(state: str, **fields: Any) -> dict[str, Any]:
    return {
        'type': 'mcp_tool_call',
        'mcp_tool_call': {
            'service_id': 'crm',
            'tool_call_id': 'mcp_1',
            'tool_name': 'lookup_customer',
            'tool_description': None,
            'parameters': {'email': 'ada@example.com'},
            'timestamp': '2026-10-16T00:00:00Z',
            'state': state,
            **fields,
        },
    }


def _native_parts(frames: list[RealtimeFrame]) -> list[NativeToolCallPart | NativeToolReturnPart]:
    return [
        event.part
        for frame in frames
        for event in frame
        if isinstance(event, PartStartEvent) and isinstance(event.part, (NativeToolCallPart, NativeToolReturnPart))
    ]


async def _next_frames(
    frames: AsyncIterator[RealtimeFrame], ws: FakeWebSocket, incoming: Sequence[dict[str, Any]]
) -> list[RealtimeFrame]:
    """The frames the connection makes of `incoming`, one each, leaving the connection open."""
    ws._incoming.extend(json.dumps(frame) for frame in incoming)  # pyright: ignore[reportPrivateUsage]
    return [await anext(frames) for _ in incoming]


async def test_an_mcp_call_is_recorded_with_its_outcome() -> None:
    connection, ws = _connection()
    frames = await _frames(
        connection,
        ws,
        [
            _mcp_call('loading'),
            _mcp_call('success', result=[{'type': 'text', 'text': 'Ada is a customer.'}]),
            _agent_response('Ada is a customer.'),
            _VOICED,
        ],
    )
    assert _native_parts(frames) == snapshot(
        [
            NativeToolCallPart(
                tool_name='mcp_server:crm',
                args={'action': 'call_tool', 'tool_name': 'lookup_customer', 'tool_args': {'email': 'ada@example.com'}},
                tool_call_id='mcp_1',
                provider_name='elevenlabs',
            ),
            NativeToolReturnPart(
                tool_name='mcp_server:crm',
                content=[{'type': 'text', 'text': 'Ada is a customer.'}],
                tool_call_id='mcp_1',
                timestamp=IsDatetime(),
                provider_name='elevenlabs',
            ),
        ]
    )


@pytest.mark.parametrize('reported', [False, True], ids=['unreported', 'reported-as-failed'])
async def test_an_mcp_call_awaiting_approval_is_denied(reported: bool) -> None:
    # Nothing in a session can approve a call ElevenLabs runs itself, and the agent waits for the answer: it is
    # denied, with a warning, and recorded as denied whether ElevenLabs reports it as failed or not at all, in the
    # response it belongs to.
    connection, ws = _connection()
    frames = connection.frames()
    with pytest.warns(UserWarning, match='cannot do, so the call was denied'):
        first = await _next_frames(frames, ws, [_mcp_call('awaiting_approval', approval_timeout_secs=300)])
    assert ws.sent_frames() == [{'type': 'mcp_tool_approval_result', 'tool_call_id': 'mcp_1', 'is_approved': False}]
    outcome = [_mcp_call('failure', error_message='Rejected by the user.')] if reported else []
    rest = await _next_frames(frames, ws, [*outcome, _agent_response('I cannot look that up.'), _VOICED])
    returns = [part for part in _native_parts(rest) if isinstance(part, NativeToolReturnPart)]
    assert [part.outcome for part in returns] == ['denied']
    assert 'Pydantic AI MCP toolset' in str(returns[0].content)
    ended = [event for frame in [*first, *rest] for event in frame if isinstance(event, ResponseEnded)]
    assert [event.response_id for event in ended] == ['pydantic_ai_response_1']
    # Only the first such call warns.
    await _next_frames(frames, ws, [_mcp_call('awaiting_approval', tool_call_id='mcp_2')])


async def test_an_mcp_call_without_a_server_id_is_named_by_kind_alone() -> None:
    connection, ws = _connection()
    frames = connection.frames()
    ws.hold_open = True
    [frame] = await _next_frames(frames, ws, [_mcp_call('loading', service_id=None)])
    assert [part.tool_name for part in _native_parts([frame])] == ['mcp_server']


async def test_an_mcp_call_denied_before_a_drop_keeps_its_denial() -> None:
    connection, ws = _connection()
    with pytest.warns(UserWarning, match='cannot do, so the call was denied'):
        ws._incoming.append(json.dumps(_mcp_call('awaiting_approval')))  # pyright: ignore[reportPrivateUsage]
        frames = [frame async for frame in connection.frames()]
    returns = [part for part in _native_parts(frames) if isinstance(part, NativeToolReturnPart)]
    assert [(part.tool_call_id, part.outcome) for part in returns] == [('mcp_1', 'denied')]
    ended = [event for frame in frames for event in frame if isinstance(event, ResponseEnded)]
    assert [event.status for event in ended] == ['lost']


async def test_a_failed_mcp_call_is_recorded_as_failed() -> None:
    connection, ws = _connection()
    frames = connection.frames()
    [frame] = await _next_frames(frames, ws, [_mcp_call('failure', error_message='The CRM is down.')])
    returns = [
        event.part
        for event in frame
        if isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolReturnPart)
    ]
    assert [(part.outcome, part.content) for part in returns] == [('failed', 'The CRM is down.')]
