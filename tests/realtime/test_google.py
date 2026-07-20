"""Tests for the Gemini Live realtime provider, all network-free."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, cast

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool, ImageGenerationTool, WebFetchTool, WebSearchTool
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    InputSpeechStartEvent,
    InputTranscript,
    ReconnectedEvent,
    SessionUsageEvent,
    ToolCall,
    ToolResult,
    Transcript,
    TurnCompleteEvent,
    TurnDetection,
)
from pydantic_ai.realtime._base import ImageInput, SessionErrorEvent, TextInput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsSameStr, IsStr, try_import

with try_import() as imports_successful:
    from google.genai import Client, errors as genai_errors, types as genai_types
    from google.genai.live import AsyncSession, ConnectionClosed

    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.realtime import google as rt_google
    from pydantic_ai.realtime.google import (
        AutomaticVAD,
        ContextCompression,
        GoogleRealtimeConnection,
        GoogleRealtimeModel,
        GoogleRealtimeModelSettings,
        MultiSpeaker,
        ReconnectPolicy,
    )

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
]


def _connect(
    model: GoogleRealtimeModel,
    instructions: str,
    *,
    messages: list[ModelMessage] | None = None,
) -> AbstractAsyncContextManager[GoogleRealtimeConnection]:
    return model.connect(
        messages=[*(messages or ()), ModelRequest(parts=[], instructions=instructions)],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


class _RecordingSession:
    """A fake `AsyncSession` that records sends and replays messages turn-by-turn.

    `receive()` mirrors the real SDK: each call yields one turn's messages and then returns; once
    the scripted turns run out it raises (defaulting to `ConnectionClosed`, as the live session does
    when the server closes the socket), so a `while`-loop over `receive()` terminates.
    """

    def __init__(self, turns: list[list[Any]] | None = None, *, close_exc: Exception | None = None) -> None:
        self._turns = list(turns or [])
        self._turn = 0
        self._close_exc = close_exc or ConnectionClosed(None, None)
        self.realtime: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.client_content: list[dict[str, Any]] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime.append(kwargs)

    async def send_client_content(self, *, turns: Any = None, turn_complete: bool = True) -> None:
        self.client_content.append({'turns': turns, 'turn_complete': turn_complete})

    async def send_tool_response(self, *, function_responses: Any) -> None:
        self.tool_responses.append(function_responses)

    async def receive(self) -> AsyncIterator[Any]:
        if self._turn >= len(self._turns):
            raise self._close_exc
        turn = self._turns[self._turn]
        self._turn += 1
        for message in turn:
            yield message


def _conn(session: _RecordingSession) -> GoogleRealtimeConnection:
    return GoogleRealtimeConnection(cast('AsyncSession', session))


# --- helpers -----------------------------------------------------------------


def test_tool_def_to_genai_with_and_without_description() -> None:
    with_desc = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})
    )
    assert with_desc.name == 'get_weather'
    assert with_desc.description == 'Weather'
    assert with_desc.parameters_json_schema == {'type': 'object'}

    without_desc = rt_google._tool_def_to_genai(  # pyright: ignore[reportPrivateUsage]
        ToolDefinition(name='ping', parameters_json_schema={'type': 'object'})
    )
    assert without_desc.description is None


def test_native_tool_web_search_maps_to_google_search() -> None:
    tool = rt_google._native_tool_to_genai(WebSearchTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.google_search is not None


def test_native_tool_web_fetch_maps_to_url_context() -> None:
    tool = rt_google._native_tool_to_genai(WebFetchTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.url_context is not None


def test_native_tool_code_execution_maps_to_code_execution() -> None:
    tool = rt_google._native_tool_to_genai(CodeExecutionTool())  # pyright: ignore[reportPrivateUsage]
    assert tool.code_execution is not None


async def test_agent_realtime_session_rejects_unsupported_native_tool() -> None:
    # A native tool outside Gemini's `supported_native_tools` fails up front with the uniform error,
    # before the Live session connects — the rejection lives in `Agent.realtime_session`, not the mapping.
    agent: Agent[None, str] = Agent()
    with pytest.raises(
        UserError,
        match=r'does not support the ImageGenerationTool native tool\(s\)\. '
        r'Supported native tools: CodeExecutionTool, WebFetchTool, WebSearchTool\.',
    ):
        async with agent.realtime_session(
            model=GoogleRealtimeModel(), capabilities=[NativeTool(ImageGenerationTool())]
        ):
            pass  # pragma: no cover - validation raises before yielding


def test_config_combines_function_and_native_tools() -> None:
    tools = [ToolDefinition(name='f', parameters_json_schema={'type': 'object'})]
    config = GoogleRealtimeModel()._config('hi', tools, None, native_tools=[WebSearchTool()])  # pyright: ignore[reportPrivateUsage]
    assert config.tools[0].function_declarations[0].name == 'f'  # type: ignore[index,union-attr]
    assert config.tools[1].google_search is not None  # type: ignore[index,union-attr]


def test_map_usage_full_and_empty() -> None:
    full = rt_google._map_usage(  # pyright: ignore[reportPrivateUsage]
        genai_types.UsageMetadata(prompt_token_count=10, response_token_count=4, cached_content_token_count=3)
    )
    assert full == RequestUsage(input_tokens=10, output_tokens=4, cache_read_tokens=3)
    assert rt_google._map_usage(genai_types.UsageMetadata()) == RequestUsage()  # pyright: ignore[reportPrivateUsage]


def test_single_ws_user_agent_noop_without_duplicate() -> None:
    # A client whose headers hold fewer than two `user-agent` entries needs no reconciliation: the
    # context manager yields without touching them. A real `GoogleProvider` always adds a capitalized
    # duplicate, so this defensive branch can't be reached through `connect` — hence a direct unit test.
    from types import SimpleNamespace

    headers = {'user-agent': 'solo'}
    client = SimpleNamespace(_api_client=SimpleNamespace(_http_options=SimpleNamespace(headers=headers)))
    with rt_google._single_ws_user_agent(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
        assert headers == {'user-agent': 'solo'}
    assert headers == {'user-agent': 'solo'}


def test_ws_trace_context_injects_and_restores_headers() -> None:
    # `google-genai` forwards the client's HTTP headers as the Live handshake headers, so trace context
    # is injected into them for the connect only, then removed so the shared client's later HTTP
    # requests don't carry a stale `traceparent`. The header dict is the SDK's private one, so this is a
    # direct unit test. (The no-op-without-a-span case is covered by the OpenAI/xAI handshake tests.)
    pytest.importorskip('opentelemetry.sdk')
    from types import SimpleNamespace

    from opentelemetry.sdk.trace import TracerProvider

    headers = {'user-agent': 'solo'}
    client = SimpleNamespace(_api_client=SimpleNamespace(_http_options=SimpleNamespace(headers=headers)))
    tracer = TracerProvider().get_tracer('test')
    with tracer.start_as_current_span('root'):
        with rt_google._ws_trace_context(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
            assert 'traceparent' in headers
        # Injected keys are removed after the handshake; the original headers are untouched.
        assert headers == {'user-agent': 'solo'}


def test_ws_trace_context_noop_without_http_options() -> None:
    # A custom/fake client without the SDK's private HTTP options simply skips injection.
    from types import SimpleNamespace

    client = SimpleNamespace(_api_client=None)
    with rt_google._ws_trace_context(cast('Any', client)):  # pyright: ignore[reportPrivateUsage]
        pass


# --- provider resolution & capabilities --------------------------------------


def test_default_provider_is_google() -> None:
    # The default `'google'` provider reads GOOGLE_API_KEY (set to a placeholder by the autouse fixture).
    model = GoogleRealtimeModel()
    assert isinstance(model.client, Client)


def test_provider_instance_is_reused() -> None:
    provider = GoogleProvider(api_key='k')
    model = GoogleRealtimeModel(provider=provider)
    assert model.client is provider.client


def test_profile() -> None:
    profile = GoogleRealtimeModel().profile
    # Gemini Live has no manual turn control or server-side interruption (automatic VAD only).
    assert (
        profile.get('supports_image_input'),
        profile.get('supports_manual_turn_control'),
        profile.get('supports_interruption'),
        profile.get('supports_session_seeding'),
    ) == (
        True,
        False,
        False,
        True,
    )
    assert profile.get('supported_native_tools') == frozenset({WebSearchTool, WebFetchTool, CodeExecutionTool})


# --- config ------------------------------------------------------------------


def test_config_full() -> None:
    settings = GoogleRealtimeModelSettings(
        max_tokens=256,
        temperature=0.5,
        top_p=0.9,
        voice='Puck',
        google_vad=AutomaticVAD(prefix_padding_ms=200, silence_duration_ms=400),
    )
    model = GoogleRealtimeModel(settings=settings)
    assert model.settings == settings
    tools = [ToolDefinition(name='get_weather', description='Weather', parameters_json_schema={'type': 'object'})]
    config = model._config('Be nice', tools, settings)  # pyright: ignore[reportPrivateUsage]

    assert model.model_name == 'gemini-2.5-flash-native-audio-latest'
    assert config.response_modalities == [genai_types.Modality.AUDIO]
    assert config.system_instruction == 'Be nice'
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Puck'  # type: ignore[union-attr]
    assert config.input_audio_transcription is not None
    assert config.output_audio_transcription is not None
    detection = config.realtime_input_config.automatic_activity_detection  # type: ignore[union-attr]
    assert detection.prefix_padding_ms == 200 and detection.silence_duration_ms == 400  # type: ignore[union-attr]
    assert config.tools[0].function_declarations[0].name == 'get_weather'  # type: ignore[index,union-attr]
    assert config.max_output_tokens == 256
    assert config.temperature == 0.5
    assert config.top_p == 0.9


def test_config_minimal_text_no_transcription_no_vad() -> None:
    model = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(
            output_modality='text', google_input_transcription=False, google_output_transcription=False
        )
    )
    config = model._config('', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config.response_modalities == [genai_types.Modality.TEXT]
    assert config.system_instruction is None  # empty instructions → not set
    assert config.speech_config is None
    assert config.input_audio_transcription is None
    assert config.output_audio_transcription is None
    assert config.realtime_input_config is None
    assert config.tools is None
    assert config.max_output_tokens is None


def test_config_forwards_only_present_model_settings() -> None:
    # `model_settings` is non-empty but carries none of the forwarded fields → all stay unset
    # (`presence_penalty` has no Gemini Live equivalent and is ignored).
    config = GoogleRealtimeModel()._config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, GoogleRealtimeModelSettings()
    )
    assert config.max_output_tokens is None
    assert config.temperature is None
    assert config.top_p is None
    assert config.top_k is None
    assert config.seed is None
    assert config.thinking_config is None
    assert config.media_resolution is None


# --- send --------------------------------------------------------------------


async def test_send_audio() -> None:
    session = _RecordingSession()
    await _conn(session).send(AudioInput(data=b'\x01\x02'))
    blob = session.realtime[0]['audio']
    assert blob.data == b'\x01\x02'
    assert blob.mime_type == 'audio/pcm;rate=16000'


async def test_send_text() -> None:
    # A typed turn is committed with `send_client_content(turn_complete=True)` so the model replies.
    session = _RecordingSession()
    await _conn(session).send(TextInput(text='hello'))
    sent = session.client_content[0]
    assert sent['turn_complete'] is True
    assert sent['turns'].role == 'user'
    assert sent['turns'].parts[0].text == 'hello'


async def test_send_image_as_video_frame() -> None:
    session = _RecordingSession()
    await _conn(session).send(ImageInput(data=b'\xff\xd8', mime_type='image/jpeg'))
    blob = session.realtime[0]['video']
    assert blob.data == b'\xff\xd8'
    assert blob.mime_type == 'image/jpeg'


async def test_send_tool_result_echoes_name() -> None:
    session = _RecordingSession()
    conn = _conn(session)
    # a prior ToolCall populates the call_id -> name map.
    conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[genai_types.FunctionCall(id='c1', name='get_weather', args={})]
            )
        )
    )
    await conn.send(ToolResult(tool_call_id='c1', output='Sunny'))
    response = session.tool_responses[0]
    assert response.id == 'c1'
    assert response.name == 'get_weather'
    assert response.response == {'output': 'Sunny'}


async def test_parallel_id_less_calls_do_not_collide() -> None:
    # Gemini may emit multiple function calls without ids; each must get a distinct internal id so
    # results echo the right name back (Gemini gets `id=None`, which is what it sent).
    session = _RecordingSession()
    conn = _conn(session)
    events = conn._map_message(  # pyright: ignore[reportPrivateUsage]
        genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(name='get_weather', args={}),
                    genai_types.FunctionCall(name='get_time', args={}),
                ]
            )
        )
    )
    call_ids = [e.tool_call_id for e in events if isinstance(e, ToolCall)]
    assert len(set(call_ids)) == 2  # distinct internal ids, no collision

    await conn.send(ToolResult(tool_call_id=call_ids[0], output='Sunny'))
    await conn.send(ToolResult(tool_call_id=call_ids[1], output='Noon'))
    assert [(r.id, r.name, r.response) for r in session.tool_responses] == [
        (None, 'get_weather', {'output': 'Sunny'}),
        (None, 'get_time', {'output': 'Noon'}),
    ]


async def test_send_unsupported_raises() -> None:
    session = _RecordingSession()
    with pytest.raises(NotImplementedError, match='object'):
        await _conn(session).send(object())  # type: ignore[arg-type]


# --- message mapping ---------------------------------------------------------


def test_map_audio_and_text_parts() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(inline_data=genai_types.Blob(data=b'\x01', mime_type='audio/pcm')),
                    genai_types.Part(text='partial'),
                    genai_types.Part(),  # neither audio nor text → produces no event
                ]
            )
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        AudioDelta(data=b'\x01'),
        Transcript(text='partial', is_final=False, output_text=True),
    ]


def test_map_skips_thought_parts() -> None:
    # Native-audio models stream their reasoning as `thought` text next to the spoken answer; it must
    # not leak into the transcript (only the real spoken text becomes a `Transcript`). Kept as a unit
    # test because a cassette can't reliably force a model to think.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(text='**Planning the greeting**', thought=True),
                    genai_types.Part(text='Hello there.'),
                ]
            )
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        Transcript(text='Hello there.', is_final=False, output_text=True)
    ]


def test_map_transcriptions_interrupt_and_turn_complete() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            input_transcription=genai_types.Transcription(text='weather?', finished=True),
            output_transcription=genai_types.Transcription(text='Sunny', finished=False),
            interrupted=True,
            turn_complete=True,
        )
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        InputTranscript(text='weather?', is_final=True),
        Transcript(text='Sunny', is_final=False),
        InputSpeechStartEvent(),
        TurnCompleteEvent(interrupted=False),
    ]


def test_map_tool_call_and_usage() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        tool_call=genai_types.LiveServerToolCall(
            function_calls=[genai_types.FunctionCall(id='c1', name='calc', args={'x': 1})]
        ),
        usage_metadata=genai_types.UsageMetadata(prompt_token_count=7, response_token_count=2),
    )
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        ToolCall(tool_call_id='c1', tool_name='calc', args=json.dumps({'x': 1})),
        SessionUsageEvent(usage=RequestUsage(input_tokens=7, output_tokens=2)),
    ]


def test_map_grounding_and_url_context_to_native_tool_part_events() -> None:
    # Grounding streams native tool parts matching the classic `GoogleModel` shapes exactly (web_search +
    # web_fetch, including a source's `domain` and a fetch's retrieval status). Kept as a unit test because a
    # cassette can't reliably force the model to ground and the recording key only exposes audio-out.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            grounding_metadata=genai_types.GroundingMetadata(
                web_search_queries=['weather rome'],
                grounding_chunks=[
                    genai_types.GroundingChunk(
                        web=genai_types.GroundingChunkWeb(
                            uri='https://example.com', title='Example', domain='example.com'
                        )
                    ),
                    genai_types.GroundingChunk(web=None),  # ignored by `SourcesEvent`: no web chunk
                    genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri=None)),  # ignored: no uri
                ],
            ),
            url_context_metadata=genai_types.UrlContextMetadata(
                url_metadata=[
                    genai_types.UrlMetadata(
                        retrieved_url='https://fetched.example',
                        url_retrieval_status=genai_types.UrlRetrievalStatus.URL_RETRIEVAL_STATUS_SUCCESS,
                    ),
                    genai_types.UrlMetadata(retrieved_url=None),  # ignored by `SourcesEvent`: no url
                ]
            ),
        )
    )
    parts = [
        NativeToolCallPart(
            tool_name='web_search',
            args={'queries': ['weather rome']},
            tool_call_id=IsStr(),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='web_search',
            content=[
                {'domain': 'example.com', 'title': 'Example', 'uri': 'https://example.com'},
                # The `web=None` chunk is dropped; the uri-less one round-trips, matching classic.
                {'domain': None, 'title': None, 'uri': None},
            ],
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
            provider_name='google',
        ),
        NativeToolCallPart(
            tool_name='web_fetch',
            args={'urls': ['https://fetched.example']},
            tool_call_id=IsStr(),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='web_fetch',
            content=[
                {
                    'retrieved_url': 'https://fetched.example',
                    'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                },
                {'retrieved_url': None, 'url_retrieval_status': None},
            ],
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
            provider_name='google',
        ),
    ]
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        event
        for index, part in enumerate(parts)
        for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
    ]


def test_map_code_execution_to_native_tool_parts() -> None:
    # When Gemini Live runs code, the executed code and its result arrive as `executable_code` /
    # `code_execution_result` parts on the model turn. They map to a `NativeToolCallPart` /
    # `NativeToolReturnPart` pair byte-identical to the classic `GoogleModel`'s (tool_name
    # `code_execution`, `args`/`content` from the SDK models' JSON dump), sharing a single `tool_call_id`
    # so the return pairs with its call, and stream as part start/end events. The spoken transcript still
    # comes through as its own `Transcript`. Kept as a unit test because
    # a cassette can't reliably force the model to run code and the recording key only exposes audio-out.
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            model_turn=genai_types.Content(
                parts=[
                    genai_types.Part(
                        executable_code=genai_types.ExecutableCode(
                            code='print(1 + 1)', language=genai_types.Language.PYTHON
                        )
                    ),
                    genai_types.Part(
                        code_execution_result=genai_types.CodeExecutionResult(
                            outcome=genai_types.Outcome.OUTCOME_OK, output='2\n'
                        )
                    ),
                    genai_types.Part(text='The answer is 2.'),
                ]
            )
        )
    )
    parts = [
        NativeToolCallPart(
            tool_name='code_execution',
            args={'code': 'print(1 + 1)', 'language': 'PYTHON'},
            tool_call_id=(code_id := IsSameStr()),
            provider_name='google',
        ),
        NativeToolReturnPart(
            tool_name='code_execution',
            content={'outcome': 'OUTCOME_OK', 'output': '2\n'},
            tool_call_id=code_id,
            timestamp=IsDatetime(),
            provider_name='google',
        ),
    ]
    assert conn._map_message(message) == [  # pyright: ignore[reportPrivateUsage]
        Transcript(text='The answer is 2.', is_final=False, output_text=True),
        *[
            event
            for index, part in enumerate(parts)
            for event in (PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part))
        ],
    ]


def test_map_grounding_absent_yields_no_sources() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            grounding_metadata=genai_types.GroundingMetadata(grounding_chunks=[]),
        )
    )
    assert conn._map_message(message) == []  # pyright: ignore[reportPrivateUsage]


def test_map_empty_message_yields_nothing() -> None:
    conn = _conn(_RecordingSession())
    assert conn._map_message(genai_types.LiveServerMessage()) == []  # pyright: ignore[reportPrivateUsage]


# --- connect -----------------------------------------------------------------


def _turn(text: str) -> genai_types.LiveServerMessage:
    return genai_types.LiveServerMessage(
        server_content=genai_types.LiveServerContent(
            output_transcription=genai_types.Transcription(text=text, finished=True), turn_complete=True
        )
    )


def _fake_client(session: _RecordingSession, captured: dict[str, Any] | None = None) -> Client:
    """A fake `google-genai` client whose `.aio.live.connect(...)` yields `session` (recording `model`/`config`)."""

    class _FakeConnect:
        async def __aenter__(self) -> _RecordingSession:
            return session

        async def __aexit__(self, *exc: object) -> bool:
            return False

    class _Live:
        def connect(self, *, model: str, config: Any) -> _FakeConnect:
            if captured is not None:
                captured['model'] = model
                captured['config'] = config
            return _FakeConnect()

    class _Aio:
        def __init__(self) -> None:
            self.live = _Live()

    class _Client:
        def __init__(self) -> None:
            self.aio = _Aio()

    return cast('Client', _Client())


def _model(session: _RecordingSession, captured: dict[str, Any] | None = None, **kwargs: Any) -> GoogleRealtimeModel:
    """A `GoogleRealtimeModel` whose provider reuses a fake client backed by `session`."""
    return GoogleRealtimeModel(provider=GoogleProvider(client=_fake_client(session, captured)), **kwargs)


async def test_connect_streams_events() -> None:
    # Two turns: `receive()` yields one turn per call, so the connection must loop to serve both
    # (a single `receive()` would stop the session after the first reply).
    session = _RecordingSession([[_turn('hi')], [_turn('bye')]])
    captured: dict[str, Any] = {}
    model = _model(session, captured)
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert captured['model'] == 'gemini-2.5-flash-native-audio-latest'
    # Both turns stream, then the server closes the socket; without a reconnect policy that surfaces a
    # non-recoverable `SessionErrorEvent` before the stream ends (see `test_iter_ends_on_api_error_close`).
    assert events[:4] == [
        Transcript(text='hi', is_final=True),
        TurnCompleteEvent(interrupted=False),
        Transcript(text='bye', is_final=True),
        TurnCompleteEvent(interrupted=False),
    ]
    assert isinstance(events[-1], SessionErrorEvent) and events[-1].recoverable is False


async def test_connect_seeds_message_history() -> None:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SpeechPart,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        UserPromptPart,
    )

    session = _RecordingSession([[_turn('hi')]])

    history = [
        ModelRequest(parts=[SystemPromptPart(content='sys'), UserPromptPart(content='earlier question')]),
        ModelResponse(parts=[TextPart(content='earlier answer'), ToolCallPart(tool_name='t', args='{}')]),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='spoken question')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='spoken answer')]),
    ]
    model = _model(session)
    async with _connect(model, 'x', messages=history) as conn:
        _ = [e async for e in conn]

    # A single seed batch of context turns, sent without `turn_complete` so the model doesn't respond.
    # System parts and tool calls are dropped (text/transcript projection only).
    seeded = session.client_content[0]
    assert seeded['turn_complete'] is False
    turns = seeded['turns']
    assert [(t.role, [p.text for p in t.parts]) for t in turns] == [
        ('user', ['earlier question']),
        ('model', ['earlier answer']),
        ('user', ['spoken question']),
        ('model', ['spoken answer']),
    ]


async def test_connect_seed_drops_non_text_and_textless_turns() -> None:
    # A list-content user prompt is projected to its text (multimodal parts dropped); a response with
    # no projectable text (only a tool call) yields no turn at all.
    from pydantic_ai.messages import ImageUrl, ModelRequest, ModelResponse, ToolCallPart, UserPromptPart

    session = _RecordingSession([[_turn('hi')]])
    history = [
        ModelRequest(parts=[UserPromptPart(content=[ImageUrl(url='https://example.com/a.png'), 'describe this'])]),
        ModelResponse(parts=[ToolCallPart(tool_name='t', args='{}')]),
    ]
    model = _model(session)
    async with _connect(model, 'x', messages=history) as conn:
        _ = [e async for e in conn]

    turns = session.client_content[0]['turns']
    assert [(t.role, [p.text for p in t.parts]) for t in turns] == [('user', ['describe this'])]


async def test_connect_wires_reconnect_only_with_resumption() -> None:
    # reconnect + session resumption → the connection can re-dial.
    on = _model(
        _RecordingSession([[_turn('hi')]]),
        reconnect=ReconnectPolicy(),
        settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True),
    )
    async with _connect(on, 'x') as conn:
        assert conn._dial is not None and conn._reconnect is not None  # pyright: ignore[reportPrivateUsage]
    # reconnect without resumption would lose state → not wired.
    off = _model(_RecordingSession([[_turn('hi')]]), reconnect=ReconnectPolicy())
    async with _connect(off, 'x') as conn:
        assert conn._dial is None and conn._reconnect is None  # pyright: ignore[reportPrivateUsage]


async def test_iter_ends_on_api_error_close() -> None:
    # The SDK surfaces a server-closed socket as an `APIError`; without a reconnect policy iteration
    # should end (not raise) but first surface a non-recoverable `SessionErrorEvent` so callers can tell a
    # dropped connection from a completed turn (mirroring the OpenAI provider).
    session = _RecordingSession([[_turn('hi')]], close_exc=genai_errors.APIError(1011, {'message': 'go away'}))
    events = [e async for e in _conn(session)]
    assert events[:2] == [Transcript(text='hi', is_final=True), TurnCompleteEvent(interrupted=False)]
    assert isinstance(events[-1], SessionErrorEvent) and events[-1].recoverable is False


# --- config: voice / tone / turn-taking knobs --------------------------------


def test_speech_config_voice_and_language() -> None:
    speech = (
        GoogleRealtimeModel(settings=GoogleRealtimeModelSettings(voice='Puck', google_language_code='pl-PL'))
        ._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
        .speech_config
    )
    assert speech is not None
    assert speech.language_code == 'pl-PL'
    assert speech.voice_config.prebuilt_voice_config.voice_name == 'Puck'  # type: ignore[union-attr]
    assert speech.multi_speaker_voice_config is None


def test_speech_config_multi_speaker_overrides_voice() -> None:
    # multi_speaker and voice are mutually exclusive in the API → multi_speaker wins, voice dropped.
    model = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(
            voice='Puck', google_multi_speaker=MultiSpeaker(voices={'Joe': 'Puck', 'Jane': 'Kore'})
        )
    )
    speech = model._config('hi', None, None).speech_config  # pyright: ignore[reportPrivateUsage]
    assert speech is not None
    assert speech.voice_config is None
    speakers = speech.multi_speaker_voice_config.speaker_voice_configs  # type: ignore[union-attr]
    assert [s.speaker for s in speakers] == ['Joe', 'Jane']  # type: ignore[union-attr]
    assert speakers[1].voice_config.prebuilt_voice_config.voice_name == 'Kore'  # type: ignore[union-attr,index]


def test_speech_config_absent_when_unset() -> None:
    assert GoogleRealtimeModel()._config('hi', None, None).speech_config is None  # pyright: ignore[reportPrivateUsage]


def test_realtime_input_full() -> None:
    model = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(
            google_vad=AutomaticVAD(
                disabled=True, start_sensitivity='high', end_sensitivity='low', silence_duration_ms=300
            ),
            google_activity_handling='no_interruption',
            google_turn_coverage='all_video',
        )
    )
    rt = model._config('hi', None, None).realtime_input_config  # pyright: ignore[reportPrivateUsage]
    assert rt is not None
    detection = rt.automatic_activity_detection
    assert detection.disabled is True  # type: ignore[union-attr]
    assert detection.start_of_speech_sensitivity == genai_types.StartSensitivity.START_SENSITIVITY_HIGH  # type: ignore[union-attr]
    assert detection.end_of_speech_sensitivity == genai_types.EndSensitivity.END_SENSITIVITY_LOW  # type: ignore[union-attr]
    assert detection.silence_duration_ms == 300  # type: ignore[union-attr]
    assert rt.activity_handling == genai_types.ActivityHandling.NO_INTERRUPTION
    assert rt.turn_coverage == genai_types.TurnCoverage.TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO


@pytest.mark.parametrize(
    ('sensitivity', 'expected_start', 'expected_end'),
    [
        (
            'low',
            genai_types.StartSensitivity.START_SENSITIVITY_LOW,
            genai_types.EndSensitivity.END_SENSITIVITY_LOW,
        ),
        ('medium', None, None),
        (
            'high',
            genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
            genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
        ),
    ],
)
def test_cross_provider_turn_detection_sensitivity(
    sensitivity: Literal['low', 'medium', 'high'],
    expected_start: genai_types.StartSensitivity | None,
    expected_end: genai_types.EndSensitivity | None,
) -> None:
    config = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(turn_detection=TurnDetection(sensitivity=sensitivity))
    )._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    realtime_input_config = config.realtime_input_config
    assert realtime_input_config is not None
    detection = realtime_input_config.automatic_activity_detection
    assert detection is not None
    assert detection.start_of_speech_sensitivity == expected_start
    assert detection.end_of_speech_sensitivity == expected_end


def test_google_vad_overrides_cross_provider_turn_detection() -> None:
    config = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(
            turn_detection=TurnDetection(sensitivity='high'),
            google_vad=AutomaticVAD(start_sensitivity='low', end_sensitivity='low'),
        )
    )._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    realtime_input_config = config.realtime_input_config
    assert realtime_input_config is not None
    detection = realtime_input_config.automatic_activity_detection
    assert detection is not None
    assert detection.start_of_speech_sensitivity == genai_types.StartSensitivity.START_SENSITIVITY_LOW
    assert detection.end_of_speech_sensitivity == genai_types.EndSensitivity.END_SENSITIVITY_LOW


def test_cross_provider_turn_detection_false_is_rejected() -> None:
    """Gemini has no manual turn controls, so disabling VAD (push-to-talk) fails loudly rather than
    producing an unusable session."""
    model = GoogleRealtimeModel(settings=GoogleRealtimeModelSettings(turn_detection=False))
    with pytest.raises(UserError, match='does not support disabling automatic turn detection'):
        model._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]


def test_realtime_input_absent_when_unset() -> None:
    # no vad, no activity handling, no turn coverage → no realtime input config at all.
    assert GoogleRealtimeModel()._config('hi', None, None).realtime_input_config is None  # pyright: ignore[reportPrivateUsage]


def test_vad_without_sensitivities() -> None:
    # a bare `AutomaticVAD()` sets a detection block but leaves sensitivities/disabled unset.
    rt = (
        GoogleRealtimeModel(settings=GoogleRealtimeModelSettings(google_vad=AutomaticVAD()))
        ._config(  # pyright: ignore[reportPrivateUsage]
            'hi', None, None
        )
        .realtime_input_config
    )
    detection = rt.automatic_activity_detection  # type: ignore[union-attr]
    assert detection.disabled is None  # type: ignore[union-attr]
    assert detection.start_of_speech_sensitivity is None  # type: ignore[union-attr]
    assert detection.end_of_speech_sensitivity is None  # type: ignore[union-attr]


def test_affective_and_proactive_audio() -> None:
    config = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(google_affective_dialog=True, google_proactive_audio=True)
    )._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config.enable_affective_dialog is True
    assert config.proactivity.proactive_audio is True  # type: ignore[union-attr]


def test_affective_and_proactive_default_off() -> None:
    config = GoogleRealtimeModel()._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config.enable_affective_dialog is None
    assert config.proactivity is None


def test_transcription_language_codes() -> None:
    config = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(google_transcription_language_codes=['pl-PL'])
    )._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    assert config.input_audio_transcription.language_codes == ['pl-PL']  # type: ignore[union-attr]
    assert config.output_audio_transcription.language_codes == ['pl-PL']  # type: ignore[union-attr]


def test_context_compression_and_session_resumption() -> None:
    model = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(
            google_context_compression=ContextCompression(trigger_tokens=8000, target_tokens=4000),
            google_enable_session_resumption=True,
        ),
    )
    config = model._config('hi', None, None)  # pyright: ignore[reportPrivateUsage]
    cwc = config.context_window_compression
    assert cwc.trigger_tokens == 8000  # type: ignore[union-attr]
    assert cwc.sliding_window.target_tokens == 4000  # type: ignore[union-attr]
    # resumption requested with no handle on first connect.
    assert config.session_resumption is not None and config.session_resumption.handle is None


def test_session_resumption_passes_handle() -> None:
    config = GoogleRealtimeModel(settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True))._config(  # pyright: ignore[reportPrivateUsage]
        'hi', None, None, resumption_handle='h9'
    )
    assert config.session_resumption.handle == 'h9'  # type: ignore[union-attr]


def test_generation_params_from_model_settings() -> None:
    settings = GoogleRealtimeModelSettings(
        temperature=0.3,
        top_p=0.8,
        top_k=20,
        max_tokens=128,
        seed=7,
        google_thinking_config={'thinking_budget': 100},
        google_video_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_LOW,
    )
    config = GoogleRealtimeModel()._config('hi', None, settings)  # pyright: ignore[reportPrivateUsage]
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.max_output_tokens == 128
    assert config.seed == 7
    assert config.thinking_config.thinking_budget == 100  # type: ignore[union-attr]
    assert config.media_resolution == genai_types.MediaResolution.MEDIA_RESOLUTION_LOW


def test_config_overrides_escape_hatch() -> None:
    model = GoogleRealtimeModel(
        settings=GoogleRealtimeModelSettings(google_config_overrides={'explicit_vad_signal': True})
    )
    assert model._config('hi', None, None).explicit_vad_signal is True  # pyright: ignore[reportPrivateUsage]


# --- reconnect via session resumption ----------------------------------------


def test_map_message_captures_resumption_handle() -> None:
    conn = _conn(_RecordingSession())
    message = genai_types.LiveServerMessage(
        session_resumption_update=genai_types.LiveServerSessionResumptionUpdate(new_handle='h-123', resumable=True)
    )
    assert conn._map_message(message) == []  # pyright: ignore[reportPrivateUsage] # internal state, not an event
    assert conn._resumption_handle == 'h-123'  # pyright: ignore[reportPrivateUsage]


def _dialer(*sessions: _RecordingSession) -> tuple[Any, list[str | None]]:
    """A `dial` that hands out `sessions` in order, then fails — records the handles it was called with."""
    handles: list[str | None] = []
    pending = iter(sessions)

    async def dial(handle: str | None) -> AsyncSession:
        handles.append(handle)
        try:
            return cast('AsyncSession', next(pending))
        except StopIteration:
            raise ConnectionClosed(None, None)

    return dial, handles


async def test_reconnect_resumes_then_gives_up() -> None:
    # s1 drops at once; reconnect resumes into s2 (one turn, then drops); reconnect then runs out.
    s1 = _RecordingSession([])
    s2 = _RecordingSession([[_turn('back')]])
    dial, handles = _dialer(s2)
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect=ReconnectPolicy(base_delay=0.0, max_attempts=2, jitter=False)
    )
    conn._resumption_handle = 'h1'  # pyright: ignore[reportPrivateUsage]
    events = [e async for e in conn]
    assert events[:3] == [
        ReconnectedEvent(),
        Transcript(text='back', is_final=True),
        TurnCompleteEvent(interrupted=False),
    ]
    assert isinstance(events[-1], SessionErrorEvent) and events[-1].recoverable is False
    # reconnect resumed from the stored handle; one success + two failed attempts.
    assert handles == ['h1', 'h1', 'h1']


async def test_reconnect_applies_jitter() -> None:
    s1 = _RecordingSession([])
    dial, _ = _dialer(_RecordingSession([[_turn('hi')]]))
    conn = GoogleRealtimeConnection(
        cast('AsyncSession', s1), dial=dial, reconnect=ReconnectPolicy(base_delay=0.0, max_attempts=1, jitter=True)
    )
    events = [e async for e in conn]
    assert events[0] == ReconnectedEvent()
    assert isinstance(events[-1], SessionErrorEvent)


async def test_connect_reconnect_closes_previous_session() -> None:
    # End-to-end through `connect()`'s own dial: a reconnect must close the previous connection's
    # context manager before opening the next, so they don't accumulate.
    sessions = iter([_RecordingSession([]), _RecordingSession([[_turn('back')]])])
    closed: list[int] = []

    class _SeqConnect:
        def __init__(self, idx: int, session: _RecordingSession) -> None:
            self._idx, self._session = idx, session

        async def __aenter__(self) -> _RecordingSession:
            return self._session

        async def __aexit__(self, *exc: object) -> bool:
            closed.append(self._idx)
            return False

    class _Live:
        def __init__(self) -> None:
            self.n = 0

        def connect(self, *, model: str, config: Any) -> _SeqConnect:
            try:
                session = next(sessions)
            except StopIteration:
                raise ConnectionClosed(None, None)  # out of sessions → reconnect ultimately fails
            cm = _SeqConnect(self.n, session)
            self.n += 1
            return cm

    class _Aio:
        def __init__(self) -> None:
            self.live = _Live()

    class _Client:
        def __init__(self) -> None:
            self.aio = _Aio()

    model = GoogleRealtimeModel(
        provider=GoogleProvider(client=cast('Client', _Client())),
        settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True),
        reconnect=ReconnectPolicy(base_delay=0.0, max_attempts=1, jitter=False),
    )
    async with _connect(model, 'x') as conn:
        events = [e async for e in conn]
    assert events[0] == ReconnectedEvent()
    assert events[1:3] == [Transcript(text='back', is_final=True), TurnCompleteEvent(interrupted=False)]
    assert isinstance(events[-1], SessionErrorEvent)
    # cm0 closed when reconnecting into cm1; cm1 closed when the next reconnect runs out of sessions.
    assert closed == [0, 1]
