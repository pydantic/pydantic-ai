"""Network-free GPT-Live tests: profile, session configuration, and codec translation.

The cassette tests in `test_openai_live_ws.py` cover a real conversation. These pin the parts a recording
can't: the guards that fire before a socket is opened, the pieces of Live's protocol our own frames
have to get right (a cassette matcher can match a recording even after the payload drifts), and the
translation rules that only show up under conditions a recorded call doesn't reliably produce.
"""

from __future__ import annotations as _annotations

import json
from contextlib import contextmanager
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    BinaryImage,
    FilePart,
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import RealtimeError, RealtimeModelProfile, infer_realtime_model
from pydantic_ai.realtime.codec import (
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    ResponseDone,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
    TruncateOutput,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    import websockets
    from openai import AsyncOpenAI
    from openai.types.live import ServerEvent, SessionConfig
    from pydantic import TypeAdapter
    from websockets.frames import Close

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import openai_live as live_module
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel
    from pydantic_ai.realtime.openai_live import (
        AUTO_BACKEND_MODEL,
        OpenAILiveConnection,
        OpenAILiveModel,
        OpenAILiveModelSettings,
        OpenAILiveResponsesDelegation,
        seed_input_items,
    )

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]


@pytest.fixture
def model() -> OpenAILiveModel:
    return OpenAILiveModel('gpt-live-1', provider='openai')


def _config(model: OpenAILiveModel, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault('instructions', '')
    kwargs.setdefault('tools', [])
    kwargs.setdefault('messages', [])
    kwargs.setdefault('settings', OpenAILiveModelSettings())
    return model._session_config(**kwargs)  # pyright: ignore[reportPrivateUsage]


def _connection(**kwargs: Any) -> OpenAILiveConnection:
    """A connection with no socket: every test here drives `_map_event` directly."""
    return OpenAILiveConnection(object(), **kwargs)  # pyright: ignore[reportArgumentType]


_server_events: TypeAdapter[ServerEvent] = TypeAdapter(ServerEvent)


def _event(payload: dict[str, Any]) -> ServerEvent:
    """Parse a raw Live frame the way the connection does, so tests drive real SDK event objects."""
    return _server_events.validate_python(payload)


def test_live_model_names_route_to_the_live_protocol(env: Any) -> None:
    """`gpt-live-*` is served by a different endpoint than the Realtime API, on the same provider."""
    env.set('OPENAI_API_KEY', 'test-key')
    assert isinstance(infer_realtime_model('openai:gpt-live-1'), OpenAILiveModel)
    assert isinstance(infer_realtime_model('openai:gpt-realtime'), OpenAIRealtimeModel)


def test_profile(model: OpenAILiveModel) -> None:
    """Live is far more constrained than the Realtime API, and the profile is what says so."""
    assert model.profile == RealtimeModelProfile(
        supports_image_input=True,
        image_input_requires_response=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_output_truncation=False,
        supports_text_output=False,
        supports_session_seeding=True,
        supports_seeding_images=False,
        supports_seeding_audio=False,
        supports_webrtc=False,
        supports_async_tool_calls=True,
        supports_thinking=False,
        supports_tool_return_schema=False,
        emits_input_speech_events=False,
        # The one model in the repo that infers its turn boundary rather than reading it off the wire.
        synthesizes_turn_boundary=True,
        supported_native_tools=frozenset(),
        audio_input_sample_rate=24000,
        audio_output_sample_rate=24000,
        context_window=None,
    )


def test_websocket_url_is_the_live_endpoint(model: OpenAILiveModel) -> None:
    """Live is a different path on the same host, and carries no model query parameter."""
    assert model._live_url() == snapshot('wss://api.openai.com/v1/live/sessions')  # pyright: ignore[reportPrivateUsage]


def test_azure_is_rejected() -> None:
    """Azure does not serve GPT-Live, so pointing at it fails with a pointer to what does."""
    provider = AzureProvider(azure_endpoint='https://example.openai.azure.com', api_key='k', api_version='v')
    with pytest.raises(UserError, match='Azure OpenAI does not serve GPT-Live'):
        OpenAILiveModel('gpt-live-1', provider=provider)


def test_agent_instructions_and_tools_configure_the_backend(model: OpenAILiveModel) -> None:
    """The agent describes the work, so it configures the backend, not the spoken conversation.

    A cassette would match this recording even if the payload drifted, so the split is pinned here.
    """
    tool = ToolDefinition(name='lookup', description='Look something up.', parameters_json_schema={'type': 'object'})
    config = _config(model, instructions='You look things up.', tools=[tool])

    assert config['delegation'] == snapshot(
        {
            'type': 'responses',
            'responses': {
                'model': AUTO_BACKEND_MODEL,
                'instructions': 'You look things up.',
                'tools': [
                    {
                        'type': 'function',
                        'name': 'lookup',
                        'parameters': {'type': 'object'},
                        'description': 'Look something up.',
                    }
                ],
            },
        }
    )
    # The spoken prompt is separate, and defaults rather than inheriting the agent's.
    assert 'You look things up.' not in config['instructions']


def test_delegation_settings_reach_the_backend(model: OpenAILiveModel) -> None:
    settings = OpenAILiveModelSettings(
        openai_live_delegation=OpenAILiveResponsesDelegation(
            model='gpt-5.6-luna',
            instructions='Prefer metric units.',
            max_output_tokens=256,
            parallel_tool_calls=False,
            reasoning_effort='low',
            verbosity='low',
            service_tier='priority',
        ),
        openai_voice='cedar',
        openai_live_instructions='Speak slowly.',
        openai_live_store=True,
    )
    config = _config(model, instructions='Be accurate.', settings=settings)

    assert config['delegation']['responses'] == snapshot(
        {
            'model': 'gpt-5.6-luna',
            # The agent's instructions come first; the setting appends to them.
            'instructions': 'Be accurate.\n\nPrefer metric units.',
            'max_output_tokens': 256,
            'parallel_tool_calls': False,
            'service_tier': 'priority',
            'reasoning': {'effort': 'low'},
            'text': {'verbosity': 'low'},
        }
    )
    assert config['instructions'] == 'Speak slowly.'
    assert config['audio']['output'] == {'voice': 'cedar'}
    assert config['store'] is True


@pytest.mark.parametrize(
    'setting,value',
    [('turn_detection', False), ('max_tokens', 100), ('input_transcription_model', 'gpt-transcribe')],
)
async def test_unsupported_settings_raise_before_connecting(model: OpenAILiveModel, setting: str, value: Any) -> None:
    """A setting Live cannot honor is a stated requirement, so it fails rather than being ignored."""
    with pytest.raises(UserError, match='GPT-Live does not support'):
        async with model.connect(
            messages=[],
            model_settings={setting: value},  # pyright: ignore[reportArgumentType]
            model_request_parameters=ModelRequestParameters(),
        ):
            pass  # pragma: no cover


@pytest.mark.parametrize('verb', [CommitAudio(), ClearAudio(), CancelResponse(), TruncateOutput(audio_end_ms=10)])
async def test_turn_control_is_rejected(verb: Any) -> None:
    """Live owns turn-taking, so every manual turn verb fails rather than being dropped.

    `CreateResponse` is the exception: the session only sends one right after an image, and on Live it
    runs the delegated backend on that image.
    """
    with pytest.raises(UserError, match='drives turn-taking itself'):
        await _connection().send(verb)


async def test_an_image_goes_to_the_backend_that_runs_on_it() -> None:
    """The voice model sees no images; the delegated backend does, as ordinary Responses input."""
    sent: list[dict[str, Any]] = []

    class _Sink(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)

    connection = _Sink(object())  # pyright: ignore[reportArgumentType]
    await connection.send(BinaryImage(data=b'png', media_type='image/png'))
    await connection.send(CreateResponse())

    assert sent == snapshot(
        [
            {
                'type': 'response.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,cG5n'}],
                },
            },
            {'type': 'response.create'},
        ]
    )


async def test_an_image_needs_respond_true(model: OpenAILiveModel) -> None:
    """Queued as context alone, an image goes unseen: Live's voice model answers questions about it blind."""
    started = json.dumps({'type': 'session.started', 'event_id': 'e', 'session': {'id': 's', 'model': 'gpt-live-1'}})
    ws = _FakeWebSocket([started])
    image = BinaryImage(data=b'png', media_type='image/png')

    with _patched_connect(ws):
        async with Agent().realtime(model).session() as session:
            with pytest.raises(UserError, match='only takes an image to respond to it'):
                await session.send(image)
            # Without manual turn control, which Live doesn't have.
            await session.send(image, respond=True)

    assert [json.loads(frame)['type'] for frame in ws.sent[1:]] == ['response.item.create', 'response.create']


def test_seeding_projects_history_to_text() -> None:
    """Live seeds from text alone, so tool rounds are rendered as readable developer/assistant text."""
    messages = [
        ModelRequest(parts=[SystemPromptPart(content='ignored, goes to instructions')]),
        ModelRequest(parts=[UserPromptPart(content='What is the weather?')]),
        ModelResponse(parts=[ThinkingPart(content='They want a forecast.'), ToolCallPart('forecast', {'city': 'A'})]),
        ModelRequest(parts=[ToolReturnPart(tool_name='forecast', content='14C', tool_call_id='1')]),
        ModelResponse(parts=[TextPart(content='Fourteen degrees.')]),
        ModelRequest(parts=[SpeechPart(speaker='user', transcript='Thanks!')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Any time.')]),
    ]

    assert seed_input_items(messages, provider_name='openai') == snapshot(
        [
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'What is the weather?'}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'They want a forecast.'}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Called `forecast` with {"city":"A"}.'}]},
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': 'Result of `forecast`: 14C'}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Fourteen degrees.'}]},
            {'role': 'user', 'content': [{'type': 'input_text', 'text': 'Thanks!'}]},
            {'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Any time.'}]},
        ]
    )


def test_seeding_refuses_media(image_content: Any) -> None:
    """Unsupported content raises rather than being silently dropped from the seeded conversation."""
    messages = [ModelRequest(parts=[UserPromptPart(content=[image_content])])]
    with pytest.raises(UserError, match='can only be seeded with text'):
        seed_input_items(messages, provider_name='openai')


def test_idle_output_audio_is_dropped_but_speech_passes() -> None:
    """Live streams a continuous audio track; only what carries sound is a reply.

    A recorded call can't pin this: the idle frames and the spoken ones are the same event type, so
    the rule only shows as a difference when the two are fed in deliberately.
    """
    connection = _connection()
    silence = {'type': 'session.output_audio.delta', 'delta': 'AAAAAAAAAAA='}
    voice = {'type': 'session.output_audio.delta', 'delta': 'f39/f39/f38='}

    assert connection._map_event(_event(silence)) == []  # pyright: ignore[reportPrivateUsage]
    assert [type(e).__name__ for e in connection._map_event(_event(voice))] == ['AudioDelta']  # pyright: ignore[reportPrivateUsage]
    # Once the model is speaking, a silent frame mid-utterance is a pause, not the end of the track.
    assert [type(e).__name__ for e in connection._map_event(_event(silence))] == ['AudioDelta']  # pyright: ignore[reportPrivateUsage]


def test_start_of_stream_dither_is_not_speech() -> None:
    """Every Live session opens with a frame or two of dither peaking around 20.

    Counting any non-zero sample as speech opened a reply before the user had said anything, and
    every idle frame after it then reached the session as assistant audio.
    """
    connection = _connection()
    dither = {'type': 'session.output_audio.delta', 'delta': 'EwD0/wcAAAD9/wEAAAAAAA=='}

    assert connection._map_event(_event(dither)) == []  # pyright: ignore[reportPrivateUsage]
    assert not connection._response_open  # pyright: ignore[reportPrivateUsage]


def test_idle_audio_is_held_back_while_the_user_speaks() -> None:
    """An assistant frame ends the user's turn, so the idle track must not reach it mid-utterance.

    Otherwise each gap between the user's words arrives as model audio and one sentence is recorded
    as a turn per word.
    """
    connection = _connection()
    silence = {'type': 'session.output_audio.delta', 'delta': 'AAAAAAAAAAA='}
    voice = {'type': 'session.output_audio.delta', 'delta': 'f39/f39/f38='}
    words = {'type': 'session.input_transcript.delta', 'delta': 'and', 'start_ms': 0, 'end_ms': 1, 'event_id': 'e1'}

    connection._map_event(_event(voice))  # pyright: ignore[reportPrivateUsage]
    connection._map_event(_event(words))  # pyright: ignore[reportPrivateUsage]
    assert connection._map_event(_event(silence)) == []  # pyright: ignore[reportPrivateUsage]


def test_an_undecodable_audio_frame_is_a_recoverable_error() -> None:
    """Bad base64 in one frame must not take down the receive loop and, with it, the call."""
    connection = _connection()
    events = connection._map_frame(  # pyright: ignore[reportPrivateUsage]
        json.dumps({'type': 'session.output_audio.delta', 'delta': 'A', 'event_id': 'e'})
    )

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.recoverable is True
    assert error.message.startswith('Failed to parse OpenAI GPT-Live event:')


def test_output_transcript_closes_the_user_turn() -> None:
    """Live marks neither turn as finished, so the model replying is what ends the user's."""
    connection = _connection()
    assert connection._map_event(  # pyright: ignore[reportPrivateUsage]
        _event(
            {'type': 'session.input_transcript.delta', 'delta': 'hello', 'start_ms': 0, 'end_ms': 1, 'event_id': 'e1'}
        )
    ) == [InputTranscript('hello')]

    events = connection._map_event(  # pyright: ignore[reportPrivateUsage]
        _event({'type': 'session.output_transcript.delta', 'delta': 'hi', 'start_ms': 1, 'end_ms': 2, 'event_id': 'e2'})
    )
    assert events == [InputTranscript('', is_final=True), OutputTranscript('hi')]


def test_client_delegation_surfaces_a_recoverable_error() -> None:
    """A session in client-delegation mode would stall silently; say so instead."""
    connection = _connection()
    events = connection._map_event(  # pyright: ignore[reportPrivateUsage]
        _event(
            {
                'type': 'session.delegation.created',
                'event_id': 'e1',
                'offset_ms': 10,
                'delegation': {'id': 'd1', 'type': 'delegation', 'target': 'client'},
            }
        )
    )
    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error.code == 'live_client_delegation'
    assert error.recoverable is True


def test_usage_is_reported_as_an_increment() -> None:
    """Live reports a cumulative total and says not to sum it; `RunUsage` adds what it is given."""
    connection = _connection()

    def usage(seconds: float) -> list[Any]:
        return connection._map_event(  # pyright: ignore[reportPrivateUsage]
            _event({'type': 'session.usage.updated', 'event_id': 'e', 'usage': {'seconds': seconds}})
        )

    assert usage(10.0) == [SessionUsage(_request_usage(10), response_scoped=False)]
    # The second report is a running total, so only the difference is new.
    assert usage(25.0) == [SessionUsage(_request_usage(15), response_scoped=False)]
    # A repeat of the same total adds nothing.
    assert usage(25.0) == []


def _request_usage(seconds: int) -> Any:
    return RequestUsage(details={'billable_audio_seconds': seconds})


async def test_text_is_sent_as_context_not_a_user_turn() -> None:
    """Live has no user-message event: speakable text is commentary, silent context is thinking."""
    sent: list[dict[str, Any]] = []

    class _Recorder(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)

    connection = _Recorder(object())  # pyright: ignore[reportArgumentType]
    await connection.send('Say this out loud.')
    await connection.send(TextContext('Know this quietly.'))

    assert sent == snapshot(
        [
            {'type': 'session.commentary.append', 'delegation_id': None, 'content': 'Say this out loud.'},
            {'type': 'session.thinking.append', 'delegation_id': None, 'content': 'Know this quietly.'},
        ]
    )


async def test_tool_result_continues_the_delegated_response() -> None:
    """A delegated response waiting on tool output does not resume on its own."""
    sent: list[dict[str, Any]] = []

    class _Recorder(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)

    connection = _Recorder(object())  # pyright: ignore[reportArgumentType]
    await connection.send(ToolResult('call_1', output='14C'))

    assert sent == snapshot(
        [
            {
                'type': 'response.item.create',
                'item': {'type': 'function_call_output', 'call_id': 'call_1', 'output': '14C'},
            },
            {'type': 'response.create'},
        ]
    )


def _backend_response(**overrides: Any) -> dict[str, Any]:
    """A Responses object as the delegated backend's lifecycle events carry it."""
    return {
        'id': 'resp_1',
        'object': 'response',
        'created_at': 0,
        'status': 'completed',
        'model': 'gpt-5.6-sol',
        'output': [],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        **overrides,
    }


def _backend_terminal(nested_type: str = 'response.completed', **response: Any) -> dict[str, Any]:
    """A backend response's terminal event: `response.completed`, `.failed`, or `.incomplete`."""
    return {'type': nested_type, 'sequence_number': 0, 'response': _backend_response(**response)}


def _backend_call(call_id: str, name: str = 'weather', arguments: str = '{}') -> dict[str, Any]:
    """The backend asking for a tool call."""
    return {
        'type': 'response.output_item.done',
        'output_index': 0,
        'sequence_number': 0,
        'item': {'type': 'function_call', 'call_id': call_id, 'name': name, 'arguments': arguments},
    }


def _open_delegation(connection: OpenAILiveConnection, *, call_ids: tuple[str, ...] = ()) -> None:
    """Open a Responses delegation and have the backend ask for `call_ids`."""
    connection._map_event(  # pyright: ignore[reportPrivateUsage]
        _event(
            {
                'type': 'session.delegation.created',
                'event_id': 'e1',
                'offset_ms': 0,
                'delegation': {'id': 'd1', 'type': 'delegation', 'target': 'responses'},
            }
        )
    )
    for call_id in call_ids:
        connection._map_response_event(_backend_call(call_id), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('nested_type', ['response.failed', 'response.incomplete'])
def test_a_backend_that_gives_up_releases_the_turn_clock(nested_type: str) -> None:
    """A delegation suspends the clock, so a backend that stops short has to close it too.

    Only `response.completed` used to clear the delegation, so a failed one left the map non-empty
    for the rest of the session and no turn could ever be reported complete again.
    """
    connection = _connection()
    _open_delegation(connection, call_ids=('c1',))
    assert connection._silence_timeout() is None  # pyright: ignore[reportPrivateUsage]

    connection._map_response_event(_backend_terminal(nested_type), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]

    assert not connection._delegations  # pyright: ignore[reportPrivateUsage]
    # The clock runs again, so this turn — and every later one — can still end.
    assert connection._silence_timeout() is not None  # pyright: ignore[reportPrivateUsage]


async def test_a_result_for_an_abandoned_call_is_not_sent() -> None:
    """A tool that finishes after its backend gave up must not restart the backend."""
    sent: list[dict[str, Any]] = []

    class _Recorder(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)  # pragma: no cover

    connection = _Recorder(object())  # pyright: ignore[reportArgumentType]
    _open_delegation(connection, call_ids=('c1',))
    connection._map_response_event(_backend_terminal('response.failed'), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]

    await connection.send(ToolResult('c1', output='too late'))

    assert sent == []


@pytest.mark.parametrize('nested_type', ['response.completed', 'response.failed'])
def test_a_tool_calls_usage_always_arrives(nested_type: str) -> None:
    """Delegated calls wait for their response's usage, so its terminal must always report some.

    A backend that fails, or reports no usage, would otherwise leave the call's `ModelResponse` open
    and hold back queued messages for the rest of the session.
    """
    connection = _connection()
    _open_delegation(connection, call_ids=('c1',))

    events = connection._map_response_event(_backend_terminal(nested_type), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]

    assert events[0] == SessionUsage(RequestUsage())
    # Only the response that asked for calls owes usage; the next one reports only what it has.
    later = connection._map_response_event(_backend_terminal(nested_type), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]
    assert not any(isinstance(event, SessionUsage) for event in later)


@pytest.mark.parametrize(
    ('nested', 'code', 'reason'),
    [
        (
            _backend_terminal('response.failed', status='failed', error={'code': 'server_error', 'message': 'boom'}),
            'live_delegation_failed',
            'server_error: boom',
        ),
        (
            _backend_terminal(
                'response.incomplete', status='incomplete', incomplete_details={'reason': 'max_output_tokens'}
            ),
            'live_delegation_incomplete',
            'incomplete: max_output_tokens',
        ),
        (_backend_terminal('response.failed', status='failed'), 'live_delegation_failed', 'response.failed'),
    ],
)
def test_a_backend_that_gives_up_is_reported(nested: dict[str, Any], code: str, reason: str) -> None:
    """Live keeps talking and the turn still ends, so the failure has to be said out loud.

    Otherwise the caller sees an ordinary turn boundary and no sign the delegated work was lost.
    """
    connection = _connection()
    _open_delegation(connection)

    events = connection._map_response_event(nested, delegation_id='d1')  # pyright: ignore[reportPrivateUsage]

    assert events == [
        RealtimeSessionErrorEvent(
            message=f'The delegated OpenAI Responses backend did not finish ({reason}).', code=code
        )
    ]
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent) and error.recoverable is True


class _Recorder(OpenAILiveConnection):
    """A connection with no socket that records what it would send."""

    def __init__(self) -> None:
        super().__init__(object())  # pyright: ignore[reportArgumentType]
        self.sent: list[str] = []

    async def _send_event(self, event: dict[str, Any]) -> None:
        self.sent.append(event['type'])

    async def terminal(self, nested_type: str = 'response.completed') -> None:
        """The backend response ends; as the receive loop does, send any continuation now due."""
        self._map_response_event(_backend_terminal(nested_type), delegation_id='d1')
        await self._send_due_continuations()

    def call(self, call_id: str) -> None:
        self._map_response_event(_backend_call(call_id), delegation_id='d1')


async def test_parallel_tool_calls_continue_once_every_result_is_in() -> None:
    """The backend resumes from all of its outputs together, so the first result is not the cue."""
    connection = _Recorder()
    _open_delegation(connection, call_ids=('c1', 'c2'))

    await connection.send(ToolResult('c1', output='14C'))
    await connection.send(ToolResult('c2', output='rainy'))
    # Every result is in, but the response that asked for them hasn't ended: it could ask for more.
    assert connection.sent == ['response.item.create', 'response.item.create']

    await connection.terminal()
    assert connection.sent == snapshot(['response.item.create', 'response.item.create', 'response.create'])


async def test_a_result_before_the_next_call_does_not_continue_early() -> None:
    """A fast tool can answer before the backend's next parallel call has even arrived.

    Continuing as soon as every call *seen so far* was answered sent `response.create` with the second
    call still to come, then a second `response.create` once it was answered.
    """
    connection = _Recorder()
    _open_delegation(connection, call_ids=('c1',))
    await connection.send(ToolResult('c1', output='14C'))
    connection.call('c2')
    await connection.terminal()
    # The response has ended, but `c2` is still unanswered.
    assert connection.sent == ['response.item.create']

    await connection.send(ToolResult('c2', output='rainy'))
    assert connection.sent == snapshot(['response.item.create', 'response.item.create', 'response.create'])


async def test_a_late_completion_does_not_end_a_delegation_mid_continuation() -> None:
    """Answering before the asking response completes must not hand the turn back early."""
    connection = _Recorder()
    _open_delegation(connection, call_ids=('c1',))
    await connection.send(ToolResult('c1', output='14C'))

    # The response that *asked* for the tool ends only now, which is what sends the continuation.
    await connection.terminal()
    assert connection.sent == ['response.item.create', 'response.create']
    assert connection._delegations  # pyright: ignore[reportPrivateUsage]
    assert connection._silence_timeout() is None  # pyright: ignore[reportPrivateUsage]

    # The continuation itself is what ends the delegation.
    await connection.terminal()
    assert not connection._delegations  # pyright: ignore[reportPrivateUsage]


async def test_tool_result_text_content_reaches_the_backend() -> None:
    """A `ToolReturn`'s text `content` goes to the backend as a message after the call's output."""
    sent: list[dict[str, Any]] = []

    class _Sink(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)

    connection = _Sink(object())  # pyright: ignore[reportArgumentType]
    await connection.send(ToolResult('c1', output='ok', content=['The user is a returning guest.']))

    assert sent == snapshot(
        [
            {'type': 'response.item.create', 'item': {'type': 'function_call_output', 'call_id': 'c1', 'output': 'ok'}},
            {
                'type': 'response.item.create',
                'item': {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'The user is a returning guest.'}],
                },
            },
            {'type': 'response.create'},
        ]
    )


async def test_tool_result_media_is_refused() -> None:
    """Live carries no media, so a result that needs it fails with nothing on the wire."""
    sent: list[dict[str, Any]] = []

    class _Recorder(OpenAILiveConnection):
        async def _send_event(self, event: dict[str, Any]) -> None:
            sent.append(event)  # pragma: no cover

    connection = _Recorder(object())  # pyright: ignore[reportArgumentType]
    result = ToolResult('c1', output='see this', content=[BinaryContent(data=b'x', media_type='image/png')])

    with pytest.raises(UserError, match='does not support media in tool results'):
        await connection.send(result)

    assert sent == []


def _session_closed(reason: str, seconds: float = 0) -> dict[str, Any]:
    return {
        'type': 'session.closed',
        'event_id': 'e',
        'reason': reason,
        'session': {'id': 's', 'expires_at': 0, 'model': 'gpt-live-1', 'status': 'active'},
        'usage': {'seconds': seconds},
    }


@pytest.mark.parametrize('reason', ['expired', 'content', 'connection_lost'])
def test_a_session_ended_by_the_provider_interrupts_the_reply(reason: str) -> None:
    """The WebSocket close that follows is clean, so the reason is the only sign the reply was cut off.

    Settled on that close, a reply stopped by the safety filter or the duration limit read as finished.
    """
    connection = _connection()
    connection._map_frame(json.dumps({'type': 'session.output_audio.delta', 'delta': 'f39/f39/f38='}))  # pyright: ignore[reportPrivateUsage]

    events = connection._map_frame(json.dumps(_session_closed(reason)))  # pyright: ignore[reportPrivateUsage]

    assert events == [
        ResponseDone(interrupted=True),
        RealtimeSessionErrorEvent(
            message=f'The OpenAI GPT-Live session ended: {reason}.', code=f'live_session_{reason}', recoverable=False
        ),
    ]


@pytest.mark.parametrize('reason', ['close_requested', 'remote_hangup'])
def test_an_ordinary_close_only_reports_final_usage(reason: str) -> None:
    connection = _connection()
    connection._map_frame(json.dumps({'type': 'session.output_audio.delta', 'delta': 'f39/f39/f38='}))  # pyright: ignore[reportPrivateUsage]

    events = connection._map_frame(json.dumps(_session_closed(reason, seconds=12)))  # pyright: ignore[reportPrivateUsage]

    assert events == [SessionUsage(_request_usage(12), response_scoped=False)]


def test_a_backend_error_event_is_reported() -> None:
    """A nested Responses `error` is the backend's diagnostic; dropping it hid why delegated work failed."""
    connection = _connection()
    error = {
        'type': 'error',
        'sequence_number': 0,
        'code': 'rate_limit_exceeded',
        'message': 'slow down',
        'param': None,
    }

    assert connection._map_response_event(error, delegation_id='d1') == [  # pyright: ignore[reportPrivateUsage]
        RealtimeSessionErrorEvent(
            message='The delegated OpenAI Responses backend reported an error: slow down', code='rate_limit_exceeded'
        )
    ]


def test_an_error_with_no_code_is_still_reported() -> None:
    """OpenAI documents `error` frames with a null `code`, which the SDK's `ServerEvent` rejects."""
    connection = _connection()
    frame = {'type': 'error', 'event_id': 'e', 'error': {'type': 'server_error', 'code': None, 'message': 'boom'}}

    assert connection._map_frame(json.dumps(frame)) == [RealtimeSessionErrorEvent(message='boom', code=None)]  # pyright: ignore[reportPrivateUsage]


def test_unknown_events_are_ignored() -> None:
    """A future event type is not a reason to end a call in progress."""
    assert _connection()._map_frame('{"type": "session.something.new"}') == []  # pyright: ignore[reportPrivateUsage]


async def test_agent_rejects_text_output(model: OpenAILiveModel) -> None:
    """Live only speaks, so asking it for text fails before the session opens."""
    agent = Agent(instructions='hi')
    with pytest.raises(UserError):
        async with agent.realtime(model, model_settings={'output_modality': 'text'}).session():
            pass  # pragma: no cover


class _FakeWebSocket:
    """A socket that yields queued frames and then goes quiet, so the turn clock can run out."""

    def __init__(self, frames: list[str], *, delay: float = 0.0) -> None:
        self._frames = list(frames)
        self._delay = delay
        self.sent: list[str] = []
        self.closed = False

    async def recv(self) -> str:
        if self._frames:
            if self._delay:
                await anyio.sleep(self._delay)
            return self._frames.pop(0)
        await anyio.sleep_forever()
        raise AssertionError('unreachable')  # pragma: no cover

    async def send(self, data: str) -> None:
        self.sent.append(data)


def _transcript_frame(delta: str, *, speaker: str = 'output') -> str:
    return json.dumps(
        {'type': f'session.{speaker}_transcript.delta', 'delta': delta, 'start_ms': 0, 'end_ms': 1, 'event_id': 'e'}
    )


async def test_quiet_stretch_ends_the_turn() -> None:
    """Live sends no end-of-turn frame, so going quiet is what finishes the reply."""
    ws = _FakeWebSocket([_transcript_frame('hello there')])
    connection = OpenAILiveConnection(ws, turn_silence_ms=10)  # pyright: ignore[reportArgumentType]

    events: list[Any] = []
    with anyio.fail_after(5):
        async for event in connection:  # pragma: no branch
            events.append(event)
            if isinstance(event, ResponseDone):
                break

    assert events == [OutputTranscript('hello there'), ResponseDone()]


async def test_delegated_work_holds_the_turn_open() -> None:
    """The model goes quiet while the backend thinks; ending the turn there would truncate the reply."""
    delegation = json.dumps(
        {
            'type': 'session.delegation.created',
            'event_id': 'e1',
            'offset_ms': 0,
            'delegation': {'id': 'd1', 'type': 'delegation', 'target': 'responses'},
        }
    )
    ws = _FakeWebSocket([delegation])
    connection = OpenAILiveConnection(ws, turn_silence_ms=10)  # pyright: ignore[reportArgumentType]

    with pytest.raises(TimeoutError):
        with anyio.fail_after(0.3):
            async for event in connection:  # pragma: no branch
                if isinstance(event, ResponseDone):  # pragma: no cover
                    break

    # Once the backend reports it is finished, the clock restarts and the turn can end.
    assert connection._delegations  # pyright: ignore[reportPrivateUsage]
    connection._map_response_event(_backend_terminal(), delegation_id='d1')  # pyright: ignore[reportPrivateUsage]
    assert not connection._delegations  # pyright: ignore[reportPrivateUsage]


async def test_response_completed_without_a_delegation_is_ignored() -> None:
    """A nested event we can't correlate is not an error; Live says it may be uncorrelated."""
    connection = _connection()
    assert connection._map_response_event(_backend_terminal(), delegation_id=None) == []  # pyright: ignore[reportPrivateUsage]
    # An output item that isn't a function call is nothing to map.
    message: dict[str, Any] = {
        **_backend_call('c'),
        'item': {'type': 'message', 'id': 'm', 'role': 'assistant', 'status': 'completed', 'content': []},
    }
    assert connection._map_response_event(message, delegation_id=None) == []  # pyright: ignore[reportPrivateUsage]
    assert connection._map_response_event(_backend_call('c', name='n', arguments=''), delegation_id='missing')[  # pyright: ignore[reportPrivateUsage]
        -1
    ] == ToolCall('c', tool_name='n', args='{}', response_usage_follows=False)


def test_a_malformed_backend_event_is_a_recoverable_error() -> None:
    """A tool call missing its `call_id` must not take down the receive loop.

    Read with dict indexing, it raised a `KeyError`, which isn't the `ValueError` `_map_frame` reports
    as recoverable. Parsed through the SDK's types, it is a `ValidationError`, which is.
    """
    connection = _connection()
    _open_delegation(connection)
    frame = {
        'type': 'response.event',
        'event_id': 'e2',
        'delegation_id': 'd1',
        'event': {**_backend_call('c1'), 'item': {'type': 'function_call', 'name': 'weather', 'arguments': '{}'}},
    }

    events = connection._map_frame(json.dumps(frame))  # pyright: ignore[reportPrivateUsage]

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent) and error.recoverable is True
    assert error.message.startswith('Failed to parse OpenAI GPT-Live event:')


def test_an_unknown_backend_event_type_is_ignored() -> None:
    """A nested event type this SDK doesn't know is not a reason to report anything."""
    connection = _connection()
    assert connection._map_response_event({'type': 'response.something_new'}, delegation_id=None) == []  # pyright: ignore[reportPrivateUsage]


async def test_aclose_cancels_the_pending_read() -> None:
    """Closing must not strand the read in flight, or its exception surfaces with no owner."""
    ws = _FakeWebSocket([])
    connection = OpenAILiveConnection(ws, turn_silence_ms=10)  # pyright: ignore[reportArgumentType]

    async def drain() -> None:
        async for _ in connection:
            pass  # pragma: no cover

    async with anyio.create_task_group() as tg:
        tg.start_soon(drain)
        await anyio.sleep(0.05)
        await connection.aclose()
        tg.cancel_scope.cancel()
    # A second close is a no-op rather than an error.
    await connection.aclose()


def test_error_events_keep_the_session_usable() -> None:
    connection = _connection()
    events = connection._map_event(  # pyright: ignore[reportPrivateUsage]
        _event(
            {
                'type': 'error',
                'event_id': 'e1',
                'error': {'type': 'invalid_request_error', 'code': 'bad_thing', 'message': 'nope'},
            }
        )
    )
    error = events[0]
    assert isinstance(error, RealtimeSessionErrorEvent)
    assert error == RealtimeSessionErrorEvent(message='nope', code='bad_thing')
    assert error.recoverable is True


def test_reconnect_does_not_restore_state() -> None:
    """A redialed Live session starts empty, so the session replays local history instead."""
    assert _connection().reconnect_restores_in_flight_state is False
    assert _connection().input_transcription_enabled is True


def test_strict_tools_and_declarative_tool_choice(model: OpenAILiveModel) -> None:
    tool = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'}, strict=True)
    config = _config(model, tools=[tool], settings=OpenAILiveModelSettings(tool_choice='required'))
    responses = config['delegation']['responses']

    assert responses['tools'] == snapshot(
        [{'type': 'function', 'name': 'lookup', 'parameters': {'type': 'object'}, 'strict': True}]
    )
    assert responses['tool_choice'] == 'required'
    # With no agent instructions there is no backend prompt to send.
    assert 'instructions' not in responses


def _backend(model: OpenAILiveModel, **settings: Any) -> str:
    return _config(model, settings=OpenAILiveModelSettings(**settings))['delegation']['responses']['model']


def test_the_backend_model_can_follow_a_plus_in_the_model_name() -> None:
    """`'gpt-live-1+gpt-6-luna'` reads as the composite it is: a voice model and the model it delegates to."""
    model = infer_realtime_model('openai:gpt-live-1+gpt-6-luna')

    assert isinstance(model, OpenAILiveModel)
    assert model.model_name == 'gpt-live-1'
    assert _backend(model) == 'gpt-6-luna'


async def _session_backend(agent: Agent[None, Any], model: OpenAILiveModel) -> str:
    """Open a session through `agent.realtime()` and read the backend its `session.start` names."""
    started = json.dumps({'type': 'session.started', 'event_id': 'e', 'session': {'id': 's', 'model': 'gpt-live-1'}})
    ws = _FakeWebSocket([started])
    with _patched_connect(ws):
        async with agent.realtime(model).session():
            pass
    return json.loads(ws.sent[0])['session']['delegation']['responses']['model']


@pytest.mark.parametrize(
    ('agent_model', 'backend'),
    [
        ('openai:gpt-6-luna', 'gpt-6-luna'),
        ('openai-responses:gpt-6-luna', 'gpt-6-luna'),
        (None, AUTO_BACKEND_MODEL),
    ],
)
async def test_the_backend_defaults_to_the_agents_own_model(
    model: OpenAILiveModel, agent_model: str | None, backend: str
) -> None:
    """The backend runs the agent's instructions and tools; an OpenAI agent model already names it."""
    assert await _session_backend(Agent(agent_model), model) == backend


@pytest.mark.parametrize(
    ('agent_route', 'live_route', 'backend'),
    [
        ('direct', 'direct', 'gpt-6-luna'),
        ('gateway', 'gateway', 'gpt-6-luna'),
        # Reached a different way, the agent's model isn't one this session can delegate to.
        ('gateway', 'direct', AUTO_BACKEND_MODEL),
        ('direct', 'gateway', AUTO_BACKEND_MODEL),
    ],
)
async def test_the_agents_model_is_used_when_reached_the_same_way(
    agent_route: str, live_route: str, backend: str
) -> None:
    """`gateway/openai:gpt-live-1` on an agent built on `gateway/openai:gpt-6-luna` delegates to it.

    Both have to go to OpenAI the same way, directly or through the same gateway route, which is what
    comparing their base URLs checks.
    """

    def provider(route: str) -> Provider[AsyncOpenAI]:
        return gateway_provider('openai', api_key='pylf_v1_us_x') if route == 'gateway' else OpenAIProvider(api_key='x')

    agent = Agent(OpenAIResponsesModel('gpt-6-luna', provider=provider(agent_route)))
    live = OpenAILiveModel('gpt-live-1', provider=provider(live_route))
    assert await _session_backend(agent, live) == backend


async def test_an_agent_model_not_at_openai_is_not_a_backend(model: OpenAILiveModel) -> None:
    """An OpenAI-compatible server speaks the protocol, but isn't where the Live session runs."""
    compatible = OpenAIChatModel('llama3', provider=OpenAIProvider(api_key='x', base_url='http://localhost:11434/v1'))
    assert await _session_backend(Agent(compatible), model) == AUTO_BACKEND_MODEL


async def test_an_unresolved_agent_model_is_not_a_backend(model: OpenAILiveModel) -> None:
    """With `defer_model_check=True` the agent's model is still a name, with no base URL to compare."""
    agent = Agent('openai:gpt-6-luna', defer_model_check=True)
    assert await _session_backend(agent, model) == AUTO_BACKEND_MODEL


async def test_a_named_backend_takes_precedence_over_the_agents_model() -> None:
    """An explicit setting beats the name's `+`, which beats the agent's model, which beats `'auto'`."""
    agent = Agent('openai:gpt-5.6-sol')
    assert await _session_backend(agent, OpenAILiveModel('gpt-live-1+gpt-6-luna', provider='openai')) == 'gpt-6-luna'

    def delegating_to(backend: str) -> OpenAILiveModel:
        delegation = OpenAILiveResponsesDelegation(model=backend)
        return OpenAILiveModel(
            'gpt-live-1', provider='openai', settings=OpenAILiveModelSettings(openai_live_delegation=delegation)
        )

    assert await _session_backend(agent, delegating_to('gpt-6-luna')) == 'gpt-6-luna'
    assert await _session_backend(agent, delegating_to('auto')) == AUTO_BACKEND_MODEL


def test_a_connection_opened_without_an_agent_uses_auto(model: OpenAILiveModel) -> None:
    """Outside `agent.realtime()` there is no run context, and so no agent to consult."""
    assert _backend(model) == AUTO_BACKEND_MODEL


def test_16khz_audio_is_chosen_through_the_profile() -> None:
    """Live runs at 16 or 24 kHz, set the way the Realtime providers set their rate: on the profile."""
    model = OpenAILiveModel(
        'gpt-live-1',
        provider='openai',
        profile={'audio_input_sample_rate': 16000, 'audio_output_sample_rate': 16000},
    )
    config = _config(model)

    assert config['audio']['format'] == {'type': 'audio/pcm', 'rate': 16000}
    TypeAdapter(SessionConfig).validate_python(config)


@pytest.mark.parametrize(
    ('input_rate', 'output_rate'),
    [(16000, 24000), (8000, 8000), (48000, 48000)],
)
def test_an_audio_rate_live_cannot_run_raises(input_rate: int, output_rate: int) -> None:
    """One format covers both directions, so the two rates must agree, and be one Live accepts."""
    model = OpenAILiveModel(
        'gpt-live-1',
        provider='openai',
        profile={'audio_input_sample_rate': input_rate, 'audio_output_sample_rate': output_rate},
    )
    with pytest.raises(UserError, match='one PCM16 audio format for both directions'):
        _config(model)


def test_tool_allow_list_trims_the_advertised_tools(model: OpenAILiveModel) -> None:
    """The backend has no list form, so an allow-list is applied by trimming, and its mode still sent.

    Dropping the mode lost what the allow-list says about whether a tool *must* be called: a
    one-tool list is a named function choice, and the backend can express exactly that.
    """
    tools = [
        ToolDefinition(name='kept', parameters_json_schema={'type': 'object'}),
        ToolDefinition(name='also_kept', parameters_json_schema={'type': 'object'}),
        ToolDefinition(name='dropped', parameters_json_schema={'type': 'object'}),
    ]
    one = _config(model, tools=tools, settings=OpenAILiveModelSettings(tool_choice=['kept']))
    assert [tool['name'] for tool in one['delegation']['responses']['tools']] == ['kept']
    assert one['delegation']['responses']['tool_choice'] == snapshot({'type': 'function', 'name': 'kept'})

    two = _config(model, tools=tools, settings=OpenAILiveModelSettings(tool_choice=['kept', 'also_kept']))
    assert [tool['name'] for tool in two['delegation']['responses']['tools']] == ['kept', 'also_kept']
    assert two['delegation']['responses']['tool_choice'] == snapshot('required')
    # Both are shapes the backend's own schema accepts.
    TypeAdapter(SessionConfig).validate_python(one)
    TypeAdapter(SessionConfig).validate_python(two)


def test_user_prompt_text_parts_are_joined() -> None:
    messages = [ModelRequest(parts=[UserPromptPart(content=['first', TextContent(content='second')])])]
    assert seed_input_items(messages, provider_name='openai') == snapshot(
        [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'first\nsecond'}]}]
    )


async def test_handshake_timeout_is_a_typed_error(model: OpenAILiveModel) -> None:
    """A session that never starts fails predictably instead of hanging."""
    with _patched_connect(_FakeWebSocket([])):
        with pytest.raises(RealtimeError, match='timed out waiting'):
            async with model.connect(
                messages=[],
                model_settings=OpenAILiveModelSettings(handshake_timeout=0.05),
                model_request_parameters=ModelRequestParameters(),
            ):
                pass  # pragma: no cover


async def test_handshake_error_is_a_typed_error(model: OpenAILiveModel) -> None:
    """A rejected configuration arrives over the open socket, so it maps like a provider error."""
    rejection = json.dumps({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'bad model'}})
    with _patched_connect(_FakeWebSocket([rejection])):
        with pytest.raises(RealtimeError, match='bad model'):
            async with model.connect(
                messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
            ):
                pass  # pragma: no cover


@contextmanager
def _patched_connect(ws: _FakeWebSocket) -> Any:
    """Stand in for `websockets.connect`, which returns an async context manager yielding the socket."""

    class _Opening:
        async def __aenter__(self) -> _FakeWebSocket:
            return ws

        async def __aexit__(self, *args: Any) -> None:
            ws.closed = True

    original = live_module.websockets.connect

    def _connect(*args: Any, **kwargs: Any) -> _Opening:
        return _Opening()

    live_module.websockets.connect = _connect
    try:
        yield
    finally:
        live_module.websockets.connect = original


async def test_a_new_turn_starts_after_the_previous_one_ended() -> None:
    """The reply clock restarts per turn: a later frame opens a new response, not a resumed one."""
    # The gap between frames has to clear the turn clock by a wide margin, or a loaded machine
    # delivers the second frame before the first turn expires and the test silently stops testing it.
    ws = _FakeWebSocket([_transcript_frame('first'), _transcript_frame('second')], delay=0.3)
    connection = OpenAILiveConnection(ws, turn_silence_ms=10)  # pyright: ignore[reportArgumentType]

    events: list[Any] = []
    with anyio.fail_after(5):
        async for event in connection:  # pragma: no branch
            events.append(event)
            if len(events) == 4:
                break

    assert events == [
        OutputTranscript('first'),
        ResponseDone(),
        OutputTranscript('second'),
        ResponseDone(),
    ]


async def test_a_user_turn_alone_is_finalized_when_the_model_stays_silent() -> None:
    """A spoken turn the model never answers must still land in history rather than hang open."""
    ws = _FakeWebSocket([_transcript_frame('are you there', speaker='input')])
    connection = OpenAILiveConnection(ws, turn_silence_ms=10)  # pyright: ignore[reportArgumentType]

    events: list[Any] = []
    with anyio.fail_after(5):
        async for event in connection:  # pragma: no branch
            events.append(event)
            if len(events) == 2:
                break

    # No `ResponseDone`: the model never replied, so there is no reply to finalize.
    assert events == [InputTranscript('are you there'), InputTranscript('', is_final=True)]


def test_seeding_skips_content_it_cannot_carry() -> None:
    """Empty transcripts and parts with no text equivalent are dropped, not sent as blanks."""
    messages = [
        ModelRequest(parts=[SpeechPart(speaker='user', transcript=None)]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='   ')]),
        ModelRequest(parts=[UserPromptPart(content='kept')]),
    ]

    assert seed_input_items(messages, provider_name='openai') == snapshot(
        [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'kept'}]}]
    )
    assert seed_input_items([ModelResponse(parts=[FilePart(content=None)])], provider_name='openai') == []  # pyright: ignore[reportArgumentType]


def test_seeding_keeps_a_failed_tool_round() -> None:
    """A retry carries the outcome of the call before it.

    Dropping it seeds the `ToolCallPart` as a call that was never answered, so the backend reads a
    round that failed as one that succeeded and does not try again.
    """
    messages = [
        ModelResponse(parts=[ToolCallPart(tool_name='weather', args={'city': 'Utrecht'}, tool_call_id='1')]),
        ModelRequest(parts=[RetryPromptPart(content='unknown city', tool_name='weather', tool_call_id='1')]),
        # A retry with no tool name is output validation rather than a tool round.
        ModelRequest(parts=[RetryPromptPart(content='not a number')]),
    ]

    assert seed_input_items(messages, provider_name='openai') == snapshot(
        [
            {
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'Called `weather` with {"city":"Utrecht"}.'}],
            },
            {
                'role': 'developer',
                'content': [
                    {'type': 'input_text', 'text': '`weather` failed: unknown city\n\nFix the errors and try again.'}
                ],
            },
            {
                'role': 'developer',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'The previous attempt failed: Validation feedback:\nnot a number\n\nFix the errors and try again.',
                    }
                ],
            },
        ]
    )


async def test_unrelated_frames_during_the_handshake_are_skipped(model: OpenAILiveModel) -> None:
    """A notice that arrives before `session.started` is not a protocol violation."""
    notice = json.dumps({'type': 'info', 'code': 'x', 'message': 'hello', 'event_id': 'e'})
    started = json.dumps({'type': 'session.started', 'event_id': 'e', 'session': {'id': 's', 'model': 'gpt-live-1'}})
    with _patched_connect(_FakeWebSocket([notice, started])):
        async with model.connect(
            messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
        ) as connection:
            assert connection.model_name == 'gpt-live-1'


@pytest.mark.parametrize('frame', ['not json at all', '["a list, not an object"]'])
async def test_a_malformed_handshake_frame_raises_a_realtime_error(model: OpenAILiveModel, frame: str) -> None:
    """The handshake promises a `RealtimeError`; a bad frame must not escape as a bare `ValueError`.

    `map_connect_errors` translates `RealtimeHandshakeError`, not the `ValueError` that parsing a
    malformed frame raises, so this only holds while the shared `expect_event` helper does the
    reading.
    """
    with _patched_connect(_FakeWebSocket([frame])):
        with pytest.raises(RealtimeError):
            async with model.connect(
                messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
            ):
                pass  # pragma: no cover


def test_session_config_matches_the_provider_schema(model: OpenAILiveModel) -> None:
    """What we build is validated against the SDK's own session shape.

    A cassette matcher can keep matching a recording after our payload drifts, so the wire shape is
    pinned against the provider's schema rather than only against a recorded conversation.
    """
    settings = OpenAILiveModelSettings(
        openai_voice='marin',
        openai_live_store=True,
        openai_live_delegation=OpenAILiveResponsesDelegation(model='gpt-5.6-sol', reasoning_effort='high'),
    )
    tool = ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'}, strict=True)
    config = _config(
        model,
        instructions='Answer carefully.',
        tools=[tool],
        messages=[ModelRequest(parts=[UserPromptPart(content='hi')])],
        settings=settings,
    )

    validated = TypeAdapter(SessionConfig).validate_python(config)
    assert validated.model == 'gpt-live-1'
    assert validated.delegation is not None and validated.delegation.type == 'responses'


def test_delegated_backend_token_usage_is_accumulated() -> None:
    """Live meters audio by the second, but the backend it delegates to is billed per token.

    Most of a call's token cost lives in the backend, so dropping this would under-report the run.
    """
    connection = _connection()
    completed: dict[str, Any] = {
        'type': 'response.event',
        'event_id': 'e1',
        'delegation_id': 'd1',
        'event': {
            'type': 'response.completed',
            'sequence_number': 0,
            'response': {
                'id': 'resp_1',
                'object': 'response',
                'created_at': 0,
                'status': 'completed',
                'model': 'gpt-5.6-sol',
                'output': [],
                'parallel_tool_calls': True,
                'tool_choice': 'auto',
                'tools': [],
                'usage': {
                    'input_tokens': 805,
                    'input_tokens_details': {'cache_write_tokens': 805, 'cached_tokens': 0},
                    'output_tokens': 19,
                    'output_tokens_details': {'reasoning_tokens': 3},
                    'total_tokens': 824,
                },
            },
        },
    }

    events = connection._map_event(_event(completed))  # pyright: ignore[reportPrivateUsage]
    assert len(events) == 1
    reported = events[0]
    assert isinstance(reported, SessionUsage)
    assert reported.usage.input_tokens == 805
    assert reported.usage.output_tokens == 19
    # The cache and reasoning breakdowns survive, so cost accounting matches a direct Responses call.
    assert reported.usage.cache_write_tokens == 805
    assert reported.usage.details['reasoning_tokens'] == 3
    # The backend's request is what a per-request input-token limit is measured against.
    assert reported.response_scoped is True
    # Priced here, against the backend's own model. The response these tokens land on carries Live's
    # name, so anything pricing it from `model_name` downstream would use the wrong rate — and a cost
    # that is already set is never recalculated.
    assert reported.usage.cost is not None and reported.usage.cost > 0


def test_unpriceable_backend_usage_is_still_accumulated() -> None:
    """A model genai-prices doesn't know leaves the cost unset; the tokens still count.

    Pricing must never cost us the usage itself, so `cost=None` has to stay distinguishable from a
    genuine zero rather than dropping the event.
    """
    connection = _connection()
    completed: dict[str, Any] = {
        'type': 'response.event',
        'event_id': 'e1',
        'delegation_id': 'd1',
        'event': {
            'type': 'response.completed',
            'sequence_number': 0,
            'response': {
                'id': 'resp_1',
                'object': 'response',
                'created_at': 0,
                'status': 'completed',
                'model': 'a-model-that-is-not-priced',
                'output': [],
                'parallel_tool_calls': True,
                'tool_choice': 'auto',
                'tools': [],
                'usage': {
                    'input_tokens': 11,
                    'input_tokens_details': {'cache_write_tokens': 0, 'cached_tokens': 0},
                    'output_tokens': 2,
                    'output_tokens_details': {'reasoning_tokens': 0},
                    'total_tokens': 13,
                },
            },
        },
    }

    events = connection._map_event(_event(completed))  # pyright: ignore[reportPrivateUsage]
    assert len(events) == 1
    reported = events[0]
    assert isinstance(reported, SessionUsage)
    assert reported.usage.input_tokens == 11
    assert reported.usage.cost is None


async def test_a_clean_close_finalizes_the_reply() -> None:
    """A graceful close ends the turn in flight rather than leaving it to settle as interrupted."""

    class _ClosingWebSocket(_FakeWebSocket):
        async def recv(self) -> str:
            if self._frames:
                return self._frames.pop(0)
            raise websockets.ConnectionClosedOK(Close(1000, ''), Close(1000, ''), True)

    connection = OpenAILiveConnection(_ClosingWebSocket([_transcript_frame('all done')]), turn_silence_ms=10_000)  # pyright: ignore[reportArgumentType]

    # Iterated to exhaustion, not broken out of: the stream ends itself when the socket closes.
    events = [event async for event in connection]
    assert events == [OutputTranscript('all done'), ResponseDone()]
