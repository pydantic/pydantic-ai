"""The lifecycle events an OpenAI-protocol connection yields: which response, spoken turn, and input each frame is about.

These are internal (see `pydantic_ai/realtime/_lifecycle.py`), so no public API reaches them yet, and the
recorded cassettes are checked against the contract they promise in `simulation/test_conformance.py`.
The tests here pin what the connection makes of the frames no recording shows: a response the provider
never announces, a request refused or dropped, a reconnect that loses what was in flight.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.realtime._lifecycle import (
    LIFECYCLE_EVENT_TYPES,
    InputAdded,
    InputLost,
    LifecycleEvent,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    UserTurnDiscarded,
    UserTurnEnded,
    UserTurnStarted,
)

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.realtime._openai_protocol import response_metadata_answers, response_request_metadata
    from pydantic_ai.realtime.codec import (
        CancelResponse,
        CreateResponse,
        RealtimeCodecEvent,
        ResponseDone,
        TextContext,
        ToolResult,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeWebSocket
    from .test_session import FakeRealtimeConnection

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


def created(response_id: str | None, *, answers: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {'object': 'realtime.response', 'status': 'in_progress', 'output': []}
    if response_id is not None:
        response['id'] = response_id
    if answers is not None:
        response['metadata'] = {'pydantic_ai_inputs': answers}
    return {'type': 'response.created', 'response': response}


def done(response_id: str | None, *, status: str = 'completed') -> dict[str, Any]:
    response: dict[str, Any] = {'object': 'realtime.response', 'status': status, 'output': []}
    if response_id is not None:
        response['id'] = response_id
    return {'type': 'response.done', 'response': response}


def transcript(response_id: str, text: str) -> dict[str, Any]:
    return {'type': 'response.output_audio_transcript.delta', 'response_id': response_id, 'delta': text}


def user_message_added(content_type: str = 'input_text') -> dict[str, Any]:
    return {
        'type': 'conversation.item.added',
        'item': {'id': 'item_user', 'type': 'message', 'role': 'user', 'content': [{'type': content_type}]},
    }


def tool_output_added(call_id: str) -> dict[str, Any]:
    return {
        'type': 'conversation.item.added',
        'item': {'id': f'item_{call_id}', 'type': 'function_call_output', 'call_id': call_id, 'output': ''},
    }


def refusal(event_id: str) -> dict[str, Any]:
    return {'type': 'error', 'error': {'type': 'invalid_request_error', 'code': 'refused', 'event_id': event_id}}


def frames(*frames: dict[str, Any]) -> list[str]:
    return [json.dumps(frame) for frame in frames]


def describe(event: RealtimeCodecEvent | LifecycleEvent) -> LifecycleEvent | str:
    """A lifecycle event as itself, and a codec event as the name of its type."""
    return event if isinstance(event, LIFECYCLE_EVENT_TYPES) else type(event).__name__


async def codec(connection: OpenAIRealtimeConnection) -> list[RealtimeCodecEvent]:
    """The codec events the connection yields to a session."""
    return [event async for event in connection]


class Stream:
    """A connection's lifecycle stream, read a few events at a time so the test can act in between."""

    def __init__(self, *incoming: dict[str, Any], **kwargs: Any) -> None:
        self.ws = FakeWebSocket(frames(*incoming))
        self.connection = OpenAIRealtimeConnection(self.ws, **kwargs)  # pyright: ignore[reportArgumentType]
        self._events = aiter(self.connection._lifecycle_events())  # pyright: ignore[reportPrivateUsage]

    def feed(self, *incoming: dict[str, Any]) -> None:
        """More frames from the server, read after those already scripted."""
        self.ws._incoming.extend(map(FakeWebSocket._normalize_frame, frames(*incoming)))  # pyright: ignore[reportPrivateUsage]

    async def take(self, count: int) -> list[LifecycleEvent | str]:
        return [describe(await anext(self._events)) for _ in range(count)]

    async def rest(self) -> list[LifecycleEvent | str]:
        """Everything up to the end of the stream (the scripted frames running out closes the connection)."""
        return [describe(event) async for event in self._events]


async def test_a_response_answers_what_its_echoed_metadata_names() -> None:
    stream = Stream(created('resp_1', answers='0'), transcript('resp_1', 'Hi.'), done('resp_1'))
    await stream.connection.send('Hello?')
    assert json.loads(stream.ws.sent[-1])['response'] == {'metadata': {'pydantic_ai_inputs': '0'}}
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1', answers=(0,)),
            'OutputTranscript',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_without_an_echo_a_response_answers_the_request_outstanding() -> None:
    """Recordings made before requests carried metadata, and servers that don't echo it."""
    stream = Stream(created('resp_1'), done('resp_1'))
    await stream.connection.send(TextContext('Some context.'))
    await stream.connection.send(CreateResponse())
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1', answers=(1,), basis='inferred'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_once_the_server_echoes_a_response_without_ours_answers_nothing() -> None:
    """Server VAD starts a response while our request is on its way: it is not ours, and ours is refused."""
    stream = Stream(created('resp_1', answers='0'), done('resp_1'))
    await stream.connection.send(CreateResponse())
    assert await stream.take(3) == snapshot(
        [
            ResponseStarted(response_id='resp_1', answers=(0,)),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
        ]
    )
    await stream.connection.send(CreateResponse())
    stream.feed(created('resp_vad'), refusal('pydantic_ai.response.1'), done('resp_vad'))
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_vad'),
            'InputRejected',
            'RealtimeSessionErrorEvent',
            ResponseRequestRefused(input_ids=(1,)),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_vad',
                status='completed',
                finish_reason='stop',
                provider_details={'status': 'completed'},
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_spoken_turn_is_bracketed_and_answered_by_the_response_vad_starts() -> None:
    stream = Stream(
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1'},
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1'},
        {'type': 'input_audio_buffer.speech_started', 'item_id': ''},
        {'type': 'input_audio_buffer.speech_stopped', 'item_id': 'item_u1'},
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_u1', 'previous_item_id': None},
        user_message_added('input_audio'),
        created('resp_1'),
        done('resp_1'),
    )
    assert await stream.rest() == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            UserTurnStarted(turn_id='item_u1'),
            'RealtimeInputSpeechStartEvent',
            'RealtimeInputSpeechStartEvent',
            'RealtimeInputSpeechEndEvent',
            UserTurnEnded(turn_id='item_u1'),
            ResponseStarted(response_id='resp_1', user_turn_id='item_u1'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_push_to_talk_commit_starts_and_ends_its_turn_and_a_clear_discards_one() -> None:
    stream = Stream(
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_u1', 'previous_item_id': None},
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u2'},
        {'type': 'input_audio_buffer.cleared'},
    )
    assert await stream.rest() == snapshot(
        [
            UserTurnStarted(turn_id='item_u1'),
            UserTurnEnded(turn_id='item_u1'),
            'RealtimeInputSpeechStartEvent',
            UserTurnStarted(turn_id='item_u2'),
            UserTurnDiscarded(turn_id='item_u2'),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_inputs_join_the_conversation_in_the_order_the_server_adds_them() -> None:
    """Seeded history is acknowledged first, and a tool output is matched by its call id."""
    stream = Stream(
        user_message_added(),
        {'type': 'conversation.item.added', 'item': {'id': 'item_a', 'type': 'message', 'role': 'assistant'}},
        user_message_added(),
        tool_output_added('call_unknown'),
        tool_output_added('call_1'),
        user_message_added('input_image'),
        user_message_added(),
        user_message_added(),
    )
    stream.connection._history_items_sent(  # pyright: ignore[reportPrivateUsage]
        [
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Earlier.'}]},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio', 'audio': ''}]},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Sure.'}]},
        ]
    )
    await stream.connection.send(TextContext('Now.'))
    await stream.connection.send(ToolResult('call_1', output='42', content=['See also.']))
    assert await stream.rest() == snapshot(
        [InputAdded(input_id=0), InputAdded(input_id=1), InputLost(input_ids=(1,)), 'RealtimeSessionErrorEvent']
    )


async def test_an_item_added_under_our_id_names_its_input_whatever_else_is_added() -> None:
    """Once the server keeps our item ids, items under other ids (a browser's, refused history) are none of ours."""
    stream = Stream(
        {**user_message_added(), 'item': {**user_message_added()['item'], 'id': 'pydantic_ai_item_1'}},
        user_message_added(),
        {**user_message_added(), 'item': {**user_message_added()['item'], 'id': 'pydantic_ai_item_1'}},
        {**user_message_added(), 'item': {**user_message_added()['item'], 'id': 'pydantic_ai_item_0'}},
    )
    await stream.connection.send(TextContext('First.'))
    await stream.connection.send(TextContext('Second.'))
    assert json.loads(stream.ws.sent[0])['item']['id'] == 'pydantic_ai_item_0'
    assert await stream.rest() == snapshot(
        [InputAdded(input_id=1), InputAdded(input_id=0), 'RealtimeSessionErrorEvent']
    )


async def test_refused_content_is_never_added() -> None:
    stream = Stream(refusal('pydantic_ai.content.0'), refusal('pydantic_ai.content.7'), user_message_added())
    await stream.connection.send(TextContext('Refused.'))
    await stream.connection.send(TextContext('Kept.'))
    assert await stream.rest() == snapshot(
        [
            'InputRejected',
            'RealtimeSessionErrorEvent',
            'InputRejected',
            'RealtimeSessionErrorEvent',
            InputAdded(input_id=1),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_refused_request_settles_every_input_it_was_made_for() -> None:
    """A tool-call batch's request is named after its last output, but answers all of them."""
    call = {'type': 'response.function_call_arguments.done', 'response_id': 'resp_1', 'name': 'lookup'}
    stream = Stream(
        created('resp_1', answers='0'), {**call, 'call_id': 'call_a'}, {**call, 'call_id': 'call_b'}, done('resp_1')
    )
    await stream.connection.send(CreateResponse())
    assert await stream.take(5) == snapshot(
        [
            ResponseStarted(response_id='resp_1', answers=(0,)),
            'ToolCall',
            'ToolCall',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
        ]
    )
    await stream.connection.send(ToolResult('call_a', output='a'))
    await stream.connection.send(ToolResult('call_b', output='b'))
    assert json.loads(stream.ws.sent[-1]) == snapshot(
        {
            'type': 'response.create',
            'event_id': 'pydantic_ai.response.2',
            'response': {'metadata': {'pydantic_ai_inputs': '1-2'}},
        }
    )
    stream.feed(refusal('pydantic_ai.response.2'), {'type': 'error', 'error': {'type': 'invalid_request_error'}})
    assert await stream.rest() == snapshot(
        [
            'InputRejected',
            'RealtimeSessionErrorEvent',
            ResponseRequestRefused(input_ids=(1, 2)),
            'RealtimeSessionErrorEvent',
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_response_never_announced_is_started_by_its_first_frame() -> None:
    """A frame naming a response that has ended is left out, and a repeated terminal ends nothing."""
    stream = Stream(
        transcript('resp_1', 'Hi.'),
        done('resp_1'),
        transcript('resp_1', ' again'),
        done('resp_1'),
        done('resp_2', status='cancelled'),
        created('resp_2'),
    )
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1', basis='inferred'),
            'OutputTranscript',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            ResponseStarted(response_id='resp_2', basis='inferred'),
            'ResponseDone',
            ResponseEnded(response_id='resp_2', status='cancelled', provider_details={'status': 'cancelled'}),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_responses_without_ids_get_ids_of_their_own() -> None:
    stream = Stream(created(None), done(None), done(None))
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='pydantic_ai_response_1'),
            'ResponseDone',
            ResponseEnded(
                response_id='pydantic_ai_response_1',
                status='completed',
                finish_reason='stop',
                provider_details={'status': 'completed'},
            ),
            'ResponseDone',
            ResponseStarted(response_id='pydantic_ai_response_2', basis='inferred'),
            ResponseEnded(
                response_id='pydantic_ai_response_2',
                status='completed',
                finish_reason='stop',
                provider_details={'status': 'completed'},
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_an_unreadable_terminal_still_ends_the_response_it_closed() -> None:
    stream = Stream(
        created('resp_1'),
        {'type': 'response.done', 'response': 'garbled'},
        {'type': 'response.done', 'response': 'garbled'},
    )
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1'),
            ResponseEnded(response_id='resp_1', status='lost'),
            'RealtimeSessionErrorEvent',
            'RealtimeSessionErrorEvent',
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_malformed_frame_only_the_lifecycle_reads_is_ignored() -> None:
    stream = Stream({'type': 'input_audio_buffer.committed'}, created('resp_1'), done('resp_1'))
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_response_first_seen_at_its_end_answers_what_its_metadata_names() -> None:
    stream = Stream(
        {**done('resp_1'), 'response': {**done('resp_1')['response'], 'metadata': {'pydantic_ai_inputs': '0'}}}
    )
    await stream.connection.send(CreateResponse())
    assert await stream.take(1) == snapshot([ResponseStarted(response_id='resp_1', answers=(0,))])


async def test_a_repeated_commit_makes_no_second_turn() -> None:
    commit = {'type': 'input_audio_buffer.committed', 'item_id': 'item_u1', 'previous_item_id': None}
    assert await Stream(commit, commit).rest() == snapshot(
        [UserTurnStarted(turn_id='item_u1'), UserTurnEnded(turn_id='item_u1'), 'RealtimeSessionErrorEvent']
    )


async def test_an_item_under_an_id_of_ours_for_no_input_waiting_leaves_order_in_charge() -> None:
    stream = Stream(
        {**user_message_added(), 'item': {**user_message_added()['item'], 'id': 'pydantic_ai_item_999'}},
        user_message_added(),
    )
    await stream.connection.send(TextContext('First.'))
    assert await stream.rest() == snapshot([InputAdded(input_id=0), 'RealtimeSessionErrorEvent'])


async def test_a_reconnect_that_never_succeeds_places_nothing() -> None:
    async def dial() -> Any:
        raise OSError('refused')

    stream = Stream(dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1})
    await stream.connection.send(TextContext('Unacknowledged.'))
    assert await stream.rest() == snapshot(['RealtimeSessionErrorEvent'])


async def test_a_frame_that_fails_to_decode_still_starts_its_response() -> None:
    stream = Stream(
        {'type': 'response.output_audio.delta', 'response_id': 'resp_1', 'delta': 'not base64!'}, done('resp_1')
    )
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1', basis='inferred'),
            'RealtimeSessionErrorEvent',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_metadata_that_is_not_ours_changes_nothing_on_the_codec_stream() -> None:
    frames_ = (
        {**created('resp_1'), 'response': {**created('resp_1')['response'], 'metadata': {'topic': 1}}},
        {**done('resp_1'), 'response': {**done('resp_1')['response'], 'metadata': {'topic': 1}}},
    )
    assert [type(event).__name__ for event in await codec(Stream(*frames_).connection)] == snapshot(
        ['ResponseDone', 'RealtimeSessionErrorEvent']
    )
    assert await Stream(*frames_).rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_request_too_large_to_name_its_inputs_is_still_taken_for_its_response() -> None:
    """Once the server echoes metadata, a response without it is not ours, unless ours carried none."""
    stream = Stream(created('resp_1', answers='0'), done('resp_1'))
    await stream.connection.send(CreateResponse())
    assert len(await stream.take(3)) == 3
    await stream.connection._request_response((10_000,), answers=range(9_000, 10_001))  # pyright: ignore[reportPrivateUsage]
    assert 'response' not in json.loads(stream.ws.sent[-1])
    stream.feed(created('resp_2'), done('resp_2'))
    started = (await stream.take(1))[0]
    assert isinstance(started, ResponseStarted)
    assert (started.basis, len(started.answers)) == snapshot(('inferred', 1001))


async def test_a_barge_in_drops_the_requests_waiting_behind_the_response_it_cut_off() -> None:
    call = {'type': 'response.function_call_arguments.done', 'response_id': 'resp_1', 'name': 'lookup'}
    stream = Stream(created('resp_1', answers='0'), {**call, 'call_id': 'call_a'})
    await stream.connection.send(CreateResponse())
    assert await stream.take(2) == snapshot([ResponseStarted(response_id='resp_1', answers=(0,)), 'ToolCall'])
    await stream.connection.send(ToolResult('call_a', output='a'))
    await stream.connection.send(CreateResponse())
    stream.feed(done('resp_1', status='cancelled'))
    assert await stream.rest() == snapshot(
        [
            'ResponseDone',
            ResponseEnded(response_id='resp_1', status='cancelled', provider_details={'status': 'cancelled'}),
            InputLost(input_ids=(2,)),
            InputLost(input_ids=(1,)),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_closed_connection_loses_what_was_open() -> None:
    stream = Stream(created('resp_1'), {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1'})
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='resp_1'),
            'RealtimeInputSpeechStartEvent',
            UserTurnStarted(turn_id='item_u1'),
            ResponseEnded(response_id='resp_1', status='lost'),
            UserTurnDiscarded(turn_id='item_u1'),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_sideband_that_ends_loses_what_was_open() -> None:
    stream = Stream(created('resp_1'), observes_output_audio=False)
    assert await stream.rest() == snapshot(
        [ResponseStarted(response_id='resp_1'), ResponseEnded(response_id='resp_1', status='lost')]
    )


async def test_a_reconnect_settles_what_the_new_socket_will_never_answer() -> None:
    """The response in flight is lost, and so is the tool-call batch no one will ask to have answered."""
    replacement = FakeWebSocket(frames(user_message_added()))
    replayed = [{'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'Earlier.'}]}]

    async def dial() -> Any:
        stream.connection._history_items_sent(replayed)  # pyright: ignore[reportPrivateUsage]
        return replacement

    call = {'type': 'response.function_call_arguments.done', 'response_id': 'resp_1', 'name': 'lookup'}
    stream = Stream(
        created('resp_1', answers='0'),
        {**call, 'call_id': 'call_a'},
        {**call, 'call_id': 'call_b'},
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
    )
    await stream.connection.send(CreateResponse())
    assert await stream.take(3) == snapshot(
        [ResponseStarted(response_id='resp_1', answers=(0,)), 'ToolCall', 'ToolCall']
    )
    # `call_b` has no output yet, so nothing has asked for the batch's answer.
    await stream.connection.send(ToolResult('call_a', output='a'))
    stream.feed(
        done('resp_1'),
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1'},
        created('resp_2', answers='2'),
        transcript('resp_2', 'Hm.'),
    )
    assert await stream.take(3) == snapshot(
        [
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeInputSpeechStartEvent',
        ]
    )
    # An unacknowledged text turn, answered by `resp_2`, and a request deferred behind it, asked for again.
    await stream.connection.send('Unacknowledged.')
    await stream.connection.send(CreateResponse())
    assert await stream.rest() == snapshot(
        [
            UserTurnStarted(turn_id='item_u1'),
            ResponseStarted(response_id='resp_2', answers=(2,)),
            'OutputTranscript',
            InputAdded(input_id=1),
            InputAdded(input_id=2),
            InputLost(input_ids=(1,)),
            ResponseEnded(response_id='resp_2', status='lost'),
            UserTurnDiscarded(turn_id='item_u1'),
            'RealtimeSessionReconnectEvent',
            InputLost(input_ids=(3,)),
            'RealtimeSessionErrorEvent',
        ]
    )
    assert [json.loads(frame) for frame in replacement.sent] == snapshot(
        [
            {
                'type': 'response.create',
                'event_id': 'pydantic_ai.response.3',
                'response': {'metadata': {'pydantic_ai_inputs': '3'}},
            }
        ]
    )


async def test_a_reconnect_loses_an_unstarted_request_the_caller_cancelled() -> None:
    replacement = FakeWebSocket([])

    async def dial() -> Any:
        return replacement

    stream = Stream(dial=dial, reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1})
    await stream.connection.send('Never mind.')
    await stream.connection.send(CancelResponse())
    assert await stream.rest() == snapshot(
        [
            InputAdded(input_id=0),
            InputLost(input_ids=(0,)),
            'RealtimeSessionReconnectEvent',
            'RealtimeSessionErrorEvent',
        ]
    )
    assert replacement.sent == []


async def test_a_version_1_connection_yields_its_codec_events_as_its_lifecycle_stream() -> None:
    connection = FakeRealtimeConnection([ResponseDone()])
    assert connection._lifecycle_version == 1  # pyright: ignore[reportPrivateUsage]
    assert [event async for event in connection._lifecycle_events()] == [ResponseDone()]  # pyright: ignore[reportPrivateUsage]


def test_response_request_metadata() -> None:
    assert response_request_metadata([1, 2]) == {'pydantic_ai_inputs': '1-2'}
    assert response_request_metadata([]) is None
    assert response_request_metadata(range(1000)) is None
    assert response_metadata_answers({'pydantic_ai_inputs': '1-2'}) == (1, 2)
    assert response_metadata_answers({'pydantic_ai_inputs': 'mine'}) is None
    assert response_metadata_answers(None) is None
