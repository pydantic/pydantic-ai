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
    InputAdded,
    InputLost,
    LifecycleEvent,
    OutputItemDetails,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    UserTurnClosed,
    UserTurnEnded,
    UserTurnStarted,
)

from ..conftest import try_import
from ..realtime_connections import lifecycle_events

with try_import() as imports_successful:
    from pydantic_ai.messages import BinaryAudio
    from pydantic_ai.realtime._openai_lifecycle import OpenAILifecycle
    from pydantic_ai.realtime._openai_protocol import (
        ACTIVE_RESPONSE_RETRIES,
        response_metadata_answers,
        response_request_metadata,
    )
    from pydantic_ai.realtime.codec import (
        CancelResponse,
        CommitAudio,
        CreateResponse,
        RealtimeCodecEvent,
        TextContext,
        ToolResult,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

    from .test_openai import FakeWebSocket

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')


def created(response_id: str | None, *, answers: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {'object': 'realtime.response', 'status': 'in_progress', 'output': []}
    if response_id is not None:
        response['id'] = response_id
    if answers is not None:
        response['metadata'] = {'pydantic_ai_inputs': answers}
    return {'type': 'response.created', 'response': response}


def done(response_id: str | None, *, status: str = 'completed', reason: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {'object': 'realtime.response', 'status': status, 'output': []}
    if reason is not None:
        response['status_details'] = {'type': status, 'reason': reason}
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
    return event if isinstance(event, LifecycleEvent) else type(event).__name__


async def codec(connection: OpenAIRealtimeConnection) -> list[RealtimeCodecEvent]:
    """The codec events the connection yields to a session."""
    return [event async for event in connection]


class Stream:
    """A connection's lifecycle stream, read a few events at a time so the test can act in between."""

    def __init__(self, *incoming: dict[str, Any], **kwargs: Any) -> None:
        self.ws = FakeWebSocket(frames(*incoming))
        self.connection = OpenAIRealtimeConnection(self.ws, **kwargs)  # pyright: ignore[reportArgumentType]
        self._events = aiter(lifecycle_events(self.connection))

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
            UserTurnClosed(turn_id='item_u2'),
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


async def test_an_unreadable_terminal_ends_a_response_created_without_an_id() -> None:
    stream = Stream(
        created(None),
        {'type': 'response.done', 'response': 'garbled'},
        created('resp_2'),
        done('resp_2'),
    )
    assert await stream.rest() == snapshot(
        [
            ResponseStarted(response_id='pydantic_ai_response_1'),
            ResponseEnded(response_id='pydantic_ai_response_1', status='lost'),
            'RealtimeSessionErrorEvent',
            ResponseStarted(response_id='resp_2'),
            'ResponseDone',
            ResponseEnded(
                response_id='resp_2', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
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
            UserTurnClosed(turn_id='item_u1'),
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
            UserTurnClosed(turn_id='item_u1'),
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


@pytest.mark.parametrize('transcribes', [True, False])
async def test_a_reconnect_ends_the_spoken_turns_whose_transcript_it_loses(transcribes: bool) -> None:
    """A committed turn stays in the conversation, but its transcript will never come on the new socket."""

    async def dial() -> Any:
        return FakeWebSocket([])

    def committed(item_id: str) -> dict[str, Any]:
        return {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None}

    kwargs: dict[str, Any] = {} if transcribes else {'input_transcription_enabled': False}
    stream = Stream(
        committed('item_u1'),
        # xAI's and Azure's interim snapshot of a transcript still to be completed.
        {
            'type': 'conversation.item.input_audio_transcription.completed',
            'item_id': 'item_u1',
            'transcript': 'So',
            'status': 'in_progress',
        },
        committed('item_u2'),
        {'type': 'conversation.item.input_audio_transcription.delta', 'item_id': 'item_u2', 'delta': 'Good'},
        committed('item_u3'),
        {'type': 'conversation.item.input_audio_transcription.completed', 'item_id': 'item_u3', 'transcript': 'Hi.'},
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
        **kwargs,
    )
    events = [event for event in await stream.rest() if isinstance(event, UserTurnClosed)]
    assert events == ([UserTurnClosed(turn_id='item_u1'), UserTurnClosed(turn_id='item_u2')] if transcribes else [])


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


def test_response_request_metadata() -> None:
    assert response_request_metadata([1, 2]) == {'pydantic_ai_inputs': '1-2'}
    assert response_request_metadata([]) is None
    assert response_request_metadata(range(1000)) is None
    assert response_metadata_answers({'pydantic_ai_inputs': '1-2'}) == (1, 2)
    assert response_metadata_answers({'pydantic_ai_inputs': 'mine'}) is None
    assert response_metadata_answers(None) is None


async def test_an_idle_timeout_nudge_is_no_user_turn() -> None:
    """With `idle_timeout_ms`, the server commits an empty audio item to nudge the model: nobody spoke."""
    stream = Stream(
        {
            'type': 'input_audio_buffer.timeout_triggered',
            'item_id': 'item_idle',
            'audio_start_ms': 0,
            'audio_end_ms': 0,
        },
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_idle', 'previous_item_id': None},
        user_message_added('input_audio'),
        created('resp_1'),
        done('resp_1'),
    )
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


async def test_speech_starts_that_never_stopped_merge_into_the_next() -> None:
    """Semantic VAD can report a burst of speech starts and commit only the last one."""
    stream = Stream(
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_a'},
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_b'},
        {'type': 'input_audio_buffer.speech_stopped', 'item_id': 'item_b'},
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_b', 'previous_item_id': None},
    )
    assert await stream.rest() == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            UserTurnStarted(turn_id='item_a'),
            'RealtimeInputSpeechStartEvent',
            UserTurnClosed(turn_id='item_a'),
            UserTurnStarted(turn_id='item_b'),
            'RealtimeInputSpeechEndEvent',
            UserTurnEnded(turn_id='item_b'),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_xai_places_a_spoken_turn_when_it_adds_its_item() -> None:
    """xAI adds the turn's item at speech start and can reply before the commit; a clear leaves it in place."""
    stream = Stream(
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1'},
        {**user_message_added('input_audio'), 'item': {**user_message_added('input_audio')['item'], 'id': 'item_u1'}},
        created('resp_1'),
        {'type': 'input_audio_buffer.cleared'},
        {'type': 'input_audio_buffer.speech_stopped', 'item_id': 'item_u1'},
        {'type': 'input_audio_buffer.speech_stopped', 'item_id': ''},
        done('resp_1'),
    )
    assert await stream.rest() == snapshot(
        [
            'RealtimeInputSpeechStartEvent',
            UserTurnStarted(turn_id='item_u1'),
            UserTurnEnded(turn_id='item_u1', still_speaking=True),
            ResponseStarted(response_id='resp_1', user_turn_id='item_u1'),
            UserTurnClosed(turn_id='item_u1'),
            'RealtimeInputSpeechEndEvent',
            'RealtimeInputSpeechEndEvent',
            'ResponseDone',
            ResponseEnded(
                response_id='resp_1', status='completed', finish_reason='stop', provider_details={'status': 'completed'}
            ),
            'RealtimeSessionErrorEvent',
        ]
    )


def active_refusal(event_id: str) -> dict[str, Any]:
    refused = refusal(event_id)
    refused['error']['code'] = 'conversation_already_has_active_response'
    return refused


def requests_sent(ws: FakeWebSocket) -> list[dict[str, Any]]:
    return [frame for frame in map(json.loads, ws.sent) if frame['type'] == 'response.create']


SPOKEN_TURN = (
    {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u'},
    {'type': 'conversation.item.added', 'item': {'id': 'item_u', 'type': 'message', 'role': 'user'}},
)
"""A spoken turn server VAD commits, which the response it starts on its own answers."""

OUR_ITEM_ADDED = {
    'type': 'conversation.item.added',
    'item': {'id': 'pydantic_ai_item_0', 'type': 'message', 'role': 'user'},
}


async def test_a_request_refused_after_the_response_that_overtook_it_is_asked_again() -> None:
    """As recorded live: server VAD starts a response before our item is added, and refuses our request after it ends.

    That response never saw the input, so it answers nothing of ours: the request is asked for again, and the
    response that starts answers the input. The refusal is handled, so it is reported to nobody.
    """
    stream = Stream(
        *SPOKEN_TURN,
        created('resp_vad'),
        OUR_ITEM_ADDED,
        done('resp_vad'),
        active_refusal('pydantic_ai.response.0'),
        created('resp_2', answers='0'),
        done('resp_2'),
    )
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert [event for event in events if isinstance(event, (ResponseStarted, InputAdded)) or 'Error' in str(event)] == (
        snapshot(
            [
                ResponseStarted(response_id='resp_vad', user_turn_id='item_u'),
                InputAdded(input_id=0),
                ResponseStarted(response_id='resp_2', answers=(0,)),
                'RealtimeSessionErrorEvent',
            ]
        )
    )
    assert [request['event_id'] for request in requests_sent(stream.ws)] == [
        'pydantic_ai.response.0',
        'pydantic_ai.response.0',
    ]


async def test_a_request_refused_while_the_response_that_overtook_it_goes_on_waits_for_its_end() -> None:
    """The refusal can also come while that response is under way, or before it is even reported."""
    for incoming in (
        (created('resp_vad'), OUR_ITEM_ADDED, active_refusal('pydantic_ai.response.0')),
        (active_refusal('pydantic_ai.response.0'), created('resp_vad'), OUR_ITEM_ADDED),
    ):
        stream = Stream(*SPOKEN_TURN, *incoming)
        await stream.connection.send('Hello?')
        await stream.take(len(incoming) + 1)
        assert len(requests_sent(stream.ws)) == 1
        stream.feed(done('resp_vad'), created('resp_2', answers='0'))
        events = await stream.rest()
        assert len(requests_sent(stream.ws)) == 2
        assert [event for event in events if isinstance(event, ResponseStarted)] == [
            ResponseStarted(response_id='resp_2', answers=(0,))
        ]


async def test_a_request_the_provider_keeps_refusing_is_given_up_on() -> None:
    """Each response server VAD starts refuses it again: after `ACTIVE_RESPONSE_RETRIES` times, the refusal is reported."""
    incoming: list[dict[str, Any]] = [*SPOKEN_TURN]
    for number in range(ACTIVE_RESPONSE_RETRIES + 1):
        incoming += [created(f'resp_vad{number}'), done(f'resp_vad{number}'), active_refusal('pydantic_ai.response.0')]
    stream = Stream(*incoming)
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert len(requests_sent(stream.ws)) == ACTIVE_RESPONSE_RETRIES + 1
    assert [
        event
        for event in events
        if isinstance(event, ResponseRequestRefused) or event in ('InputRejected', 'RealtimeSessionErrorEvent')
    ] == snapshot(
        [
            'InputRejected',
            'RealtimeSessionErrorEvent',
            ResponseRequestRefused(input_ids=(0,)),
            'RealtimeSessionErrorEvent',
        ]
    )


async def test_a_reconnect_loses_a_request_the_connection_no_longer_counts_as_active() -> None:
    """The old socket neither started nor refused it, and the new one isn't asked again: it is lost."""

    async def dial() -> Any:
        return FakeWebSocket([])

    # A stray `response.done` for an earlier response ends what the connection thought was active.
    stream = Stream(
        created('resp_1', answers='0'),
        done('resp_1'),
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
    )
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
    stream.feed(done('resp_1'))
    events = await stream.rest()
    assert [event for event in events if isinstance(event, InputLost)] == snapshot([InputLost(input_ids=(1,))])


async def test_an_assistant_messages_phase_is_kept_for_its_part() -> None:
    """gpt-realtime-2.x labels each assistant message `commentary` or `final_answer`."""

    def item_added(item: object) -> dict[str, Any]:
        return {'type': 'response.output_item.added', 'response_id': 'resp_1', 'item': item}

    message: dict[str, Any] = {'type': 'message', 'role': 'assistant', 'content': []}
    stream = Stream(
        created('resp_1'),
        item_added({**message, 'id': 'item_a1', 'phase': 'commentary'}),
        item_added({**message, 'id': 'item_a2'}),
        item_added({**message, 'phase': 'final_answer'}),
        item_added('not an item'),
        {
            'type': 'response.output_item.added',
            'response_id': '',
            'item': {**message, 'id': 'item_a3', 'phase': 'final_answer'},
        },
        done('resp_1'),
    )
    events = [event for event in await stream.rest() if isinstance(event, OutputItemDetails)]
    assert events == snapshot(
        [OutputItemDetails(response_id='resp_1', item_id='item_a1', provider_details={'phase': 'commentary'})]
    )


async def test_what_we_sent_ahead_of_our_commit_joins_the_conversation_before_its_turn() -> None:
    """The provider handles frames in order: inputs not acknowledged yet when our commit went out came first."""
    stream = Stream(
        {'type': 'conversation.item.added', 'item': user_message_added()['item'] | {'id': 'pydantic_ai_item_1'}},
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_u1', 'previous_item_id': None},
        {'type': 'conversation.item.added', 'item': user_message_added()['item'] | {'id': 'pydantic_ai_item_0'}},
        tool_output_added('call_a'),
    )
    await stream.connection.send('First.')
    await stream.connection.send('Second.')
    await stream.connection.send(ToolResult('call_a', output='a'))
    await stream.connection.send(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    await stream.connection.send(CommitAudio())
    assert [event for event in await stream.rest() if not isinstance(event, str)] == snapshot(
        [
            InputAdded(input_id=1),
            InputAdded(input_id=0),
            InputAdded(input_id=2),
            UserTurnStarted(turn_id='item_u1'),
            UserTurnEnded(turn_id='item_u1'),
            InputLost(input_ids=(1, 2, 0)),
        ]
    )


async def test_each_of_our_commits_places_what_was_sent_ahead_of_it() -> None:
    """Two commits sent before the first is acknowledged: each turn follows only what went out before its commit."""

    def committed(item_id: str) -> dict[str, Any]:
        return {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None}

    stream = Stream(committed('item_u1'), committed('item_u2'))
    audio = BinaryAudio(data=b'\x00\x00', media_type='audio/pcm')
    await stream.connection.send('First.')
    await stream.connection.send(audio)
    await stream.connection.send(CommitAudio())
    await stream.connection.send('Second.')
    await stream.connection.send(audio)
    await stream.connection.send(CommitAudio())
    events = [event for event in await stream.rest() if isinstance(event, (InputAdded, UserTurnEnded))]
    assert events == snapshot(
        [
            InputAdded(input_id=0),
            UserTurnEnded(turn_id='item_u1'),
            InputAdded(input_id=3),
            UserTurnEnded(turn_id='item_u2'),
        ]
    )


async def test_a_commit_that_fails_to_go_out_places_nothing() -> None:
    class _FailingCommit(FakeWebSocket):
        async def send(self, data: str) -> None:
            if 'input_audio_buffer.commit' in data:
                raise OSError('gone')
            await super().send(data)

    ws = _FailingCommit(
        frames({'type': 'input_audio_buffer.committed', 'item_id': 'item_u1', 'previous_item_id': None})
    )
    connection = OpenAIRealtimeConnection(ws)  # pyright: ignore[reportArgumentType]
    await connection.send('First.')
    with pytest.raises(OSError):
        await connection.send(CommitAudio())
    events = [event async for event in lifecycle_events(connection) if isinstance(event, (InputAdded, UserTurnEnded))]
    assert events == snapshot([UserTurnEnded(turn_id='item_u1')])


async def test_a_turn_cleared_before_it_joined_never_does() -> None:
    """A late `speech_stopped` for audio already cleared makes no turn of it, and an idle item is forgotten once used."""
    stream = Stream(
        {'type': 'input_audio_buffer.speech_started', 'item_id': 'item_u1', 'audio_start_ms': 0},
        {'type': 'input_audio_buffer.cleared'},
        {'type': 'input_audio_buffer.speech_stopped', 'item_id': 'item_u1', 'audio_end_ms': 500},
        {
            'type': 'input_audio_buffer.timeout_triggered',
            'item_id': 'item_idle',
            'audio_start_ms': 0,
            'audio_end_ms': 0,
        },
        {'type': 'input_audio_buffer.committed', 'item_id': 'item_idle', 'previous_item_id': None},
    )
    events = [event for event in await stream.rest() if not isinstance(event, str)]
    assert events == snapshot([UserTurnStarted(turn_id='item_u1'), UserTurnClosed(turn_id='item_u1')])
    assert stream.connection._lifecycle._idle_items == set()  # pyright: ignore[reportPrivateUsage]


def test_a_failed_commit_drops_only_what_it_noted() -> None:
    """Another commit noted meanwhile stays, whichever of the two fails; one a reconnect already forgot is nothing to drop."""
    lifecycle = OpenAILifecycle()
    lifecycle.message_sent(0)
    first = lifecycle.audio_commit_sent()
    second = lifecycle.audio_commit_sent()
    # The two note the same inputs, so only identity tells them apart.
    assert first.sent_before == second.sent_before
    lifecycle.audio_commit_failed(second)
    assert [id(noted) for noted in lifecycle._commits] == [id(first)]  # pyright: ignore[reportPrivateUsage]
    third = lifecycle.audio_commit_sent()
    lifecycle.audio_commit_failed(first)
    assert [id(noted) for noted in lifecycle._commits] == [id(third)]  # pyright: ignore[reportPrivateUsage]
    lifecycle.socket_replaced()
    lifecycle.audio_commit_failed(third)
    assert not lifecycle._commits  # pyright: ignore[reportPrivateUsage]


async def test_a_reconnect_asks_again_for_a_request_a_provider_response_overtook() -> None:
    """The old socket dropped before refusing (or starting) our request: the new one is asked for it."""
    replacement = FakeWebSocket([])

    async def dial() -> Any:
        return replacement

    stream = Stream(
        *SPOKEN_TURN,
        created('resp_vad'),
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
    )
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert ResponseStarted(response_id='resp_vad', user_turn_id='item_u') in events
    assert [request['event_id'] for request in requests_sent(replacement)] == ['pydantic_ai.response.0']


def test_giving_up_on_a_request_already_settled_refuses_nothing() -> None:
    lifecycle = OpenAILifecycle()
    lifecycle.requests_dropped([0])
    lifecycle.requests_refused([0])
    assert lifecycle.take_pending() == snapshot([InputLost(input_ids=(0,))])


async def test_a_request_a_provider_response_overtook_and_then_refused_for_good_is_not_asked_again() -> None:
    """Refused for any other reason, it is settled: a reconnect doesn't ask for it again."""
    replacement = FakeWebSocket([])

    async def dial() -> Any:
        return replacement

    stream = Stream(
        *SPOKEN_TURN,
        created('resp_vad'),
        refusal('pydantic_ai.response.0'),
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
    )
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert ResponseRequestRefused(input_ids=(0,)) in events
    assert requests_sent(replacement) == []


async def test_a_request_a_provider_response_overtook_is_lost_with_the_connection() -> None:
    """Neither refused nor started when the connection closes for good: no response will answer it."""
    stream = Stream(*SPOKEN_TURN, created('resp_vad'))
    await stream.connection.send('Hello?')
    assert InputLost(input_ids=(0,)) in await stream.rest()


# The frames below are cut from live probes of OpenAI's and Azure's GA API under server VAD (October 2026): a commit
# of ours went out at the moment marked, and the frames are those the server sent, in order, with their ids
# shortened. Under server VAD, the provider commits only speech its VAD heard: VAD's own commit always comes right
# after the `speech_stopped` for its item, and the acknowledgement of a commit of ours never does.


def speech_started(item_id: str) -> dict[str, Any]:
    return {'type': 'input_audio_buffer.speech_started', 'item_id': item_id, 'audio_start_ms': 0}


def speech_stopped(item_id: str) -> dict[str, Any]:
    return {'type': 'input_audio_buffer.speech_stopped', 'item_id': item_id, 'audio_end_ms': 4992}


def audio_committed(item_id: str) -> dict[str, Any]:
    return {'type': 'input_audio_buffer.committed', 'item_id': item_id, 'previous_item_id': None}


def audio_item_added(item_id: str) -> dict[str, Any]:
    return {
        'type': 'conversation.item.added',
        'item': {'id': item_id, 'type': 'message', 'role': 'user', 'content': [{'type': 'input_audio'}]},
    }


def commit_empty(event_id: str | None) -> dict[str, Any]:
    return {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'code': 'input_audio_buffer_commit_empty',
            'message': 'Error committing input audio buffer: buffer too small.',
            'event_id': event_id,
        },
    }


_AUDIO = BinaryAudio(data=b'\x01\x00' * 2_400, media_type='audio/pcm')


def turn_events(events: list[LifecycleEvent | str]) -> list[LifecycleEvent | str]:
    return [
        event for event in events if isinstance(event, (UserTurnStarted, UserTurnEnded, UserTurnClosed, InputAdded))
    ]


async def test_vad_commit_and_a_refusal_naming_ours_leave_our_mid_speech_commit_waiting() -> None:
    """`race_then_speech`: a commit sent as VAD stopped (refused as empty, naming it), the user speaking on, a second
    commit mid-speech, and the session closing before that one's acknowledgement (`committed` for the new turn) is
    read. VAD's commit acknowledges neither of ours, and the refusal takes back the commit it names, so the second
    is still waiting: it ends the turn under way."""
    stream = Stream(
        speech_started('item_x'),
        speech_stopped('item_x'),
        audio_committed('item_x'),
        audio_item_added('item_x'),
        commit_empty('pydantic_ai.commit.1'),
        speech_started('item_y'),
    )
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    assert [json.loads(sent).get('event_id') for sent in stream.ws.sent if 'commit' in sent] == snapshot(
        ['pydantic_ai.commit.1', 'pydantic_ai.commit.3']
    )
    assert turn_events(await stream.rest()) == snapshot(
        [
            UserTurnStarted(turn_id='item_x'),
            UserTurnEnded(turn_id='item_x'),
            UserTurnStarted(turn_id='item_y'),
            UserTurnEnded(turn_id='item_y'),
        ]
    )


async def test_a_refusal_naming_no_commit_of_ours_takes_back_the_oldest() -> None:
    """A provider that echoes no id of ours (or none at all) refuses its oldest commit not acknowledged: the one sent
    as VAD stopped, so the one mid-speech still ends the turn under way."""
    stream = Stream(
        speech_started('item_x'),
        speech_stopped('item_x'),
        audio_committed('item_x'),
        commit_empty(None),
        speech_started('item_y'),
    )
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    assert turn_events(await stream.rest()) == snapshot(
        [
            UserTurnStarted(turn_id='item_x'),
            UserTurnEnded(turn_id='item_x'),
            UserTurnStarted(turn_id='item_y'),
            UserTurnEnded(turn_id='item_y'),
        ]
    )


async def test_an_unacknowledged_commit_under_vad_with_no_speech_under_way_makes_no_turn() -> None:
    """`after_commit`: VAD committed the turn (and started its reply), then a commit of ours went out of the audio
    streamed after it: silence, which the provider refuses as empty. The session closing before the refusal is read
    records no turn for it."""
    stream = Stream(speech_started('item_x'), speech_stopped('item_x'), audio_committed('item_x'), created('resp_1'))
    await stream.connection.send(_AUDIO)
    assert turn_events(await stream.take(5)) == snapshot(
        [UserTurnStarted(turn_id='item_x'), UserTurnEnded(turn_id='item_x')]
    )
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    assert turn_events(await stream.rest()) == snapshot([])


async def test_a_commit_mid_speech_is_acknowledged_with_the_turn_under_way() -> None:
    """`mid_speech`: our commit commits the turn VAD heard start (no `speech_stopped` before its `committed`), and VAD
    goes on with a new item, which it stops and commits itself, never having reported its start."""
    stream = Stream(
        speech_started('item_x'),
        audio_committed('item_x'),
        audio_item_added('item_x'),
        speech_stopped('item_z'),
        audio_committed('item_z'),
        audio_item_added('item_z'),
    )
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    assert turn_events(await stream.rest()) == snapshot(
        [
            UserTurnStarted(turn_id='item_x'),
            UserTurnEnded(turn_id='item_x'),
            UserTurnStarted(turn_id='item_z'),
            UserTurnEnded(turn_id='item_z'),
        ]
    )
    assert not stream.connection._lifecycle._commits  # pyright: ignore[reportPrivateUsage]


async def test_a_commit_as_new_speech_starts_is_acknowledged_with_it() -> None:
    """`onset`: VAD committed the turn, our commit went out as the user started again, and the provider reports that
    speech starting and commits it for ours; VAD then commits the rest itself."""
    stream = Stream(
        speech_started('item_x'),
        speech_stopped('item_x'),
        audio_committed('item_x'),
        audio_item_added('item_x'),
        speech_started('item_y'),
        audio_committed('item_y'),
        audio_item_added('item_y'),
        speech_stopped('item_z'),
        audio_committed('item_z'),
    )
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    assert turn_events(await stream.rest()) == snapshot(
        [
            UserTurnStarted(turn_id='item_x'),
            UserTurnEnded(turn_id='item_x'),
            UserTurnStarted(turn_id='item_y'),
            UserTurnEnded(turn_id='item_y'),
            UserTurnStarted(turn_id='item_z'),
            UserTurnEnded(turn_id='item_z'),
        ]
    )
    assert not stream.connection._lifecycle._commits  # pyright: ignore[reportPrivateUsage]


async def test_input_sent_ahead_of_a_commit_is_placed_by_its_own_acknowledgement_after_a_vad_commit() -> None:
    """VAD's commit comes before the acknowledgement of a text input sent ahead of our commit: the provider had the
    input after it committed the turn (it acknowledges each in order), so the input follows the reply VAD started."""
    stream = Stream(
        speech_started('item_x'),
        speech_stopped('item_x'),
        audio_committed('item_x'),
        audio_item_added('item_x'),
        created('resp_1'),
        user_message_added(),
        commit_empty('pydantic_ai.commit.2'),
    )
    await stream.connection.send(TextContext('Some context.'))
    await stream.connection.send(_AUDIO)
    await stream.connection.send(CommitAudio())
    events = await stream.rest()
    assert [event for event in events if isinstance(event, (UserTurnEnded, InputAdded, ResponseStarted))] == snapshot(
        [
            UserTurnEnded(turn_id='item_x'),
            ResponseStarted(response_id='resp_1', user_turn_id='item_x'),
            InputAdded(input_id=0),
        ]
    )


@pytest.mark.parametrize('refused_first', [False, True])
async def test_a_request_overtaken_by_a_response_the_user_cut_off_is_dropped(refused_first: bool) -> None:
    """The user barged in on the response that overtook our request: their speech takes the turn, as it does from a
    request deferred behind that response, so the request is dropped whenever its refusal comes."""
    cancelled = done('resp_vad', status='cancelled', reason='turn_detected')
    refused = active_refusal('pydantic_ai.response.0')
    stream = Stream(
        *SPOKEN_TURN,
        created('resp_vad'),
        OUR_ITEM_ADDED,
        *((refused, cancelled) if refused_first else (cancelled, refused)),
    )
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert InputLost(input_ids=(0,)) in events
    assert 'RealtimeSessionErrorEvent' not in events[:-1]
    assert len(requests_sent(stream.ws)) == 1


@pytest.mark.parametrize(
    ('reason', 'asked_again'),
    [
        # The user barged in on the response that overtook the request: their speech took the turn.
        ('turn_detected', False),
        # Cancelled for another reason (xAI cancels a response of its own to start ours): still owed.
        (None, True),
    ],
)
async def test_a_reconnect_asks_again_for_an_overtaken_request_unless_the_user_cut_that_response_off(
    reason: str | None, asked_again: bool
) -> None:
    replacement = FakeWebSocket([])

    async def dial() -> Any:
        return replacement

    stream = Stream(
        *SPOKEN_TURN,
        created('resp_vad'),
        OUR_ITEM_ADDED,
        done('resp_vad', status='cancelled', reason=reason),
        dial=dial,
        reconnect={'base_delay': 0.0, 'max_attempts': 1, 'max_reconnects': 1},
    )
    await stream.connection.send('Hello?')
    events = await stream.rest()
    assert bool(requests_sent(replacement)) is asked_again
    # Lost with the reconnect, or (asked again) with the replacement, which closes before answering.
    assert InputLost(input_ids=(0,)) in events
