"""Every recorded WebSocket cassette, run through its adapter, obeys the codec lifecycle contract.

The simulator checks the contract on the traces its fake servers produce; this checks it on the real
ones. Each cassette's provider frames are replayed through a fresh connection of the right class (with no
session; the recorded client frames are read only to count the inputs they sent), and the codec events it
yields are fed to the same `LifecycleChecker` the simulator uses. See `_conformance.py` for the rules.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import RealtimeSessionErrorEvent, RealtimeSessionReconnectEvent

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.realtime._lifecycle import (
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
    from pydantic_ai.realtime.codec import (
        AudioDelta,
        RealtimeCodecEvent,
        ResponseDone,
        SessionUsage,
        ToolCall,
        ToolCallCancelled,
    )
    from pydantic_ai.usage import RequestUsage

    from ..ws_cassettes import CassetteClose, CassetteMessage, RealtimeCassette
    from ._cassette_replay import CASSETTES_DIR, replay_codec_events, replay_lifecycle_events, websocket_cassettes
    from ._conformance import LifecycleChecker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='realtime provider SDKs not installed')


@pytest.mark.parametrize(
    'recording',
    [pytest.param(path, id=f'{path.parent.name}/{path.stem}') for path in websocket_cassettes()]
    if imports_successful()
    else [],
)
async def test_cassette_obeys_the_codec_lifecycle(recording: Path) -> None:
    for events in await replay_codec_events(recording):
        checker = LifecycleChecker()
        for event in events:
            checker.feed(event)
        assert checker.issues == []
    for events, inputs_sent in await replay_lifecycle_events(recording):
        checker = LifecycleChecker(lifecycle=True, inputs_sent=lambda sent=inputs_sent: sent)
        for event in events:
            checker.feed(event)
        checker.finish()
        assert checker.issues == []


def _response(event_type: str, response_id: str, status: str, inputs: str) -> dict[str, Any]:
    """A Voice Live `response.created`/`response.done` frame echoing the `response.create` metadata."""
    return {
        'event_id': f'event_{event_type}_{response_id}',
        'type': event_type,
        'response': {
            'object': 'realtime.response',
            'id': response_id,
            'status': status,
            'status_details': None,
            'output': [],
            'usage': None,
            'metadata': {'pydantic_ai_inputs': inputs},
        },
    }


async def test_replay_counts_the_inputs_the_client_sent(tmp_path: Path) -> None:
    """A response's echoed metadata names the inputs it answers, checked against what the client sent.

    The server echoes `response.create`'s metadata, so the lifecycle stream says which inputs each
    response answers. The replay counts the inputs the recorded `response.create` frames name, so an
    answer to one of them isn't flagged as never sent, while an answer naming an input the client never
    sent still is.
    """
    session = {'event_id': 'event_session', 'type': 'session.created', 'session': {'model': 'gpt-realtime'}}
    item = {'id': 'pydantic_ai_item_0', 'type': 'message', 'role': 'user'}
    cassette = RealtimeCassette(
        interactions=[
            CassetteMessage('received', session),
            CassetteMessage(
                'sent', {'type': 'conversation.item.create', 'event_id': 'pydantic_ai.content.0', 'item': item}
            ),
            CassetteMessage(
                'sent',
                {
                    'type': 'response.create',
                    'event_id': 'pydantic_ai.response.0',
                    'response': {'metadata': {'pydantic_ai_inputs': '0'}},
                },
            ),
            CassetteMessage('received', _response('response.created', 'resp_1', 'in_progress', '0')),
            CassetteMessage('received', _response('response.done', 'resp_1', 'completed', '0')),
            CassetteClose(code=1000, reason='', ok=True),
            # Input indexes count up across a session, so input 0 still counts on the next socket, but input
            # 1 was never sent.
            CassetteMessage('received', session),
            CassetteMessage('received', _response('response.created', 'resp_2', 'in_progress', '1')),
            CassetteMessage('received', _response('response.done', 'resp_2', 'completed', '1')),
        ]
    )
    path = tmp_path / 'test_azure_voice_live_ws' / 'echoed_metadata.yaml'
    cassette.dump(path)

    issues: list[list[str]] = []
    for events, inputs_sent in await replay_lifecycle_events(path):
        checker = LifecycleChecker(lifecycle=True, inputs_sent=lambda sent=inputs_sent: sent)
        for event in events:
            checker.feed(event)
        checker.finish()
        issues.append([issue.code for issue in checker.issues])
    assert issues == snapshot([[], ['lifecycle.unknown_answer']])


def feed_all(*events: RealtimeCodecEvent) -> list[str]:
    checker = LifecycleChecker()
    for event in events:
        checker.feed(event)
    return [issue.code for issue in checker.issues]


def test_lifecycle_rules() -> None:
    call = ToolCall('call_1', tool_name='lookup', args='{}', response_id='resp_1')
    done = ResponseDone(provider_response_id='resp_1')
    assert feed_all(call, call) == snapshot(['codec.duplicate_tool_call'])
    assert feed_all(call, ToolCallCancelled(['call_1', 'call_2'])) == snapshot(['codec.unknown_cancellation'])
    assert feed_all(call, ToolCallCancelled(['call_1'])) == snapshot([])
    assert feed_all(done, AudioDelta(b'\x00', response_id='resp_1')) == snapshot(['codec.content_after_terminal'])
    assert feed_all(done, done) == snapshot(['codec.duplicate_terminal'])
    second = AudioDelta(b'\x00', response_id='resp_2')
    assert feed_all(call, second) == snapshot(['codec.overlapping_responses'])
    assert feed_all(call, done, second) == snapshot([])
    assert feed_all(done, RealtimeSessionReconnectEvent(), done) == snapshot([])
    fatal = RealtimeSessionErrorEvent('gone', recoverable=False)
    assert feed_all(fatal, done) == snapshot(['codec.event_after_fatal'])


def feed_lifecycle(*events: RealtimeCodecEvent | LifecycleEvent, inputs_sent: int = 0) -> list[str]:
    checker = LifecycleChecker(lifecycle=True, inputs_sent=lambda: inputs_sent)
    for event in events:
        checker.feed(event)
    checker.finish()
    return [issue.code for issue in checker.issues]


def test_lifecycle_contract_rules() -> None:
    start = ResponseStarted(response_id='resp_1', answers=(0,))
    end = ResponseEnded(response_id='resp_1', status='completed')
    audio = AudioDelta(b'\x00', response_id='resp_1')
    assert feed_lifecycle(start, audio, ResponseDone(provider_response_id='resp_1'), end, inputs_sent=1) == snapshot([])
    assert feed_lifecycle(audio, start, end, start, end, end, inputs_sent=1) == snapshot(
        [
            'lifecycle.content_outside_response',
            'lifecycle.duplicate_start',
            'lifecycle.input_settled_twice',
            'lifecycle.duplicate_end',
            'lifecycle.duplicate_end',
        ]
    )
    assert feed_lifecycle(ResponseEnded(response_id='resp_2', status='lost')) == snapshot(
        ['lifecycle.end_without_start']
    )
    assert feed_lifecycle(start) == snapshot(['lifecycle.unknown_answer', 'lifecycle.unended_at_close'])
    assert feed_lifecycle(
        ResponseStarted(response_id='resp_4', answers=(-1,)),
        ResponseEnded(response_id='resp_4', status='completed'),
        InputLost(input_ids=(0, 0)),
        inputs_sent=1,
    ) == snapshot(['lifecycle.unknown_answer', 'lifecycle.input_settled_twice'])
    assert feed_lifecycle(
        SessionUsage(RequestUsage(), provider_response_id='resp_3'),
        SessionUsage(RequestUsage(), response_scoped=False),
        InputLost(input_ids=(0,)),
        ResponseRequestRefused(input_ids=(0, 1)),
        InputAdded(input_id=0),
        InputAdded(input_id=0),
        inputs_sent=2,
    ) == snapshot(
        ['lifecycle.content_outside_response', 'lifecycle.input_settled_twice', 'lifecycle.input_added_twice']
    )
    turn = UserTurnStarted(turn_id='item_u1')
    # A turn that joined the conversation can still be discarded (it gets no more audio), but only once.
    assert feed_lifecycle(
        turn,
        UserTurnEnded(turn_id='item_u1'),
        UserTurnDiscarded(turn_id='item_u1'),
        UserTurnDiscarded(turn_id='item_u1'),
    ) == snapshot(['lifecycle.turn_end_without_start'])
    assert feed_lifecycle(turn, turn) == snapshot(['lifecycle.turn_started_twice', 'lifecycle.turn_unended_at_close'])
    assert feed_lifecycle(UserTurnEnded(turn_id='item_u2')) == snapshot(['lifecycle.turn_end_without_start'])


async def test_recorded_abnormal_close_is_replayed_as_one() -> None:
    """A socket the recording saw drop with 1011 ends its replay the same way, not as a clean close."""
    # The xAI reconnect recording, whatever it is called: the test drops it with a 1011.
    recording = next(
        path for path in sorted((CASSETTES_DIR / 'test_xai_ws').glob('*.yaml')) if 'code: 1011' in path.read_text()
    )
    dropped, resumed = await replay_codec_events(recording)
    assert isinstance(error := dropped[-1], RealtimeSessionErrorEvent)
    assert '1011' in error.message
    assert isinstance(error := resumed[-1], RealtimeSessionErrorEvent)
    assert '1011' not in error.message
