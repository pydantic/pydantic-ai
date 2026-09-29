"""Every recorded WebSocket cassette, run through its adapter, obeys the codec lifecycle contract.

The simulator checks the contract on the traces its fake servers produce; this checks it on the real
ones. Each cassette's provider frames are replayed through a fresh connection of the right class (with no
session, and the recorded client frames ignored), and the codec events it yields are fed to the same
`LifecycleChecker` the simulator uses. See `_conformance.py` for the rules.
"""

from __future__ import annotations as _annotations

from pathlib import Path

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

    from ._cassette_replay import replay_codec_events, replay_lifecycle_events, websocket_cassettes
    from ._conformance import LifecycleChecker

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider SDKs not installed'),
]


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
    for events in await replay_lifecycle_events(recording):
        checker = LifecycleChecker(lifecycle=True)
        for event in events:
            checker.feed(event)
        checker.finish()
        assert checker.issues == []


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
    assert feed_lifecycle(turn, UserTurnEnded(turn_id='item_u1'), UserTurnDiscarded(turn_id='item_u1')) == snapshot(
        ['lifecycle.turn_end_without_start']
    )
    assert feed_lifecycle(turn, turn) == snapshot(['lifecycle.turn_started_twice', 'lifecycle.turn_unended_at_close'])
