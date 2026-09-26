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
    from pydantic_ai.realtime.codec import AudioDelta, RealtimeCodecEvent, ResponseDone, ToolCall, ToolCallCancelled

    from ._cassette_replay import replay_codec_events, websocket_cassettes
    from ._conformance import LifecycleChecker

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider SDKs not installed'),
]


@pytest.mark.parametrize(
    'cassette',
    [pytest.param(path, id=f'{path.parent.name}/{path.stem}') for path in websocket_cassettes()]
    if imports_successful()
    else [],
)
async def test_cassette_obeys_the_codec_lifecycle(cassette: Path) -> None:
    for events in await replay_codec_events(cassette):
        checker = LifecycleChecker()
        for event in events:
            checker.feed(event)
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
    assert feed_all(done, RealtimeSessionReconnectEvent(), done) == snapshot([])
    fatal = RealtimeSessionErrorEvent('gone', recoverable=False)
    assert feed_all(fatal, done) == snapshot(['codec.event_after_fatal'])
