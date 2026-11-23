"""Falsifiers for `tests/stream_contract.py`: one hand-built stream per violation code, and the shapes it allows.

These are unit tests because no provider streams most of these shapes on demand; the autouse
`fail_stream_contract_violations` applies the same checker to every stream the rest of the suite iterates.
"""

from __future__ import annotations

import pytest

from pydantic_ai.messages import (
    ModelResponsePart,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from .stream_contract import KNOWN, StreamContractCode, check_stream_events

CALL = ToolCallPart('get_weather', '{"city": ', tool_call_id='call_1')
CALL_DONE = ToolCallPart('get_weather', '{"city": "Paris"}', tool_call_id='call_1')
ARGS = ToolCallPartDelta(args_delta='"Paris"}', tool_call_id='call_1')


@pytest.mark.parametrize(
    ('events', 'built', 'codes'),
    [
        pytest.param(
            [PartDeltaEvent(index=0, delta=TextPartDelta('hi'))],
            [TextPart('hi')],
            ['delta-before-start'],
            id='delta-before-start',
        ),
        pytest.param(
            [PartEndEvent(index=0, part=TextPart('hi'))],
            [TextPart('hi')],
            ['end-before-start'],
            id='end-before-start',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=CALL),
                PartStartEvent(index=1, part=TextPart('and')),
                PartEndEvent(index=0, part=CALL),
                PartDeltaEvent(index=0, delta=ARGS),
                PartEndEvent(index=1, part=TextPart('and')),
            ],
            [CALL_DONE, TextPart('and')],
            ['delta-after-end'],
            id='delta-after-end',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=CALL),
                PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"Paris"}')),
                PartEndEvent(index=0, part=CALL_DONE),
            ],
            [CALL_DONE],
            ['tool-delta-without-id'],
            id='tool-delta-without-id',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=CALL),
                PartEndEvent(index=0, part=ToolCallPart('get_weather', '{"city": "London"}', tool_call_id='call_1')),
                PartStartEvent(index=0, part=ToolCallPart('get_weather', '{"city": "London"}', tool_call_id='call_1')),
                PartEndEvent(index=0, part=ToolCallPart('get_weather', '{"city": "London"}', tool_call_id='call_1')),
            ],
            [ToolCallPart('get_weather', '{"city": "London"}', tool_call_id='call_1')],
            ['end-differs-from-stream'],
            id='end-differs-from-stream',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=NativeToolCallPart('web_search', '{"query": ', tool_call_id='ws_1')),
                PartEndEvent(index=0, part=NativeToolCallPart('web_search', '{"query": ', tool_call_id='ws_1')),
            ],
            [NativeToolCallPart('web_search', '{"query": ', tool_call_id='ws_1')],
            ['native-args-not-json'],
            id='native-args-not-json',
        ),
        pytest.param(
            [PartStartEvent(index=0, part=TextPart('hi')), PartEndEvent(index=0, part=TextPart('hi'))],
            [TextPart('hi'), TextPart('there')],
            ['stream-differs-from-get'],
            id='stream-differs-from-get',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=TextPart('a')),
                PartStartEvent(index=2, part=TextPart('c')),
                PartStartEvent(index=1, part=TextPart('b')),
            ],
            [TextPart('a'), TextPart('b'), TextPart('c')],
            ['replay-differs-from-get'],
            id='replay-differs-from-get',
        ),
    ],
)
def test_each_code_has_a_falsifier(
    events: list[ModelResponseStreamEvent], built: list[ModelResponsePart], codes: list[StreamContractCode]
) -> None:
    assert check_stream_events(events, built) == codes


@pytest.mark.parametrize(
    ('events', 'built'),
    [
        pytest.param(
            [
                PartStartEvent(index=0, part=CALL),
                PartDeltaEvent(index=0, delta=ARGS),
                PartEndEvent(index=0, part=CALL_DONE),
                PartStartEvent(index=1, part=TextPart('done')),
                PartEndEvent(index=1, part=TextPart('done')),
            ],
            [CALL_DONE, TextPart('done')],
            id='streamed-parts',
        ),
        pytest.param(
            [
                PartStartEvent(index=0, part=TextPart('draft')),
                PartEndEvent(index=0, part=TextPart('draft')),
                PartStartEvent(index=0, part=TextPart('final')),
                PartEndEvent(index=0, part=TextPart('final')),
            ],
            [TextPart('final')],
            id='restarted-after-its-end',
        ),
        pytest.param(
            [PartStartEvent(index=2, part=TextPart('new')), PartEndEvent(index=2, part=TextPart('new'))],
            [TextPart('earlier'), TextPart('run'), TextPart('new')],
            id='resumed-after-carried-in-parts',
        ),
    ],
)
def test_allowed_shapes(events: list[ModelResponseStreamEvent], built: list[ModelResponsePart]) -> None:
    assert check_stream_events(events, built) == []


def test_known_entries_name_a_tracking_issue_and_a_repro() -> None:
    for (stream, _code, _part), known in KNOWN.items():
        assert stream.endswith('StreamedResponse')
        assert known.reason.strip() and known.tracked_by.strip()
        assert known.repro.startswith('tests/') and '::' in known.repro
