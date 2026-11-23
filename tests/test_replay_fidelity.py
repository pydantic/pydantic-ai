"""Checks of `tests/replay_fidelity.py`: its table against the wire shapes and recordings, and one falsifier per code.

The falsifiers are unit tests because each needs a conversation that drops a field or misses a cache read, and a
cassette edited to hold one would be an exchange no provider produced. The autouse `fail_replay_fidelity_violations`
applies the same checks to every cassette the rest of the suite replays.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import get_args

import pytest
from cassetter import Body, Cassette, HttpInteraction, HttpRequest, HttpResponse, RecordedRequest, RecordMode

from .cassette_utils import TESTS_DIR
from .manifests import assert_classified
from .replay_fidelity import (
    ECHO_FIELDS,
    Echoes,
    EchoName,
    ReplayFidelityCode,
    WireShape,
    iter_echo_checks,
    replay_fidelity_violations,
)

ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages'
MODEL = 'claude-sonnet-4-5'
QUESTION: dict[str, object] = {'role': 'user', 'content': [{'type': 'text', 'text': 'What is the weather in Paris?'}]}
TOOL_RESULT: dict[str, object] = {
    'role': 'user',
    'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_01', 'content': 'Sunny'}],
}
CACHED_SYSTEM: list[object] = [
    {'type': 'text', 'text': 'You are a weather assistant.', 'cache_control': {'type': 'ephemeral'}}
]


def tool_use(tool_use_id: str) -> dict[str, object]:
    return {'type': 'tool_use', 'id': tool_use_id, 'name': 'get_weather', 'input': {'city': 'Paris'}}


def usage(*, read: int, written: int) -> dict[str, object]:
    return {
        'input_tokens': 20,
        'output_tokens': 10,
        'cache_read_input_tokens': read,
        'cache_creation_input_tokens': written,
    }


def exchange(request: Mapping[str, object], response: Mapping[str, object]) -> HttpInteraction:
    return HttpInteraction(
        HttpRequest('POST', ANTHROPIC_URL, {}, Body('json', request)), HttpResponse(200, {}, Body('json', response)), ''
    )


def sent(request: Mapping[str, object]) -> RecordedRequest:
    return RecordedRequest('POST', ANTHROPIC_URL, {}, json.dumps(request))


def tool_conversation(*, echoed_id: str) -> tuple[list[HttpInteraction], list[RecordedRequest]]:
    """A played tool call, and a next request that sends its call back under `echoed_id`."""
    first: dict[str, object] = {'model': MODEL, 'messages': [QUESTION]}
    second: dict[str, object] = {
        'model': MODEL,
        'messages': [QUESTION, {'role': 'assistant', 'content': [tool_use('toolu_01')]}, TOOL_RESULT],
    }
    sent_second: dict[str, object] = {
        'model': MODEL,
        'messages': [QUESTION, {'role': 'assistant', 'content': [tool_use(echoed_id)]}, TOOL_RESULT],
    }
    interactions = [
        exchange(first, {'content': [tool_use('toolu_01')], 'usage': usage(read=0, written=0)}),
        exchange(second, {'content': [{'type': 'text', 'text': 'Sunny.'}], 'usage': usage(read=0, written=0)}),
    ]
    return interactions, [sent(first), sent(sent_second)]


def cached_conversation(*, read: int) -> tuple[list[HttpInteraction], list[RecordedRequest]]:
    """A request that writes a 2,000-token prefix to the cache, and a next request that extends it and reads `read`."""
    first: dict[str, object] = {'model': MODEL, 'system': CACHED_SYSTEM, 'messages': [QUESTION]}
    second: dict[str, object] = {
        'model': MODEL,
        'system': CACHED_SYSTEM,
        'messages': [
            QUESTION,
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Sunny.'}]},
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'And in Oslo?', 'cache_control': {'type': 'ephemeral'}}],
            },
        ],
    }
    interactions = [
        exchange(first, {'content': [{'type': 'text', 'text': 'Sunny.'}], 'usage': usage(read=0, written=2000)}),
        exchange(second, {'content': [{'type': 'text', 'text': 'Cloudy.'}], 'usage': usage(read=read, written=40)}),
    ]
    return interactions, [sent(first), sent(second)]


@dataclass(frozen=True)
class Case:
    id: str
    conversation: tuple[list[HttpInteraction], list[RecordedRequest]]
    codes: list[ReplayFidelityCode]


CASES = [
    Case('tool-call-id-sent-back', tool_conversation(echoed_id='toolu_01'), []),
    Case('tool-call-id-replaced', tool_conversation(echoed_id='toolu_02'), ['field-not-echoed']),
    Case('cache-write-read', cached_conversation(read=2000), []),
    Case('cache-write-unread', cached_conversation(read=0), ['cache-not-read']),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
def test_violation_codes(case: Case):
    interactions, requests = case.conversation
    assert [violation.code for violation in replay_fidelity_violations(interactions, requests)] == case.codes


def test_every_wire_shape_is_classified():
    assert_classified(get_args(WireShape), ECHO_FIELDS, complete_walk=True, table_name='ECHO_FIELDS')


@pytest.fixture(scope='module')
def fields_sent_back() -> set[tuple[WireShape, EchoName]]:
    """The `ECHO_FIELDS` entries a recorded conversation sends back, its recorded requests standing in for sent ones.

    Cassettes are scanned in path order until every entry is found.
    """
    wanted = {
        (shape, name) for shape, echoes in ECHO_FIELDS.items() if isinstance(echoes, Echoes) for name in echoes.fields
    }
    found: set[tuple[WireShape, EchoName]] = set()
    for path in sorted(TESTS_DIR.rglob('*.yaml')):
        if found >= wanted:
            break
        if '/cassettes/' not in path.as_posix() or path.read_text(encoding='utf-8').count('\n- request:') < 2:
            continue
        cassette = Cassette(path, record_mode=RecordMode.NONE)
        cassette.load()
        found |= {
            (check.shape, check.field)
            for check in iter_echo_checks(cassette.interactions, cassette.requests)
            if check.echoed
        }
    return found


def test_every_echo_field_is_sent_back_in_a_recording(fields_sent_back: set[tuple[WireShape, EchoName]]):
    stale = [
        f'{shape}: {name}'
        for shape, echoes in ECHO_FIELDS.items()
        if isinstance(echoes, Echoes)
        for name in echoes.fields
        if (shape, name) not in fields_sent_back
    ]
    assert not stale, (
        f'`ECHO_FIELDS` entries {stale} are sent back in no recorded conversation: fix how the check reads them, or '
        'remove them'
    )
