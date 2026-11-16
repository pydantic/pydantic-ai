"""One hand-built input per violation code in `tests/history_invariants.py`, showing each checker can fail.

A property whose checker no input can fail checks nothing, so every code the property tests in
`test_history_properties.py` rely on has an input here that triggers exactly that code. The body
checkers read request bodies as each provider's mapper produces them, normalized to JSON.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
from pydantic import JsonValue

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    repair_messages,
)

from .conftest import legacy_retry_prompt_part
from .history_invariants import (
    WireBlock,
    WireMessage,
    anthropic_wire,
    bedrock_wire,
    body_changes,
    call_pairing_violations,
    chat_completions_wire,
    duplicate_call_id_violations,
    empty_message_violations,
    empty_text_violations,
    google_wire,
    idempotence_violations,
    openai_responses_wire,
    part_changes,
    round_trip_violations,
    tool_pairing_violations,
)


def _call(call_id: str = 'call_1') -> ToolCallPart:
    return ToolCallPart('lookup', {'q': 'a'}, call_id)


def _return(call_id: str = 'call_1') -> ToolReturnPart:
    return ToolReturnPart('lookup', 'a', call_id)


@pytest.mark.parametrize(
    'history,code',
    [
        pytest.param([ModelRequest(parts=[_return()])], 'tool.orphan_return', id='orphan_return'),
        pytest.param(
            [ModelRequest.user_text_prompt('Look it up'), ModelResponse(parts=[_call()])],
            'tool.missing_return',
            id='missing_return',
        ),
        pytest.param(
            [
                ModelRequest.user_text_prompt('Look it up'),
                ModelResponse(parts=[_call()]),
                ModelRequest.user_text_prompt('Never mind'),
            ],
            'tool.missing_return',
            id='unanswered_by_the_next_request',
        ),
        pytest.param(
            [
                ModelRequest.user_text_prompt('Look it up'),
                ModelResponse(parts=[_call()]),
                ModelRequest(parts=[_return(), _return()]),
            ],
            'tool.duplicate_return',
            id='duplicate_return',
        ),
    ],
)
def test_tool_pairing(history: list[ModelMessage], code: str):
    assert tool_pairing_violations(history) == [code]


def test_tool_pairing_accepts_a_call_answered_in_the_next_request():
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Look it up'),
        ModelResponse(parts=[_call()]),
        ModelRequest(parts=[_return()]),
        ModelResponse(parts=[TextPart('Found it')]),
    ]
    assert tool_pairing_violations(history) == []


def test_part_lost():
    assert part_changes([ModelRequest.user_text_prompt('Hello')], []) == ['history.part_lost']


def test_part_added():
    assert part_changes([], [ModelRequest.user_text_prompt('Hello')]) == ['history.part_added']


def test_part_changes_exempts_what_repair_drops_and_adds():
    """An orphan return may go, and a synthesized return may come."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[_return('call_gone'), UserPromptPart('Look it up')]),
        ModelResponse(parts=[_call()]),
    ]
    repaired = repair_messages(history)
    assert len(repaired) == 3
    assert part_changes(history, repaired) == []


def test_python_round_trip():
    """A deprecated `RetryPromptPart` loads back as the part it always meant, so it doesn't dump the same."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[legacy_retry_prompt_part('Try again', tool_name='lookup', tool_call_id='call_1')])
    ]
    assert round_trip_violations(history, mode='python') == ['history.python_round_trip']


def test_json_round_trip():
    """JSON has no NaN: a NaN in a tool return comes back as `None`."""
    history: list[ModelMessage] = [ModelRequest(parts=[ToolReturnPart('lookup', [float('nan')], 'call_1')])]
    assert round_trip_violations(history, mode='python') == []
    assert round_trip_violations(history, mode='json') == ['history.json_round_trip']


def test_not_idempotent():
    """A repair that appends to every history it is given never settles."""
    history: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    assert idempotence_violations(lambda messages: [*messages, ModelRequest.user_text_prompt('Again')], history) == [
        'history.not_idempotent'
    ]


def test_not_idempotent_in_place():
    """A transform that grows its input in place and returns it is caught against a snapshot of its first output."""

    def grow(messages: list[ModelMessage]) -> list[ModelMessage]:
        messages.append(ModelRequest.user_text_prompt('Again'))
        return messages

    assert idempotence_violations(grow, [ModelRequest.user_text_prompt('Hi')]) == ['history.not_idempotent']


def test_body_changed():
    assert body_changes({'role': 'user', 'content': 'Hello'}, {'role': 'user', 'content': 'Hello!'}) == [
        'wire.body_changed'
    ]


@dataclass(frozen=True)
class WireCase:
    """A well-formed body's messages for one shape, holding every kind of block its projection tells apart."""

    id: str
    project: Callable[[JsonValue], list[WireMessage]]
    messages: JsonValue
    expected: list[WireMessage]


WIRE_CASES = [
    WireCase(
        'chat-completions',
        chat_completions_wire,
        [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Look it up'}, {'type': 'image_url'}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function'}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'a'},
            {'role': 'assistant', 'content': 'Found it'},
        ],
        [
            WireMessage('user', [WireBlock('text', 'Look it up'), WireBlock('other')]),
            WireMessage('assistant', [WireBlock('call', 'call_1')]),
            WireMessage('tool', [WireBlock('result', 'call_1')]),
            WireMessage('assistant', [WireBlock('text', 'Found it')]),
        ],
    ),
    WireCase(
        'openai-responses',
        openai_responses_wire,
        [
            {'role': 'user', 'content': 'Look it up'},
            {'type': 'reasoning', 'summary': []},
            {'type': 'function_call', 'call_id': 'call_1', 'name': 'lookup', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'a'},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Found it'}]},
        ],
        [
            WireMessage('user', [WireBlock('text', 'Look it up')]),
            WireMessage('reasoning', [WireBlock('other')]),
            WireMessage('function_call', [WireBlock('call', 'call_1')]),
            WireMessage('function_call_output', [WireBlock('result', 'call_1')]),
            WireMessage('assistant', [WireBlock('text', 'Found it')]),
        ],
    ),
    WireCase(
        'anthropic',
        anthropic_wire,
        [
            {'role': 'user', 'content': 'Look it up'},
            {
                'role': 'assistant',
                'content': [{'type': 'thinking', 'thinking': 'Hm'}, {'type': 'tool_use', 'id': 'call_1'}],
            },
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Thanks'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Found it'}]},
        ],
        [
            WireMessage('user', [WireBlock('text', 'Look it up')]),
            WireMessage('assistant', [WireBlock('other'), WireBlock('call', 'call_1')]),
            WireMessage('user', [WireBlock('result', 'call_1'), WireBlock('text', 'Thanks')]),
            WireMessage('assistant', [WireBlock('text', 'Found it')]),
        ],
    ),
    WireCase(
        'bedrock',
        bedrock_wire,
        [
            {'role': 'user', 'content': [{'text': 'Look it up'}]},
            {'role': 'assistant', 'content': [{'reasoningContent': {}}, {'toolUse': {'toolUseId': 'call_1'}}]},
            {'role': 'user', 'content': [{'toolResult': {'toolUseId': 'call_1', 'content': []}}]},
            {'role': 'assistant', 'content': [{'text': 'Found it'}]},
        ],
        [
            WireMessage('user', [WireBlock('text', 'Look it up')]),
            WireMessage('assistant', [WireBlock('other'), WireBlock('call', 'call_1')]),
            WireMessage('user', [WireBlock('result', 'call_1')]),
            WireMessage('assistant', [WireBlock('text', 'Found it')]),
        ],
    ),
    WireCase(
        'google',
        google_wire,
        [
            {'role': 'user', 'parts': [{'text': 'Look it up'}]},
            {'role': 'model', 'parts': [{'text': 'Hm', 'thought': True}, {'function_call': {'id': 'call_1'}}]},
            {'role': 'user', 'parts': [{'function_response': {'id': 'call_1'}}]},
            {'role': 'model', 'parts': [{'text': 'Found it'}]},
        ],
        [
            WireMessage('user', [WireBlock('text', 'Look it up')]),
            WireMessage('model', [WireBlock('other'), WireBlock('call', 'call_1')]),
            WireMessage('user', [WireBlock('result', 'call_1')]),
            WireMessage('model', [WireBlock('text', 'Found it')]),
        ],
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in WIRE_CASES])
def test_well_formed_body_passes_every_body_checker(case: WireCase):
    wire = case.project(case.messages)
    assert wire == case.expected
    assert duplicate_call_id_violations(wire, across_messages=True) == []
    assert call_pairing_violations(wire, results_follow_their_call=True) == []
    assert (
        empty_message_violations(
            wire, system_suffices=False, empty_assistant_turns_pass=False, empty_user_turns_pass=False
        )
        == []
    )
    assert empty_text_violations(wire) == []


def test_duplicate_call_id():
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}, {'id': 'call_0'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'b'},
        ]
    )
    assert duplicate_call_id_violations(wire, across_messages=False) == ['wire.duplicate_call_id']


def test_duplicate_call_id_across_messages():
    """An id an earlier message used counts only where the API rejects it across messages too."""
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'b'},
        ]
    )
    assert duplicate_call_id_violations(wire, across_messages=False) == []
    assert duplicate_call_id_violations(wire, across_messages=True) == ['wire.duplicate_call_id']


def test_orphan_result_without_a_call():
    wire = openai_responses_wire([{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'a'}])
    assert call_pairing_violations(wire, results_follow_their_call=False) == ['wire.orphan_result']


def test_orphan_result_after_another_message():
    """Where results must answer the message right before theirs, an earlier call isn't enough."""
    wire = anthropic_wire(
        [
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'call_1'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Wait'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Waiting'}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1'}]},
        ]
    )
    assert call_pairing_violations(wire, results_follow_their_call=False) == []
    assert call_pairing_violations(wire, results_follow_their_call=True) == ['wire.orphan_result']


def test_results_in_their_own_messages_follow_their_call():
    """Chat Completions sends each result as its own `tool` message; the run answers the assistant message before it."""
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}, {'id': 'call_1'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'b'},
        ]
    )
    assert call_pairing_violations(wire, results_follow_their_call=True) == []


def test_result_after_a_text_turn_does_not_follow_its_call():
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}]},
            {'role': 'user', 'content': 'meanwhile'},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
        ]
    )
    assert call_pairing_violations(wire, results_follow_their_call=False) == []
    assert call_pairing_violations(wire, results_follow_their_call=True) == ['wire.orphan_result']


def test_repeated_call_answered_once():
    """A call repeated under one id needs a result per call; one result leaves the other call unanswered."""
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}, {'id': 'call_0'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
        ]
    )
    assert call_pairing_violations(wire, results_follow_their_call=False) == ['wire.unanswered_call']
    assert call_pairing_violations(wire, results_follow_their_call=True) == ['wire.unanswered_call']


def test_second_result_for_one_call_is_an_orphan():
    wire = chat_completions_wire(
        [
            {'role': 'assistant', 'tool_calls': [{'id': 'call_0'}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'a'},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'b'},
        ]
    )
    assert call_pairing_violations(wire, results_follow_their_call=False) == ['wire.orphan_result']
    assert call_pairing_violations(wire, results_follow_their_call=True) == ['wire.orphan_result']


def test_result_after_a_mixed_message_does_not_follow_its_call():
    """A message that mixes results with text closes the run of result messages that follows a call.

    Built from wire messages directly: the projections combine consecutive same-role turns, as their APIs do.
    """
    wire = [
        WireMessage('assistant', [WireBlock('call', 'call_0'), WireBlock('call', 'call_1')]),
        WireMessage('user', [WireBlock('result', 'call_0'), WireBlock('text', 'And?')]),
        WireMessage('user', [WireBlock('result', 'call_1')]),
    ]
    assert call_pairing_violations(wire, results_follow_their_call=False) == []
    assert call_pairing_violations(wire, results_follow_their_call=True) == ['wire.orphan_result']


def test_unanswered_call():
    wire = google_wire([{'role': 'model', 'parts': [{'function_call': {'id': 'call_1', 'name': 'lookup'}}]}])
    assert call_pairing_violations(wire, results_follow_their_call=False) == ['wire.unanswered_call']


def test_empty_body():
    wire = anthropic_wire([])
    assert empty_message_violations(
        wire, system_suffices=False, empty_assistant_turns_pass=False, empty_user_turns_pass=False
    ) == ['wire.empty_body']
    assert (
        empty_message_violations(
            wire, system_suffices=True, empty_assistant_turns_pass=False, empty_user_turns_pass=False
        )
        == []
    )


def test_empty_message():
    wire = chat_completions_wire([{'role': 'assistant', 'content': None}])
    assert empty_message_violations(
        wire, system_suffices=False, empty_assistant_turns_pass=False, empty_user_turns_pass=False
    ) == ['wire.empty_message']
    assert (
        empty_message_violations(
            wire, system_suffices=False, empty_assistant_turns_pass=True, empty_user_turns_pass=False
        )
        == []
    )


def test_empty_user_turn():
    wire = chat_completions_wire([{'role': 'user', 'content': []}])
    assert empty_message_violations(
        wire, system_suffices=False, empty_assistant_turns_pass=False, empty_user_turns_pass=False
    ) == ['wire.empty_message']
    assert (
        empty_message_violations(
            wire, system_suffices=False, empty_assistant_turns_pass=False, empty_user_turns_pass=True
        )
        == []
    )


def test_empty_message_outside_an_assistant_turn():
    wire = chat_completions_wire([{'role': 'user', 'content': None}])
    assert empty_message_violations(
        wire, system_suffices=False, empty_assistant_turns_pass=True, empty_user_turns_pass=False
    ) == ['wire.empty_message']


def test_empty_text():
    wire = anthropic_wire([{'role': 'user', 'content': [{'type': 'text', 'text': ''}]}])
    assert empty_text_violations(wire) == ['wire.empty_text']


def test_whitespace_only_text():
    wire = bedrock_wire([{'role': 'user', 'content': [{'text': ' \n'}]}])
    assert empty_text_violations(wire) == ['wire.empty_text']


def test_whitespace_only_text_passes_in_an_assistant_turn():
    """Anthropic and Bedrock take a whitespace-only text block in an assistant turn, but not an empty one."""
    assert empty_text_violations(bedrock_wire([{'role': 'assistant', 'content': [{'text': ' '}]}])) == []
    assert empty_text_violations(bedrock_wire([{'role': 'assistant', 'content': [{'text': ''}]}])) == [
        'wire.empty_text'
    ]
