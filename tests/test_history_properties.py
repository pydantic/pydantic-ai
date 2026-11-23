"""Properties of message histories, of `repair_messages`, and of the request bodies each model maps them to.

Histories are drawn by `tests/history_strategies.py`, and every property is a checker in
`tests/history_invariants.py` that `test_history_invariants.py` shows can fail.

The body properties call each model's private message mapper instead of recording requests. They are
about the request body itself, over many histories per run: no request is sent, and a cassette holds
one recorded body per test and isn't always matched on it. Each history goes through what an agent run
does before mapping, in its order: `repair_messages`, `Model.prepare_messages` with the run's parameters,
`repair_messages` again, then `Model.prepare_request`, whose parameters the mapper gets. Every body test also
checks P1, that mapping doesn't raise: no drawn history is one a model should refuse, so a `UserError` fails a
test too, except the one refusal a case declares (`MapperCase.refusal`).

A property that fails on main for drawn histories is a known finding: `KNOWN_FINDINGS` leaves the damage that
draws it out of that property's histories, and a strict expected failure on a hand-built reproducer pins it.
"""

from __future__ import annotations as _annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, assert_never

import httpx2
import pydantic_core
import pytest
from _pytest.mark import ParameterSet
from hypothesis import Phase, assume, given, settings, strategies as st
from pydantic import JsonValue

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import (
    FilePart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.models import Model, override_allow_model_requests
from pydantic_ai.models.decision import DecisionModel

from .conftest import try_import
from .history_invariants import (
    body_changes,
    call_field_violations,
    call_pairing_violations,
    duplicate_call_id_violations,
    empty_frame_violations,
    empty_message_violations,
    empty_text_violations,
    idempotence_violations,
    json_strings,
    part_changes,
    round_trip_violations,
    stale_reference_violations,
    tool_pairing_violations,
)
from .history_mappers import BASELINE, BODY_CASES, MAPPER_CASES, NO_TEXT_TO_JUDGE, MapperCase, prepare
from .history_strategies import (
    IGNORE_RANDOM_STATE_NOTE,
    PART_STRATEGIES,
    PROPERTY_SETTINGS,
    SERVER_REF_SHAPES,
    Damage,
    Drawn,
    ServerRefField,
    histories,
)
from .manifests import assert_classified, union_members
from .model_adapter_cases import SdkCell, sdk_cells, shipped_model_classes
from .model_adapter_transports import XaiTransport

with try_import():
    import grpc
    import grpc.aio

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = IGNORE_RANDOM_STATE_NOTE


@PROPERTY_SETTINGS
@given(histories())
def test_history_survives_serialization(history: list[ModelMessage]):
    """`ModelMessagesTypeAdapter` loads back what it dumped, in Python and in JSON."""
    assert round_trip_violations(history, mode='python') == []
    assert round_trip_violations(history, mode='json') == []


def _params(cases: list[MapperCase]) -> list[ParameterSet]:
    return [pytest.param(case, id=case.id, marks=case.marks) for case in cases]


@pytest.mark.parametrize('case', _params(BODY_CASES))
async def test_projection_reads_the_mapped_body(case: MapperCase):
    assert case.wire((await case.body(BASELINE)).messages) == case.baseline


CheckedProperty = Literal[
    'call_ids_are_unique',
    'results_answer_calls',
    'messages_are_not_empty',
    'text_is_not_empty',
    'deterministic',
    'frames_are_not_empty',
    'call_fields_survive',
    'calls_pair_after_preparation',
    'references_stay_on_their_api',
]

_ENDPOINT_BOUND_REFS: frozenset[ServerRefField] = frozenset(
    {'response_id', 'message_id', 'reasoning_id', 'function_call_id'}
)
"""The server references that name state on the endpoint that issued them. A tool search call id isn't one: it is also
the call's `tool_call_id`, which another model sends back as the id of the local call it maps the search to."""


def _foreign_references(case: MapperCase, history: list[ModelMessage]) -> set[str]:
    """The server references in `history` that an API other than `case.api` issued."""
    patterns = [
        re.compile(shape.pattern)
        for (api, field), shape in SERVER_REF_SHAPES.items()
        if api != case.api and field in _ENDPOINT_BOUND_REFS
    ]
    dumped: JsonValue = pydantic_core.from_json(ModelMessagesTypeAdapter.dump_json(history))
    return {value for value in json_strings(dumped) if any(pattern.fullmatch(value) for pattern in patterns)}


async def _violations(checked: CheckedProperty, case: MapperCase, history: list[ModelMessage]) -> list[str]:
    """What the `checked` property finds wrong with the body `case` maps `history` to."""
    body = await case.body(history)
    wire = case.wire(body.messages)
    match checked:
        case 'call_ids_are_unique':
            return duplicate_call_id_violations(
                wire, across_messages=case.id in REPEATED_CALL_ID_ACROSS_RESPONSES_REJECTIONS
            )
        case 'results_answer_calls':
            return call_pairing_violations(wire, results_follow_their_call=case.results_follow_their_call)
        case 'messages_are_not_empty':
            return empty_message_violations(
                wire,
                system_suffices=case.takes_system_alone and body.system is not None,
                empty_assistant_turns_pass=case.takes_empty_assistant_turns,
                empty_user_turns_pass=case.takes_empty_user_turns,
            )
        case 'text_is_not_empty':
            return empty_text_violations(wire)
        case 'deterministic':
            stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
            return body_changes(body, await case.body(history)) + body_changes(body, await case.body(stored))
        case 'frames_are_not_empty':
            return empty_frame_violations(wire)
        case 'call_fields_survive':
            calls = [
                part
                for message in body.projected
                for part in message.parts
                if isinstance(part, ToolCallPart | NativeToolCallPart)
            ]
            return call_field_violations(calls, wire, system=case.model().system)
        case 'calls_pair_after_preparation':
            return tool_pairing_violations(body.projected)
        case 'references_stay_on_their_api':
            return stale_reference_violations(_foreign_references(case, history), [body.system, body.messages])
        case _:
            assert_never(checked)


# --- known findings ------------------------------------------------------------------------------------
#
# A known finding is a property that fails on main for drawn histories. Its entry leaves the damage that draws it
# out of that property on the providers it fails on, and `test_known_finding_is_still_present` pins it there with
# a hand-built reproducer, as a strict expected failure that starts passing once the finding is fixed: then delete
# the entry. Each pin quotes how the probed API rejected the reproducer, probed 2026-10-23 (UTC). Bedrock was
# probed on us.anthropic.claude-sonnet-4-6, as the probe account can't call the us.anthropic.claude-sonnet-5 the
# suite maps with. A finding whose body no API rejects, such as an empty frame, quotes what the body sends.


@dataclass(frozen=True)
class Pin:
    """A hand-built reproducer of a known finding, on one provider."""

    case_id: str | None
    """The mapper case the reproducer fails on, or `None` for a history `repair_messages` leaves broken. Every
    mapper maps what repair leaves, so the finding's damage is then left out of its property on every case, and
    out of the repair property."""
    history: list[ModelMessage]
    rejection: str
    """The probed model and its error, verbatim, or, for a body no API rejects, what it sends instead."""
    label: str = ''
    """Tells apart two reproducers on one provider."""


@dataclass(frozen=True)
class KnownFinding:
    id: str
    fails: CheckedProperty
    damage: frozenset[Damage]
    """The damage that draws the finding, left out of `fails` on the providers it is pinned on."""
    tracked_by: str
    pins: tuple[Pin, ...]

    @property
    def breaks_repair(self) -> bool:
        return any(pin.case_id is None for pin in self.pins)

    @property
    def providers(self) -> frozenset[str]:
        if self.breaks_repair:
            return frozenset(case.id for case in BODY_CASES)
        return frozenset(pin.case_id for pin in self.pins if pin.case_id is not None)


_REPEATED_CALL_ID: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Look both up')]),
    ModelResponse(parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0'), ToolCallPart('lookup', {'q': 'b'}, 'call_0')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), ToolReturnPart('lookup', 'b', 'call_0')]),
]

_REPEATED_CALL_ID_ACROSS_RESPONSES: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Look a up, then b')]),
    ModelResponse(parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0')]),
    ModelResponse(parts=[ToolCallPart('lookup', {'q': 'b'}, 'call_0')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'b', 'call_0')]),
]

_WHITESPACE_PROMPT: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Hi')]),
    ModelResponse(parts=[TextPart('Hello')]),
    ModelRequest(parts=[UserPromptPart(' ')]),
]

REPEATED_CALL_ID_REJECTIONS = {
    'openai-chat': "gpt-5.5: 400 \"Invalid parameter: messages with role 'tool' must be a response to a "
    'preceeding message '  # codespell:ignore preceeding
    "with 'tool_calls'.\" (the same body with distinct ids was accepted)",
    'anthropic': 'claude-sonnet-5-5: 400 "messages.1.content.1: `tool_use` ids must be unique"',
    'google': 'gemini-3.5-flash: 400 "Request contains an invalid argument." '
    '(the same body with distinct ids was accepted)',
    'bedrock': 'us.anthropic.claude-sonnet-4-6: 400 ValidationException '
    '"The toolUse blocks at messages.2.content contain duplicate Ids: call_0"',
    'mistral': 'mistral-large-latest: 400 "Duplicate tool call id in assistant message"',
    'cohere': 'command-r7b-12-2024: 400 "invalid tool call provided in messages[1].tool_calls[1]: '
    "cannot have duplicate tool ids 'call_0'\"",
}
"""The APIs that rejected a call id repeated in one response, and how. OpenAI Responses (gpt-5.5), Groq
(openai/gpt-oss-120b) and the Hugging Face router (Qwen/Qwen3-235B-A22B through `provider_name='auto'`) accepted it.
The misspelling in OpenAI's error is its own."""

REPEATED_CALL_ID_ACROSS_RESPONSES_REJECTIONS = {
    'cohere': 'command-r7b-12-2024: 400 "invalid tool call provided in messages[3].tool_calls[0]: '
    "cannot have duplicate tool ids 'call_0'\"",
}
"""The APIs that also rejected a call id an earlier response used, and how. OpenAI Chat, OpenAI Responses, Anthropic,
Google, Bedrock, Groq, Mistral and the Hugging Face router accepted it, so on them an id is unique per message."""

KNOWN_FINDINGS = [
    KnownFinding(
        'F1',
        'call_ids_are_unique',
        frozenset({'repeated_call_id'}),
        'https://github.com/pydantic/pydantic-ai/issues/11120',
        tuple(Pin(case_id, _REPEATED_CALL_ID, rejection) for case_id, rejection in REPEATED_CALL_ID_REJECTIONS.items()),
    ),
    KnownFinding(
        'F1b',
        'call_ids_are_unique',
        frozenset({'repeated_call_id_across_responses'}),
        'https://github.com/pydantic/pydantic-ai/issues/11120',
        tuple(
            Pin(case_id, _REPEATED_CALL_ID_ACROSS_RESPONSES, rejection)
            for case_id, rejection in REPEATED_CALL_ID_ACROSS_RESPONSES_REJECTIONS.items()
        ),
    ),
    KnownFinding(
        'F3',
        'text_is_not_empty',
        frozenset({'empty_system_prompt'}),
        'https://github.com/pydantic/pydantic-ai/issues/11121',
        (
            Pin(
                'anthropic',
                [
                    ModelRequest(parts=[UserPromptPart('Hi')]),
                    ModelResponse(parts=[TextPart('Hello')]),
                    ModelRequest(parts=[UserPromptPart('Again'), SystemPromptPart('')]),
                ],
                'claude-sonnet-5-5: 400 "messages.3: system text blocks must contain non-whitespace text"',
            ),
        ),
    ),
    KnownFinding(
        'F4',
        'text_is_not_empty',
        frozenset({'empty_prompt', 'empty_response_text'}),
        'https://github.com/pydantic/pydantic-ai/issues/11122',
        (
            Pin(
                'bedrock',
                [ModelRequest(parts=[UserPromptPart('')])],
                'us.anthropic.claude-sonnet-4-6: 400 ValidationException "The text field in the ContentBlock object '
                'at messages.0.content.0 is blank. Add text to the text field, and try again."',
                label='empty-prompt',
            ),
            Pin(
                'bedrock',
                [
                    ModelRequest(parts=[UserPromptPart('Hi')]),
                    ModelResponse(parts=[TextPart('Hello'), TextPart('')]),
                    ModelRequest(parts=[UserPromptPart('Again')]),
                ],
                'us.anthropic.claude-sonnet-4-6: 400 ValidationException "The model returned the following errors: '
                'messages: text content blocks must be non-empty"',
                label='empty-response-text',
            ),
            Pin(
                'bedrock',
                _WHITESPACE_PROMPT,
                'us.anthropic.claude-sonnet-4-6: 400 ValidationException "The model returned the following errors: '
                'messages: text content blocks must contain non-whitespace text"',
                label='whitespace-prompt',
            ),
        ),
    ),
    KnownFinding(
        'F5',
        'messages_are_not_empty',
        frozenset({'empty_prompt', 'empty_parts'}),
        'https://github.com/pydantic/pydantic-ai/issues/11123',
        (
            Pin(
                'anthropic',
                [ModelRequest(parts=[UserPromptPart('')])],
                'claude-sonnet-5-5: 400 "messages: at least one message is required"',
                label='empty-prompt',
            ),
            Pin(
                'anthropic',
                [ModelRequest(parts=[], instructions='Be brief.')],
                'claude-sonnet-5-5: 400 "messages: at least one message is required"',
                label='instructions-only',
            ),
        ),
    ),
    KnownFinding(
        'F6',
        'text_is_not_empty',
        frozenset({'empty_prompt'}),
        'https://github.com/pydantic/pydantic-ai/issues/11124',
        (
            Pin(
                'anthropic',
                _WHITESPACE_PROMPT,
                'claude-sonnet-5-5: 400 "messages: text content blocks must contain non-whitespace text"',
            ),
        ),
    ),
    KnownFinding(
        'F7',
        'results_answer_calls',
        frozenset({'reanswered_call_id'}),
        'https://github.com/pydantic/pydantic-ai/issues/11180',
        (
            Pin(
                None,
                [
                    ModelRequest(parts=[UserPromptPart('Look it up')]),
                    ModelResponse(parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0')]),
                    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0')]),
                    ModelResponse(parts=[TextPart('It is a.')]),
                    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), UserPromptPart('Thanks')]),
                ],
                'repair keeps the second return; claude-sonnet-5-5: 400 "messages.4.content.0: unexpected '
                '`tool_use_id` found in `tool_result` blocks: call_0. Each `tool_result` block must have a '
                'corresponding `tool_use` block in the previous message."',
                label='repair',
            ),
        ),
    ),
    KnownFinding(
        'F8',
        'messages_are_not_empty',
        frozenset({'empty_prompt'}),
        'https://github.com/pydantic/pydantic-ai/issues/11181',
        tuple(
            Pin(case_id, [ModelRequest(parts=[UserPromptPart([])])], rejection)
            for case_id, rejection in {
                'groq': "openai/gpt-oss-120b: 400 \"'messages.0' : for 'role:user' the following must be satisfied"
                "[('messages.0.content' : one of the following must be satisfied[('messages.0.content' : value must "
                "be a string) OR ('messages.0.content' : minimum number of items is 1)])]\"",
                'mistral': 'mistral-large-latest: 400 "No messages contain any content"',
                'cohere': 'command-r7b-12-2024: 400 "missing required parameter: \'messages[0].content\'"',
            }.items()
        ),
    ),
    KnownFinding(
        'F9',
        'frames_are_not_empty',
        frozenset({'empty_system_prompt'}),
        'to file: an empty mid-conversation SystemPromptPart is wrapped as <system></system>',
        tuple(
            Pin(case_id, [ModelRequest(parts=[UserPromptPart('Hi'), SystemPromptPart('')])], sent)
            for case_id, sent in {
                'google': 'not probed; sends {"text": "<system></system>"} in the user turn',
                'bedrock': 'not probed; sends {"text": "<system></system>"} in the user turn',
                'snowflake': 'not probed; sends a user message "<system></system>"',
                'xai': 'not probed; sends a ROLE_USER message "<system verified></system>"',
            }.items()
        ),
    ),
    KnownFinding(
        'F10',
        'frames_are_not_empty',
        frozenset({'empty_signed_thinking'}),
        'to file: Groq and Hugging Face send an empty ThinkingPart as bare think tags',
        tuple(
            Pin(
                case_id,
                [
                    ModelRequest(parts=[UserPromptPart('Hi')]),
                    ModelResponse(
                        parts=[ThinkingPart('', signature='AA==', provider_name='openai'), TextPart('The answer.')]
                    ),
                    ModelRequest(parts=[UserPromptPart('Again')]),
                ],
                'not probed; sends the assistant turn "<think>\\n\\n</think>\\n\\nThe answer."',
            )
            for case_id in ('groq', 'huggingface')
        ),
    ),
    KnownFinding(
        'F11',
        'results_answer_calls',
        frozenset({'empty_parts'}),
        'to file: the framing note lands between tool calls and their results when a history opens with speech',
        (
            Pin(
                'snowflake',
                [
                    ModelResponse(
                        parts=[
                            SpeechPart(speaker='assistant', transcript='Hi! Let me check.'),
                            ToolCallPart('lookup', {}, 'call_1'),
                        ]
                    ),
                    ModelRequest(parts=[ToolReturnPart('lookup', 'sunny', 'call_1')]),
                    ModelResponse(parts=[ThinkingPart('They asked about the weather.'), TextPart('It is sunny.')]),
                    ModelRequest(parts=[UserPromptPart('Thanks')], instructions='Be brief.'),
                ],
                'not probed; sends roles system, assistant (tool_calls), system, tool',
            ),
        ),
    ),
    KnownFinding(
        'F12',
        'messages_are_not_empty',
        frozenset({'empty_parts'}),
        'to file: OpenRouterModel sends messages: [] when the only content is a thinking-only response',
        (
            Pin(
                'openrouter',
                [ModelRequest(parts=[]), ModelResponse(parts=[ThinkingPart('Hmm.')])],
                'not probed; sends "messages": []',
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    'finding,pin',
    [
        pytest.param(
            finding,
            pin,
            id='-'.join(filter(None, [finding.id, pin.case_id, pin.label])),
            marks=(
                *next((case.marks for case in BODY_CASES if case.id == pin.case_id), ()),
                pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason=f'{finding.id} ({finding.tracked_by}): {pin.rejection}',
                ),
            ),
        )
        for finding in KNOWN_FINDINGS
        for pin in finding.pins
    ],
)
async def test_known_finding_is_still_present(finding: KnownFinding, pin: Pin):
    if pin.case_id is None:
        assert tool_pairing_violations(repair_messages(pin.history)) == []
    else:
        case = next(case for case in BODY_CASES if case.id == pin.case_id)
        assert await _violations(finding.fails, case, pin.history) == []


# --- properties --------------------------------------------------------------------------------------


_REPAIR_BREAKING = frozenset[Damage]().union(*(finding.damage for finding in KNOWN_FINDINGS if finding.breaks_repair))
"""The damage of every known finding repair leaves broken, which every mapper maps as repair leaves it."""


@PROPERTY_SETTINGS
@given(histories(well_formed=False, without=_REPAIR_BREAKING))
def test_repair_makes_tool_calls_pair_up(history: list[ModelMessage]):
    """Repair answers every call once in the next request, keeps every part but orphan returns, and settles."""
    repaired = repair_messages(history)
    assert tool_pairing_violations(repaired) == []
    assert part_changes(history, repaired) == []
    assert idempotence_violations(repair_messages, history) == []


# The body properties draw each history inside the test, since a known finding's damage is left out only for the
# cases it fails on. They are sync: Hypothesis ends a draw that runs out of data with a `BaseException`, which raised
# inside an async test breaks the anyio test runner for every later example.


def _check(checked: CheckedProperty, case: MapperCase, data: st.DataObject) -> list[str]:
    """Draw a damaged history, less the damage of each known finding of `checked` on `case`, and check its body."""
    known = [finding.damage for finding in KNOWN_FINDINGS if finding.fails == checked and case.id in finding.providers]
    if checked == 'calls_pair_after_preparation':
        # A history repair leaves unpaired fails before preparation runs: that is the repair finding's to pin.
        known.append(_REPAIR_BREAKING)
    history = data.draw(histories(well_formed=False, without=frozenset[Damage]().union(*known)))
    with asyncio.Runner() as runner:
        try:
            return runner.run(_violations(checked, case, history))
        except UserError as error:
            assume(case.refusal is None or not str(error).startswith(case.refusal))
            raise


@pytest.mark.parametrize('case', _params([case for case in BODY_CASES if case.id in REPEATED_CALL_ID_REJECTIONS]))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_call_ids_are_unique(case: MapperCase, data: st.DataObject):
    """P2: no call id is sent twice in one message, nor in one body where the API rejects that too, on the APIs that
    reject one sent twice."""
    assert _check('call_ids_are_unique', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_results_answer_calls(case: MapperCase, data: st.DataObject):
    """P2: every result answers an earlier call (where required, one of the message right before), and every call
    is answered."""
    assert _check('results_answer_calls', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_messages_are_not_empty(case: MapperCase, data: st.DataObject):
    """P4: the body has a message, and every message has content, except what the API takes without."""
    assert _check('messages_are_not_empty', case, data) == []


@pytest.mark.parametrize('case', _params([case for case in BODY_CASES if case.rejects_empty_text]))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_text_is_not_empty(case: MapperCase, data: st.DataObject):
    """P4: no text block is empty or whitespace-only, where the API rejects one."""
    assert _check('text_is_not_empty', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapping_is_deterministic(case: MapperCase, data: st.DataObject):
    """P5: mapping a history again, or after a JSON round trip, gives the same body."""
    assert _check('deterministic', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_frames_are_not_empty(case: MapperCase, data: st.DataObject):
    """P6: no part with nothing to say is sent in its frame, such as empty thinking as `<think>\\n\\n</think>`."""
    assert _check('frames_are_not_empty', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_calls_keep_their_fields(case: MapperCase, data: st.DataObject):
    """P7: every call the body sends goes with the name and arguments the history holds for it, a native call under
    the function its provider ran."""
    assert _check('call_fields_survive', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_prepared_calls_pair_up(case: MapperCase, data: st.DataObject):
    """P8: on an API that rejects an unpaired tool call, what `prepare_messages` makes of a repaired history still
    answers every call once, before the run's second repair could hide a call or result it lost, as when it turns
    another provider's native tool search into a local one. Elsewhere it keeps the history as translated."""
    if not case.rejects_unpaired_tool_calls:
        pytest.skip(f'`{case.id}` does not set `rejects_unpaired_tool_calls`')
    assert _check('calls_pair_after_preparation', case, data) == []


@pytest.mark.parametrize('case', _params(BODY_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_references_stay_on_their_api(case: MapperCase, data: st.DataObject):
    """P9: a body carries no response, item or reasoning id another API issued: the state it names lives on that API's
    endpoint, as a Responses reasoning id means nothing to Chat Completions."""
    assert _check('references_stay_on_their_api', case, data) == []


# --- manifests -----------------------------------------------------------------------------------------


def test_every_part_kind_is_drawn_or_exempt():
    """Every request and response part kind has a strategy or a reason it has none."""
    members = union_members(ModelRequestPart) | union_members(ModelResponsePart)
    assert_classified(members, PART_STRATEGIES, complete_walk=True, table_name='PART_STRATEGIES')


def _holds(kind: type, history: list[ModelMessage]) -> bool:
    """Whether `history` holds a part of `kind` that no more specific kind in `PART_STRATEGIES` claims."""
    return any(
        isinstance(part, kind)
        and not any(
            other is not kind and issubclass(other, kind) and isinstance(part, other) for other in PART_STRATEGIES
        )
        for message in history
        for part in message.parts
    )


def test_every_drawn_part_kind_appears():
    """Every kind the table says is drawn turns up in a batch of drawn histories."""
    drawn = [kind for kind, entry in PART_STRATEGIES.items() if isinstance(entry, Drawn)]
    seen: set[type] = set()

    # Generation only: one pass over a fixed batch, without shrinking, which `find` would run for each kind.
    @settings(max_examples=200, derandomize=True, database=None, deadline=None, phases=[Phase.generate])
    @given(histories())
    def draw(history: list[ModelMessage]):
        seen.update(kind for kind in drawn if _holds(kind, history))

    draw()
    assert [kind.__name__ for kind in drawn if kind not in seen] == []


_TESTS = Path(__file__).parent


@pytest.mark.parametrize('key', list(SERVER_REF_SHAPES), ids='-'.join)
def test_server_ref_shapes_match_their_cassettes(key: tuple[str, str]):
    """Each drawn server reference has the shape its cassette recorded."""
    shape = SERVER_REF_SHAPES[key]  # pyright: ignore[reportArgumentType]
    recorded = (_TESTS / shape.cassette).read_text()
    assert re.search(rf'(?<![A-Za-z0-9_]){shape.pattern}(?![A-Za-z0-9_])', recorded), shape


_REQUEST_CELLS = [cell for cell in sdk_cells() if cell.entry_point == 'request']


def test_every_requesting_model_class_has_a_mapper_case():
    """Every model class that sends a request has a mapper case, or a reason it has none."""
    _, skipped = shipped_model_classes()
    assert_classified(
        {cell.model_class.__name__ for cell in _REQUEST_CELLS},
        MAPPER_CASES,
        complete_walk=not skipped,
        table_name='MAPPER_CASES',
    )


@dataclass
class _Refusal:
    """A provider that answers every request it receives with a 400, so a request that reaches it ends there."""

    reached: int = 0

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.reached += 1
        return httpx2.Response(400, json={'error': {'message': 'probe', 'type': 'probe', 'code': 'probe'}})

    async def unary(self) -> object:
        self.reached += 1
        raise grpc.aio.AioRpcError(
            grpc.StatusCode.INVALID_ARGUMENT, grpc.aio.Metadata(), grpc.aio.Metadata(), details='probe'
        )

    async def stream(self) -> AsyncIterator[object]:
        yield await self.unary()


async def _send(model: Model, cell: SdkCell, history: list[ModelMessage]) -> UserError | None:
    """Send `history` through `model` as an agent run prepares it, with `cell`'s parameters."""
    prepared = prepare(model, history, cell.case.parameters)
    try:
        with override_allow_model_requests(True):
            await model.request(prepared.messages, None, cell.case.parameters)
    except ModelAPIError:
        pass
    except UserError as error:
        return error
    return None


def _refused_by_design(model: Model, history: list[ModelMessage], error: UserError) -> bool:
    """Whether `model` refuses `history` as it documents: a file where it reads none, or, for a decision model, a
    request with no user text to judge."""
    if 'Files are not supported' in str(error):
        holds_file = any(isinstance(part, FilePart) for message in history for part in message.parts)
        return holds_file and not model.profile.get('supports_image_input', True)
    return isinstance(model, DecisionModel) and str(error).startswith(NO_TEXT_TO_JUDGE)


@pytest.mark.parametrize('cell', [pytest.param(cell, id=cell.id) for cell in _REQUEST_CELLS])
def test_prepared_history_reaches_the_provider(cell: SdkCell):
    """P1, for every model class: a history prepared as an agent run prepares it reaches the provider.

    No drawn history is one a model should refuse, except as `_refused_by_design` says. The model is built once, as
    building some clients costs more than mapping a history (Bedrock's takes about 170 ms).
    """
    refusal = _Refusal()
    transport = cell.case.transport
    built = transport.model(refusal) if isinstance(transport, XaiTransport) else transport.model(refusal.handler)
    with asyncio.Runner() as runner:
        stack = AsyncExitStack()
        model = runner.run(stack.enter_async_context(built))
        assert isinstance(model, Model)

        @PROPERTY_SETTINGS
        @given(histories(well_formed=False, without=_REPAIR_BREAKING))
        def reaches_the_provider(history: list[ModelMessage]):
            refusal.reached = 0
            error = runner.run(_send(model, cell, history))
            if error is not None and _refused_by_design(model, history, error):
                return
            assert error is None
            assert refusal.reached

        try:
            reaches_the_provider()
        finally:
            runner.run(stack.aclose())


@st.composite
def _redacted_histories(draw: st.DrawFn) -> tuple[list[ModelMessage], set[str]]:
    """A history in which one response's text was redacted after it was stored, and the response ids that no longer
    hold the history up to them: that response's and every later one's."""
    history = draw(histories())
    responses = [index for index, message in enumerate(history) if isinstance(message, ModelResponse) and message.parts]
    if not responses:
        return history, set()
    edited = draw(st.sampled_from(responses))
    response = history[edited]
    assert isinstance(response, ModelResponse)
    history[edited] = replace(
        response,
        parts=[replace(part, content='[redacted]') if isinstance(part, TextPart) else part for part in response.parts],
    )
    stale = {
        message.provider_response_id
        for message in history[edited:]
        if isinstance(message, ModelResponse) and message.provider_response_id is not None
    }
    return history, stale


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@PROPERTY_SETTINGS
@given(_redacted_histories())
def test_responses_continuation_skips_edited_history(edit: tuple[list[ModelMessage], set[str]]):
    """P10: OpenAI Responses continues a stored response only from a history it can tell is unchanged: after a response
    is redacted, no request names it or a later response as `previous_response_id`, since the server would answer
    from the unredacted copy it stored."""
    history, stale = edit
    model = OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))
    messages = prepare(model, history).messages
    previous_response_id = model._get_previous_response_id_and_new_messages(messages)[0]  # pyright: ignore[reportPrivateUsage]
    assert stale_reference_violations(stale, [previous_response_id]) == []
