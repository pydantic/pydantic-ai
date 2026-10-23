"""Properties of message histories, of `repair_messages`, and of the request bodies each model maps them to.

Histories are drawn by `tests/history_strategies.py`, and every property is a checker in
`tests/history_invariants.py` that `test_history_invariants.py` shows can fail.

The body properties call each model's private message mapper instead of recording requests. They are
about the request body itself, over many histories per run: no request is sent, and a cassette holds
one recorded body per test and isn't always matched on it. Each history goes through what an agent run
does before mapping: `repair_messages`, `Model.prepare_request`, then `Model.prepare_messages`, repaired
again as the run repairs it. Every body test also checks P1, that mapping doesn't raise: no drawn history
is one a model should refuse, so a `UserError` fails a test too.

A property that fails on main for drawn histories is a known finding: `KNOWN_FINDINGS` leaves the damage that
draws it out of that property's histories, and a strict expected failure on a hand-built reproducer pins it.
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from functools import cache
from typing import Literal

import pydantic_core
import pytest
from _pytest.mark import ParameterSet
from hypothesis import given, strategies as st
from pydantic import JsonValue

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.models import Model, ModelRequestParameters

from .conftest import try_import
from .history_invariants import (
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
from .history_strategies import Damage, histories

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as groq_imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as huggingface_imports_successful:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as mistral_imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider


@given(histories())
def test_history_survives_serialization(history: list[ModelMessage]):
    """`ModelMessagesTypeAdapter` loads back what it dumped, in Python and in JSON."""
    assert round_trip_violations(history, mode='python') == []
    assert round_trip_violations(history, mode='json') == []


@given(histories(well_formed=False))
def test_repair_makes_tool_calls_pair_up(history: list[ModelMessage]):
    """Repair answers every call once in the next request, keeps every part but orphan returns, and settles."""
    repaired = repair_messages(history)
    assert tool_pairing_violations(repaired) == []
    assert part_changes(history, repaired) == []
    assert idempotence_violations(repair_messages, history) == []


@dataclass(frozen=True)
class Body:
    """A mapped request body, normalized to JSON: the system prompt, where the API takes it apart, and the messages."""

    system: JsonValue
    messages: JsonValue


def _prepare(model: Model, history: list[ModelMessage]) -> tuple[ModelRequestParameters, list[ModelMessage]]:
    """What an agent run does to a history before the model maps it."""
    _, parameters = model.prepare_request(None, ModelRequestParameters())
    return parameters, repair_messages(model.prepare_messages(repair_messages(history), parameters))


def _body(system: object, messages: object) -> Body:
    return Body(
        pydantic_core.to_jsonable_python(system, bytes_mode='base64'),
        pydantic_core.to_jsonable_python(messages, bytes_mode='base64'),
    )


@cache
def _openai_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_chat_body(history: list[ModelMessage]) -> Body:
    model = _openai_chat_model()
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _openai_responses_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_responses_body(history: list[ModelMessage]) -> Body:
    model = _openai_responses_model()
    parameters, messages = _prepare(model, history)
    instructions, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, OpenAIResponsesModelSettings(), parameters
    )
    return _body(instructions if isinstance(instructions, str) else None, items)


@cache
def _anthropic_model() -> AnthropicModel:
    return AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key='test-key'))


async def _anthropic_body(history: list[ModelMessage]) -> Body:
    model = _anthropic_model()
    parameters, messages = _prepare(model, history)
    system, mapped = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        messages, parameters, AnthropicModelSettings()
    )
    return _body(system, mapped)


@cache
def _google_model() -> GoogleModel:
    return GoogleModel('gemini-3.5-flash', provider=GoogleProvider(api_key='test-key'))


async def _google_body(history: list[ModelMessage]) -> Body:
    model = _google_model()
    parameters, messages = _prepare(model, history)
    system, contents = await model._map_messages(messages, parameters)  # pyright: ignore[reportPrivateUsage]
    return _body(system, contents)


@cache
def _bedrock_model() -> BedrockConverseModel:
    return BedrockConverseModel(
        'us.anthropic.claude-sonnet-5', provider=BedrockProvider(api_key='test-key', region_name='us-east-1')
    )


async def _bedrock_body(history: list[ModelMessage]) -> Body:
    model = _bedrock_model()
    parameters, messages = _prepare(model, history)
    system, mapped = await model._map_messages(messages, parameters, None)  # pyright: ignore[reportPrivateUsage]
    return _body(system, mapped)


@cache
def _groq_model() -> GroqModel:
    return GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key='test-key'))


async def _groq_body(history: list[ModelMessage]) -> Body:
    model = _groq_model()
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _mistral_model() -> MistralModel:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test-key'))


async def _mistral_body(history: list[ModelMessage]) -> Body:
    model = _mistral_model()
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _cohere_model() -> CohereModel:
    return CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='test-key'))


async def _cohere_body(history: list[ModelMessage]) -> Body:
    model = _cohere_model()
    parameters, messages = _prepare(model, history)
    return _body(None, model._map_messages(messages, parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _huggingface_model() -> HuggingFaceModel:
    return HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='test-key'))


async def _huggingface_body(history: list[ModelMessage]) -> Body:
    model = _huggingface_model()
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, parameters))  # pyright: ignore[reportPrivateUsage]


@dataclass(frozen=True)
class MapperCase:
    id: str
    body: Callable[[list[ModelMessage]], Coroutine[object, object, Body]]
    wire: Callable[[JsonValue], list[WireMessage]]
    results_follow_their_call: bool = False
    """Whether a tool result must answer a call of the message right before its own."""
    rejects_duplicate_call_ids: bool = False
    """Whether the API rejects a call id sent twice, as each case that sets it did when probed on 2026-10-23 (UTC).

    OpenAI Responses and Groq accepted one; the Hugging Face probe failed for an unrelated reason.
    """
    rejects_empty_text: bool = False
    """Whether the API rejects a text block with no text, which the adapter's own handling of empty text shows."""
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _needs(imports_successful: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not imports_successful(), reason=f'{package} not installed'),)


MAPPER_CASES = [
    MapperCase(
        'openai-chat',
        _openai_chat_body,
        chat_completions_wire,
        rejects_duplicate_call_ids=True,
        marks=_needs(openai_imports_successful, 'openai'),
    ),
    MapperCase(
        'openai-responses',
        _openai_responses_body,
        openai_responses_wire,
        marks=_needs(openai_imports_successful, 'openai'),
    ),
    MapperCase(
        'anthropic',
        _anthropic_body,
        anthropic_wire,
        results_follow_their_call=True,
        rejects_duplicate_call_ids=True,
        rejects_empty_text=True,
        marks=_needs(anthropic_imports_successful, 'anthropic'),
    ),
    MapperCase(
        'google',
        _google_body,
        google_wire,
        rejects_duplicate_call_ids=True,
        marks=_needs(google_imports_successful, 'google-genai'),
    ),
    MapperCase(
        'bedrock',
        _bedrock_body,
        bedrock_wire,
        results_follow_their_call=True,
        rejects_duplicate_call_ids=True,
        rejects_empty_text=True,
        marks=_needs(bedrock_imports_successful, 'boto3'),
    ),
    MapperCase('groq', _groq_body, chat_completions_wire, marks=_needs(groq_imports_successful, 'groq')),
    MapperCase(
        'mistral',
        _mistral_body,
        chat_completions_wire,
        rejects_duplicate_call_ids=True,
        marks=_needs(mistral_imports_successful, 'mistralai'),
    ),
    MapperCase(
        'cohere',
        _cohere_body,
        chat_completions_wire,
        rejects_duplicate_call_ids=True,
        marks=_needs(cohere_imports_successful, 'cohere'),
    ),
    MapperCase(
        'huggingface',
        _huggingface_body,
        chat_completions_wire,
        marks=_needs(huggingface_imports_successful, 'huggingface-hub'),
    ),
]


def _params(cases: list[MapperCase]) -> list[ParameterSet]:
    return [pytest.param(case, id=case.id, marks=case.marks) for case in cases]


def _pinned(rejections: dict[str, str]) -> list[ParameterSet]:
    """One strict expected failure per case named in `rejections`, the reason quoting how its API rejected the body."""
    cases = {case.id: case for case in MAPPER_CASES}
    return [
        pytest.param(
            cases[case_id], id=case_id, marks=(*cases[case_id].marks, pytest.mark.xfail(strict=True, reason=reason))
        )
        for case_id, reason in rejections.items()
    ]


# --- known findings, pinned ------------------------------------------------------------------------
#
# Each known finding is a property that fails on main for drawn histories. It is pinned here by a strict
# expected failure on a hand-built reproducer, which starts passing once the finding is fixed: then delete
# the pinned test and its `KNOWN_FINDINGS` entry. The reasons quote the API's response to the reproducer.

_PROBED = 'probed 2026-10-23 (UTC)'
_ON_BEDROCK = (
    'us.anthropic.claude-sonnet-4-6 on Bedrock '
    '(the suite maps with us.anthropic.claude-sonnet-5, which the probe account cannot call)'
)

_REPEATED_CALL_ID: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Look both up')]),
    ModelResponse(parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0'), ToolCallPart('lookup', {'q': 'b'}, 'call_0')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), ToolReturnPart('lookup', 'b', 'call_0')]),
]


@pytest.mark.parametrize(
    'case',
    _pinned(
        {
            # The misspelling is the API's own, quoted verbatim.
            'openai-chat': f"F1: gpt-5.5 rejected it, {_PROBED}: 400 \"Invalid parameter: messages with role 'tool' "
            'must be a response to a preceeding message '  # codespell:ignore preceeding
            "with 'tool_calls'.\" (the same body with distinct ids was accepted)",
            'anthropic': f'F1: claude-sonnet-5-5 rejected it, {_PROBED}: 400 '
            '"messages.1.content.1: `tool_use` ids must be unique"',
            'google': f'F1: gemini-3.5-flash rejected it, {_PROBED}: 400 "Request contains an invalid argument." '
            '(the same body with distinct ids was accepted)',
            'bedrock': f'F1: {_ON_BEDROCK} rejected it, {_PROBED}: 400 ValidationException "The toolUse blocks at '
            'messages.2.content contain duplicate Ids: call_0"',
            'mistral': f'F1: mistral-large-latest rejected it, {_PROBED}: 400 '
            '"Duplicate tool call id in assistant message"',
            'cohere': f'F1: command-r7b-12-2024 rejected it, {_PROBED}: 400 "invalid tool call provided in '
            "messages[1].tool_calls[1]: cannot have duplicate tool ids 'call_0'\"",
        }
    ),
)
async def test_f1_a_call_id_repeated_in_one_response_is_sent_repeated(case: MapperCase):
    """A call id repeated within one response, as `end_strategy='graceful'` records across the batches it splits at
    an output call (#9815) and as a model numbering its calls per response sends."""
    assert duplicate_call_id_violations(case.wire((await case.body(_REPEATED_CALL_ID)).messages)) == []


@pytest.mark.parametrize(
    'case',
    _pinned(
        {
            'anthropic': f'F3: claude-sonnet-5-5 rejected it, {_PROBED}: 400 '
            '"messages.3: system text blocks must contain non-whitespace text"'
        }
    ),
)
async def test_f3_an_empty_mid_conversation_system_prompt_is_sent_as_empty_text(case: MapperCase):
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi')]),
        ModelResponse(parts=[TextPart('Hello')]),
        ModelRequest(parts=[UserPromptPart('Again'), SystemPromptPart('')]),
    ]
    assert empty_text_violations(case.wire((await case.body(history)).messages)) == []


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
@pytest.mark.parametrize(
    'history',
    [
        pytest.param(
            [ModelRequest(parts=[UserPromptPart('')])],
            id='empty-prompt',
            marks=pytest.mark.xfail(
                strict=True,
                reason=f'F4: {_ON_BEDROCK} rejected it, {_PROBED}: 400 ValidationException "The text field in the '
                'ContentBlock object at messages.0.content.0 is blank. Add text to the text field, and try again."',
            ),
        ),
        pytest.param(
            [
                ModelRequest(parts=[UserPromptPart('Hi')]),
                ModelResponse(parts=[TextPart('Hello'), TextPart('')]),
                ModelRequest(parts=[UserPromptPart('Again')]),
            ],
            id='empty-response-text',
            marks=pytest.mark.xfail(
                strict=True,
                reason=f'F4: {_ON_BEDROCK} rejected it, {_PROBED}: 400 ValidationException "The model returned the '
                'following errors: messages: text content blocks must be non-empty"',
            ),
        ),
    ],
)
async def test_f4_empty_text_is_sent_as_an_empty_text_block(history: list[ModelMessage]):
    assert empty_text_violations(bedrock_wire((await _bedrock_body(history)).messages)) == []


@pytest.mark.parametrize(
    'case',
    _pinned(
        {'anthropic': f'F5: claude-sonnet-5-5 rejected it, {_PROBED}: 400 "messages: at least one message is required"'}
    ),
)
async def test_f5_a_lone_empty_prompt_is_sent_as_no_messages(case: MapperCase):
    """A run's first request with an empty prompt, as `agent.run('')` makes."""
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('')])]
    assert empty_message_violations(case.wire((await case.body(history)).messages)) == []


CheckedProperty = Literal[
    'call_ids_are_unique', 'results_answer_calls', 'messages_are_not_empty', 'text_is_not_empty', 'deterministic'
]


@dataclass(frozen=True)
class KnownFinding:
    """A property that fails on main: the damage that draws it, left out of the property where it fails, and its pin."""

    id: str
    fails: CheckedProperty
    damage: frozenset[Damage]
    providers: frozenset[str]
    pinned_by: Callable[..., Awaitable[None]]


KNOWN_FINDINGS = [
    KnownFinding(
        'F1',
        'call_ids_are_unique',
        frozenset({'repeated_call_id'}),
        frozenset(case.id for case in MAPPER_CASES if case.rejects_duplicate_call_ids),
        test_f1_a_call_id_repeated_in_one_response_is_sent_repeated,
    ),
    KnownFinding(
        'F3',
        'text_is_not_empty',
        frozenset({'empty_system_prompt'}),
        frozenset({'anthropic'}),
        test_f3_an_empty_mid_conversation_system_prompt_is_sent_as_empty_text,
    ),
    KnownFinding(
        'F4',
        'text_is_not_empty',
        frozenset({'empty_prompt', 'empty_response_text'}),
        frozenset({'bedrock'}),
        test_f4_empty_text_is_sent_as_an_empty_text_block,
    ),
    KnownFinding(
        'F5',
        'messages_are_not_empty',
        frozenset({'empty_prompt'}),
        frozenset({'anthropic'}),
        test_f5_a_lone_empty_prompt_is_sent_as_no_messages,
    ),
]


def _drawn(case: MapperCase, checked: CheckedProperty) -> st.SearchStrategy[list[ModelMessage]]:
    """Damaged histories, without the damage of a known finding of `checked` on `case`."""
    known = [finding.damage for finding in KNOWN_FINDINGS if finding.fails == checked and case.id in finding.providers]
    return histories(well_formed=False, without=frozenset[Damage]().union(*known))


# --- properties --------------------------------------------------------------------------------------
#
# Each history is drawn inside the test, since a known finding's damage is left out only for the cases it fails on.
# The tests are sync: Hypothesis ends a draw that runs out of data with a `BaseException`, which raised inside an
# async test breaks the anyio test runner for every later example.


def _map(case: MapperCase, history: list[ModelMessage]) -> Body:
    with asyncio.Runner() as runner:
        return runner.run(case.body(history))


@pytest.mark.parametrize('case', _params([case for case in MAPPER_CASES if case.rejects_duplicate_call_ids]))
@given(data=st.data())
def test_mapped_call_ids_are_unique(case: MapperCase, data: st.DataObject):
    """P2: no call id is sent twice."""
    history = data.draw(_drawn(case, 'call_ids_are_unique'))
    assert duplicate_call_id_violations(case.wire(_map(case, history).messages)) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(data=st.data())
def test_mapped_results_answer_calls(case: MapperCase, data: st.DataObject):
    """P2: every result answers an earlier call (where required, one of the message right before), and every call
    is answered."""
    history = data.draw(_drawn(case, 'results_answer_calls'))
    wire = case.wire(_map(case, history).messages)
    assert call_pairing_violations(wire, results_follow_their_call=case.results_follow_their_call) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(data=st.data())
def test_mapped_messages_are_not_empty(case: MapperCase, data: st.DataObject):
    """P4: the body has a message, and every message has content."""
    history = data.draw(_drawn(case, 'messages_are_not_empty'))
    assert empty_message_violations(case.wire(_map(case, history).messages)) == []


@pytest.mark.parametrize('case', _params([case for case in MAPPER_CASES if case.rejects_empty_text]))
@given(data=st.data())
def test_mapped_text_is_not_empty(case: MapperCase, data: st.DataObject):
    """P4: no text block is empty, where the API rejects one."""
    history = data.draw(_drawn(case, 'text_is_not_empty'))
    assert empty_text_violations(case.wire(_map(case, history).messages)) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(data=st.data())
def test_mapping_is_deterministic(case: MapperCase, data: st.DataObject):
    """P5: mapping a history again, or after a JSON round trip, gives the same body."""
    history = data.draw(_drawn(case, 'deterministic'))
    body = _map(case, history)
    assert body_changes(body, _map(case, history)) == []
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    assert body_changes(body, _map(case, stored)) == []
