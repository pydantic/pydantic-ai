"""Properties of message histories, of `repair_messages`, and of the request bodies each model maps them to.

Histories are drawn by `tests/history_strategies.py`, and every property is a checker in
`tests/history_invariants.py` that `test_history_invariants.py` shows can fail.

The body properties call each model's private message mapper instead of recording requests. They are
about the request body itself, over many histories per run: no request is sent, and a cassette holds
one recorded body per test and isn't always matched on it. Each history goes through what an agent run
does before mapping: `repair_messages`, `Model.prepare_request`, then `Model.prepare_messages`, repaired
again as the run repairs it. Every body test also checks P1, that mapping doesn't raise: no drawn history
is one a model should refuse, so a `UserError` fails a test too.

A property that fails on main for a history in the drawn domain stays, marked as a strict expected failure
that names the input, so it starts failing (and its mark has to go) once the behavior is fixed.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache

import pydantic_core
import pytest
from _pytest.mark import ParameterSet
from hypothesis import Phase, given, settings
from pydantic import JsonValue

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
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
    role_violations,
    round_trip_violations,
    tool_pairing_violations,
)
from .history_strategies import histories

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
    body: Callable[[list[ModelMessage]], Awaitable[Body]]
    wire: Callable[[JsonValue], list[WireMessage]]
    results_follow_their_call: bool = False
    """Whether a tool result must answer a call of the message right before its own."""
    alternates_roles: bool = False
    """Whether no two adjacent messages may have the same role.

    Anthropic combines consecutive same-role turns, and its adapter relies on that
    (`tests/models/anthropic/test_carried_thinking.py`), so this is the Bedrock Converse adapter's own rule:
    it merges consecutive user turns, and parts the ones it can't merge with an assistant turn.
    """
    rejects_empty_text: bool = False
    """Whether the API rejects a text block with no text, which the adapter's own handling of empty text shows."""
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _needs(imports_successful: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not imports_successful(), reason=f'{package} not installed'),)


MAPPER_CASES = [
    MapperCase(
        'openai-chat', _openai_chat_body, chat_completions_wire, marks=_needs(openai_imports_successful, 'openai')
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
        rejects_empty_text=True,
        marks=_needs(anthropic_imports_successful, 'anthropic'),
    ),
    MapperCase('google', _google_body, google_wire, marks=_needs(google_imports_successful, 'google-genai')),
    MapperCase(
        'bedrock',
        _bedrock_body,
        bedrock_wire,
        results_follow_their_call=True,
        alternates_roles=True,
        rejects_empty_text=True,
        marks=_needs(bedrock_imports_successful, 'boto3'),
    ),
    MapperCase('groq', _groq_body, chat_completions_wire, marks=_needs(groq_imports_successful, 'groq')),
    MapperCase('mistral', _mistral_body, chat_completions_wire, marks=_needs(mistral_imports_successful, 'mistralai')),
    MapperCase('cohere', _cohere_body, chat_completions_wire, marks=_needs(cohere_imports_successful, 'cohere')),
    MapperCase(
        'huggingface',
        _huggingface_body,
        chat_completions_wire,
        marks=_needs(huggingface_imports_successful, 'huggingface-hub'),
    ),
]


def _params(cases: list[MapperCase]) -> list[ParameterSet]:
    return [pytest.param(case, id=case.id, marks=case.marks) for case in cases]


_NO_SHRINKING = settings(phases=[Phase.explicit, Phase.reuse, Phase.generate])
"""For a test every case of which is a known failure: shrinking the failing history would only cost time."""


@pytest.mark.xfail(
    strict=True,
    reason='A tool call id repeated within one response, as `end_strategy="graceful"` records across the batches it '
    'splits at an output call and as a model numbering its calls per response sends, reaches every body repeated.',
)
@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@_NO_SHRINKING
@given(history=histories(well_formed=False))
async def test_mapped_call_ids_are_unique(case: MapperCase, history: list[ModelMessage]):
    """P2: no call id is sent twice."""
    assert duplicate_call_id_violations(case.wire((await case.body(history)).messages)) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(history=histories(well_formed=False))
async def test_mapped_results_answer_calls(case: MapperCase, history: list[ModelMessage]):
    """P2: every result answers an earlier call (where required, one of the message right before), and every call
    is answered."""
    wire = case.wire((await case.body(history)).messages)
    assert call_pairing_violations(wire, results_follow_their_call=case.results_follow_their_call) == []


@pytest.mark.parametrize('case', _params([case for case in MAPPER_CASES if case.alternates_roles]))
@given(history=histories(well_formed=False))
async def test_mapped_roles_alternate(case: MapperCase, history: list[ModelMessage]):
    """P3: no two adjacent messages have the same role, where the adapter keeps them apart."""
    assert role_violations(case.wire((await case.body(history)).messages)) == []


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
@pytest.mark.xfail(
    strict=True,
    reason='Bedrock is sent two consecutive assistant turns once an answered request that held only a request-scoped '
    'prompt drops out, since repair keeps responses from an API apart.',
)
async def test_bedrock_roles_alternate_around_an_answered_request_scoped_prompt():
    """A run continued without a user prompt, where a capability enqueued a request-scoped nudge.

    Drawn by `histories()` too, but not within the default profile's examples, so it is pinned here.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi')]),
        ModelResponse(parts=[TextPart('Hello')], model_name='claude-sonnet-5', provider_name='bedrock'),
        ModelRequest(parts=[SystemPromptPart('Be brief.', scope='request')]),
        ModelResponse(parts=[TextPart('OK')], model_name='claude-sonnet-5', provider_name='bedrock'),
        ModelRequest(parts=[UserPromptPart('Thanks')]),
    ]
    assert role_violations(bedrock_wire((await _bedrock_body(history)).messages)) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(history=histories(well_formed=False))
async def test_mapped_messages_are_not_empty(case: MapperCase, history: list[ModelMessage]):
    """P4: every message has content."""
    assert empty_message_violations(case.wire((await case.body(history)).messages)) == []


_EMPTY_TEXT_ON_THE_WIRE = {
    'anthropic': "A mid-conversation `SystemPromptPart('')` is sent as an empty text block in a `system` entry.",
    'bedrock': "A `UserPromptPart('')` or `TextPart('')` is sent as an empty text block.",
}


@pytest.mark.parametrize(
    'case',
    [
        pytest.param(
            case,
            id=case.id,
            marks=(*case.marks, pytest.mark.xfail(strict=True, reason=_EMPTY_TEXT_ON_THE_WIRE[case.id])),
        )
        for case in MAPPER_CASES
        if case.rejects_empty_text
    ],
)
@_NO_SHRINKING
@given(history=histories(well_formed=False))
async def test_mapped_text_is_not_empty(case: MapperCase, history: list[ModelMessage]):
    """P4: no text block is empty, where the API rejects one."""
    assert empty_text_violations(case.wire((await case.body(history)).messages)) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@given(history=histories(well_formed=False))
async def test_mapping_is_deterministic(case: MapperCase, history: list[ModelMessage]):
    """P5: mapping a history again, or after a JSON round trip, gives the same body."""
    body = await case.body(history)
    assert body_changes(body, await case.body(history)) == []
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    assert body_changes(body, await case.body(stored)) == []
