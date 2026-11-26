"""Properties of message histories, of `repair_messages`, and of the request bodies each model maps them to.

Histories are drawn by `tests/history_strategies.py`, and every property is a checker in
`tests/history_invariants.py` that `test_history_invariants.py` shows can fail.

The body properties call each model's private message mapper instead of recording requests. They are
about the request body itself, over many histories per run: no request is sent, and a cassette holds
one recorded body per test and isn't always matched on it. Each history goes through what an agent run
does before mapping, in its order: `repair_messages`, `Model.prepare_messages` with the run's parameters,
`repair_messages` again, then `Model.prepare_request`, whose parameters the mapper gets. Every body test also
checks P1, that mapping doesn't raise: no drawn history is one a model should refuse, so a `UserError` fails a
test too.

A property that fails on main for drawn histories is a known finding: `KNOWN_FINDINGS` leaves the damage that
draws it out of that property's histories, and a strict expected failure on a hand-built reproducer pins it.

The histories in `tests/history_cases.py` are fixed cases for two more properties: every request a history leads to
starts with what the previous request settled, and a history dumped and loaded renders the same request.
"""

from __future__ import annotations as _annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from functools import cache
from typing import Any, Generic, Literal, TypeVar, assert_never

import pydantic_core
import pytest
from _pytest.mark import ParameterSet
from hypothesis import HealthCheck, given, settings, strategies as st
from pydantic import JsonValue

from pydantic_ai._framing import lookalike_note_announcement
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
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    _request_scoped_tail_texts,  # pyright: ignore[reportPrivateUsage]
)

from .conftest import try_import
from .history_cases import HISTORIES, HistoryCase, message_json
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
    google_interactions_wire,
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
    from pydantic_ai.models.google_interactions import GoogleInteractionsModel
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
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

# Hypothesis compares the global `random` state before and after each draw and notes this deprecation when it
# changed. Its own `control.py` documents the false positive: a different thread touching `random` during a draw
# trips it, and the guard it keeps covers only the states it set itself. Under xdist this module shares a worker
# with tests that run threads, so the note fired on CI as a flaky error (`filterwarnings = error`). No strategy
# in `tests/history_strategies.py` uses `random`.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
)

_long_run_examples = os.environ.get('HISTORY_PROPERTY_EXAMPLES')
PROPERTY_SETTINGS = settings(
    max_examples=int(_long_run_examples) if _long_run_examples else 50,
    derandomize=_long_run_examples is None,
    database=None,
    deadline=None,
    print_blob=_long_run_examples is not None,
    suppress_health_check=[HealthCheck.too_slow],
)
"""A small, derandomized batch by default, so a run is reproducible, fits the per-test budget in `cost_guards`, and
covers the same branches on every xdist worker. `HISTORY_PROPERTY_EXAMPLES` sets a long, randomized run to hunt for
new failures, as `REALTIME_SIMULATION_EXAMPLES` does for the realtime simulation. `cost_guards` bounds each test's
time, so Hypothesis's own check on slow drawing, which a loaded worker can trip, is off."""


@PROPERTY_SETTINGS
@given(histories())
def test_history_survives_serialization(history: list[ModelMessage]):
    """`ModelMessagesTypeAdapter` loads back what it dumped, in Python and in JSON."""
    assert round_trip_violations(history, mode='python') == []
    assert round_trip_violations(history, mode='json') == []


@dataclass(frozen=True)
class Body:
    """A mapped request body, normalized to JSON: the system prompt, where the API takes it apart, and the messages."""

    system: JsonValue
    messages: JsonValue


def _prepare(model: Model, history: list[ModelMessage]) -> tuple[ModelRequestParameters, list[ModelMessage]]:
    """What an agent run does to a history before the model maps it, in its order: the run repairs the history,
    prepares it with the run's parameters, and repairs it again, then the model's `request` prepares the
    parameters it maps the messages with."""
    run_parameters = ModelRequestParameters()
    messages = repair_messages(model.prepare_messages(repair_messages(history), run_parameters))
    _, parameters = model.prepare_request(None, run_parameters)
    return parameters, messages


def _body(system: object, messages: object) -> Body:
    return Body(
        pydantic_core.to_jsonable_python(system, bytes_mode='base64'),
        pydantic_core.to_jsonable_python(messages, bytes_mode='base64'),
    )


@cache
def _openai_chat_model() -> OpenAIChatModel:
    return OpenAIChatModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_chat_body(model: OpenAIChatModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=OpenAIChatModelSettings()
    )
    return _body(None, mapped)


@cache
def _openrouter_model() -> OpenRouterModel:
    """Sends thinking back in a `reasoning` field (`openai_chat_send_back_thinking_parts='field'`)."""
    return OpenRouterModel('anthropic/claude-haiku-4.5', provider=OpenRouterProvider(api_key='test-key'))


@cache
def _openai_responses_model() -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-5.5', provider=OpenAIProvider(api_key='test-key'))


async def _openai_responses_body(model: OpenAIResponsesModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    instructions, items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_settings=OpenAIResponsesModelSettings(), model_request_parameters=parameters
    )
    return _body(instructions if isinstance(instructions, str) else None, items)


@cache
def _anthropic_model() -> AnthropicModel:
    # A model with the native mid-conversation `system` role, which the rejections below were probed against on
    # `claude-sonnet-5-5` before it moved to the sealed `<system>` rendering.
    return AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key='test-key'))


async def _anthropic_body(model: AnthropicModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    system, mapped = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=AnthropicModelSettings()
    )
    return _body(system, mapped)


@cache
def _google_model() -> GoogleModel:
    return GoogleModel('gemini-3.5-flash', provider=GoogleProvider(api_key='test-key'))


async def _google_body(model: GoogleModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    system, contents = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters
    )
    return _body(system, contents)


@cache
def _google_interactions_model() -> GoogleInteractionsModel:
    return GoogleInteractionsModel('gemini-3.8-flash', provider=GoogleProvider(api_key='test-key'))


async def _google_interactions_body(model: GoogleInteractionsModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    system, steps = await model._map_messages(messages, parameters)  # pyright: ignore[reportPrivateUsage]
    return _body(system, steps)


@cache
def _bedrock_model() -> BedrockConverseModel:
    return BedrockConverseModel(
        'us.anthropic.claude-sonnet-5', provider=BedrockProvider(api_key='test-key', region_name='us-east-1')
    )


async def _bedrock_body(model: BedrockConverseModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    system, mapped = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        messages, model_request_parameters=parameters, model_settings=None
    )
    return _body(system, mapped)


@cache
def _groq_model() -> GroqModel:
    return GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key='test-key'))


async def _groq_body(model: GroqModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _mistral_model() -> MistralModel:
    return MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test-key'))


async def _mistral_body(model: MistralModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _cohere_model() -> CohereModel:
    return CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='test-key'))


async def _cohere_body(model: CohereModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    return _body(None, model._map_messages(messages, model_request_parameters=parameters))  # pyright: ignore[reportPrivateUsage]


@cache
def _huggingface_model() -> HuggingFaceModel:
    return HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='test-key'))


async def _huggingface_body(model: HuggingFaceModel, history: list[ModelMessage]) -> Body:
    parameters, messages = _prepare(model, history)
    return _body(None, await model._map_messages(messages, model_request_parameters=parameters))  # pyright: ignore[reportPrivateUsage]


ModelT = TypeVar('ModelT', bound=Model)


@dataclass(frozen=True)
class MapperCase(Generic[ModelT]):
    id: str
    model: Callable[[], ModelT]
    mapper: Callable[[ModelT, list[ModelMessage]], Coroutine[object, object, Body]]
    wire: Callable[[JsonValue], list[WireMessage]]
    baseline: list[WireMessage]
    """What `wire` reads off the body `_BASELINE` maps to: if a provider SDK renames a field, the checkers stop
    seeing calls or text, and this tells."""
    results_follow_their_call: bool = False
    """Whether a tool result must answer a call of the nearest earlier message that is not made of results: Anthropic
    and Bedrock document it; OpenAI Chat (gpt-5.5) rejected a `tool` message after an assistant turn without
    `tool_calls`, probed 2026-10-23 (UTC)."""
    rejects_empty_text: bool = False
    """Whether the API rejects a text block with no text, which the adapter's own handling of empty text shows."""
    takes_system_alone: bool = False
    """Whether the API takes a system prompt with no message as a request: OpenAI Responses (gpt-5.5) accepted
    instructions with an empty `input`, probed 2026-10-23 (UTC)."""
    takes_empty_user_turns: bool = False
    """Whether the API takes a user turn with no content, which the adapter sends for `UserPromptPart([])`: OpenAI
    Chat and OpenAI Responses (gpt-5.5) and the Hugging Face router (Qwen/Qwen3-235B-A22B through
    `provider_name='auto'`, with a tool declared) accepted one, probed 2026-10-23 (UTC)."""
    takes_empty_assistant_turns: bool = False
    """Whether the API takes an assistant turn with no content, which the adapter sends for a response with no
    parts: Groq (openai/gpt-oss-120b) and the Hugging Face router (Qwen/Qwen3-235B-A22B through `provider_name='auto'`,
    with a tool declared to clear #11177) accepted one, probed 2026-10-23 (UTC)."""
    marks: tuple[pytest.MarkDecorator, ...] = ()

    async def body(self, history: list[ModelMessage]) -> Body:
        """The body this case's model maps `history` to."""
        return await self.mapper(self.model(), history)


def _needs(imports_successful: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not imports_successful(), reason=f'{package} not installed'),)


_BASELINE: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('Look it up')]),
    ModelResponse(parts=[TextPart('Looking'), ToolCallPart('lookup', {'q': 'a'}, 'call_1')]),
    ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_1')]),
    ModelResponse(parts=[TextPart('Found it')]),
]


def _baseline_turns(*, assistant: str, results: str, call_id: str = 'call_1') -> list[WireMessage]:
    """`_BASELINE` as a body of role-tagged messages projects to, with the roles and call id the provider uses."""
    return [
        WireMessage('user', [WireBlock('text', 'Look it up')]),
        WireMessage(assistant, [WireBlock('text', 'Looking'), WireBlock('call', call_id)]),
        WireMessage(results, [WireBlock('result', call_id)]),
        WireMessage(assistant, [WireBlock('text', 'Found it')]),
    ]


MAPPER_CASES = [
    MapperCase(
        'openai-chat',
        _openai_chat_model,
        _openai_chat_body,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        results_follow_their_call=True,
        takes_empty_user_turns=True,
        marks=_needs(openai_imports_successful, 'openai'),
    ),
    MapperCase(
        'openai-responses',
        _openai_responses_model,
        _openai_responses_body,
        openai_responses_wire,
        [
            WireMessage('user', [WireBlock('text', 'Look it up')]),
            WireMessage('assistant', [WireBlock('text', 'Looking')]),
            WireMessage('function_call', [WireBlock('call', 'call_1')]),
            WireMessage('function_call_output', [WireBlock('result', 'call_1')]),
            WireMessage('assistant', [WireBlock('text', 'Found it')]),
        ],
        takes_system_alone=True,
        takes_empty_user_turns=True,
        marks=_needs(openai_imports_successful, 'openai'),
    ),
    MapperCase(
        'anthropic',
        _anthropic_model,
        _anthropic_body,
        anthropic_wire,
        _baseline_turns(assistant='assistant', results='user'),
        results_follow_their_call=True,
        rejects_empty_text=True,
        marks=_needs(anthropic_imports_successful, 'anthropic'),
    ),
    MapperCase(
        'google',
        _google_model,
        _google_body,
        google_wire,
        _baseline_turns(assistant='model', results='user'),
        marks=_needs(google_imports_successful, 'google-genai'),
    ),
    MapperCase(
        'google-interactions',
        _google_interactions_model,
        _google_interactions_body,
        google_interactions_wire,
        [
            WireMessage('user_input', [WireBlock('text', 'Look it up')]),
            WireMessage('model_output', [WireBlock('text', 'Looking')]),
            WireMessage('function_call', [WireBlock('call', 'call_1')]),
            WireMessage('function_result', [WireBlock('result', 'call_1')]),
            WireMessage('model_output', [WireBlock('text', 'Found it')]),
        ],
        marks=_needs(google_imports_successful, 'google-genai'),
    ),
    MapperCase(
        'bedrock',
        _bedrock_model,
        _bedrock_body,
        bedrock_wire,
        _baseline_turns(assistant='assistant', results='user'),
        results_follow_their_call=True,
        rejects_empty_text=True,
        marks=_needs(bedrock_imports_successful, 'boto3'),
    ),
    MapperCase(
        'groq',
        _groq_model,
        _groq_body,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        takes_empty_assistant_turns=True,
        marks=_needs(groq_imports_successful, 'groq'),
    ),
    MapperCase(
        'mistral',
        _mistral_model,
        _mistral_body,
        chat_completions_wire,
        # Mistral takes only 9-character alphanumeric ids, so its profile rewrites `call_1` to a stable hash.
        _baseline_turns(assistant='assistant', results='tool', call_id='RksXkbv4v'),
        marks=_needs(mistral_imports_successful, 'mistralai'),
    ),
    MapperCase(
        'cohere',
        _cohere_model,
        _cohere_body,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        marks=_needs(cohere_imports_successful, 'cohere'),
    ),
    MapperCase(
        'huggingface',
        _huggingface_model,
        _huggingface_body,
        chat_completions_wire,
        _baseline_turns(assistant='assistant', results='tool'),
        takes_empty_user_turns=True,
        takes_empty_assistant_turns=True,
        marks=_needs(huggingface_imports_successful, 'huggingface-hub'),
    ),
]


def _params(cases: list[MapperCase[Any]]) -> list[ParameterSet]:
    return [pytest.param(case, id=case.id, marks=case.marks) for case in cases]


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
async def test_projection_reads_the_mapped_body(case: MapperCase[Any]):
    assert case.wire((await case.body(_BASELINE)).messages) == case.baseline


CheckedProperty = Literal[
    'call_ids_are_unique', 'results_answer_calls', 'messages_are_not_empty', 'text_is_not_empty', 'deterministic'
]


async def _violations(checked: CheckedProperty, case: MapperCase[Any], history: list[ModelMessage]) -> list[str]:
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
        case _:
            assert_never(checked)


# --- known findings ------------------------------------------------------------------------------------
#
# A known finding is a property that fails on main for drawn histories. Its entry leaves the damage that draws it
# out of that property on the providers it fails on, and `test_known_finding_is_still_present` pins it there with
# a hand-built reproducer, as a strict expected failure that starts passing once the finding is fixed: then delete
# the entry. Each pin quotes how the probed API rejected the reproducer, probed 2026-10-23 (UTC). Bedrock was
# probed on us.anthropic.claude-sonnet-4-6, as the probe account can't call the us.anthropic.claude-sonnet-5 the
# suite maps with.


@dataclass(frozen=True)
class Pin:
    """A hand-built reproducer of a known finding, on one provider."""

    case_id: str | None
    """The mapper case the reproducer fails on, or `None` for a history `repair_messages` leaves broken. Every
    mapper maps what repair leaves, so the finding's damage is then left out of its property on every case, and
    out of the repair property."""
    history: list[ModelMessage]
    rejection: str
    """The probed model and its error, verbatim."""
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
            return frozenset(case.id for case in MAPPER_CASES)
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
                'google-interactions',
                [ModelRequest(parts=[UserPromptPart([])])],
                'gemini-3.8-flash: 404 "Model \'gemini-3.8-flash\' not found" for input=[], probed 2026-11-06; '
                'an explicit empty user_input instead gives 400 "GenerateContentRequest.contents: contents is not specified"',
                label='empty-prompt',
            ),
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
]


@pytest.mark.parametrize(
    'finding,pin',
    [
        pytest.param(
            finding,
            pin,
            id='-'.join(filter(None, [finding.id, pin.case_id, pin.label])),
            marks=(
                *next((case.marks for case in MAPPER_CASES if case.id == pin.case_id), ()),
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
        case = next(case for case in MAPPER_CASES if case.id == pin.case_id)
        assert await _violations(finding.fails, case, pin.history) == []


# --- properties --------------------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    histories(
        well_formed=False,
        without=frozenset[Damage]().union(*(finding.damage for finding in KNOWN_FINDINGS if finding.breaks_repair)),
    )
)
def test_repair_makes_tool_calls_pair_up(history: list[ModelMessage]):
    """Repair answers every call once in the next request, keeps every part but orphan returns, and settles."""
    repaired = repair_messages(history)
    assert tool_pairing_violations(repaired) == []
    assert part_changes(history, repaired) == []
    assert idempotence_violations(repair_messages, history) == []


# The body properties draw each history inside the test, since a known finding's damage is left out only for the
# cases it fails on. They are sync: Hypothesis ends a draw that runs out of data with a `BaseException`, which raised
# inside an async test breaks the anyio test runner for every later example.


def _check(checked: CheckedProperty, case: MapperCase[Any], data: st.DataObject) -> list[str]:
    """Draw a damaged history, less the damage of each known finding of `checked` on `case`, and check its body."""
    known = [finding.damage for finding in KNOWN_FINDINGS if finding.fails == checked and case.id in finding.providers]
    history = data.draw(histories(well_formed=False, without=frozenset[Damage]().union(*known)))
    with asyncio.Runner() as runner:
        return runner.run(_violations(checked, case, history))


@pytest.mark.parametrize('case', _params([case for case in MAPPER_CASES if case.id in REPEATED_CALL_ID_REJECTIONS]))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_call_ids_are_unique(case: MapperCase[Any], data: st.DataObject):
    """P2: no call id is sent twice in one message, nor in one body where the API rejects that too, on the APIs that
    reject one sent twice."""
    assert _check('call_ids_are_unique', case, data) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_results_answer_calls(case: MapperCase[Any], data: st.DataObject):
    """P2: every result answers an earlier call (where required, one of the message right before), and every call
    is answered."""
    assert _check('results_answer_calls', case, data) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_messages_are_not_empty(case: MapperCase[Any], data: st.DataObject):
    """P4: the body has a message, and every message has content, except what the API takes without."""
    assert _check('messages_are_not_empty', case, data) == []


@pytest.mark.parametrize('case', _params([case for case in MAPPER_CASES if case.rejects_empty_text]))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapped_text_is_not_empty(case: MapperCase[Any], data: st.DataObject):
    """P4: no text block is empty or whitespace-only, where the API rejects one."""
    assert _check('text_is_not_empty', case, data) == []


@pytest.mark.parametrize('case', _params(MAPPER_CASES))
@PROPERTY_SETTINGS
@given(data=st.data())
def test_mapping_is_deterministic(case: MapperCase[Any], data: st.DataObject):
    """P5: mapping a history again, or after a JSON round trip, gives the same body."""
    assert _check('deterministic', case, data) == []


# --- hand-built histories ------------------------------------------------------------------------------
#
# Fixed cases from `tests/history_cases.py`, rendered for every mapper and for OpenRouter, which sends thinking back in
# a `reasoning` field. They aren't drawn: each holds the record kinds `prepare_messages` translates or drops, in the
# shapes a run stores them in.

_OPENROUTER = MapperCase(
    'openrouter',
    _openrouter_model,
    _openai_chat_body,
    chat_completions_wire,
    _baseline_turns(assistant='assistant', results='tool'),
    marks=_needs(openai_imports_successful, 'openai'),
)

HISTORY_CASE_TARGETS = [*MAPPER_CASES, _OPENROUTER]


def history_case_params() -> list[ParameterSet]:
    """Every hand-built history on every target, skipped where the target's package isn't installed."""
    return [
        pytest.param(history, case, id=f'{history.id}-{case.id}', marks=case.marks)
        for history in HISTORIES
        for case in HISTORY_CASE_TARGETS
    ]


@dataclass(frozen=True)
class Rendered:
    """One request a history leads to: what `prepare_messages` returned, and the provider body's messages."""

    messages: list[JsonValue]
    request_scoped_tail: int
    """How many of the trailing request's last parts are request-scoped prompts, which later requests drop."""
    body: list[JsonValue]


async def _render(case: MapperCase[Any], history: list[ModelMessage]) -> Rendered:
    _, messages = _prepare(case.model(), history)
    body = (await case.body(history)).messages
    assert isinstance(body, list)
    return Rendered(
        messages=[message_json(message) for message in messages],
        request_scoped_tail=len(_request_scoped_tail_texts(messages)),
        body=body,
    )


_ANNOUNCEMENTS = {lookalike_note_announcement(system_role=False, sealed=sealed) for sealed in (False, True)}


def _parts(message: JsonValue) -> list[JsonValue]:
    assert isinstance(message, dict) and isinstance(message['parts'], list)
    return message['parts']


def _is_announcement(part: JsonValue) -> bool:
    return isinstance(part, dict) and part.get('part_kind') == 'system-prompt' and part.get('content') in _ANNOUNCEMENTS


def _without_new_announcement(earlier: Rendered, later: Rendered) -> Rendered:
    """`later` without the framing announcement, when it joined the standing prompt after `earlier`.

    With `'when_needed'`, on a model without mid-conversation system messages, the announcement goes ahead of the first
    request's standing prompt at the first request whose history needs it, and stays there: the cached prefix changes
    once, on purpose (`Model._announce_in_standing_prompt`). That's the one change to a settled message allowed.
    """
    later_first, *later_rest = later.messages
    if any(_is_announcement(part) for part in _parts(earlier.messages[0])):
        return later
    first_parts = _parts(later_first)
    if not (first_parts and _is_announcement(first_parts[0])):
        return later
    assert isinstance(later_first, dict)
    return replace(later, messages=[{**later_first, 'parts': first_parts[1:]}, *later_rest])


def _assert_settled_messages_kept(earlier: Rendered, later: Rendered) -> None:
    """`later` renders every message `earlier` rendered the same, and the trailing request's parts as the start of
    its own, so parts a later turn adds to it (such as a note ahead of carried reasoning) only extend it. The trailing
    request's request-scoped prompts are left out: later requests drop them."""
    *settled, trailing = earlier.messages
    assert later.messages[: len(settled)] == settled
    parts = _parts(trailing)[: len(_parts(trailing)) - earlier.request_scoped_tail]
    later_trailing = later.messages[len(settled)]
    assert isinstance(trailing, dict) and isinstance(later_trailing, dict)
    assert {**later_trailing, 'parts': _parts(later_trailing)[: len(parts)]} == {**trailing, 'parts': parts}


def _conversation(body: list[JsonValue]) -> list[JsonValue]:
    """The body's items after the leading system ones: those carry the request's instructions, which an instruction
    change rewrites by design, and which the cache-prefix tests cover."""
    start = next(
        (
            i
            for i, item in enumerate(body)
            if not (isinstance(item, dict) and item.get('role') in ('system', 'developer'))
        ),
        len(body),
    )
    return body[start:]


def _is_user_side(item: JsonValue) -> bool:
    """Whether a body item is the request's side of the conversation, rather than a turn the model produced."""
    assert isinstance(item, dict)
    role = item.get('role')
    if role is not None:
        return role not in ('assistant', 'model')
    # Items without a role, from OpenAI Responses and Google Interactions: only user input and tool results are the
    # request's side.
    return item.get('type') in ('function_call_output', 'user_input', 'function_result')


def _settled_body(body: list[JsonValue]) -> list[JsonValue]:
    """The conversation items up to and including the last turn the model produced."""
    conversation = _conversation(body)
    last = max((i for i, item in enumerate(conversation) if not _is_user_side(item)), default=-1)
    return conversation[: last + 1]


@pytest.mark.parametrize(('history_case', 'case'), history_case_params())
async def test_earlier_messages_render_the_same_in_every_later_request(
    history_case: HistoryCase, case: MapperCase[Any]
):
    """Each request starts with what the previous request settled: the prepared messages (see
    `_assert_settled_messages_kept`) and the body up to its last assistant turn. Rendering a turn depends only on that
    turn and what came before it, never on later ones, except for the framing announcement joining the standing
    prompt once (see `_without_new_announcement`)."""
    history = await history_case.history()
    request_ends = [i + 1 for i, message in enumerate(history) if isinstance(message, ModelRequest)]
    renderings = [await _render(case, history[:end]) for end in request_ends]
    for earlier, later in zip(renderings, renderings[1:]):
        _assert_settled_messages_kept(earlier, _without_new_announcement(earlier, later))
        settled_body = _settled_body(earlier.body)
        assert _conversation(later.body)[: len(settled_body)] == settled_body


@pytest.mark.parametrize(('history_case', 'case'), history_case_params())
async def test_round_trip_renders_the_same_request(history_case: HistoryCase, case: MapperCase[Any]):
    """A history dumped and loaded with `ModelMessagesTypeAdapter` renders the same request bytes."""
    history = await history_case.history()
    loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    assert await _render(case, loaded) == await _render(case, history)
