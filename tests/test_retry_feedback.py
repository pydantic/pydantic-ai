"""How Pydantic AI's own retry feedback reaches the model.

A retry that answers a tool call rides that call's result — a `ToolReturnPart` with
`outcome='retried'`. One that doesn't (structured output failed validation, an output validator
raised `ModelRetry`, the response held nothing usable) is a `RetryFeedbackPart`: stored
model-neutrally, and rendered per model at `prepare_messages` time into the system voice, so the
model can tell harness feedback from something a person wrote
(https://github.com/pydantic/pydantic-ai/issues/6404).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pytest
from pydantic import BaseModel
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.exceptions import CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryFeedbackPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelProfile, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr, message_part, try_import

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.anyio


class Answer(BaseModel):
    count: int


@dataclass(frozen=True)
class Case:
    """One way a response can turn out unusable, and what the model is told about it."""

    id: str
    cause: Literal['validation_error', 'no_output', 'model_retry']
    first_response: ModelResponse
    second_response: ModelResponse
    output_type: Any = str
    validator: Callable[[Any], Any] | None = None
    stored_content: Any = None
    """`RetryFeedbackPart.content` as it is kept in history — model-neutral, no wording."""
    rendered: str = ''
    """The text the model is shown, under whichever voice its profile allows."""


def _reject_bad(output: str) -> str:
    if output == 'bad':
        raise ModelRetry('the answer has to be a number')
    return output


CASES = [
    Case(
        id='validation_error',
        cause='validation_error',
        output_type=PromptedOutput(Answer),
        first_response=ModelResponse(parts=[TextPart('{"count": "lots"}')]),
        second_response=ModelResponse(parts=[TextPart('{"count": 3}')]),
        stored_content=snapshot(
            [
                {
                    'type': 'int_parsing',
                    'loc': ('count',),
                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                    'input': 'lots',
                }
            ]
        ),
        rendered=snapshot("""\
The response failed validation:
1 validation error:
```json
[
  {
    "type": "int_parsing",
    "loc": [
      "count"
    ],
    "msg": "Input should be a valid integer, unable to parse string as an integer"
  }
]
```\
"""),
    ),
    Case(
        id='no_output',
        cause='no_output',
        first_response=ModelResponse(parts=[ThinkingPart(content='hmm')]),
        second_response=ModelResponse(parts=[TextPart('3')]),
        stored_content=snapshot('Please return text.'),
        rendered=snapshot('The response contained no usable output. Please return text.'),
    ),
    Case(
        id='model_retry',
        cause='model_retry',
        first_response=ModelResponse(parts=[TextPart('bad')]),
        second_response=ModelResponse(parts=[TextPart('3')]),
        validator=_reject_bad,
        stored_content=snapshot('the answer has to be a number'),
        rendered=snapshot("""\
The response was not accepted:
the answer has to be a number\
"""),
    ),
]


@pytest.mark.parametrize('inline_system_prompts', [True, False], ids=['inline-system', 'system-wrapped'])
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_retry_feedback_is_stored_neutral_and_rendered_in_the_system_voice(
    case: Case, inline_system_prompts: bool
):
    """Every cause keeps its wording out of history and lands in the system voice on the wire.

    A model whose profile takes a mid-conversation system message gets a real one; elsewhere
    `_wrap_non_leading_system_prompts` degrades it to `<system>`-tagged user text. That the wrapped
    side is `<system>`-tagged rather than a bare `SystemPromptPart` is what pins the rendering as
    running *before* the wrap, not after it.
    """
    requests: list[Sequence[ModelRequestPart]] = []

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        requests.append(request.parts)
        return case.first_response if len(requests) == 1 else case.second_response

    model = FunctionModel(respond, profile=ModelProfile(supports_inline_system_prompts=inline_system_prompts))
    agent = Agent(model, output_type=case.output_type)
    if case.validator is not None:
        agent.output_validator(case.validator)

    result = await agent.run('how many?')

    feedback = message_part(result.all_messages(), RetryFeedbackPart, message_index=2)
    assert feedback == RetryFeedbackPart(content=case.stored_content, cause=case.cause, timestamp=IsDatetime())

    rendered_part = requests[1][-1]
    if inline_system_prompts:
        assert isinstance(rendered_part, SystemPromptPart)
        shown = rendered_part.content
    else:
        assert isinstance(rendered_part, UserPromptPart)
        assert isinstance(rendered_part.content, str)
        assert rendered_part.content.startswith('<system>') and rendered_part.content.endswith('</system>')
        shown = rendered_part.content[len('<system>') : -len('</system>')]
    assert shown == case.rendered


async def test_feedback_is_replaced_in_place_and_leaves_the_standing_prompt_alone():
    """The renderer swaps the part where it was authored, so a request that also holds a user
    prompt keeps its order, and the run's standing system prompt is not wrapped along with it."""
    model = FunctionModel(lambda _m, _i: ModelResponse(parts=[TextPart('ok')]))
    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='be terse'), UserPromptPart(content='first')]),
        ModelResponse(parts=[TextPart('nope')]),
        ModelRequest(
            parts=[
                UserPromptPart(content='second'),
                RetryFeedbackPart(content='the answer has to be a number', cause='model_retry'),
            ]
        ),
    ]

    assert model.prepare_messages(history, ModelRequestParameters()) == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='be terse', timestamp=IsDatetime()),
                    UserPromptPart(content='first', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(parts=[TextPart(content='nope')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[
                    UserPromptPart(content='second', timestamp=IsDatetime()),
                    UserPromptPart(
                        content="""\
<system>The response was not accepted:
the answer has to be a number</system>\
""",
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
        ]
    )


def test_retry_feedback_part_round_trips_through_the_type_adapter():
    """A stored history keeps the part's `cause` and raw `ErrorDetails`, not just its rendering."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                RetryFeedbackPart(
                    content=[{'type': 'int_parsing', 'loc': ('count',), 'msg': 'not an int', 'input': 'lots'}],
                    cause='validation_error',
                )
            ]
        )
    ]

    restored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert restored == messages


async def test_a_retried_tool_return_suppresses_an_otherwise_valid_output():
    """Retry-wins keys off the outcome, not the part class: a function tool asking to be called
    again holds back an output produced in the same response."""

    class Output(BaseModel):
        value: str

    responses: list[ModelResponse] = []

    def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        response = (
            ModelResponse(parts=[ToolCallPart('flaky', {}), ToolCallPart('final_result', {'value': 'first'})])
            if not responses
            else ModelResponse(parts=[ToolCallPart('final_result', {'value': 'second'})])
        )
        responses.append(response)
        return response

    agent = Agent(FunctionModel(respond), output_type=Output)

    @agent.tool_plain
    def flaky() -> str:
        if len(responses) == 1:
            raise ModelRetry('not ready yet')
        return 'ready'  # pragma: no cover

    result = await agent.run('go')

    assert result.output == Output(value='second')
    retried = message_part(result.all_messages(), ToolReturnPart, message_index=2)
    assert (retried.tool_name, retried.outcome, retried.content) == ('flaky', 'retried', 'not ready yet')


def _anthropic_tool_result(body: Any) -> Any:
    """Anthropic flags the `tool_result` block itself with `is_error`."""
    return body['messages'][-1]['content'][0]


def _google_tool_result(body: Any) -> Any:
    """Google's function response carries an `error` key where a successful one carries `output`."""
    return body['contents'][-1]['parts'][0]['functionResponse']


def _bedrock_tool_result(body: Any) -> Any:
    """Bedrock's Converse API puts a `status` on the `toolResult` block."""
    return body['messages'][-1]['content'][0]['toolResult']


NATIVE_ERROR_CHANNELS: dict[str, Callable[[Any], Any]] = {
    'anthropic': _anthropic_tool_result,
    'google': _google_tool_result,
    'bedrock': _bedrock_tool_result,
}

NATIVE_ERROR_CHANNEL_EXPECTATIONS: dict[str, Any] = {
    'anthropic': snapshot(
        {
            'tool_use_id': IsStr(),
            'type': 'tool_result',
            'content': [{'text': 'The country is not supported. Use "France" instead.', 'type': 'text'}],
            'is_error': True,
        }
    ),
    'google': snapshot(
        {
            'id': IsStr(),
            'name': 'get_capital',
            'response': {'error': 'The country is not supported. Use "France" instead.'},
        }
    ),
    'bedrock': snapshot(
        {
            'toolUseId': IsStr(),
            'content': [{'text': 'The country is not supported. Use "France" instead.'}],
            'status': 'error',
        }
    ),
}


def _error_channel_model(provider: str, anthropic_api_key: str, gemini_api_key: str, bedrock_provider: Any) -> Model:
    """The three models with a native tool-result error channel, built here rather than taken from
    the shared `model` fixture, whose Google entry pins a model id the API has since retired."""
    if provider == 'anthropic':
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    elif provider == 'google':
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        return GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
    else:
        from pydantic_ai.models.bedrock import BedrockConverseModel

        return BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)


def _model_call_bodies(vcr: Cassette) -> list[Any]:
    """The recorded model-call bodies, in order.

    Not every recorded request is a model call: Bedrock's client signs through a form-encoded STS
    exchange first, so indexing `vcr.requests` directly would read the wrong body on one provider and
    the right one on the others.
    """
    bodies: list[Any] = []
    for request in vcr.requests:  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        try:
            body = json.loads(request.body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        except (TypeError, ValueError):
            continue
        if isinstance(body, dict) and ('messages' in body or 'contents' in body):
            bodies.append(body)
    return bodies


@pytest.mark.vcr
@pytest.mark.parametrize('provider', ['anthropic', 'google', 'bedrock'])
async def test_a_retried_tool_return_takes_the_provider_native_error_channel(
    allow_model_requests: None,
    provider: str,
    anthropic_api_key: str,
    gemini_api_key: str,
    bedrock_provider: Any,
    vcr: Cassette,
):
    """A retry that answers a tool call reaches the provider the way a failure does.

    That is wire parity with the `RetryPromptPart` this replaces, which every one of these mappers
    already rendered as an error, and it is what stops a retry from reading as a result the model
    should build on. Each provider expresses it differently — `is_error`, an `error` key, a `status`
    — so the outbound body is pinned per provider rather than the mapping asserted in the abstract.
    Recorded live, so the assertion is also that each API accepts the retry there.
    """
    calls = 0

    agent = Agent(_error_channel_model(provider, anthropic_api_key, gemini_api_key, bedrock_provider))

    @agent.tool_plain
    def get_capital(country: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry('The country is not supported. Use "France" instead.')
        return 'Paris'

    result = await agent.run('What is the capital of Frankreich? Use the get_capital tool.')

    assert calls >= 1
    retried = message_part(result.all_messages(), ToolReturnPart, message_index=2)
    assert retried.outcome == 'retried'

    # Whether the model retries the call or gives up is its own business; the claim under test is
    # what the second model call carries, which is the retried result either way.
    body = _model_call_bodies(vcr)[1]
    assert NATIVE_ERROR_CHANNELS[provider](body) == NATIVE_ERROR_CHANNEL_EXPECTATIONS[provider]


async def test_legacy_retry_prompt_part_handed_back_through_deferred_results():
    """User code can still answer a deferred call with a `RetryPromptPart`, and it still reaches
    the model as it always did — instruction suffix included."""

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('buy', {'fruit': 'pear'}, tool_call_id='buy_pear')])
        return ModelResponse(parts=[TextPart('understood')])

    agent = Agent(FunctionModel(respond), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def buy(fruit: str) -> str:
        raise CallDeferred

    deferred = await agent.run('buy me a pear')
    assert isinstance(deferred.output, DeferredToolRequests)

    result = await agent.run(
        message_history=deferred.all_messages(),
        deferred_tool_results=DeferredToolResults(
            calls={'buy_pear': RetryPromptPart(content='pears are out of stock')}
        ),
    )

    assert result.output == 'understood'
    retry = message_part(result.all_messages(), RetryPromptPart, message_index=2)
    assert retry.model_response() == snapshot("""\
pears are out of stock

Fix the errors and try again.\
""")


@pytest.mark.skipif(not groq_available(), reason='groq not installed')
async def test_a_legacy_retry_prompt_part_still_maps_unchanged():
    """A stored history holding either shape of `RetryPromptPart` renders exactly as it always did.

    Pinned on Groq because it is a text-only tool API that carries both shapes — the tool-bound one
    as a tool message, the tool-less one as the bare user text that motivated this redesign. No
    cassette would catch a regression here: nothing the framework emits reaches this branch any
    more, so only a history handed in by a user does.
    """
    model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='test-key'))
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                RetryPromptPart(content='pears are out of stock', tool_name='buy', tool_call_id='buy_pear'),
                RetryPromptPart(content='the answer has to be a number'),
            ]
        )
    ]

    mapped = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assert mapped == snapshot(
        [
            {
                'role': 'tool',
                'tool_call_id': 'buy_pear',
                'content': """\
pears are out of stock

Fix the errors and try again.\
""",
            },
            {
                'role': 'user',
                'content': """\
Validation feedback:
the answer has to be a number

Fix the errors and try again.\
""",
            },
        ]
    )


class Nested(BaseModel):
    x: int


async def test_root_level_input_is_serialized_once_per_distinct_value():
    """A retried tool return echoes the arguments the model sent without repeating them per error.

    Root-level errors share one `input` — the whole arguments object — so serializing it into each
    one multiplies a large payload by the error count
    (https://github.com/pydantic/pydantic-ai/issues/7171). The first keeps it, a later one carrying
    that same value drops it, and an error whose `input` is its own offending value keeps it,
    whether it sits at the root or nested.
    """

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('check', {'b': 'oops', 'nested': {'x': 'bad'}})])
        return ModelResponse(parts=[TextPart('understood')])

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def check(a: int, b: int, c: int, nested: Nested) -> int:
        return a + b + c + nested.x  # pragma: no cover

    result = await agent.run('go')

    retried = message_part(result.all_messages(), ToolReturnPart, message_index=2)
    assert retried.outcome == 'retried'
    assert retried.content == snapshot(
        [
            {'type': 'missing', 'loc': ['a'], 'msg': 'Field required', 'input': {'b': 'oops', 'nested': {'x': 'bad'}}},
            {
                'type': 'int_parsing',
                'loc': ['b'],
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'input': 'oops',
            },
            {'type': 'missing', 'loc': ['c'], 'msg': 'Field required'},
            {
                'type': 'int_parsing',
                'loc': ['nested', 'x'],
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'input': 'bad',
            },
        ]
    )


async def test_an_unrendered_feedback_part_reaching_a_model_directly_raises():
    """`prepare_messages` is where a `RetryFeedbackPart` becomes something a model can send, so a
    direct `Model.request()` that skipped it gets told to run it — mirroring the same contract for
    `ToolAvailabilityDeltaPart`."""
    model = TestModel()
    messages: list[ModelMessage] = [ModelRequest(parts=[RetryFeedbackPart(content='no good', cause='model_retry')])]

    with pytest.raises(UserError, match=r'Call `model.prepare_messages\(messages\)` first'):
        await model.request(messages, None, ModelRequestParameters())


@pytest.mark.vcr
@pytest.mark.parametrize(
    ('model', 'expected_turns'),
    [
        pytest.param(
            'openai',
            snapshot(
                [
                    ('user', 'How many continents are there? Answer with just the number.'),
                    ('assistant', '7'),
                    (
                        'system',
                        """\
The response was not accepted:
answer with the number spelled out as a word, nothing else\
""",
                    ),
                ]
            ),
            id='openai-inline-system',
        ),
        pytest.param(
            'anthropic',
            snapshot(
                [
                    ('user', [{'text': 'How many continents are there? Answer with just the number.', 'type': 'text'}]),
                    ('assistant', [{'text': '7', 'type': 'text'}]),
                    (
                        'user',
                        [
                            {
                                'text': """\
<system>The response was not accepted:
answer with the number spelled out as a word, nothing else</system>\
""",
                                'type': 'text',
                            }
                        ],
                    ),
                ]
            ),
            id='anthropic-system-wrapped',
        ),
    ],
    indirect=['model'],
)
async def test_retry_feedback_reaches_the_provider(
    allow_model_requests: None, model: Model, expected_turns: Any, vcr: Cassette
):
    """The rendering is not just internal bookkeeping: each provider accepts the voice its profile
    allows, and answers the feedback.

    `o3-mini` honors a mid-conversation `{'role': 'system'}` entry, so the feedback goes out as one;
    `claude-sonnet-4-5` does not, so it degrades to `<system>`-tagged user text. Both sides of that
    profile flag are pinned here because a cassette matcher that ignores the body would replay green
    either way.
    """
    rejected = False

    agent = Agent(model)

    @agent.output_validator
    def spell_it_out(output: str) -> str:
        nonlocal rejected
        if not rejected:
            rejected = True
            raise ModelRetry('answer with the number spelled out as a word, nothing else')
        return output

    result = await agent.run('How many continents are there? Answer with just the number.')

    assert rejected
    feedback = message_part(result.all_messages(), RetryFeedbackPart, message_index=2)
    assert feedback.cause == 'model_retry'

    second_request = json.loads(vcr.requests[1].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert [(turn['role'], turn['content']) for turn in second_request['messages']] == expected_turns
