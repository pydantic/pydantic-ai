"""Tests for the `Choices` helper: picking one of a set of options known only at run time."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    Choice,
    Choices,
    ModelRetry,
    NativeOutput,
    ToolOutput,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


INTENTS = {
    'refund': 'The customer wants their money back.',
    'replace': 'The customer wants a working unit instead.',
    'escalate': 'Nobody on this tier can resolve it.',
}


def pick(key: str) -> FunctionModel:
    """A model that fills the single output tool with `key`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': key})])

    return FunctionModel(respond)


def output_tool(output_type: Any) -> dict[str, Any]:
    """The output tool definition the model is shown, captured from inside a real run."""
    captured: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        captured.update(name=tool.name, description=tool.description, parameters=tool.parameters_json_schema)
        response = captured['parameters']['properties']['response']
        first = response['enum'][0] if 'enum' in response else response['anyOf'][0]['const']
        return ModelResponse(parts=[ToolCallPart(tool.name, {'response': first})])

    Agent(FunctionModel(respond), output_type=output_type).run_sync('hello')
    return captured


def test_described_choices_are_an_any_of_of_consts():
    """Descriptions produce the same shape `GenerateToolJsonSchema.enum_schema` emits for a described enum."""
    assert output_tool(
        Choices(INTENTS, name='customer_intent', description='What the customer is asking for.')
    ) == snapshot(
        {
            'name': 'final_result',
            'description': 'The final response which ends this conversation',
            'parameters': {
                'properties': {
                    'response': {
                        'anyOf': [
                            {'const': 'refund', 'description': 'The customer wants their money back.'},
                            {'const': 'replace', 'description': 'The customer wants a working unit instead.'},
                            {'const': 'escalate', 'description': 'Nobody on this tier can resolve it.'},
                        ],
                        'description': 'What the customer is asking for.',
                        'type': 'string',
                    }
                },
                'required': ['response'],
                'type': 'object',
            },
        }
    )


def test_undescribed_choices_degrade_to_a_plain_enum():
    """With nothing to say about the options, the maximally supported `enum` form is what goes out."""
    assert output_tool(Choices(['yes', 'no']))['parameters'] == snapshot(
        {
            'properties': {'response': {'enum': ['yes', 'no'], 'type': 'string'}},
            'required': ['response'],
            'type': 'object',
        }
    )


def test_partially_described_choices_describe_only_what_was_described():
    schema = output_tool(Choices({'yes': Choice(), 'no': Choice('Only when you are sure.')}))
    assert schema['parameters']['properties']['response'] == snapshot(
        {'anyOf': [{'const': 'yes'}, {'const': 'no', 'description': 'Only when you are sure.'}], 'type': 'string'}
    )


def test_a_pick_is_the_key():
    agent = Agent(pick('replace'), output_type=Choices(INTENTS))
    assert agent.run_sync('My blender arrived in pieces.').output == 'replace'


def test_an_invented_option_is_rejected():
    """The model can't answer with something that isn't on the list, unlike a hand-built `StructuredDict` schema."""
    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
        Agent(pick('incinerate'), output_type=Choices(INTENTS), retries=0).run_sync('x')


class Doc(BaseModel):
    id: str
    title: str


DOCS = [Doc(id='rfc-6265', title='HTTP State Management'), Doc(id='rfc-9110', title='HTTP Semantics')]


def test_a_choice_can_stand_for_a_value():
    Cited = Choices({doc.id: Choice(doc.title, value=doc) for doc in DOCS})
    assert Agent(pick('rfc-9110'), output_type=Cited).run_sync('Which one covers semantics?').output == snapshot(
        Doc(id='rfc-9110', title='HTTP Semantics')
    )


@dataclass
class Screen:
    clicks: list[str]

    def click(self, target: str) -> str:
        self.clicks.append(target)
        return f'clicked {target}'

    async def observe(self) -> str:
        return 'looked again'


def actions(screen: Screen) -> type[str]:
    return Choices(
        {
            'login': Choice('The Login button, top right.', value=partial(screen.click, 'login')),
            'reobserve': Choice('Look again before deciding.', value=screen.observe),
            'abstain': 'Do nothing, because none of these is safe.',
        },
        description='Which action to take on the screen.',
    )


@pytest.mark.parametrize(
    ('key', 'expected'),
    [('login', 'clicked login'), ('reobserve', 'looked again'), ('abstain', 'abstain')],
)
def test_a_callable_choice_value_is_called(key: str, expected: str):
    """The pick *is* the action: sync and async callables both run, and a plain option still yields its key."""
    screen = Screen(clicks=[])
    result = Agent(pick(key), output_type=actions(screen)).run_sync('The login page is showing.')
    assert result.output == expected
    assert screen.clicks == (['login'] if key == 'login' else [])


def test_a_callable_choice_value_does_not_change_the_schema():
    """However a set was authored, the model is asked the same question."""
    assert output_tool(actions(Screen(clicks=[])))['parameters']['properties']['response'] == snapshot(
        {
            'anyOf': [
                {'const': 'login', 'description': 'The Login button, top right.'},
                {'const': 'reobserve', 'description': 'Look again before deciding.'},
                {'const': 'abstain', 'description': 'Do nothing, because none of these is safe.'},
            ],
            'description': 'Which action to take on the screen.',
            'type': 'string',
        }
    )


async def test_a_picked_action_runs_once_across_a_stream():
    """Streaming re-validates a completed value on every later chunk, so the action runs for the final output only."""
    runs: list[str] = []

    def act() -> str:
        runs.append('ran')
        return 'acted'

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        name = info.output_tools[0].name
        for index, chunk in enumerate(['{"resp', 'onse": "a', 'ct', '"}']):
            yield {0: DeltaToolCall(name=name if index == 0 else None, json_args=chunk, tool_call_id='pick')}

    agent = Agent(FunctionModel(stream_function=stream), output_type=Choices({'act': Choice('Do it.', value=act)}))
    async with agent.run_stream('x') as result:
        streamed = [output async for output in result.stream_output(debounce_by=None)]
        assert await result.get_output() == 'acted'
    # Until the pick is final the key stands in for itself, so the action's side effect happens exactly once.
    assert streamed == snapshot(['act', 'act', 'acted'])
    assert runs == ['ran']


async def test_a_picked_action_can_ask_for_a_retry():
    """`ModelRetry` from an action reaches the model as a retry prompt, the way it does from an output function."""
    attempts: list[str] = []

    def flaky() -> str:
        attempts.append('tried')
        raise ModelRetry('That one is out of stock, pick another.')

    keys = iter(['replace', 'refund'])

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': next(keys)})])

    agent = Agent(
        FunctionModel(respond),
        output_type=Choices({'replace': Choice('Send another.', value=flaky), 'refund': 'Give the money back.'}),
    )
    result = await agent.run('x')
    assert result.output == 'refund'
    assert attempts == ['tried']
    assert [
        part.content for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
    ] == snapshot(['That one is out of stock, pick another.'])


async def test_a_picked_action_cannot_ask_for_a_retry_while_streaming():
    """`run_stream()` doesn't support retries, so an action's `ModelRetry` surfaces the way any other does."""

    def nope() -> str:
        raise ModelRetry('not this one')

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"response": "act"}', tool_call_id='pick')}

    agent = Agent(FunctionModel(stream_function=stream), output_type=Choices({'act': Choice('Do it.', value=nope)}))
    with pytest.raises(UnexpectedModelBehavior, match='Output validation failed during streaming'):
        async with agent.run_stream('x') as result:
            await result.get_output()


@pytest.mark.parametrize('with_actions', [False, True])
def test_output_json_schema_describes_the_keys(with_actions: bool):
    """What a set asks the model for is a key, even when the key stands for an action to run."""
    choices = (
        Choices({'refund': INTENTS['refund'], 'replace': Choice(INTENTS['replace'], value=lambda: 'shipped')})
        if with_actions
        else Choices({'refund': INTENTS['refund'], 'replace': INTENTS['replace']})
    )
    assert Agent(pick('refund'), output_type=choices).output_json_schema() == snapshot(
        {
            'anyOf': [
                {'const': 'refund', 'description': 'The customer wants their money back.'},
                {'const': 'replace', 'description': 'The customer wants a working unit instead.'},
            ],
            'type': 'string',
        }
    )


def test_a_choices_set_with_actions_can_be_one_of_several_output_types():
    """Each union member gets its own output tool, and picking the set's tool still runs the action."""
    captured: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(tool.name for tool in info.output_tools)
        return ModelResponse(parts=[ToolCallPart('final_result_action', {'response': 'login'})])

    action = Choices({'login': Choice('Click login.', value=lambda: 'clicked login')}, name='action')
    result = Agent(FunctionModel(respond), output_type=[action, Doc]).run_sync('x')
    assert result.output == 'clicked login'
    assert captured == snapshot(['final_result_action', 'final_result_Doc'])


ACTIONS = Choices({'act': Choice('Do it.', value=lambda: 'acted')})


def test_a_callable_choice_value_is_refused_outside_an_output_type():
    """There is nothing to call the action as a field or a parameter, so the schema refuses to be built."""
    with pytest.raises(UserError, match="can only be used as an agent's `output_type`"):

        class Plan(BaseModel):  # pyright: ignore[reportUnusedClass]
            action: ACTIONS  # pyright: ignore[reportInvalidTypeForm]

    agent = Agent(pick('act'))

    with pytest.raises(UserError, match="can only be used as an agent's `output_type`"):

        @agent.tool_plain  # pyright: ignore[reportUnknownArgumentType]
        def take(action: ACTIONS) -> str:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
            return 'done'  # pragma: no cover


SEVERITIES = Choices(
    {'blocker': 'Nothing else can proceed until this is fixed.', 'minor': 'Cosmetic or rare.'},
    description='How bad it is.',
)


class Report(BaseModel):
    summary: str
    severity: SEVERITIES  # pyright: ignore[reportInvalidTypeForm]


def test_choices_works_as_a_model_field():
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {'summary': 'Login loops', 'severity': 'blocker'})]
        )

    result = Agent(FunctionModel(respond), output_type=Report).run_sync('x')
    assert result.output == snapshot(Report(summary='Login loops', severity='blocker'))


def test_choices_works_as_a_tool_parameter():
    captured: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not captured:
            captured.update(info.function_tools[0].parameters_json_schema)
            return ModelResponse(parts=[ToolCallPart('triage', {'severity': 'minor'}, tool_call_id='triage')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), output_type=str)

    @agent.tool_plain  # pyright: ignore[reportUnknownArgumentType]
    def triage(severity: SEVERITIES) -> str:  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
        return f'filed as {severity}'

    agent.run_sync('x')
    assert captured == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'severity': {
                    'anyOf': [
                        {'const': 'blocker', 'description': 'Nothing else can proceed until this is fixed.'},
                        {'const': 'minor', 'description': 'Cosmetic or rare.'},
                    ],
                    'description': 'How bad it is.',
                    'type': 'string',
                }
            },
            'required': ['severity'],
            'type': 'object',
        }
    )


def test_name_and_description_name_the_picked_field_not_the_tool():
    """`ToolOutput`'s own `name`/`description` describe the tool around the set, and take precedence."""
    schema = output_tool(
        ToolOutput(
            Choices(INTENTS, name='customer_intent', description='What the customer is asking for.'),
            name='classify',
            description='Work out what they want.',
        )
    )
    assert (schema['name'], schema['description']) == snapshot(('classify', 'Work out what they want.'))


def test_choices_names_the_object_in_native_output():
    captured: dict[str, Any] = {}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        output_object = info.model_request_parameters.output_object
        assert output_object is not None
        captured.update(name=output_object.name, json_schema=output_object.json_schema)
        return ModelResponse(parts=[TextPart('{"response": "refund"}')])

    result = Agent(
        FunctionModel(respond),
        output_type=NativeOutput(Choices(INTENTS, name='customer_intent', description='What the customer wants.')),
    ).run_sync('x')
    assert result.output == 'refund'
    assert captured['name'] == snapshot('customer_intent')
    assert captured['json_schema']['properties']['response']['description'] == snapshot('What the customer wants.')


def test_choices_construction_is_checked():
    with pytest.raises(UserError, match=re.escape('`Choices` requires at least one choice.')):
        Choices({})

    with pytest.raises(UserError, match='not a single string'):
        Choices('abc')

    with pytest.raises(UserError, match='requires some'):
        Choice('The Login button.', value=Screen(clicks=[]).click)


def test_a_callable_without_an_introspectable_signature_is_taken_at_its_word():
    class Opaque:
        __signature__ = 'not a signature'

        def __call__(self) -> str:
            return 'ran'

    assert Agent(pick('go'), output_type=Choices({'go': Choice('Go.', value=Opaque())})).run_sync('x').output == 'ran'


# `pick` is what each provider actually answered when the cassettes were recorded. The claim under test is
# that the `anyOf`-of-`const`s schema survives the wire -- OpenAI receives it under `strict: true` and accepts
# it -- and comes back as one of the options, not that a given model reads a complaint the way a person would.
@pytest.mark.parametrize(('model', 'pick'), [('openai', 'escalate'), ('anthropic', 'replace')], indirect=['model'])
async def test_choices_round_trips_on_the_wire(allow_model_requests: None, model: Model, pick: str):
    """A described `Choices` set is understood by real providers, which answer with one of its options."""
    agent = Agent(
        model,
        output_type=Choices(INTENTS, name='customer_intent', description='What the customer is asking for.'),
    )
    result = await agent.run('The blender arrived smashed. Please send me a replacement unit.')
    assert result.output in INTENTS
    assert result.output == pick
