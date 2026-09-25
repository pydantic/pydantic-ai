from __future__ import annotations

import pickle
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, RunContext, Tool, ToolOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    ToolCallProposed,
    UnsureRoute,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsStr


class InMemoryDecisionModel(DecisionModel[None]):
    def __init__(self):
        super().__init__()
        self.requests: list[DecisionRequest] = []

    @property
    def model_name(self) -> str:
        return 'in-memory-decisions'

    @property
    def system(self) -> str:
        return 'test-decisions'

    @property
    def base_url(self) -> str:
        return 'https://example.test/decisions'

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        self.requests.append(request)
        answers: dict[str, DecisionAnswer] = {}
        for name, question in request.questions.items():
            if isinstance(question, ScoreQuestion):
                answers[name] = ScoreAnswer(
                    score=10,
                    confidence=1,
                    probabilities={level: float(level == 10) for level in range(11)},
                )
            elif isinstance(question, NoulQuestion):
                answers[name] = NoulAnswer(noul=0.8)
            else:
                assert isinstance(question, ChoiceQuestion)
                choice = 'review' if 'review' in question.criteria else next(iter(question.criteria))
                answers[name] = ChoiceAnswer(
                    choice=choice,
                    confidence=0.9,
                    probabilities={option: float(option == choice) for option in question.criteria},
                )
        return DecisionResponse(
            answers=answers,
            model_name=self.model_name,
            usage=RequestUsage(input_tokens=4, output_tokens=2),
        )


class Triage(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need an immediate response?')
    action: Literal['approve', 'review'] = Field(description='What should happen next?')


@pytest.mark.anyio
async def test_decision_model_extension_point(allow_model_requests: None):
    model = InMemoryDecisionModel()
    result = await Agent(model, output_type=Triage).run('The customer cannot sign in.')

    assert result.output == Triage(urgent=True, action='review')
    assert model.requests == snapshot(
        [
            DecisionRequest(
                state='The customer cannot sign in.',
                questions={
                    'urgent': NoulQuestion(
                        instructions={
                            'field': 'urgent',
                            'question': 'Does this need an immediate response?',
                            'goal': 'Triage a support ticket.',
                        }
                    ),
                    'action': ChoiceQuestion(
                        criteria={'approve': None, 'review': None},
                        instructions={
                            'field': 'action',
                            'question': 'What should happen next?',
                            'goal': 'Triage a support ticket.',
                        },
                    ),
                },
            )
        ]
    )
    assert result.response.usage == RequestUsage(input_tokens=4, output_tokens=2)
    assert result.response.provider_name == 'test-decisions'
    assert result.response.provider_url == 'https://example.test/decisions'


@pytest.mark.anyio
async def test_thinking_goes_into_the_history(allow_model_requests: None):
    """A model's thinking is sent with the rest of its response, in the order it was produced.

    A unit test pins the exact `state`, which a cassette matched without its body would not.
    """
    model = InMemoryDecisionModel()
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Where is my order?')]),
        ModelResponse(
            parts=[
                ThinkingPart('I should look the order up.'),
                TextPart('Let me check.'),
                ThinkingPart('', signature='encrypted-by-the-provider'),
                ToolCallPart('look_up_order', {'order_id': 42}, tool_call_id='call_1'),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart('look_up_order', 'Shipped.', tool_call_id='call_1')]),
        ModelResponse(parts=[TextPart('It has shipped.')]),
    ]
    await Agent(model, output_type=Triage).run('Thanks!', message_history=history)

    assert model.requests[0].state == snapshot(
        {
            'history': [
                {'user': 'Where is my order?'},
                {'thinking': 'I should look the order up.'},
                {'assistant': 'Let me check.'},
                {'tool_call': {'name': 'look_up_order', 'args': {'order_id': 42}}},
                {'tool_return': {'name': 'look_up_order', 'content': 'Shipped.'}},
                {'assistant': 'It has shipped.'},
            ],
            'text': 'Thanks!',
        }
    )


@pytest.mark.anyio
async def test_judging_what_a_model_thought(allow_model_requests: None):
    """A judge given another run's messages sees that run's thinking, which can be the very thing judged."""
    model = InMemoryDecisionModel()
    judge = Agent(model, output_type=bool, instructions='Did the assistant consider getting around the tests?')
    conversation: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Make the tests pass.')]),
        ModelResponse(
            parts=[
                ThinkingPart('The quickest way is to skip the failing test.'),
                TextPart('Done: all tests pass.'),
            ]
        ),
    ]
    await judge.run(message_history=conversation)

    assert model.requests[0].state == snapshot(
        {
            'history': [
                {'user': 'Make the tests pass.'},
                {'thinking': 'The quickest way is to skip the failing test.'},
                {'assistant': 'Done: all tests pass.'},
            ]
        }
    )


@pytest.mark.anyio
async def test_no_choice_limit(allow_model_requests: None):
    model = InMemoryDecisionModel()
    tools = [
        ToolDefinition(name=f'tool_{index}', description=None, parameters_json_schema={'type': 'object'})
        for index in range(256)
    ]

    response = await model.request(
        [ModelRequest(parts=[UserPromptPart('Pick a tool.')])],
        None,
        ModelRequestParameters(function_tools=tools, allow_text_output=False),
    )

    question = model.requests[0].questions['route']
    assert isinstance(question, ChoiceQuestion)
    assert len(question.criteria) == 256
    assert len(response.parts) == 1
    part = response.parts[0]
    assert isinstance(part, ToolCallPart)
    assert part.tool_name == 'tool_0'
    assert part.args == {}


Rubric = Annotated[
    Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    WithJsonSchema(
        {'type': 'integer', 'anyOf': [{'const': level, 'description': f'Level {level}'} for level in range(11)]}
    ),
]


class ElevenLevelReview(BaseModel):
    score: Rubric


@pytest.mark.anyio
async def test_no_score_limit(allow_model_requests: None):
    model = InMemoryDecisionModel()
    result = await Agent(model, output_type=ElevenLevelReview).run('Score this.')

    assert result.output == ElevenLevelReview(score=10)
    question = model.requests[0].questions['score']
    assert isinstance(question, ScoreQuestion)
    assert len(question.criteria) == 11


class TenLevelDecisionModel(InMemoryDecisionModel):
    max_score_levels = 10


class OptionalElevenLevelReview(BaseModel):
    score: Rubric | None


@pytest.mark.anyio
async def test_levels_over_score_limit_are_a_pick_one(allow_model_requests: None):
    model = TenLevelDecisionModel()
    result = await Agent(model, output_type=ElevenLevelReview).run('Score this.')

    assert result.output == ElevenLevelReview(score=0)
    question = model.requests[0].questions['score']
    assert isinstance(question, ChoiceQuestion)
    assert question.criteria == {str(level): f'Level {level}' for level in range(11)}


@pytest.mark.anyio
async def test_levels_over_score_limit_can_be_optional(allow_model_requests: None):
    model = TenLevelDecisionModel()
    result = await Agent(model, output_type=OptionalElevenLevelReview).run('Score this.')

    assert result.output == OptionalElevenLevelReview(score=0)
    question = model.requests[0].questions['score']
    assert isinstance(question, ChoiceQuestion)
    assert len(question.criteria) == 12


@pytest.mark.anyio
async def test_rubric_without_score_limit_cannot_be_optional(allow_model_requests: None):
    with pytest.raises(UserError, match="Output field 'score' is a rubric, and a rubric cannot be optional"):
        await Agent(InMemoryDecisionModel(), output_type=OptionalElevenLevelReview).run('Score this.')


class MappingResult(BaseModel):
    flags: dict[Literal['a', 'b'], bool] = Field(description='Which flags apply?')


@pytest.mark.anyio
async def test_mapping_output(allow_model_requests: None):
    result = await Agent(InMemoryDecisionModel(), output_type=MappingResult).run('Both apply.')
    assert result.output == MappingResult(flags={'a': True, 'b': True})


class RecursiveResult(BaseModel):
    child: RecursiveResult


class LimitedListResult(BaseModel):
    values: list[Literal['a', 'b']] = Field(description='Which apply?', max_length=1)


class SteppedProbabilityResult(BaseModel):
    value: float = Field(description='How likely?', ge=0, le=1, multiple_of=0.1)


class LimitedMappingResult(BaseModel):
    values: dict[Literal['a', 'b'], bool] = Field(description='Which apply?', max_length=1)


class WrongMappingValueResult(BaseModel):
    values: dict[Literal['a', 'b'], str] = Field(description='Which apply?')


class UnboundedMappingResult(BaseModel):
    values: dict[str, bool] = Field(description='Which apply?')


class OneMappingOptionResult(BaseModel):
    values: dict[Literal['a'], bool] = Field(description='Which apply?')


@pytest.mark.anyio
@pytest.mark.parametrize(
    'output_type',
    [
        RecursiveResult,
        LimitedListResult,
        SteppedProbabilityResult,
        LimitedMappingResult,
        WrongMappingValueResult,
        UnboundedMappingResult,
        OneMappingOptionResult,
    ],
)
async def test_unsupported_decision_shapes(allow_model_requests: None, output_type: type[BaseModel]):
    with pytest.raises(UserError):
        await Agent(InMemoryDecisionModel(), output_type=output_type).run('Anything.')


@pytest.mark.anyio
async def test_mapping_needs_two_options(allow_model_requests: None):
    output_tool = ToolDefinition(
        name='final_result',
        description='Return the result.',
        kind='output',
        parameters_json_schema={
            'type': 'object',
            'properties': {
                'values': {
                    'type': 'object',
                    'description': 'Which apply?',
                    'additionalProperties': {'type': 'boolean'},
                    'propertyNames': {'enum': ['a']},
                }
            },
        },
    )

    with pytest.raises(UserError, match='a mapping must be keyed by two or more options'):
        await InMemoryDecisionModel().request(
            [ModelRequest(parts=[UserPromptPart('Anything.')])],
            None,
            ModelRequestParameters(output_tools=[output_tool], output_mode='tool', allow_text_output=False),
        )


@pytest.mark.anyio
async def test_text_output_is_refused(allow_model_requests: None):
    """A decision model cannot write text, so a `str` branch is refused rather than silently never taken."""
    model = InMemoryDecisionModel()
    assert model.profile.get('supports_text_output') is False
    with pytest.raises(UserError, match='Text output is not supported by this model'):
        await Agent(model, output_type=[Triage, str]).run('The checkout page returns a 500 for every customer.')
    assert model.requests == []


class Escalation(BaseModel):
    """Hand the ticket to a person."""

    security: bool = Field(description='Is this a security issue?')


def refund(amount: float) -> str:
    """Return a payment to the customer."""
    return f'Refunded {amount}'  # pragma: no cover


async def escalate(ctx: RunContext[None]) -> str:
    """Escalate to a person on the support team."""
    return 'escalated'


class Priority(str, Enum):
    """How soon the ticket needs a reply."""

    now = 'now'
    later = 'later'


def assign(team: Literal['billing', 'technical']) -> str:
    """Assign the ticket to a team.

    Args:
        team: Which team should handle it?
    """
    return team


def route_question(model: InMemoryDecisionModel) -> ChoiceQuestion:
    question = model.requests[0].questions['route']
    assert isinstance(question, ChoiceQuestion)
    return question


@pytest.mark.anyio
@pytest.mark.parametrize(
    'output_type,labels',
    [
        pytest.param(Triage, ['Triage', 'refund'], id='a single output type goes by its class name'),
        pytest.param(bool, ['output', 'refund'], id='a wrapped bare output type goes by `output`'),
        pytest.param(Priority, ['Priority', 'refund'], id='a wrapped `Enum` goes by its class name'),
        pytest.param(
            ToolOutput(Triage, name='triage_it'), ['triage_it', 'refund'], id='a named output goes by its name'
        ),
        pytest.param(
            [Triage, Escalation, None],
            ['Triage', 'Escalation', 'None', 'refund'],
            id='union members go by their own names',
        ),
        pytest.param([Triage, escalate], ['Triage', 'escalate', 'refund'], id='a hand-off goes by its name'),
        pytest.param(escalate, ['escalate', 'refund'], id='a single hand-off goes by its name'),
        pytest.param(assign, ['assign', 'refund'], id='a single output function goes by its name'),
        pytest.param(ToolOutput(Triage), ['Triage', 'refund'], id='an unnamed `ToolOutput` goes by its title'),
    ],
)
async def test_route_labels(allow_model_requests: None, output_type: Any, labels: list[str]):
    """Each route is offered under the name the user gave it, never the name of the tool Pydantic AI made for it."""
    model = InMemoryDecisionModel()
    await Agent(model, output_type=output_type, tools=[refund], instructions='Is it urgent?').run('Charged twice.')
    assert list(route_question(model).criteria) == labels


@pytest.mark.anyio
async def test_the_fill_calls_the_route_what_the_route_question_did(allow_model_requests: None):
    """One route, one name: the fill's `chosen` is the label the route question offered and the model answered."""
    model = InMemoryDecisionModel()
    result = await Agent(model, output_type=[Escalation, Triage]).run('Someone else can see my invoices.')

    assert result.output == Escalation(security=True)
    assert model.requests == snapshot(
        [
            DecisionRequest(
                state='Someone else can see my invoices.',
                questions={
                    'Escalation.security': NoulQuestion(
                        instructions={
                            'field': 'security',
                            'premise': "If the user's request calls for Escalation: Hand the ticket to a person.",
                            'question': 'Is this a security issue?',
                        }
                    ),
                    'Triage.urgent': NoulQuestion(
                        instructions={
                            'field': 'urgent',
                            'premise': "If the user's request calls for Triage: Triage a support ticket.",
                            'question': 'Does this need an immediate response?',
                        }
                    ),
                    'Triage.action': ChoiceQuestion(
                        criteria={'approve': None, 'review': None},
                        instructions={
                            'field': 'action',
                            'premise': "If the user's request calls for Triage: Triage a support ticket.",
                            'question': 'What should happen next?',
                        },
                    ),
                    'route': ChoiceQuestion(
                        criteria={'Escalation': 'Hand the ticket to a person.', 'Triage': 'Triage a support ticket.'},
                        instructions='Which of these does this call for?',
                    ),
                },
            )
        ]
    )
    # `provider_details` names routes by their labels too, not by the tools Pydantic AI made for them.
    assert (result.response.provider_details or {})['route'] == snapshot(
        {
            'choice': 'Escalation',
            'probabilities': {'Escalation': 1.0, 'Triage': 0.0},
            'offered': ['Escalation', 'Triage'],
        }
    )


@pytest.mark.anyio
async def test_a_route_label_collision_renames_the_output_route(allow_model_requests: None):
    """A tool keeps its name; an output route that would share it gets ` (output)`, and is still read back right."""
    output_tools = [
        ToolDefinition(
            name=f'final_result_{name}',
            description=f'{name} the ticket.',
            kind='output',
            parameters_json_schema={
                'type': 'object',
                'properties': {'urgent': {'type': 'boolean', 'description': 'Is it urgent?'}},
            },
        )
        for name in ('Refund', 'Triage')
    ]
    function_tool = ToolDefinition(
        name='Refund', description='Refund the customer.', parameters_json_schema={'type': 'object'}
    )
    model = InMemoryDecisionModel()

    response = await model.request(
        [ModelRequest(parts=[UserPromptPart('Charged twice.')])],
        None,
        ModelRequestParameters(
            output_mode='tool', output_tools=output_tools, function_tools=[function_tool], allow_text_output=False
        ),
    )

    assert route_question(model).criteria == snapshot(
        {'Refund (output)': 'Refund the ticket.', 'Triage': 'Triage the ticket.', 'Refund': 'Refund the customer.'}
    )
    assert response.parts == [ToolCallPart('final_result_Refund', {'urgent': True}, tool_call_id=IsStr())]
    # Both output routes were asked up front, each under its label, and only the taken one's answers were read.
    assert list(model.requests[0].questions) == snapshot(['Refund (output).urgent', 'Triage.urgent', 'route'])
    assert model.requests[0].questions['Refund (output).urgent'].instructions == snapshot(
        {
            'field': 'urgent',
            'premise': "If the user's request calls for Refund (output): Refund the ticket.",
            'question': 'Is it urgent?',
        }
    )


class Routed(BaseModel):
    """Route a support ticket."""

    route: bool = Field(description='Does it name a delivery route?')


@pytest.mark.anyio
async def test_the_route_question_stays_clear_of_a_field_named_route(allow_model_requests: None):
    """A field asked beside the route question is keyed under its route's label, so `route` is never a field's."""
    model = InMemoryDecisionModel()
    await Agent(model, output_type=Routed, tools=[refund]).run('Take the A2.')
    assert list(model.requests[0].questions) == ['Routed.route', 'route']
    assert list(route_question(model).criteria) == ['Routed', 'refund']


class RoutingDecisionModel(InMemoryDecisionModel):
    """Answers the route question, which every request to it carries, from a fixed distribution over the routes offered."""

    def __init__(self, route: dict[str, float]):
        super().__init__()
        self.route = route

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        question = request.questions.get('route')
        if not isinstance(question, ChoiceQuestion):
            # A fill, or a single output type left with nothing else on offer: there is no route to pick.
            return response
        # A route that is no longer offered keeps its probability out of the answer, as a real model's would.
        probabilities = {label: self.route[label] for label in question.criteria}
        choice = max(probabilities, key=lambda label: probabilities[label])
        response.answers['route'] = ChoiceAnswer(
            choice=choice, confidence=probabilities[choice], probabilities=probabilities
        )
        return response


def look_up_order() -> str:
    """Look up the customer's order."""
    return 'Order #1 shipped yesterday.'


def issue_refund() -> str:
    """Refund the customer's last payment."""
    return 'Refunded.'  # pragma: no cover


@pytest.mark.anyio
async def test_the_likeliest_route_is_taken_however_unsure(allow_model_requests: None):
    """With no `decision_route_threshold`, the pick is taken at any probability: here a tool at 0.46."""
    model = RoutingDecisionModel({'Triage': 0.44, 'look_up_order': 0.46, 'issue_refund': 0.1})
    result = await Agent(model, output_type=Triage, tools=[look_up_order]).run('Where is my order?')

    [first, *_] = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert [part.tool_name for part in first.parts if isinstance(part, ToolCallPart)] == ['look_up_order']
    # The output's fields were asked beside the route question, but the tool call was not built from them.
    assert first.provider_details == snapshot(
        {
            'confidence': {},
            'probabilities': {},
            'scores': {},
            'route': {
                'choice': 'look_up_order',
                'probabilities': {'Triage': 0.44, 'look_up_order': 0.46},
                'offered': ['Triage', 'look_up_order'],
            },
        }
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    'output_type,route,picked',
    [
        pytest.param(Triage, {'Triage': 0.3, 'look_up_order': 0.45, 'issue_refund': 0.25}, 'look_up_order', id='tool'),
        pytest.param(Triage, {'Triage': 0.6, 'look_up_order': 0.3, 'issue_refund': 0.1}, 'Triage', id='output'),
        pytest.param(
            [Triage, None], {'Triage': 0.3, 'None': 0.5, 'look_up_order': 0.2, 'issue_refund': 0.0}, 'None', id='None'
        ),
    ],
)
async def test_a_pick_below_the_route_threshold_is_unsure(
    allow_model_requests: None, output_type: Any, route: dict[str, float], picked: str
):
    """Every kind of route is held to the bar, and the step is handed on before anything is filled."""
    model = RoutingDecisionModel(route)
    agent = Agent(
        model,
        output_type=output_type,
        tools=[look_up_order, issue_refund],
        model_settings=DecisionModelSettings(decision_route_threshold=0.7),
    )
    with pytest.raises(UnsureRoute) as exc_info:
        await agent.run('Where is my order?')

    error = exc_info.value
    assert (error.model_name, error.route, error.threshold) == ('in-memory-decisions', picked, 0.7)
    assert error.probability == route[picked]
    assert error.probabilities == route
    assert len(model.requests) == 1


@pytest.mark.anyio
async def test_an_unsure_route_says_what_to_do_about_it(allow_model_requests: None):
    model = RoutingDecisionModel({'Triage': 0.3, 'look_up_order': 0.7})
    agent = Agent(
        model,
        output_type=Triage,
        tools=[look_up_order],
        model_settings=DecisionModelSettings(decision_route_threshold=0.75),
    )
    with pytest.raises(UnsureRoute) as exc_info:
        await agent.run('Where is my order?')
    assert str(exc_info.value) == snapshot(
        "in-memory-decisions picked 'look_up_order' with probability 0.70, below `decision_route_threshold` (0.75). "
        'Put a model behind it to take the steps it is unsure of: `FallbackModel(decision_model, language_model)` '
        'hands `language_model` this step.'
    )


@pytest.mark.anyio
@pytest.mark.parametrize('threshold', [0.7, 0.5])
async def test_a_pick_at_or_above_the_route_threshold_is_taken(allow_model_requests: None, threshold: float):
    model = RoutingDecisionModel({'Triage': 0.3, 'look_up_order': 0.7})
    agent = Agent(
        model,
        output_type=Triage,
        tools=[look_up_order],
        model_settings=DecisionModelSettings(decision_route_threshold=threshold),
    )
    result = await agent.run('Where is my order?')

    assert result.output == Triage(urgent=True, action='review')
    [first, *_] = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert [part.tool_name for part in first.parts if isinstance(part, ToolCallPart)] == ['look_up_order']


@pytest.mark.anyio
async def test_a_fallback_model_takes_the_unsure_step(allow_model_requests: None):
    """`UnsureRoute` is a `ModelAPIError`, so the default `FallbackModel` hands the whole step to the next model."""
    decision_model = RoutingDecisionModel({'Triage': 0.55, 'look_up_order': 0.45})
    model = FallbackModel(decision_model, TestModel(call_tools=[]))
    agent = Agent(
        model,
        output_type=Triage,
        tools=[look_up_order],
        model_settings=DecisionModelSettings(decision_route_threshold=0.7),
    )
    result = await agent.run('Where is my order?')

    assert result.response.model_name == 'test'
    assert len(decision_model.requests) == 1


def approve() -> str:
    """Approve the request as it stands."""
    return 'approved'  # pragma: no cover


def set_urgency(urgent: bool) -> str:
    """Set how urgent the ticket is."""
    return f'urgent={urgent}'  # pragma: no cover


@pytest.mark.anyio
@pytest.mark.parametrize(
    'tool',
    [pytest.param(approve, id='with nothing to fill'), pytest.param(set_urgency, id='with arguments to fill')],
)
async def test_the_last_route_left_is_not_held_to_the_route_threshold(allow_model_requests: None, tool: Any):
    """With every other route returned this turn, nothing was picked, so there is no pick to be unsure of."""
    history = [
        ModelRequest(parts=[UserPromptPart('Where is my order?')]),
        ModelResponse(parts=[ToolCallPart('look_up_order', {}, 'call_1')]),
        ModelRequest(parts=[ToolReturnPart('look_up_order', 'Order #1 shipped yesterday.', 'call_1')]),
    ]
    model = InMemoryDecisionModel()
    response = await model.request(
        history,
        DecisionModelSettings(decision_route_threshold=1.0),
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(name='look_up_order', description='Look up the customer order.'),
                Tool(tool).tool_def,
            ],
            allow_text_output=False,
        ),
    )

    assert [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)] == [tool.__name__]
    name = tool.__name__
    assert (response.provider_details or {})['route'] == {
        'choice': name,
        'probabilities': {name: 1.0},
        'offered': [name],
    }


@pytest.mark.anyio
async def test_a_single_output_type_is_not_held_to_the_route_threshold(allow_model_requests: None):
    """With nothing else on offer there is no route question, so no pick to be unsure of."""
    model = InMemoryDecisionModel()
    agent = Agent(model, output_type=Triage, model_settings=DecisionModelSettings(decision_route_threshold=1.0))
    result = await agent.run('Where is my order?')

    assert result.output == Triage(urgent=True, action='review')
    assert 'route' not in model.requests[0].questions


def test_unsure_route_pickles():
    exc = pickle.loads(pickle.dumps(UnsureRoute('jev-latest', 'refund', {'refund': 0.4, 'Ticket': 0.6}, 0.7)))
    assert (exc.model_name, exc.route, exc.probability, exc.probabilities, exc.threshold) == (
        'jev-latest',
        'refund',
        0.4,
        {'refund': 0.4, 'Ticket': 0.6},
        0.7,
    )


class Reprioritise(str, Enum):
    """Change how soon the ticket needs a reply."""

    now = 'now'
    later = 'later'


@pytest.mark.anyio
async def test_a_union_member_enum_is_described_by_its_docstring(allow_model_requests: None):
    """An `Enum` is wrapped as a `$ref` to its definition, and its docstring is there rather than on the route."""
    model = InMemoryDecisionModel()
    await Agent(model, output_type=[Triage, Reprioritise]).run('Can this wait until Monday?')

    assert route_question(model).criteria == snapshot(
        {'Triage': 'Triage a support ticket.', 'Reprioritise': 'Change how soon the ticket needs a reply.'}
    )


def look_up(**kwargs: Any) -> str:
    return 'found'  # pragma: no cover


@pytest.mark.anyio
@pytest.mark.parametrize(
    'schema',
    [
        pytest.param({'$ref': '#/$defs/Anything'}, id='a `true` definition'),
        pytest.param({'$ref': '#/$defs/Nothing'}, id='a `false` definition'),
        pytest.param(True, id='a `true` property'),
    ],
)
async def test_a_boolean_schema_is_an_unsupported_argument(allow_model_requests: None, schema: Any):
    """JSON Schema allows `true` and `false` wherever a schema goes (#8621); an argument of either is proposed."""
    tool = Tool.from_schema(
        look_up,
        name='look_up',
        description='Look the order up.',
        json_schema={
            'type': 'object',
            'properties': {'query': schema},
            '$defs': {'Anything': True, 'Nothing': False},
        },
    )
    model = RoutingDecisionModel({'Triage': 0.1, 'look_up': 0.9})

    with pytest.raises(ToolCallProposed, match="proposed calling 'look_up'"):
        await Agent(model, output_type=Triage, tools=[tool]).run('Where is my order?')


@pytest.mark.anyio
@pytest.mark.parametrize('schema', [True, False, {'$ref': '#/$defs/Anything'}, {'$ref': '#/$defs/Nothing'}])
async def test_a_boolean_schema_is_an_unsupported_output_field(allow_model_requests: None, schema: Any):
    output_tool = ToolDefinition(
        name='final_result',
        description='Look the order up.',
        kind='output',
        parameters_json_schema={
            'type': 'object',
            'properties': {'query': schema},
            '$defs': {'Anything': True, 'Nothing': False},
        },
    )

    with pytest.raises(UserError, match="Output field 'query' is not supported by this model"):
        await InMemoryDecisionModel().request(
            [ModelRequest(parts=[UserPromptPart('Where is my order?')])],
            None,
            ModelRequestParameters(output_tools=[output_tool], output_mode='tool', allow_text_output=False),
        )


@pytest.mark.anyio
async def test_the_route_question_carries_the_agent_instructions(allow_model_requests: None):
    model = InMemoryDecisionModel()
    await Agent(model, output_type=Triage, tools=[refund], instructions='Handle support tickets.').run('Charged twice.')
    assert route_question(model).instructions == snapshot(
        {'question': 'Which of these does this call for?', 'background': 'Handle support tickets.'}
    )


class Bank(str, Enum):
    """The banks the customer's accounts can be with."""

    ing = 'ing'
    rabobank = 'rabobank'


class Party(BaseModel):
    """One side of a payment."""

    bank: Bank = Field(description='Which bank?')


class Transfer(BaseModel):
    """Send money from one of the customer's accounts to someone else."""

    source: Party = Field(description='The account the money leaves.')
    target: Party


@pytest.mark.anyio
async def test_a_nested_field_carries_what_it_sits_in(allow_model_requests: None):
    """Flattening a model drops what its fields and models say about themselves, which is what tells leaves apart.

    `source.bank` and `target.bank` both ask "Which bank?"; only the chain above them, root to leaf, says which
    bank each one is. The field's own description, beside its `$ref`, and the nested model's docstring each go in,
    and an `Enum` whose docstring the leaf's own description hides comes last. A unit test pins the exact questions.
    """
    model = InMemoryDecisionModel()
    await Agent(model, output_type=Transfer, instructions='Payments desk.').run('Send 300 from my ING to Rabobank.')
    assert model.requests[0].questions == snapshot(
        {
            'source.bank': ChoiceQuestion(
                criteria={'ing': None, 'rabobank': None},
                instructions={
                    'field': 'source.bank',
                    'context': [
                        'source: The account the money leaves.',
                        'Party: One side of a payment.',
                        "Bank: The banks the customer's accounts can be with.",
                    ],
                    'question': 'Which bank?',
                    'goal': "Send money from one of the customer's accounts to someone else.",
                    'background': 'Payments desk.',
                },
            ),
            'target.bank': ChoiceQuestion(
                criteria={'ing': None, 'rabobank': None},
                instructions={
                    'field': 'target.bank',
                    'context': [
                        'Party: One side of a payment.',
                        "Bank: The banks the customer's accounts can be with.",
                    ],
                    'question': 'Which bank?',
                    'goal': "Send money from one of the customer's accounts to someone else.",
                    'background': 'Payments desk.',
                },
            ),
        }
    )


class Colours(BaseModel):
    """Pick colours."""

    chosen: list[Bank] = Field(description='Does this bank apply?')
    maybe: Bank | None = Field(description='Which bank, if any?')
    plain: Bank


@pytest.mark.anyio
async def test_an_enum_docstring_the_field_hides_is_context_however_the_enum_is_reached(allow_model_requests: None):
    """An optional `Enum` and a `list` of one hide the docstring behind their own description the same way.

    A field without a description of its own is asked the `Enum`'s docstring as its question, so there is nothing
    hidden to add.
    """
    model = InMemoryDecisionModel()
    await Agent(model, output_type=Colours).run('ING.')
    questions = model.requests[0].questions
    bank = "Bank: The banks the customer's accounts can be with."
    assert questions['chosen.ing'].instructions == snapshot(
        {
            'field': 'chosen',
            'context': [bank],
            'question': 'Does this bank apply?',
            'goal': 'Pick colours.',
            'option': 'ing',
        }
    )
    assert questions['maybe'].instructions == snapshot(
        {'field': 'maybe', 'context': [bank], 'question': 'Which bank, if any?', 'goal': 'Pick colours.'}
    )
    assert questions['plain'].instructions == snapshot(
        {'field': 'plain', 'question': "The banks the customer's accounts can be with.", 'goal': 'Pick colours.'}
    )


class Book(BaseModel):
    """Book a new appointment."""

    day: Literal['monday', 'tuesday'] = Field(description='Which day?')


class Cancel(BaseModel):
    """Cancel an existing appointment."""

    day: Literal['monday', 'tuesday'] = Field(description='Which day?')


@pytest.mark.anyio
async def test_every_route_is_asked_up_front_under_its_premise(allow_model_requests: None):
    """Each fillable route's fields ride beside the route question, keyed and premised by the route's label.

    One request instead of a pick and a fill. Only the taken route's answers are read, so the other route's answer
    to the same field name reaches neither the output nor `provider_details`, and `requests` is not reported.
    """
    model = RoutingDecisionModel({'Book': 0.2, 'Cancel': 0.8})
    result = await Agent(model, output_type=[Book, Cancel]).run('I cannot make it on Tuesday.')
    assert len(model.requests) == 1
    assert model.requests[0].questions == snapshot(
        {
            'Book.day': ChoiceQuestion(
                criteria={'monday': None, 'tuesday': None},
                instructions={
                    'field': 'day',
                    'premise': "If the user's request calls for Book: Book a new appointment.",
                    'question': 'Which day?',
                },
            ),
            'Cancel.day': ChoiceQuestion(
                criteria={'monday': None, 'tuesday': None},
                instructions={
                    'field': 'day',
                    'premise': "If the user's request calls for Cancel: Cancel an existing appointment.",
                    'question': 'Which day?',
                },
            ),
            'route': ChoiceQuestion(
                criteria={'Book': 'Book a new appointment.', 'Cancel': 'Cancel an existing appointment.'},
                instructions='Which of these does this call for?',
            ),
        }
    )
    assert result.output == Cancel(day='monday')
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'day': 0.9},
            'probabilities': {'day': {'monday': 1.0, 'tuesday': 0.0}},
            'scores': {},
            'route': {
                'choice': 'Cancel',
                'probabilities': {'Book': 0.2, 'Cancel': 0.8},
                'offered': ['Book', 'Cancel'],
            },
        }
    )


@pytest.mark.anyio
async def test_a_request_too_large_to_ask_every_route_in_picks_then_fills(allow_model_requests: None):
    """Past the size cutoff, a route is picked first and filled in a second request, as a union always used to be.

    A state this long costs more to send twice than the other route's questions do to ask, so it is asked up
    front; past 16k tokens the request would answer more slowly than two small ones, so it is not.
    """
    long = 'I cannot make it on Tuesday. ' + 'Some detail nobody asked about. ' * 3000
    model = RoutingDecisionModel({'Book': 0.2, 'Cancel': 0.8})
    result = await Agent(model, output_type=[Book, Cancel]).run(long)
    assert [list(request.questions) for request in model.requests] == snapshot([['route'], ['day']])
    assert model.requests[1].questions['day'].instructions == snapshot(
        {
            'field': 'day',
            'premise': "If the user's request calls for Cancel: Cancel an existing appointment.",
            'question': 'Which day?',
        }
    )
    assert (result.response.provider_details or {})['requests'] == 2


class Wide(BaseModel):
    """Record the ticket in full."""

    a: bool = Field(description='Is it about billing, invoices, charges, refunds, or anything to do with money?')
    b: bool = Field(description='Is it about shipping, delivery, tracking, parcels, or anything to do with transport?')
    c: bool = Field(description='Is it about the product itself, a defect, a missing part, or how to use it at all?')
    d: bool = Field(description='Is it about the account, logging in, passwords, or two-factor authentication at all?')


@pytest.mark.anyio
async def test_other_routes_questions_costing_more_than_a_second_request_are_not_asked_up_front(
    allow_model_requests: None,
):
    """When the questions thrown away would cost more than sending the short state again, pick first, fill after.

    A single output type beside tools is still asked up front, as it always has been.
    """

    def record(
        a: bool, b: bool, c: bool, d: bool, e: Literal['low', 'medium', 'high', 'critical', 'unknown', 'other']
    ) -> str:
        """Record the ticket in the other system.

        Args:
            a: Is it about billing, invoices, charges, refunds, or anything to do with money at all?
            b: Is it about shipping, delivery, tracking, parcels, or anything to do with transport?
            c: Is it about the product itself, a defect, a missing part, or how to use it at all?
            d: Is it about the account, logging in, passwords, or two-factor authentication at all?
            e: How bad is it, from low to critical, or unknown, or something else entirely?
        """
        return 'recorded'  # pragma: no cover

    model = InMemoryDecisionModel()
    await Agent(model, output_type=Wide, tools=[record]).run('Hi.')
    assert list(model.requests[0].questions) == snapshot(['Wide.a', 'Wide.b', 'Wide.c', 'Wide.d', 'route'])


@pytest.mark.anyio
async def test_a_label_with_a_dot_in_it_keeps_its_questions_apart(allow_model_requests: None):
    """A question's key is for reading its answer back: `a.b` + `c` and `a` + `b.c` would both be `a.b.c`.

    The one asked second gets `_` appended, and each route's answers are read back by the keys it was given.
    """
    # A hand-written schema can nest an object in place rather than by `$ref`: it has a description, and no model.
    nested = {
        'type': 'object',
        'properties': {
            'b': {
                'type': 'object',
                'description': 'The B part.',
                'properties': {'c': {'type': 'boolean', 'description': 'Is it C?'}},
            }
        },
    }
    flat = {'type': 'object', 'properties': {'c': {'type': 'boolean', 'description': 'Is it C?'}}}
    output_tools = [
        ToolDefinition(name='a', description='Do A.', kind='output', parameters_json_schema=nested),
        ToolDefinition(name='a.b', description='Do A.B.', kind='output', parameters_json_schema=flat),
    ]
    model = RoutingDecisionModel({'a': 0.3, 'a.b': 0.7})
    response = await model.request(
        [ModelRequest(parts=[UserPromptPart('C.')])],
        None,
        ModelRequestParameters(output_mode='tool', output_tools=output_tools, allow_text_output=False),
    )
    assert list(model.requests[0].questions) == snapshot(['a.b.c', 'a.b.c_', 'route'])
    assert model.requests[0].questions['a.b.c'].instructions == snapshot(
        {
            'field': 'b.c',
            'premise': "If the user's request calls for a: Do A.",
            'context': ['b: The B part.'],
            'question': 'Is it C?',
        }
    )
    assert response.parts == [ToolCallPart('a.b', {'c': True}, tool_call_id=IsStr())]


@pytest.mark.anyio
async def test_after_a_tool_returns_the_turn_is_told_apart_from_the_text(allow_model_requests: None):
    """The latest prompt stays the text under judgement, and the calls made for it since go under `done`.

    Without the split, a request after a tool call has no `text` at all: the prompt sits in `history` with the
    calls, and a question about the text has none to be about.
    """
    model = RoutingDecisionModel({'Triage': 0.4, 'look_up_order': 0.6})
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi.')]),
        ModelResponse(parts=[TextPart('Hello, how can I help?')]),
    ]
    await Agent(model, output_type=Triage, tools=[look_up_order]).run('Where is my order?', message_history=history)
    assert model.requests[-1].state == snapshot(
        {
            'history': [{'user': 'Hi.'}, {'assistant': 'Hello, how can I help?'}],
            'text': 'Where is my order?',
            'done': [
                {'tool_call': {'name': 'look_up_order', 'args': {}}},
                {'tool_return': {'name': 'look_up_order', 'content': 'Order #1 shipped yesterday.'}},
            ],
        }
    )


@pytest.mark.anyio
async def test_a_message_history_that_ends_mid_turn_is_split_at_its_latest_prompt(allow_model_requests: None):
    """A `message_history` passed in can end partway through a turn, and the split still loses and repeats nothing.

    What came before the latest prompt in the same request (the previous turn's last result, a system prompt) is
    history; what came after it is this turn's. Every entry lands in exactly one of the three.
    """
    model = RoutingDecisionModel({'Triage': 0.9, 'look_up_order': 0.05, 'issue_refund': 0.05})
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Where is order 1?')]),
        ModelResponse(parts=[ToolCallPart('look_up_order', {}, 'c1')]),
        ModelRequest(
            parts=[
                ToolReturnPart('look_up_order', 'Order #1 shipped yesterday.', 'c1'),
                UserPromptPart('Thanks. It arrived broken.'),
                UserPromptPart('Please refund it.'),
                RetryPromptPart('Pick something else.'),
            ]
        ),
        ModelResponse(parts=[ThinkingPart('A refund, then.'), ToolCallPart('issue_refund', {}, 'c2')]),
        ModelRequest(parts=[ToolReturnPart('issue_refund', 'Refunded.', 'c2')]),
    ]
    await Agent(model, output_type=Triage, tools=[look_up_order, issue_refund]).run(message_history=history)
    [request] = model.requests
    assert request.state == snapshot(
        {
            'history': [
                {'user': 'Where is order 1?'},
                {'tool_call': {'name': 'look_up_order', 'args': {}}},
                {'tool_return': {'name': 'look_up_order', 'content': 'Order #1 shipped yesterday.'}},
            ],
            'text': """\
Thanks. It arrived broken.

Please refund it.\
""",
            'done': [
                {'retry': IsStr()},
                {'thinking': 'A refund, then.'},
                {'tool_call': {'name': 'issue_refund', 'args': {}}},
                {'tool_return': {'name': 'issue_refund', 'content': 'Refunded.'}},
            ],
        }
    )
    # The previous turn's result does not withhold `look_up_order`; this turn's withholds `issue_refund`.
    assert list(route_question(model).criteria) == ['Triage', 'look_up_order']


@pytest.mark.anyio
async def test_a_turn_with_no_earlier_history_has_no_history_entry(allow_model_requests: None):
    """The split only names what is there: the first turn's calls leave nothing before the prompt."""
    model = RoutingDecisionModel({'Triage': 0.3, 'look_up_order': 0.6, 'issue_refund': 0.1})
    await Agent(model, output_type=Triage, tools=[look_up_order, issue_refund]).run('Where is my order?')
    assert model.requests[-1].state == snapshot(
        {
            'text': 'Where is my order?',
            'done': [
                {'tool_call': {'name': 'look_up_order', 'args': {}}},
                {'tool_return': {'name': 'look_up_order', 'content': 'Order #1 shipped yesterday.'}},
            ],
        }
    )


@pytest.mark.anyio
async def test_a_tool_result_with_no_prompt_before_it_is_all_history(allow_model_requests: None):
    """Resuming a call made elsewhere can leave nothing but the call and its result: no prompt to split the turn at."""
    model = InMemoryDecisionModel()
    history: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('look_up_order', {}, 'c1')]),
        ModelRequest(parts=[ToolReturnPart('look_up_order', 'Order #1 shipped yesterday.', 'c1')]),
    ]
    await Agent(model, output_type=Triage, tools=[look_up_order]).run(message_history=history)
    assert model.requests[0].state == snapshot(
        {
            'history': [
                {'tool_call': {'name': 'look_up_order', 'args': {}}},
                {'tool_return': {'name': 'look_up_order', 'content': 'Order #1 shipped yesterday.'}},
            ]
        }
    )


class Short(BaseModel):
    """Note the ticket."""

    urgent: bool = Field(description='Is it urgent?')


@pytest.mark.anyio
async def test_a_route_with_nothing_to_ask_could_be_the_one_taken(allow_model_requests: None):
    """If the pick can land on a route with no questions, every question asked up front may be thrown away.

    Two routes' questions that cost less than a second request each still are not asked up front beside a `None`,
    since declining would discard both.
    """
    model = RoutingDecisionModel({'Wide': 0.1, 'Short': 0.1, 'None': 0.8})
    await Agent(model, output_type=[Wide, Short]).run('Hi.')
    assert list(model.requests[0].questions) == snapshot(
        ['Wide.a', 'Wide.b', 'Wide.c', 'Wide.d', 'Short.urgent', 'route']
    )
    model = RoutingDecisionModel({'Wide': 0.1, 'Short': 0.1, 'None': 0.8})
    await Agent(model, output_type=[Wide, Short, None]).run('Hi.')
    assert list(model.requests[0].questions) == snapshot(['route'])
