from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, RunContext, Tool, ToolOutput
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
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
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage


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
    return 'escalated'  # pragma: no cover


class Priority(str, Enum):
    """How soon the ticket needs a reply."""

    now = 'now'
    later = 'later'


def assign(team: Literal['billing', 'technical']) -> str:
    """Assign the ticket to a team.

    Args:
        team: Which team should handle it?
    """
    return team  # pragma: no cover


def route_question(model: InMemoryDecisionModel, key: str = 'route') -> ChoiceQuestion:
    question = model.requests[0].questions[key]
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
                    'route': ChoiceQuestion(
                        criteria={'Escalation': 'Hand the ticket to a person.', 'Triage': 'Triage a support ticket.'},
                        instructions='Which of these does this call for?',
                    )
                },
            ),
            DecisionRequest(
                state='Someone else can see my invoices.',
                questions={
                    'security': NoulQuestion(
                        instructions={
                            'field': 'security',
                            'question': 'Is this a security issue?',
                            'chosen': 'Escalation',
                            'goal': 'Hand the ticket to a person.',
                        }
                    )
                },
            ),
        ]
    )
    # `provider_details` names routes by their labels too, not by the tools Pydantic AI made for them.
    assert (result.response.provider_details or {})['route'] == snapshot(
        {
            'choice': 'Escalation',
            'probabilities': {'Escalation': 1.0, 'Triage': 0.0},
            'offered': ['Escalation', 'Triage'],
            'taken': 'Escalation',
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
    assert [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)] == ['final_result_Refund']
    assert model.requests[1].questions == snapshot(
        {
            'urgent': NoulQuestion(
                instructions={
                    'field': 'urgent',
                    'question': 'Is it urgent?',
                    'chosen': 'Refund (output)',
                    'goal': 'Refund the ticket.',
                }
            )
        }
    )


class Routed(BaseModel):
    """Route a support ticket."""

    route: bool = Field(description='Does it name a delivery route?')


@pytest.mark.anyio
async def test_the_route_question_stays_clear_of_a_field_named_route(allow_model_requests: None):
    model = InMemoryDecisionModel()
    await Agent(model, output_type=Routed, tools=[refund]).run('Take the A2.')
    assert list(model.requests[0].questions) == ['route', 'route_']
    assert list(route_question(model, 'route_').criteria) == ['Routed', 'refund']


class RoutingDecisionModel(InMemoryDecisionModel):
    """Answers the route question from a fixed distribution over the routes offered, and the rest as its parent does."""

    def __init__(self, route: dict[str, float]):
        super().__init__()
        self.route = route

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        if isinstance(question := request.questions.get('route'), ChoiceQuestion):
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
async def test_the_lean_weighs_every_function_tool_together(allow_model_requests: None):
    """Probability split between two tools still says a tool is wanted, though neither clears the bar alone."""
    model = RoutingDecisionModel({'Triage': 0.0, 'look_up_order': 0.59, 'issue_refund': 0.41})
    result = await Agent(model, output_type=Triage, tools=[look_up_order, issue_refund]).run('Where is my order?')

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
                'probabilities': {'Triage': 0.0, 'look_up_order': 0.59, 'issue_refund': 0.41},
                'offered': ['Triage', 'look_up_order', 'issue_refund'],
                'taken': 'look_up_order',
            },
        }
    )


@pytest.mark.anyio
async def test_the_function_tools_together_below_the_bar_are_a_lean(allow_model_requests: None):
    """A tool is picked, but the tools together fall short of the bar, so the output is filled and the lean reported."""
    model = RoutingDecisionModel({'Triage': 0.44, 'look_up_order': 0.46, 'issue_refund': 0.1})
    result = await Agent(model, output_type=Triage, tools=[look_up_order, issue_refund]).run('Where is my order?')

    assert result.output == Triage(urgent=True, action='review')
    assert len(model.requests) == 1
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'urgent': 0.6, 'action': 0.9},
            'probabilities': {'action': {'approve': 0.0, 'review': 1.0}},
            'scores': {},
            'route': {
                'choice': 'look_up_order',
                'probabilities': {'Triage': 0.44, 'look_up_order': 0.46, 'issue_refund': 0.1},
                'offered': ['Triage', 'look_up_order', 'issue_refund'],
                'taken': 'Triage',
            },
        }
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
