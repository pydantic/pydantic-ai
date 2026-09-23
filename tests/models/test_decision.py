from __future__ import annotations

from typing import Annotated, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, BoolCriteria
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
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
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsStr, try_import

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire


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
                    legend={level: criterion for level, criterion in enumerate(question.criteria)},
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


class Release(BaseModel):
    """Decide whether a change can ship."""

    ship: Annotated[bool, BoolCriteria(true='It can go out today.', false='It has to wait.')] = Field(
        description='Can this change ship?'
    )


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span(allow_model_requests: None, capfire: CaptureLogfire):
    """The `decide` span belongs to the base class, so any decision model gets one, with the protocol's shapes.

    A history makes the state JSON rather than text, and a described yes/no sends criteria, both on the span as
    on the wire.
    """
    history = [
        ModelRequest(parts=[UserPromptPart('The migration is reviewed.')]),
        ModelResponse(parts=[TextPart('Noted.')]),
    ]
    agent = Agent(InMemoryDecisionModel(), output_type=Release, capabilities=[Instrumentation()])
    result = await agent.run('And the tests pass.', message_history=history)

    assert result.output == Release(ship=True)
    [span] = [
        span
        for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        if span['name'] == 'decide in-memory-decisions'
    ]
    assert {key: value for key, value in span['attributes'].items() if not key.startswith('logfire.')} == snapshot(
        {
            'gen_ai.operation.name': 'decide',
            'gen_ai.provider.name': 'test-decisions',
            'gen_ai.system': 'test-decisions',
            'server.address': 'example.test',
            'gen_ai.request.model': 'in-memory-decisions',
            'pydantic_ai.decision.questions': {
                'ship': {
                    'type': 'noul',
                    'instructions': {
                        'field': 'ship',
                        'question': 'Can this change ship?',
                        'goal': 'Decide whether a change can ship.',
                    },
                    'criteria': {'true': 'It can go out today.', 'false': 'It has to wait.'},
                }
            },
            'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'tool_call': 0.6},
            'pydantic_ai.decision.state': {
                'history': [{'user': 'The migration is reviewed.'}, {'assistant': 'Noted.'}],
                'text': 'And the tests pass.',
            },
            'gen_ai.agent.name': 'agent',
            'gen_ai.agent.call.id': IsStr(),
            'gen_ai.conversation.id': IsStr(),
            'gen_ai.response.model': 'in-memory-decisions',
            'pydantic_ai.decision.usage.input_tokens': 4,
            'pydantic_ai.decision.usage.output_tokens': 2,
            'pydantic_ai.decision.answers': {'ship': {'type': 'noul', 'noul': 0.8}},
        }
    )


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_no_decide_span_without_instrumentation(allow_model_requests: None, capfire: CaptureLogfire):
    """Outside an instrumented request there is no `chat` span to hang a `decide` span from, so none is made."""
    result = await Agent(InMemoryDecisionModel(), output_type=Triage).run('The customer cannot sign in.')

    assert result.output == Triage(urgent=True, action='review')
    assert capfire.exporter.exported_spans_as_dict() == []


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

    question = model.requests[0].questions['tool']
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
