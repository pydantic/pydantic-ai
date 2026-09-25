from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, WithJsonSchema

from pydantic_ai import Agent, BoolCriteria, RunContext, Tool, ToolOutput
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
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
    DecisionHandOff,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    UnfillableRoute,
    UnsureRoute,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
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
            'pydantic_ai.decision.thresholds': {'boolean': 0.5},
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
            'pydantic_ai.decision.confidence': {'ship': 0.6},
        }
    )


class NotAnAnswerDecisionModel(InMemoryDecisionModel):
    """Breaks the protocol's contract, answering a question with something that is not an answer."""

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        response.answers['ship'] = cast(DecisionAnswer, None)
        return response


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span_leaves_out_what_is_not_an_answer(allow_model_requests: None, capfire: CaptureLogfire):
    """Instrumentation doesn't change how the run rejects a malformed answer, and the span records the error."""
    agent = Agent(NotAnAnswerDecisionModel(), output_type=Release, capabilities=[Instrumentation()])
    with pytest.raises(UnexpectedModelBehavior, match="Unexpected answer from the model for output field 'ship'"):
        await agent.run('And the tests pass.')

    [span] = [
        span
        for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        if span['name'] == 'decide in-memory-decisions'
    ]
    assert span['attributes']['pydantic_ai.decision.answers'] == snapshot({})
    assert [event['attributes']['exception.type'] for event in span['events']] == snapshot(
        ['pydantic_ai.exceptions.UnexpectedModelBehavior']
    )


class MistypedDecisionModel(InMemoryDecisionModel):
    """Answers with a `type` the protocol doesn't have, which the run reads by the answer's class regardless."""

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        response.answers['ship'] = NoulAnswer(noul=0.8, type=cast(Literal['noul'], 'bogus'))
        return response


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span_without_content_keeps_only_an_unknown_type(
    allow_model_requests: None, capfire: CaptureLogfire
):
    """An answer the run accepts is never failed by its telemetry, even with a `type` the protocol doesn't have."""
    agent = Agent(
        MistypedDecisionModel(),
        output_type=Release,
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=False))],
    )
    result = await agent.run('And the tests pass.')

    assert result.output == Release(ship=True)
    [span] = [
        span
        for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        if span['name'] == 'decide in-memory-decisions'
    ]
    assert span['attributes']['pydantic_ai.decision.answers'] == snapshot({'ship': {'type': 'bogus'}})


class UnsureDecisionModel(InMemoryDecisionModel):
    """Picks the tool on the route question, less surely than the route threshold asks."""

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        response.answers['route'] = ChoiceAnswer(
            choice='escalate_to_team', confidence=0.1, probabilities={'Triage': 0.45, 'escalate_to_team': 0.55}
        )
        return response


def escalate_to_team(team: Literal['billing', 'security']) -> str:
    """Hand the ticket to a specialist team."""
    return f'Escalated to {team}.'  # pragma: no cover - picked below the threshold, so never called


def _span_tree(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """Each span's name, status and exception events, nested under its parent, for the hand-off tests."""
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)

    def node(span: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {'name': span['name']}
        if level := span['attributes'].get('logfire.level_num'):
            result['level'] = level
        if events := span.get('events'):
            # The stack trace repeats the message, and its frames are this test's own.
            result['events'] = [
                {
                    'name': event['name'],
                    **{key: value for key, value in event['attributes'].items() if key != 'exception.stacktrace'},
                }
                for event in events
            ]
        if children := [
            child for child in spans if child['parent'] and child['parent']['span_id'] == span['context']['span_id']
        ]:
            result['children'] = [node(child) for child in children]
        return result

    return [node(span) for span in spans if not span['parent']]


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span_records_an_unsure_route_handed_to_a_fallback(
    allow_model_requests: None, capfire: CaptureLogfire
):
    """A pick below `decision_route_threshold` is recorded on the `decide` span that picked it, even without content.

    The `FallbackModel` hands the step to the model behind the decision model, so the `chat` span ends without an
    error: the `decide` span is the one place the hand-off shows, with the picked route's label on its exception.
    """
    fallback = FallbackModel(UnsureDecisionModel(), TestModel(call_tools=[]))
    agent = Agent(
        fallback,
        output_type=Triage,
        tools=[escalate_to_team],
        model_settings=DecisionModelSettings(decision_route_threshold=0.6),
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=False))],
    )
    result = await agent.run('The customer cannot sign in.')

    assert result.response.model_name == 'test'
    assert _span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent agent',
                'children': [
                    {
                        'name': 'chat test',
                        'children': [
                            {
                                'name': 'decide in-memory-decisions',
                                'level': 17,
                                'events': [
                                    {
                                        'name': 'exception',
                                        'exception.type': 'pydantic_ai.models.decision.UnsureRoute',
                                        'exception.escaped': 'False',
                                        'pydantic_ai.decision.route': 'escalate_to_team',
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )
    [decide] = [
        span
        for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        if span['name'] == 'decide in-memory-decisions'
    ]
    assert {key: value for key, value in decide['attributes'].items() if key.startswith('pydantic_ai.decision.')} == (
        snapshot(
            {
                'pydantic_ai.decision.questions': {
                    'urgent': {'type': 'noul'},
                    'action': {'type': 'choice'},
                    'route': {'type': 'choice'},
                },
                'pydantic_ai.decision.thresholds': {'boolean': 0.5, 'route': 0.6},
                'pydantic_ai.decision.route': 'Triage',
                'pydantic_ai.decision.route_question': 'route',
                'pydantic_ai.decision.route_options': ['Triage', 'escalate_to_team'],
                'pydantic_ai.decision.usage.input_tokens': 4,
                'pydantic_ai.decision.usage.output_tokens': 2,
                'pydantic_ai.decision.answers': {
                    'urgent': {'type': 'noul', 'noul': 0.8},
                    'action': {'type': 'choice', 'confidence': 0.9},
                    'route': {
                        'type': 'choice',
                        'choice': 'escalate_to_team',
                        'confidence': 0.1,
                        'probabilities': {'Triage': 0.45, 'escalate_to_team': 0.55},
                    },
                },
            }
        )
    )


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span_records_an_unsure_route_without_a_fallback(
    allow_model_requests: None, capfire: CaptureLogfire
):
    """Without a model behind it, the unsure pick fails the run, and both the `decide` and `chat` spans record it."""
    agent = Agent(
        UnsureDecisionModel(),
        output_type=Triage,
        tools=[escalate_to_team],
        model_settings=DecisionModelSettings(decision_route_threshold=0.6),
        capabilities=[Instrumentation()],
    )
    with pytest.raises(UnsureRoute):
        await agent.run('The customer cannot sign in.')

    assert _span_tree(capfire) == snapshot(
        [
            {
                'name': 'invoke_agent agent',
                'level': 17,
                'events': [
                    {
                        'name': 'exception',
                        'exception.type': 'pydantic_ai.models.decision.UnsureRoute',
                        'exception.message': "in-memory-decisions picked 'escalate_to_team' with probability 0.55, below `decision_route_threshold` (0.60). Put a model behind it to take the steps it is unsure of: `FallbackModel(decision_model, language_model)` hands `language_model` this step.",
                        'exception.escaped': 'False',
                    }
                ],
                'children': [
                    {
                        'name': 'chat in-memory-decisions',
                        'level': 17,
                        'events': [
                            {
                                'name': 'exception',
                                'exception.type': 'pydantic_ai.models.decision.UnsureRoute',
                                'exception.message': "in-memory-decisions picked 'escalate_to_team' with probability 0.55, below `decision_route_threshold` (0.60). Put a model behind it to take the steps it is unsure of: `FallbackModel(decision_model, language_model)` hands `language_model` this step.",
                                'exception.escaped': 'False',
                            }
                        ],
                        'children': [
                            {
                                'name': 'decide in-memory-decisions',
                                'level': 17,
                                'events': [
                                    {
                                        'name': 'exception',
                                        'exception.type': 'pydantic_ai.models.decision.UnsureRoute',
                                        'exception.message': "in-memory-decisions picked 'escalate_to_team' with probability 0.55, below `decision_route_threshold` (0.60). Put a model behind it to take the steps it is unsure of: `FallbackModel(decision_model, language_model)` hands `language_model` this step.",
                                        'exception.escaped': 'False',
                                        'pydantic_ai.decision.route': 'escalate_to_team',
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    )


class TaggedReview(BaseModel):
    """Review a support ticket."""

    tags: list[Literal['billing', 'security']] = Field(description='Which teams does this concern?')
    action: Literal['approve', 'review'] = Field(description='What should happen next?')
    severity: Annotated[
        Literal[0, 1, 2],
        WithJsonSchema(
            {'type': 'integer', 'anyOf': [{'const': level, 'description': f'Level {level}'} for level in range(3)]}
        ),
    ] = Field(description='How severe is it?')


class TaggedReviewDecisionModel(InMemoryDecisionModel):
    """Sure of one option of the list and unsure of the other, and answers the rubric near its top level."""

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        response = await super().decide(request, model_settings)
        response.answers['tags.billing'] = NoulAnswer(noul=0.95)
        response.answers['tags.security'] = NoulAnswer(noul=0.4)
        response.answers['severity'] = ScoreAnswer(
            score=1.8, confidence=0.7, probabilities={0: 0.05, 1: 0.1, 2: 0.85}, legend={2: 'Level 2'}
        )
        return response


@pytest.mark.anyio
@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_decide_span_per_question_confidence_without_content(allow_model_requests: None, capfire: CaptureLogfire):
    """Confidence is keyed like the questions, so each option of a list gets its own, not the field's least sure.

    Without content, a score keeps its probabilities, which are keyed by level, but a choice does not, since its
    are keyed by option label. With one output type and nothing to choose between, there is no `route`.
    """
    agent = Agent(
        TaggedReviewDecisionModel(),
        output_type=TaggedReview,
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_content=False))],
    )
    result = await agent.run('I was charged twice.')

    assert result.output == TaggedReview(tags=['billing'], action='review', severity=2)
    assert result.response.provider_details is not None
    assert result.response.provider_details['confidence'] == snapshot({'tags': 0.2, 'action': 0.9, 'severity': 0.7})
    [span] = [
        span
        for span in capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
        if span['name'] == 'decide in-memory-decisions'
    ]
    assert {
        key: value for key, value in span['attributes'].items() if key.startswith('pydantic_ai.decision.')
    } == snapshot(
        {
            'pydantic_ai.decision.questions': {
                'tags.billing': {'type': 'noul'},
                'tags.security': {'type': 'noul'},
                'action': {'type': 'choice'},
                'severity': {'type': 'score'},
            },
            'pydantic_ai.decision.thresholds': {'boolean': 0.5},
            'pydantic_ai.decision.usage.input_tokens': 4,
            'pydantic_ai.decision.usage.output_tokens': 2,
            'pydantic_ai.decision.answers': {
                'tags.billing': {'type': 'noul', 'noul': 0.95},
                'tags.security': {'type': 'noul', 'noul': 0.4},
                'action': {'type': 'choice', 'confidence': 0.9},
                'severity': {
                    'type': 'score',
                    'score': 1.8,
                    'confidence': 0.7,
                    'probabilities': {'0': 0.05, '1': 0.1, '2': 0.85},
                },
            },
            'pydantic_ai.decision.confidence': {
                'tags.billing': 0.9,
                'tags.security': 0.2,
                'action': 0.9,
                'severity': 0.7,
            },
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


class Reply(BaseModel):
    """Write the customer a reply."""

    body: str


def write_note(note: str) -> str:
    """Leave a note on the ticket."""
    return note  # pragma: no cover


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
        pytest.param(
            Reply,
            {'Reply': 0.6, 'look_up_order': 0.3, 'issue_refund': 0.1},
            'Reply',
            id='an output the model cannot fill',
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

    with pytest.raises(UnfillableRoute, match="picked 'look_up'"):
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
        {'question': 'Which of these does this call for?', 'instructions': 'Handle support tickets.'}
    )


class Area(str, Enum):
    billing = 'billing'
    bug = 'bug'


class Customer(BaseModel):
    area: Area = Field(description='Which part of the product is this about?')


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    customer: Customer


class PartReply(BaseModel):
    """Reply, and say whether it is urgent."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    body: str


def assign_with_note(team: Literal['billing', 'technical'], note: str) -> str:
    """Assign the ticket to a team, with a note."""
    return team  # pragma: no cover


def set_urgency_with_note(urgent: bool, note: str) -> str:
    """Set how urgent the ticket is, with a note."""
    return note  # pragma: no cover


@dataclass(frozen=True)
class Cell:
    """One output type and tool set, and the route the model picks, if it is asked to pick one."""

    id: str
    output_type: Any
    expected: Any
    tools: list[Any] = field(default_factory=list[Any])
    pick: str | None = None


# Every kind of route, each one the model can fill (`Ticket`, with a nested model and an `Enum`; `assign`;
# `set_urgency`), can fill partly (`PartReply`, `assign_with_note`, `set_urgency_with_note`) or cannot fill at all
# (`Reply`, `write_note`, `refund`), and the ones with nothing to fill (`escalate`, `None`, `look_up_order`). Each
# cell is the questions of each request the step made and what the step did.
CELLS = [
    # A single output type, alone.
    Cell(
        'Ticket',
        Ticket,
        snapshot(([['urgent', 'customer.area']], "final_result({'urgent': True, 'customer': {'area': 'billing'}})")),
    ),
    Cell(
        'PartReply',
        PartReply,
        snapshot(
            (
                [],
                "UserError: Output field 'body' is not supported by this model",
            )
        ),
    ),
    Cell(
        'Reply',
        Reply,
        snapshot(
            (
                [],
                "UserError: Output field 'body' is not supported by this model",
            )
        ),
    ),
    Cell('assign', assign, snapshot(([['team']], "final_result({'team': 'billing'})"))),
    Cell(
        'write_note',
        write_note,
        snapshot(
            (
                [],
                "UserError: Output field 'note' is not supported by this model",
            )
        ),
    ),
    Cell(
        'escalate',
        escalate,
        snapshot(
            (
                [],
                'UserError: An `output_type` with no fields is not supported by this model; there is nothing to ask the model',
            )
        ),
    ),
    # A single output type beside tools.
    Cell(
        'Ticket + set_urgency: Ticket',
        Ticket,
        snapshot(
            ([['urgent', 'customer.area', 'route']], "final_result({'urgent': True, 'customer': {'area': 'billing'}})")
        ),
        [set_urgency],
        'Ticket',
    ),
    Cell(
        'Ticket + set_urgency: set_urgency',
        Ticket,
        snapshot(([['urgent', 'customer.area', 'route'], ['urgent']], "set_urgency({'urgent': True})")),
        [set_urgency],
        'set_urgency',
    ),
    Cell(
        'Ticket + refund: refund',
        Ticket,
        snapshot(([['urgent', 'customer.area', 'route']], "hands off 'refund'")),
        [refund],
        'refund',
    ),
    Cell(
        'assign + set_urgency: assign',
        assign,
        snapshot(([['team', 'route']], "final_result({'team': 'billing'})")),
        [set_urgency],
        'assign',
    ),
    Cell(
        'PartReply + set_urgency: PartReply',
        PartReply,
        snapshot(([['route']], "hands off 'PartReply'")),
        [set_urgency],
        'PartReply',
    ),
    Cell(
        'PartReply + set_urgency: set_urgency',
        PartReply,
        snapshot(([['route'], ['urgent']], "set_urgency({'urgent': True})")),
        [set_urgency],
        'set_urgency',
    ),
    Cell(
        'Reply + look_up_order: Reply',
        Reply,
        snapshot(([['route']], "hands off 'Reply'")),
        [look_up_order],
        'Reply',
    ),
    Cell(
        'Reply + look_up_order: look_up_order',
        Reply,
        snapshot(([['route']], 'look_up_order({})')),
        [look_up_order],
        'look_up_order',
    ),
    Cell(
        'write_note + set_urgency: write_note',
        write_note,
        snapshot(([['route']], "hands off 'write_note'")),
        [set_urgency],
        'write_note',
    ),
    Cell(
        'write_note + set_urgency: set_urgency',
        write_note,
        snapshot(([['route'], ['urgent']], "set_urgency({'urgent': True})")),
        [set_urgency],
        'set_urgency',
    ),
    Cell(
        'Reply + refund',
        Reply,
        snapshot(
            (
                [],
                "UserError: Output field 'body' is not supported by this model",
            )
        ),
        [refund],
    ),
    Cell(
        'PartReply + set_urgency_with_note',
        PartReply,
        snapshot(
            (
                [],
                "UserError: Output field 'body' is not supported by this model",
            )
        ),
        [set_urgency_with_note],
    ),
    # A single output type beside `None` or an output function with nothing to fill.
    Cell(
        'Ticket | None: Ticket',
        [Ticket, None],
        snapshot(
            (
                [['urgent', 'customer.area', 'route']],
                "final_result_Ticket({'urgent': True, 'customer': {'area': 'billing'}})",
            )
        ),
        pick='Ticket',
    ),
    Cell(
        'Ticket | None: None',
        [Ticket, None],
        snapshot(([['urgent', 'customer.area', 'route']], "final_result_None({'response': None})")),
        pick='None',
    ),
    Cell('Reply | None: Reply', [Reply, None], snapshot(([['route']], "hands off 'Reply'")), pick='Reply'),
    Cell(
        'Reply | None: None',
        [Reply, None],
        snapshot(([['route']], "final_result_None({'response': None})")),
        pick='None',
    ),
    Cell(
        'PartReply | escalate: PartReply',
        [PartReply, escalate],
        snapshot(([['route']], "hands off 'PartReply'")),
        pick='PartReply',
    ),
    Cell(
        'PartReply | escalate: escalate',
        [PartReply, escalate],
        snapshot(([['route']], 'final_result_escalate({})')),
        pick='escalate',
    ),
    # A union of output types.
    Cell(
        'Ticket | Escalation: Escalation',
        [Ticket, Escalation],
        snapshot(([['route'], ['security']], "final_result_Escalation({'security': True})")),
        pick='Escalation',
    ),
    Cell(
        'Ticket | Reply: Ticket',
        [Ticket, Reply],
        snapshot(
            (
                [['route'], ['urgent', 'customer.area']],
                "final_result_Ticket({'urgent': True, 'customer': {'area': 'billing'}})",
            )
        ),
        pick='Ticket',
    ),
    Cell(
        'Ticket | Reply: Reply',
        [Ticket, Reply],
        snapshot(([['route']], "hands off 'Reply'")),
        pick='Reply',
    ),
    Cell(
        'Ticket | assign_with_note: assign_with_note',
        [Ticket, assign_with_note],
        snapshot(([['route']], "hands off 'assign_with_note'")),
        pick='assign_with_note',
    ),
    Cell(
        'Ticket | assign: assign',
        [Ticket, assign],
        snapshot(([['route'], ['team']], "final_result_assign({'team': 'billing'})")),
        pick='assign',
    ),
    Cell(
        'PartReply | Reply',
        [PartReply, Reply],
        snapshot(
            (
                [],
                'UserError: None of the output types can be filled by this model, so every answer would be handed off and the request asking which would be wasted',
            )
        ),
    ),
    Cell(
        'PartReply | Reply | None: Reply',
        [PartReply, Reply, None],
        snapshot(([['route']], "hands off 'Reply'")),
        pick='Reply',
    ),
    Cell(
        'PartReply | Reply | None: None',
        [PartReply, Reply, None],
        snapshot(([['route']], "final_result_None({'response': None})")),
        pick='None',
    ),
    Cell(
        'PartReply | Reply + set_urgency: PartReply',
        [PartReply, Reply],
        snapshot(([['route']], "hands off 'PartReply'")),
        [set_urgency],
        'PartReply',
    ),
    Cell(
        'PartReply | Reply + set_urgency: set_urgency',
        [PartReply, Reply],
        snapshot(([['route'], ['urgent']], "set_urgency({'urgent': True})")),
        [set_urgency],
        'set_urgency',
    ),
    Cell(
        'PartReply | Reply + refund',
        [PartReply, Reply],
        snapshot(
            (
                [],
                'UserError: None of the output types can be filled by this model, so every answer would be handed off and the request asking which would be wasted',
            )
        ),
        [refund],
    ),
    Cell(
        'Ticket | Escalation + refund: refund',
        [Ticket, Escalation],
        snapshot(([['route']], "hands off 'refund'")),
        [refund],
        'refund',
    ),
    # Only routes with nothing to fill.
    Cell(
        'escalate | None: escalate',
        [escalate, None],
        snapshot(([['route']], 'final_result_escalate({})')),
        pick='escalate',
    ),
    Cell(
        'escalate + set_urgency: set_urgency',
        escalate,
        snapshot(([['route'], ['urgent']], "set_urgency({'urgent': True})")),
        [set_urgency],
        'set_urgency',
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize('cell', [pytest.param(cell, id=cell.id) for cell in CELLS])
async def test_the_route_matrix(allow_model_requests: None, cell: Cell):
    """What each combination of routes asks, and what each pick does.

    A route the model cannot fill is still offered, and picking it hands the step off; the agent is refused
    before any request only when no route on offer could be taken without a hand-off. A unit test, because the
    claim is about what is asked across dozens of combinations, which the TypeSafe tests cover by example.
    """
    model = RoutingDecisionModel(defaultdict(float, {cell.pick: 0.9} if cell.pick else {}))
    agent = Agent(model, output_type=cell.output_type, tools=cell.tools, instructions='Handle the ticket.')
    try:
        async with agent.iter('I was charged twice for my order.') as run:
            prompt_node = run.next_node
            assert Agent.is_user_prompt_node(prompt_node)
            request_node = await run.next(prompt_node)
            assert Agent.is_model_request_node(request_node)
            tools_node = await run.next(request_node)
        assert Agent.is_call_tools_node(tools_node)
        [call] = tools_node.model_response.parts
        assert isinstance(call, ToolCallPart)
        outcome = f'{call.tool_name}({call.args})'
    except UnfillableRoute as e:
        outcome = f'hands off {e.route!r}'
    except UserError as e:
        outcome = f'UserError: {str(e).split(". ")[0]}'
    assert ([list(request.questions) for request in model.requests], outcome) == cell.expected


@pytest.mark.anyio
async def test_an_output_type_the_model_cannot_fill_is_left_to_the_model_behind_it(allow_model_requests: None):
    """Beside a tool, an output type the model cannot fill is a route, where alone it would be refused.

    The model picks the tool, and once it has returned the output type is the one route left. It is taken without
    a route question, like the last tool left, and cannot be filled, so it is handed off without a request.
    """
    decision_model = RoutingDecisionModel({'Reply': 0.1, 'look_up_order': 0.9})
    with pytest.raises(UnfillableRoute) as exc_info:
        await Agent(decision_model, output_type=Reply, tools=[look_up_order]).run('Where is my order?')
    assert (exc_info.value.route, exc_info.value.probability) == ('Reply', 1.0)
    assert [list(request.questions) for request in decision_model.requests] == [['route']]

    decision_model = RoutingDecisionModel({'Reply': 0.1, 'look_up_order': 0.9})
    agent = Agent(FallbackModel(decision_model, TestModel(call_tools=[])), output_type=Reply, tools=[look_up_order])
    result = await agent.run('Where is my order?')

    assert result.output == Reply(body='a')
    assert [
        (response.model_name, [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)])
        for response in result.all_messages()
        if isinstance(response, ModelResponse)
    ] == snapshot([('in-memory-decisions', ['look_up_order']), ('test', ['final_result'])])


@pytest.mark.anyio
async def test_unfillable_output_types_left_after_the_tools_return_are_still_asked_about(allow_model_requests: None):
    """Once the tool has returned, every route left hands off, but with several of them the model is still asked
    which: the answer names the route the hand-off reports, and there is no one route to name without it."""
    decision_model = RoutingDecisionModel({'PartReply': 0.3, 'Reply': 0.6, 'look_up_order': 0.9})
    with pytest.raises(UnfillableRoute) as exc_info:
        await Agent(decision_model, output_type=[PartReply, Reply], tools=[look_up_order]).run('Where is my order?')

    assert (exc_info.value.route, exc_info.value.probability) == ('Reply', 0.6)
    assert [list(request.questions) for request in decision_model.requests] == [['route'], ['route']]
    assert list(route_question(decision_model).criteria) == ['PartReply', 'Reply', 'look_up_order']


class UnavailableDecisionModel(InMemoryDecisionModel):
    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        raise ModelAPIError(self.model_name, 'The backend is down.')


@pytest.mark.anyio
async def test_a_fallback_on_decision_hand_offs_takes_only_those(allow_model_requests: None):
    """`fallback_on=DecisionHandOff` hands the language model the steps the decision model hands off, and nothing
    else: an error from the decision model's backend fails the run rather than quietly costing a language model call."""

    async def run(decision_model: InMemoryDecisionModel, settings: DecisionModelSettings | None = None) -> str | None:
        model = FallbackModel(decision_model, TestModel(call_tools=[]), fallback_on=DecisionHandOff)
        agent = Agent(model, output_type=Triage, tools=[look_up_order, refund])
        result = await agent.run('Where is my order?', model_settings=settings)
        return result.response.model_name

    # The model picks a tool it cannot fill: `UnfillableRoute`.
    assert await run(RoutingDecisionModel({'Triage': 0.1, 'look_up_order': 0.1, 'refund': 0.8})) == 'test'
    # The model is unsure of its pick: `UnsureRoute`.
    unsure = RoutingDecisionModel({'Triage': 0.3, 'look_up_order': 0.5, 'refund': 0.2})
    assert await run(unsure, DecisionModelSettings(decision_route_threshold=0.7)) == 'test'
    # The backend fails: not a hand-off, so it is not handed on.
    with pytest.raises(ModelAPIError, match='The backend is down') as exc_info:
        await run(UnavailableDecisionModel())
    assert not isinstance(exc_info.value, DecisionHandOff)
