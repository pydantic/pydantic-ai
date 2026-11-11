"""Tests for `GoogleDecisionsModel`, the Gemini API's decisions endpoint as a decision model.

The recorded tests run against `gemini-decision-1`. The mocked ones cover what no real response produces, with bodies
shaped like the live ones: answers keyed by the question IDs the request used, unrounded float64 probabilities, and
usage in `total_input_tokens`.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, WithJsonSchema

from pydantic_ai import (
    Agent,
    BinaryContent,
    BinaryImage,
    BoolCriteria,
    ModelAPIError,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import infer_model, infer_model_profile
from pydantic_ai.models.decision import (
    ChoiceQuestion,
    DecisionModelSettings,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)
from pydantic_ai.providers import Provider

from .._inline_snapshot import snapshot
from ..conftest import IsStr, RequestCapture, TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google_decisions import GoogleDecisionsModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

MODEL_NAME = 'gemini-decision-1'

Handler = Callable[[httpx2.Request], httpx2.Response]
Answer = dict[str, JsonValue]

_json: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class Captured:
    """A decisions endpoint that records each request and answers it with `respond`."""

    def __init__(self, respond: Handler):
        self.requests: list[httpx2.Request] = []
        self._respond = respond

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self._respond(request)

    @property
    def bodies(self) -> list[JsonValue]:
        return [_json.validate_json(request.content) for request in self.requests]


def mock_model(handler: Handler) -> GoogleDecisionsModel:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return GoogleDecisionsModel(MODEL_NAME, provider=GoogleProvider(api_key='test-key', http_client=http_client))


def decisions_response(answers: Mapping[str, Answer], input_tokens: int = 280) -> httpx2.Response:
    usage = {
        'total_tokens': input_tokens,
        'total_input_tokens': input_tokens,
        'input_tokens_by_modality': [{'modality': 'text', 'tokens': input_tokens}],
        'total_output_tokens': 0,
    }
    return httpx2.Response(200, json={'answers': answers, 'usage': usage})


def _answer(question: JsonValue, picks: tuple[str, ...]) -> Answer:
    """An answer to one question as the endpoint words it: the first of `picks` it offers, else its first option."""
    assert isinstance(question, dict)
    criteria = question.get('criteria')
    if question['type'] == 'noul':
        return {'type': 'noul', 'noul': 0.97, 'confidence': 0.97}
    if question['type'] == 'choice':
        assert isinstance(criteria, dict)
        labels = list(criteria)
        choice = next((label for label in picks if label in labels), labels[0])
        rest = 0.1 / (len(labels) - 1)
        probabilities: dict[str, JsonValue] = {label: 0.9 if label == choice else rest for label in labels}
        return {'type': 'choice', 'choice': choice, 'confidence': 0.9, 'probabilities': probabilities}
    assert isinstance(criteria, list)
    levels = range(len(criteria))
    level_probabilities = [0.8 if level == 1 else 0.2 / (len(levels) - 1) for level in levels]
    return {
        'type': 'score',
        'score': sum(level * p for level, p in zip(levels, level_probabilities)),
        'confidence': 0.8,
        'probabilities': {str(level): p for level, p in zip(levels, level_probabilities)},
        'legend': {str(level): description for level, description in zip(levels, criteria)},
    }


def answering(*picks: str) -> Handler:
    """An endpoint that answers every question it is asked, under the ID it was asked under."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = _json.validate_json(request.content)
        assert isinstance(body, dict)
        questions = body['questions']
        assert isinstance(questions, dict)
        return decisions_response({key: _answer(question, picks) for key, question in questions.items()})

    return respond


Urgency = Annotated[
    Literal[0, 1, 2],
    WithJsonSchema(
        {
            'type': 'integer',
            'anyOf': [
                {'const': 0, 'description': 'Can wait a week'},
                {'const': 1, 'description': 'Should be handled today'},
                {'const': 2, 'description': 'Needs a reply within the hour'},
            ],
        }
    ),
]


class Ticket(BaseModel):
    """Triage a support ticket."""

    refund: Annotated[bool, BoolCriteria(true='They ask for their money back.', false='They do not.')] = Field(
        description='Do they want a refund?'
    )
    area: Literal['billing', 'product defect'] = Field(description='Which team owns it?')
    urgency: Urgency = Field(description='How urgent is it?')


@pytest.fixture
def capture_model(request_capture: RequestCapture, gemini_api_key: str) -> GoogleDecisionsModel:
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=30))
    return GoogleDecisionsModel(MODEL_NAME, provider=provider)


@pytest.mark.vcr
async def test_output_type(
    allow_model_requests: None, capture_model: GoogleDecisionsModel, request_capture: RequestCapture
):
    """Each field goes as a question under its own ID, with its wording as text and its meanings as strings.

    The answers come back under those IDs and fill the output: a yes/no as a `bool`, a pick-one as its label, a
    rubric as its level.
    """
    agent = Agent(capture_model, output_type=Ticket)

    result = await agent.run('The blender arrived with a cracked jar. I want my money back today.')

    assert request_capture.paths == ['/v1beta/decisions']
    assert request_capture.bodies('/decisions') == snapshot(
        [
            {
                'model': 'gemini-decision-1',
                'input': 'The blender arrived with a cracked jar. I want my money back today.',
                'questions': {
                    'q0': {
                        'type': 'noul',
                        'instructions': '{"field": "refund", "question": "Do they want a refund?", "goal": "Triage a support ticket."}',
                        'criteria': {
                            'true_description': 'They ask for their money back.',
                            'false_description': 'They do not.',
                        },
                    },
                    'q1': {
                        'type': 'choice',
                        'instructions': '{"field": "area", "question": "Which team owns it?", "goal": "Triage a support ticket."}',
                        'criteria': {'billing': None, 'product defect': None},
                    },
                    'q2': {
                        'type': 'score',
                        'instructions': '{"field": "urgency", "question": "How urgent is it?", "goal": "Triage a support ticket."}',
                        'criteria': ['Can wait a week', 'Should be handled today', 'Needs a reply within the hour'],
                    },
                },
            }
        ]
    )
    assert result.output == snapshot(Ticket(refund=True, area='product defect', urgency=1))
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'refund': 0.966166, 'area': 0.8778519632794022, 'urgency': 0.7177949412677748},
            'probabilities': {
                'area': {'billing': 0.12214803672059774, 'product defect': 0.8778519632794022},
                'urgency': {'0': 0.0825415728655953, '1': 0.7177949412677748, '2': 0.19966348586662996},
            },
            'scores': {'urgency': 1.1171219130010348},
        }
    )
    assert (result.response.provider_name, result.response.model_name) == ('google-decisions', MODEL_NAME)
    assert result.usage.input_tokens == snapshot(303)


@pytest.mark.vcr
async def test_route_keys_round_trip(
    allow_model_requests: None, capture_model: GoogleDecisionsModel, request_capture: RequestCapture
):
    """The route question and each route's fields go under plain IDs, and their answers come back under their names.

    The planner keys a route's fields `'<route>.<field>'`, which the endpoint's ID pattern refuses: the tool's
    argument here is the answer to `escalate_to_human.reason`, asked beside the route question.
    """

    def escalate_to_human(reason: Literal['legal threat', 'safety issue']) -> str:
        """Hand the ticket to a person."""
        return f'escalated: {reason}'

    agent = Agent(capture_model, output_type=Ticket, tools=[escalate_to_human])

    result = await agent.run('Your heater caught fire in my kitchen. My lawyer will be in touch.')

    questions = request_capture.body('/decisions')['questions']
    assert isinstance(questions, dict)
    assert list(questions) == [f'q{index}' for index in range(len(questions))]
    tool_call = result.all_messages()[1]
    assert isinstance(tool_call, ModelResponse)
    assert tool_call.parts == snapshot(
        [
            ToolCallPart(
                tool_name='escalate_to_human',
                args={'reason': 'legal threat'},
                tool_call_id=IsStr(),
            )
        ]
    )
    assert tool_call.provider_details is not None
    assert tool_call.provider_details['route'] == snapshot(
        {
            'choice': 'escalate_to_human',
            'probabilities': {'escalate_to_human': 0.9813419372969718, 'Ticket': 0.018658062703028166},
            'offered': ['Ticket', 'escalate_to_human'],
        }
    )
    assert result.output == snapshot(Ticket(refund=False, area='product defect', urgency=2))


@pytest.mark.vcr
async def test_image_input(
    allow_model_requests: None,
    capture_model: GoogleDecisionsModel,
    request_capture: RequestCapture,
    image_content: BinaryImage,
):
    """An image goes inline after its `<image N>` label, with its `resolution` from `vendor_metadata`."""
    agent = Agent(
        capture_model, output_type=Literal['kiwi', 'apple', 'banana'], instructions='Which fruit is in the photo?'
    )
    image = BinaryContent(image_content.data, media_type='image/jpeg', vendor_metadata={'resolution': 'low'})

    result = await agent.run(['Photo from the customer:', image])

    assert result.output == 'kiwi'
    assert request_capture.body('/decisions')['input'] == snapshot(
        [
            {
                'type': 'text',
                'text': """\
Photo from the customer:

<image 1>\
""",
            },
            {'type': 'text', 'text': '<image 1>:'},
            {
                'type': 'image',
                'data': IsStr(),
                'mime_type': 'image/jpeg',
                'resolution': 'low',
            },
        ]
    )
    assert result.usage.input_tokens == snapshot(378)


@pytest.mark.vcr
async def test_history_goes_as_state_text(
    allow_model_requests: None, capture_model: GoogleDecisionsModel, request_capture: RequestCapture
):
    """The endpoint takes no conversation turns, so the history goes in the one text it is sent."""
    agent = Agent(capture_model, output_type=bool, instructions='Is the customer still angry?')
    history = [ModelRequest(parts=[UserPromptPart('My order is two weeks late!')])]

    result = await agent.run('Thanks, it just arrived.', message_history=history)

    assert request_capture.body('/decisions')['input'] == snapshot(
        'My order is two weeks late!\n\nThanks, it just arrived.'
    )
    assert result.output == snapshot(False)


async def test_question_mapping(allow_model_requests: None):
    """Wording that is not text goes as JSON, and an undescribed side of a yes/no is left out.

    Called through `decide`, since the planner always words its questions and describes every rubric level: a
    question without instructions, or with a level of `None`, is the API's to refuse.
    """
    endpoint = Captured(answering())
    request = DecisionRequest(
        state={'ticket': 'Refund please'},
        questions={
            'refund': NoulQuestion(criteria=NoulCriteria(false='They want a replacement.')),
            'area': ChoiceQuestion(
                instructions={'question': 'Which team?'}, criteria={'billing': {'owns': 'money'}, 'shipping': None}
            ),
            'urgency': ScoreQuestion(instructions='How urgent?', criteria=[None, ['today', 'now']]),
        },
    )

    await mock_model(endpoint).decide(request, DecisionModelSettings())

    assert endpoint.bodies == snapshot(
        [
            {
                'model': 'gemini-decision-1',
                'input': '{"ticket": "Refund please"}',
                'questions': {
                    'q0': {'type': 'noul', 'criteria': {'false_description': 'They want a replacement.'}},
                    'q1': {
                        'type': 'choice',
                        'instructions': '{"question": "Which team?"}',
                        'criteria': {'billing': '{"owns": "money"}', 'shipping': None},
                    },
                    'q2': {'type': 'score', 'instructions': 'How urgent?', 'criteria': [None, '["today", "now"]']},
                },
            }
        ]
    )


async def test_settings_are_forwarded(allow_model_requests: None):
    """`extra_headers` go on the request and `extra_body` into its body."""
    endpoint = Captured(answering())
    agent = Agent(mock_model(endpoint), output_type=bool, instructions='Is this urgent?')

    await agent.run(
        'Server is down',
        model_settings={'extra_headers': {'x-team': 'support'}, 'extra_body': {'labels': {'team': 'support'}}},
    )

    assert endpoint.requests[0].headers['x-team'] == 'support'
    body = endpoint.bodies[0]
    assert isinstance(body, dict)
    assert body['labels'] == {'team': 'support'}


@pytest.mark.parametrize(
    ('extra_body', 'message'),
    [
        ('labels', '`extra_body` must be a dict'),
        ({'questions': {}}, '`extra_body` cannot set `questions`'),
        ({'Ques_tions': {}}, '`extra_body` cannot set `questions`'),
    ],
)
async def test_extra_body_refused(allow_model_requests: None, extra_body: object, message: str):
    endpoint = Captured(answering())
    agent = Agent(mock_model(endpoint), output_type=bool, instructions='Is this urgent?')

    with pytest.raises(UserError, match=message):
        await agent.run('Server is down', model_settings={'extra_body': extra_body})
    assert endpoint.requests == []


async def test_rate_limit_error(allow_model_requests: None):
    """A 429 raises a `ModelHTTPError` with its body, for a `FallbackModel` to take over.

    Mocked with the body a live 429 had: the free tier's limit can't be hit on demand.
    """
    body = {
        'error': {
            'message': 'Rate limit exceeded for model gemini-decision-1 (limit: 15 requests per minute on Free Tier). '
            'Please retry in 20s or upgrade your tier at https://ai.dev/rate-limit.',
            'code': 'too_many_requests',
        }
    }
    agent = Agent(mock_model(lambda _: httpx2.Response(429, json=body)), output_type=bool, instructions='Urgent?')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Server is down')

    assert (exc_info.value.status_code, exc_info.value.model_name, exc_info.value.body) == (429, MODEL_NAME, body)


class Handling(StrEnum):
    refund = 'refund'
    long = 'refund the customer ' + 'x' * 120


@pytest.mark.vcr
async def test_option_label_too_long(allow_model_requests: None, capture_model: GoogleDecisionsModel):
    """The API takes option labels of up to 128 characters, and refuses a longer one with a 400."""
    agent = Agent(capture_model, output_type=Handling, instructions='How should we handle this?')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('I want my money back.')

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'error': {
                'message': '* CreateDecisionRequest.questions[q0].choice.criteria[refund the customer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]: must have an id of at most 128 characters\n',
                'code': 'invalid_request',
            }
        }
    )


@pytest.mark.vcr
async def test_unknown_model(allow_model_requests: None, gemini_api_key: str):
    model = GoogleDecisionsModel('gemini-decision-0', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(model, output_type=bool, instructions='Urgent?')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Server is down')

    assert exc_info.value.status_code == 404
    assert exc_info.value.body == snapshot(
        {
            'error': {
                'message': 'Requested model not found. See https://ai.google.dev/gemini-api/docs/models for a list of available models.',
                'code': 'not_found',
            }
        }
    )


async def test_connection_error(allow_model_requests: None):
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError('connection refused', request=request)

    agent = Agent(mock_model(refuse), output_type=bool, instructions='Urgent?')

    with pytest.raises(ModelAPIError, match='connection refused'):
        await agent.run('Server is down')


def _not_json(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, text='not json')


def _unknown_id(request: httpx2.Request) -> httpx2.Response:
    return decisions_response({'other': {'type': 'noul', 'noul': 0.5, 'confidence': 0.5}})


def _not_a_probability(request: httpx2.Request) -> httpx2.Response:
    return decisions_response({'q0': {'type': 'noul', 'noul': 1.5, 'confidence': 1.5}})


@pytest.mark.parametrize('respond', [_not_json, _unknown_id, _not_a_probability])
async def test_invalid_response(allow_model_requests: None, respond: Handler):
    """A body that can't be read as answers to the questions asked raises `UnexpectedModelBehavior`.

    No real response is shaped like these, so they are mocked.
    """
    agent = Agent(mock_model(respond), output_type=bool, instructions='Urgent?')

    with pytest.raises(UnexpectedModelBehavior, match='Invalid response from the Gemini API'):
        await agent.run('Server is down')


def _short_of_one(request: httpx2.Request) -> httpx2.Response:
    return decisions_response(
        {
            'q0': {
                'type': 'choice',
                'choice': 'billing',
                'confidence': 0.9,
                'probabilities': {'billing': 0.9, 'bug': 0.095},
            }
        }
    )


async def test_probabilities_must_sum_to_one(allow_model_requests: None):
    """A distribution short of one is refused, though a backend that rounds each probability could send it.

    The endpoint sends unrounded probabilities, which sum to one within float error. Mocked: no real response does.
    """
    agent = Agent(mock_model(_short_of_one), output_type=Literal['billing', 'bug'], instructions='Which team?')

    with pytest.raises(UnexpectedModelBehavior, match="answer 'response' does not match its question"):
        await agent.run('I was charged twice')


async def test_non_image_evidence_refused():
    """Only `decide` can be handed a non-image: the planner maps only images into `request.images`."""
    model = mock_model(answering())
    request = DecisionRequest(
        state='See the attachment',
        questions={'urgent': NoulQuestion(instructions='Urgent?')},
        images=(BinaryContent(b'%PDF-1.4', media_type='application/pdf'),),
    )

    with pytest.raises(UserError, match='image evidence only'):
        await model.decide(request, DecisionModelSettings())


async def test_question_limit():
    """More questions than the endpoint accepts raise `ModelAPIError` before anything is sent."""
    endpoint = Captured(answering())
    request = DecisionRequest(
        state='Server is down',
        questions={f'q{index}': NoulQuestion(instructions='Urgent?') for index in range(1001)},
    )

    with pytest.raises(ModelAPIError, match='accepts at most 1000 questions; got 1001'):
        await mock_model(endpoint).decide(request, DecisionModelSettings())
    assert endpoint.requests == []


def test_infer_model(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')

    model = infer_model(f'google-decisions:{MODEL_NAME}')

    assert isinstance(model, GoogleDecisionsModel)
    assert (model.model_id, model.system, model.base_url) == (
        f'google-decisions:{MODEL_NAME}',
        'google-decisions',
        'https://generativelanguage.googleapis.com/',
    )
    profile = infer_model_profile(f'google-decisions:{MODEL_NAME}')
    assert (profile.get('supports_text_output'), profile.get('default_structured_output_mode')) == (False, 'tool')
    assert model.profile.get('supports_image_input') is True


class OtherProvider(Provider[httpx2.AsyncClient]):
    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        raise NotImplementedError

    @property
    def client(self) -> httpx2.AsyncClient:
        raise NotImplementedError


def test_infer_model_refuses_another_provider():
    with pytest.raises(UserError, match='require a `GoogleProvider`'):
        infer_model(f'google-decisions:{MODEL_NAME}', provider_factory=lambda _: OtherProvider())
