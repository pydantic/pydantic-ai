"""Tests for `GoogleDecisionsModel`, the Gemini API's decisions endpoint as a decision model.

The endpoint's responses here are shaped from live requests to it: answers keyed by the question IDs the request
used, unrounded float64 probabilities, and usage in `total_input_tokens`.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, WithJsonSchema

from pydantic_ai import (
    Agent,
    BinaryContent,
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
from ..conftest import IsStr, TestEnv, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google_decisions import GoogleDecisionsModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

MODEL_NAME = 'gemini-decision-1'
DECISIONS_URL = 'https://generativelanguage.googleapis.com/v1beta/decisions'

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


async def test_output_type(allow_model_requests: None):
    """Each field goes as a question under its own ID, with its wording as text and its meanings as strings.

    A mocked endpoint, not a recording: the request body is what this pins.
    """
    endpoint = Captured(answering('product defect'))
    agent = Agent(mock_model(endpoint), output_type=Ticket)

    result = await agent.run('The blender arrived with a cracked jar. I want my money back today.')

    assert result.output == Ticket(refund=True, area='product defect', urgency=1)
    assert [request.url for request in endpoint.requests] == [DECISIONS_URL]
    assert endpoint.bodies == snapshot(
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
    assert result.response.provider_name == 'google-decisions'
    assert result.response.model_name == MODEL_NAME
    assert result.usage.input_tokens == 280
    assert result.response.provider_details == snapshot(
        {
            'confidence': {'refund': 0.94, 'area': 0.9, 'urgency': 0.8},
            'probabilities': {
                'area': {'billing': 0.1, 'product defect': 0.9},
                'urgency': {'0': 0.1, '1': 0.8, '2': 0.1},
            },
            'scores': {'urgency': 1.0},
        }
    )


async def test_route_keys_round_trip(allow_model_requests: None):
    """The route question and each route's fields go under plain IDs, and their answers come back under their names.

    The planner keys a route's fields `'<route>.<field>'`, which the endpoint's ID pattern refuses.
    """
    endpoint = Captured(answering('escalate_to_human', 'legal threat'))

    def escalate_to_human(reason: Literal['legal threat', 'safety issue']) -> str:
        """Hand the ticket to a person."""
        return f'escalated: {reason}'

    agent = Agent(mock_model(endpoint), output_type=Ticket, tools=[escalate_to_human])

    result = await agent.run('Your heater caught fire in my kitchen. My lawyer will be in touch.')

    first, *_ = endpoint.bodies
    assert isinstance(first, dict)
    questions = first['questions']
    assert isinstance(questions, dict)
    assert list(questions) == [f'q{index}' for index in range(len(questions))]
    tool_call = result.all_messages()[1]
    assert isinstance(tool_call, ModelResponse)
    assert tool_call.parts == [ToolCallPart('escalate_to_human', {'reason': 'legal threat'}, tool_call_id=IsStr())]
    assert tool_call.provider_details is not None
    assert tool_call.provider_details['route'] == snapshot(
        {
            'choice': 'escalate_to_human',
            'probabilities': {'Ticket': 0.1, 'escalate_to_human': 0.9},
            'offered': ['Ticket', 'escalate_to_human'],
        }
    )
    assert result.output == Ticket(refund=True, area='billing', urgency=1)


async def test_image_input(allow_model_requests: None):
    """Each image goes inline after its `<image N>` label, with its `resolution` from `vendor_metadata`."""
    endpoint = Captured(answering('green'))
    agent = Agent(mock_model(endpoint), output_type=Literal['red', 'green'], instructions='What colour is the product?')
    low = BinaryContent(b'\x89PNG\r\n\x1a\nlow', media_type='image/png', vendor_metadata={'resolution': 'low'})
    default = BinaryContent(b'\xff\xd8\xffdefault', media_type='image/jpeg')

    result = await agent.run(['Photos from the customer:', low, default])

    assert result.output == 'green'
    body = endpoint.bodies[0]
    assert isinstance(body, dict)
    assert body['input'] == snapshot(
        [
            {
                'type': 'text',
                'text': """\
Photos from the customer:

<image 1>

<image 2>\
""",
            },
            {'type': 'text', 'text': '<image 1>:'},
            {'type': 'image', 'data': 'iVBORw0KGgpsb3c=', 'mime_type': 'image/png', 'resolution': 'low'},
            {'type': 'text', 'text': '<image 2>:'},
            {'type': 'image', 'data': '/9j/ZGVmYXVsdA==', 'mime_type': 'image/jpeg'},
        ]
    )


async def test_history_goes_as_state_text(allow_model_requests: None):
    """The endpoint takes no conversation turns, so the history goes in the one text it is sent."""
    endpoint = Captured(answering())
    agent = Agent(mock_model(endpoint), output_type=bool, instructions='Is the customer still angry?')
    history = [ModelRequest(parts=[UserPromptPart('My order is two weeks late!')])]

    await agent.run('Thanks, it just arrived.', message_history=history)

    body = endpoint.bodies[0]
    assert isinstance(body, dict)
    assert body['input'] == snapshot('My order is two weeks late!\n\nThanks, it just arrived.')


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
    ],
)
async def test_extra_body_refused(allow_model_requests: None, extra_body: object, message: str):
    endpoint = Captured(answering())
    agent = Agent(mock_model(endpoint), output_type=bool, instructions='Is this urgent?')

    with pytest.raises(UserError, match=message):
        await agent.run('Server is down', model_settings={'extra_body': extra_body})
    assert endpoint.requests == []


@pytest.mark.parametrize(
    ('status', 'body'),
    [
        pytest.param(
            400,
            {'error': {'message': "Question 'q0' must have non-empty instructions.", 'code': 'invalid_request'}},
            id='invalid-request',
        ),
        pytest.param(
            404,
            {'error': {'message': 'Requested model not found.', 'code': 'not_found'}},
            id='not-found',
        ),
        pytest.param(
            429,
            {'error': {'message': 'Rate limit exceeded. Please retry in 20s.', 'code': 'too_many_requests'}},
            id='rate-limit',
        ),
    ],
)
async def test_http_error(allow_model_requests: None, status: int, body: dict[str, object]):
    """An error response raises a `ModelHTTPError` with its status and body, for a `FallbackModel` to take over."""
    agent = Agent(mock_model(lambda _: httpx2.Response(status, json=body)), output_type=bool, instructions='Urgent?')

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Server is down')

    assert exc_info.value.status_code == status
    assert exc_info.value.model_name == MODEL_NAME
    assert exc_info.value.body == body


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
