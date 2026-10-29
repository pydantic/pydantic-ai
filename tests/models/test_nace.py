from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelTimeoutError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.decision import (
    ChoiceAnswer as DecisionChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer as DecisionNoulAnswer,
    NoulCriteria as DecisionNoulCriteria,
    NoulQuestion,
    ScoreAnswer as DecisionScoreAnswer,
    ScoreQuestion,
    _decide_response_attributes,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.usage import RequestUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from nace_sdk import (
        Choice,
        ChoiceAnswer,
        NaceAPIConnectionError,
        NaceAPIResponseValidationError,
        NaceAPITimeoutError,
        NaceError,
        NaceInsufficientCreditError,
        NaceOverloadedError,
        NaceRateLimitError,
        NaceUnprocessableEntityError,
        Noul,
        NoulAnswer,
        NoulCriteria,
        RetryPolicy,
        Score,
        ScoreAnswer,
        image_input,
    )

    from pydantic_ai.models.nace import NaceModel
    from pydantic_ai.providers.nace import NaceProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='nace-sdk not installed')

_PNG = bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
    '0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082'
)
_JPEG = bytes.fromhex(
    'ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707'
    '070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c'
    '1c2837292c30313434341f27393d38323c2e333432ffc0000b080001000101011100'
    'ffc4001f0000010501010101010100000000000000000102030405060708090a0bff'
    'da00080001000100003f00fbffd9'
)


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int = 0, image_tokens: int | None = None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.image_tokens = image_tokens


class _Response:
    def __init__(
        self,
        answers: dict[str, Any],
        *,
        model: str,
        usage: _Usage | None = None,
        request_id: str = 'req_test',
    ) -> None:
        self.answers = answers
        self.model = model
        self.usage = usage or _Usage(4)
        self.request_id = request_id


class _Client:
    base_url = 'https://console.nace.ai'

    def __init__(self, response: _Response | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, state: Any, questions: Any, **kwargs: Any) -> _Response:
        self.calls.append({'state': state, 'questions': questions, **kwargs})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _model(model_name: str, client: _Client) -> NaceModel:
    return NaceModel(model_name, provider=NaceProvider(nace_client=client))  # type: ignore[arg-type]


def _request(**overrides: Any) -> DecisionRequest:
    questions = overrides.pop('questions', {'billing': NoulQuestion(instructions='Is this about billing?')})
    return DecisionRequest(state=overrides.pop('state', 'I was charged twice.'), questions=questions, **overrides)


async def test_noul_choice_and_score_on_drex_v2_and_drex_max() -> None:
    for model_name in ('drex-v2.0', 'drex-max'):
        client = _Client(
            _Response(
                {
                    'billing': NoulAnswer(noul=0.8),
                    'tone': ChoiceAnswer(choice='calm', confidence=0.7, probabilities={'calm': 0.7, 'angry': 0.3}),
                    'severity': ScoreAnswer(
                        score=1.2,
                        confidence=0.6,
                        probabilities={0: 0.2, 1: 0.4, 2: 0.4},
                        legend=('low', 'mid', 'high'),
                    ),
                },
                model=model_name,
                usage=_Usage(11, 2),
                request_id='req_1',
            )
        )
        response = await _model(model_name, client).decide(
            _request(
                questions={
                    'billing': NoulQuestion(
                        instructions='Is this about billing?',
                        criteria=DecisionNoulCriteria(true='a charge', false=None),
                    ),
                    'tone': ChoiceQuestion(instructions='What is the tone?', criteria={'calm': None, 'angry': 'upset'}),
                    'severity': ScoreQuestion(instructions='How severe?', criteria=['low', 'mid', 'high']),
                }
            ),
            {},
        )
        sent = client.calls[0]['questions']
        assert sent['billing'] == Noul(
            instructions='Is this about billing?',
            criteria=NoulCriteria(true='a charge', false=None),
        )
        assert sent['tone'] == Choice(instructions='What is the tone?', criteria={'calm': None, 'angry': 'upset'})
        assert sent['severity'] == Score(instructions='How severe?', criteria=['low', 'mid', 'high'])
        assert client.calls[0]['model'] == model_name
        assert client.calls[0]['images'] is None
        assert response.provider_response_id == 'req_1'
        assert response.usage.input_tokens == 11
        assert 'image_tokens' not in response.usage.details
        assert response.answers['billing'] == DecisionNoulAnswer(noul=0.8)
        assert response.answers['tone'] == DecisionChoiceAnswer(
            choice='calm', confidence=0.7, probabilities={'calm': 0.7, 'angry': 0.3}
        )
        assert response.answers['severity'] == DecisionScoreAnswer(
            score=1.2,
            confidence=0.6,
            probabilities={0: 0.2, 1: 0.4, 2: 0.4},
            legend={0: 'low', 1: 'mid', 2: 'high'},
        )


async def test_mixed_batch_preserves_question_order_names() -> None:
    client = _Client(_Response({'a': NoulAnswer(noul=0.1), 'b': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    await _model('drex-v2.0', client).decide(
        _request(questions={'a': NoulQuestion(instructions='A?'), 'b': NoulQuestion(instructions='B?')}),
        {'extra_headers': {'x-test': '1'}},
    )
    assert list(client.calls[0]['questions']) == ['a', 'b']
    assert client.calls[0]['extra_headers'] == {'x-test': '1'}


async def test_drex_max_sends_one_image_two_images_and_reversed_images() -> None:
    one = BinaryContent(_PNG, media_type='image/png')
    two = BinaryContent(_JPEG, media_type='image/jpeg')
    cases = (
        (one,),
        (one, two),
        (two, one),
    )
    for images in cases:
        client = _Client(
            _Response(
                {'seen': NoulAnswer(noul=0.4)},
                model='drex-max',
                usage=_Usage(9, image_tokens=3 * len(images)),
            )
        )
        response = await _model('drex-max', client).decide(
            _request(questions={'seen': NoulQuestion(instructions='Is the charge visible?')}, images=images),
            {},
        )
        assert client.calls[0]['images'] == [image_input(image.data) for image in images]
        assert response.usage.details['image_tokens'] == 3 * len(images)


async def test_drex_v2_rejects_images_before_the_client_is_called() -> None:
    client = _Client()
    with pytest.raises(ModelAPIError, match='does not accept images'):
        await _model('drex-v2.0', client).decide(_request(images=(BinaryContent(_PNG, media_type='image/png'),)), {})
    assert client.calls == []


async def test_seventeenth_drex_max_question_fails_before_the_client_is_called() -> None:
    client = _Client()
    questions = {f'q{index}': NoulQuestion(instructions='Yes?') for index in range(17)}
    with pytest.raises(ModelAPIError, match='at most 16 questions'):
        await _model('drex-max', client).decide(_request(questions=questions), {})
    assert client.calls == []


async def test_drex_v2_does_not_use_the_drex_max_question_cap() -> None:
    questions = {f'q{index}': NoulQuestion(instructions='Yes?') for index in range(17)}
    client = _Client(_Response({name: NoulAnswer(noul=0.5) for name in questions}, model='drex-v2.0'))
    response = await _model('drex-v2.0', client).decide(_request(questions=questions), {})
    assert set(response.answers) == set(questions)
    assert len(client.calls) == 1


async def test_unprocessable_entity_keeps_issue_path_and_message() -> None:
    body = {'error': {'issues': [{'path': 'questions.tone.criteria', 'message': 'required'}]}}
    client = _Client(error=NaceUnprocessableEntityError(422, body, httpx.Headers({'x-request-id': 'req_422'})))
    with pytest.raises(ModelRequestRejectedError) as exc_info:
        await _model('drex-v2.0', client).decide(_request(), {})
    error = exc_info.value
    assert error.body == body
    assert 'questions.tone.criteria' in str(error)
    assert 'required' in str(error)


async def test_payment_required_keeps_the_response_body() -> None:
    body = {'error': {'message': 'insufficient credit', 'balance': 0}}
    client = _Client(error=NaceInsufficientCreditError(402, body, httpx.Headers()))
    with pytest.raises(ModelQuotaExceededError) as exc_info:
        await _model('drex-max', client).decide(_request(), {})
    assert exc_info.value.body == body


async def test_retry_policy_covers_429_and_529_and_respects_retry_after() -> None:
    client = _Client(_Response({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    await _model('drex-v2.0', client).decide(_request(), {'max_retries': 3})
    policy = client.calls[0]['retry']
    assert isinstance(policy, RetryPolicy)
    assert policy.max_retries == 3
    assert policy.respect_retry_after is True
    assert 429 in policy.http_statuses
    assert 529 in policy.http_statuses


@pytest.mark.parametrize(
    ('make_error', 'category'),
    [
        (lambda: NaceRateLimitError(429, {'error': 'slow down'}, httpx.Headers()), ModelRateLimitError),
        (lambda: NaceOverloadedError(529, {'error': 'overloaded'}, httpx.Headers()), ModelOverloadedError),
    ],
)
async def test_rate_limit_and_overloaded_map_to_model_http_errors(
    make_error: Callable[[], NaceError], category: type[Exception]
) -> None:
    error = make_error()
    client = _Client(error=error)
    with pytest.raises(category) as exc_info:
        await _model('drex-v2.0', client).decide(_request(), {})
    assert exc_info.value.body == error.body  # type: ignore[attr-defined]


def test_decide_span_adds_image_tokens_only_when_the_sdk_reports_them() -> None:
    request = DecisionRequest(state='x', questions={'q': NoulQuestion(instructions='Yes?')})
    with_images = DecisionResponse(
        answers={},
        model_name='drex-max',
        usage=RequestUsage(input_tokens=9, output_tokens=0, details={'image_tokens': 6}),
        provider_response_id='req_img',
    )
    attributes = _decide_response_attributes(with_images, request, None, False)
    assert attributes['pydantic_ai.decision.usage.input_tokens'] == 9
    assert attributes['pydantic_ai.decision.usage.image_tokens'] == 6
    assert attributes['gen_ai.response.id'] == 'req_img'

    without_images = DecisionResponse(
        answers={},
        model_name='drex-v2.0',
        usage=RequestUsage(input_tokens=4, output_tokens=0),
        provider_response_id='req_text',
    )
    plain = _decide_response_attributes(without_images, request, None, False)
    assert 'pydantic_ai.decision.usage.image_tokens' not in plain
    assert plain['gen_ai.response.id'] == 'req_text'


async def test_omitted_model_sends_drex_v2_and_drex_latest_is_sent_unchanged() -> None:
    """The SDK's own default is `drex-latest`, so the model id is always passed, never left to the client."""
    client = _Client(_Response({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    await NaceModel(provider=NaceProvider(nace_client=client)).decide(_request(), {})  # type: ignore[arg-type]
    await _model('drex-latest', client).decide(_request(), {})
    assert [call['model'] for call in client.calls] == ['drex-v2.0', 'drex-latest']


async def test_retry_policy_keeps_the_180_second_budget_without_max_retries() -> None:
    client = _Client(_Response({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    await _model('drex-v2.0', client).decide(_request(), {})
    policy = client.calls[0]['retry']
    assert isinstance(policy, RetryPolicy)
    assert policy.timeout == 180.0
    assert policy.max_retries == RetryPolicy().max_retries
    assert {429, 529} <= set(policy.http_statuses)


async def test_a_profile_cannot_raise_the_drex_max_cap_or_enable_images_elsewhere() -> None:
    """Unit test: these guards run before any request, so no recording can show them."""
    client = _Client()
    widened = {'nace_max_questions': 64, 'nace_accepts_images': True}
    questions = {f'q{index}': NoulQuestion(instructions='Yes?') for index in range(17)}
    drex_max = NaceModel('drex-max', provider=NaceProvider(nace_client=client), profile=widened)  # type: ignore[arg-type]
    with pytest.raises(ModelAPIError, match='at most 16 questions'):
        await drex_max.decide(_request(questions=questions), {})
    drex_v2 = NaceModel('drex-v2.0', provider=NaceProvider(nace_client=client), profile=widened)  # type: ignore[arg-type]
    with pytest.raises(ModelAPIError, match='does not accept images'):
        await drex_v2.decide(_request(images=(BinaryContent(_PNG, media_type='image/png'),)), {})
    assert client.calls == []


async def test_oversized_image_sets_are_refused_with_a_workspace_file_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The caps are patched down so the test does not build megabytes of images."""
    from pydantic_ai.models import nace as nace_module

    client = _Client()
    images = (BinaryContent(_PNG, media_type='image/png'), BinaryContent(_JPEG, media_type='image/jpeg'))
    monkeypatch.setattr(nace_module, '_MAX_IMAGE_SET_BYTES', len(_PNG) + len(_JPEG) - 1)
    with pytest.raises(UserError, match='workspace_file'):
        await _model('drex-max', client).decide(_request(images=images), {})
    monkeypatch.undo()
    monkeypatch.setattr(nace_module, '_MAX_IMAGE_BODY_BYTES', 10)
    with pytest.raises(UserError, match='workspace_file'):
        await _model('drex-max', client).decide(_request(images=images), {})
    monkeypatch.undo()
    monkeypatch.setattr(nace_module, '_MAX_IMAGE_BYTES', len(_PNG) - 1)
    with pytest.raises(UserError, match='workspace_file'):
        await _model('drex-max', client).decide(_request(images=images), {})
    assert client.calls == []


async def test_omitted_instructions_stay_none_on_the_wire() -> None:
    client = _Client(_Response({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    await _model('drex-v2.0', client).decide(_request(questions={'billing': NoulQuestion(instructions=None)}), {})
    question = client.calls[0]['questions']['billing']
    assert isinstance(question, Noul)
    assert question.instructions is None


async def test_an_image_the_sdk_cannot_read_is_refused_before_the_client_is_called() -> None:
    client = _Client()
    with pytest.raises(UserError, match='could not accept this image'):
        await _model('drex-max', client).decide(
            _request(images=(BinaryContent(b'not an image', media_type='image/png'),)), {}
        )
    assert client.calls == []


async def test_timeout_defaults_to_60_seconds_and_a_set_timeout_is_forwarded() -> None:
    """Unit test: the timeout is a client argument, and a recording does not show it."""
    client = _Client(_Response({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    custom = httpx.Timeout(5.0, connect=2.0)
    await _model('drex-v2.0', client).decide(_request(), {})
    await _model('drex-v2.0', client).decide(_request(), {'timeout': custom})
    assert [call['timeout'] for call in client.calls] == [60.0, custom]


@pytest.mark.parametrize(
    ('make_error', 'category'),
    [
        (
            lambda: NaceAPIResponseValidationError(200, {'answers': 1}, httpx.Headers(), 'answers'),
            UnexpectedModelBehavior,
        ),
        (lambda: NaceAPIConnectionError('connection reset'), ModelConnectionError),
        (lambda: NaceAPITimeoutError(60.0), ModelTimeoutError),
        (lambda: NaceError('no API key'), UserError),
    ],
)
async def test_sdk_errors_map_to_pydantic_ai_errors(
    make_error: Callable[[], NaceError], category: type[Exception]
) -> None:
    """Unit test: these failures come from the SDK's own checks or the transport, which a recording cannot replay."""
    with pytest.raises(category):
        await _model('drex-v2.0', _Client(error=make_error())).decide(_request(), {})


async def test_a_response_without_a_request_id_has_no_provider_response_id() -> None:
    class _NoRequestId(_Response):
        @property
        def request_id(self) -> str:
            raise NaceError('no x-request-id header')

        @request_id.setter
        def request_id(self, value: str) -> None:
            pass

    client = _Client(_NoRequestId({'billing': NoulAnswer(noul=0.2)}, model='drex-v2.0'))
    response = await _model('drex-v2.0', client).decide(_request(), {})
    assert response.provider_response_id is None


async def test_non_text_instructions_are_sent_as_text_and_non_text_levels_are_refused() -> None:
    client = _Client(
        _Response(
            {'tone': ScoreAnswer(score=1, confidence=0.9, probabilities={}, legend=('calm', 'angry'))},
            model='drex-v2.0',
        )
    )
    model = _model('drex-v2.0', client)
    await model.decide(_request(questions={'tone': ScoreQuestion(criteria=['calm', 'angry'], instructions=3)}), {})
    assert client.calls[0]['questions']['tone'].instructions == '3'
    with pytest.raises(UserError, match='score level must be a string'):
        await model.decide(_request(questions={'tone': ScoreQuestion(criteria=['calm', 2])}), {})


async def test_an_unknown_answer_type_is_unexpected_model_behavior() -> None:
    client = _Client(_Response({'billing': object()}, model='drex-v2.0'))
    with pytest.raises(UnexpectedModelBehavior, match='Unexpected answer from Nace'):
        await _model('drex-v2.0', client).decide(_request(), {})


def test_the_nace_provider_is_inferred_from_its_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NACE_API_KEY', 'test-key')
    model = NaceModel('drex-max')
    assert model.system == 'nace'
    assert model.base_url == 'https://console.nace.ai'
    assert model.model_name == 'drex-max'
