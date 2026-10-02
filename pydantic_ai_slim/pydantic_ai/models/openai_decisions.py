from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal

import httpx2
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

from .._http import to_httpx2_timeout
from ..exceptions import UnexpectedModelBehavior, UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.openai import OpenAIProvider
from ..settings import ModelSettings
from ..usage import RequestUsage
from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from .openai import _map_api_errors  # pyright: ignore[reportPrivateUsage]

try:
    from openai import NOT_GIVEN, AsyncOpenAI
    from openai._base_client import make_request_options
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI Decisions model, '
        'you can use the `openai` optional group: `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('OpenAIDecisionsModel',)


@dataclass(init=False)
class OpenAIDecisionsModel(DecisionModel[AsyncOpenAI]):
    """A decision model using OpenAI's `/v1/decisions` API.

    Uses the same output types, tool routing and settings as
    [`DecisionModel`][pydantic_ai.models.decision.DecisionModel]. Requires access to the Decisions API.
    Generic sampling settings are ignored; `timeout`, `extra_headers` and `extra_body` are forwarded.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    _model_name: str = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Decisions model.

        Args:
            model_name: The model to use, such as `gpt-6-luna`.
            provider: The provider for authentication, the base URL and the OpenAI client.
            profile: The model profile. Defaults to one selected by the provider.
            settings: Model settings used as defaults for this model.
        """
        self._model_name = model_name
        self._provider = OpenAIProvider() if isinstance(provider, str) else provider
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to OpenAI's Decisions API."""
        body = {
            'model': self._model_name,
            'input': _text(request.state),
            'questions': [_question(name, question) for name, question in request.questions.items()],
        }
        with _map_api_errors(self._model_name):
            try:
                response = await self.client.post(
                    '/decisions',
                    cast_to=httpx2.Response,
                    body=body,
                    options=make_request_options(
                        timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
                        extra_headers=model_settings.get('extra_headers'),
                        extra_body=model_settings.get('extra_body'),
                    ),
                )
            except (TypeError, ValueError) as e:
                raise UserError(f'Could not send this request to the OpenAI Decisions API: {e}') from e

        try:
            parsed = _response_adapter.validate_json(response.content)
            if (
                len(parsed.answers) != len(request.questions)
                or {a.name for a in parsed.answers} != request.questions.keys()
            ):
                raise ValueError('answer names do not match the questions')
            answers = {answer.name: _answer(answer, request.questions[answer.name]) for answer in parsed.answers}
        except (ValidationError, ValueError) as e:
            raise UnexpectedModelBehavior(f'Invalid response from the OpenAI Decisions API: {e}', response.text) from e

        usage = RequestUsage()
        if parsed.usage is not None:
            usage.input_tokens = parsed.usage.input_tokens
            usage.output_tokens = parsed.usage.output_tokens
            if parsed.usage.input_tokens_details is not None:
                usage.cache_read_tokens = parsed.usage.input_tokens_details.cached_tokens
                usage.cache_write_tokens = parsed.usage.input_tokens_details.cache_write_tokens
            if parsed.usage.output_tokens_details is not None:
                usage.details['reasoning_tokens'] = parsed.usage.output_tokens_details.reasoning_tokens
        return DecisionResponse(
            answers=answers,
            model_name=parsed.model,
            usage=usage,
            provider_response_id=response.headers.get('x-request-id'),
        )


def _text(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _question(name: str, question: DecisionQuestion) -> dict[str, object]:
    instructions = [] if question.instructions is None else [_text(question.instructions)]
    body: dict[str, object] = {'name': name}
    if isinstance(question, NoulQuestion):
        body['type'] = 'predicate'
        if question.criteria is not None:
            for label, description in (('Yes', question.criteria.true), ('No', question.criteria.false)):
                if description is not None:
                    instructions.append(f'{label}: {_text(description)}')
    elif isinstance(question, ChoiceQuestion):
        body['type'] = 'choice'
        body['choices'] = [
            {'value': value, **({'description': _text(description)} if description is not None else {})}
            for value, description in question.criteria.items()
        ]
    elif isinstance(question, ScoreQuestion):
        body['type'] = 'score'
        body['levels'] = [
            {'label': str(index), 'description': _text(description) if description is not None else None}
            for index, description in enumerate(question.criteria)
        ]
    else:
        assert_never(question)
    if instructions:
        body['instructions'] = '\n'.join(instructions)
    return body


_Probability = Annotated[float, Field(ge=0, le=1)]
_TokenCount = Annotated[int, Field(ge=0)]


@dataclass(kw_only=True)
class _PredicateAnswer:
    name: str
    type: Literal['predicate']
    probability: _Probability


@dataclass(kw_only=True)
class _ChoiceProbability:
    value: str
    probability: _Probability


@dataclass(kw_only=True)
class _ChoiceAnswer:
    name: str
    type: Literal['choice']
    choice: str
    confidence: _Probability
    probabilities: list[_ChoiceProbability]


@dataclass(kw_only=True)
class _ScoreProbability:
    label: str
    probability: _Probability


@dataclass(kw_only=True)
class _ScoreAnswer:
    name: str
    type: Literal['score']
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    confidence: _Probability
    probabilities: list[_ScoreProbability]


@dataclass(kw_only=True)
class _InputTokensDetails:
    cached_tokens: _TokenCount = 0
    cache_write_tokens: _TokenCount = 0


@dataclass(kw_only=True)
class _OutputTokensDetails:
    reasoning_tokens: _TokenCount = 0


@dataclass(kw_only=True)
class _Usage:
    input_tokens: _TokenCount
    output_tokens: _TokenCount
    input_tokens_details: _InputTokensDetails | None = None
    output_tokens_details: _OutputTokensDetails | None = None


@dataclass(kw_only=True)
class _Response:
    model: str
    answers: list[Annotated[_PredicateAnswer | _ChoiceAnswer | _ScoreAnswer, Field(discriminator='type')]]
    usage: _Usage | None = None


_response_adapter = TypeAdapter(_Response)


def _sums_to_one(probabilities: list[float]) -> bool:
    # Allow each probability half a unit of two-decimal rounding, as `SystemOneModel` does.
    return abs(sum(probabilities) - 1) <= 1e-6 + len(probabilities) * 0.005


def _answer(answer: _PredicateAnswer | _ChoiceAnswer | _ScoreAnswer, question: DecisionQuestion) -> DecisionAnswer:
    if isinstance(answer, _PredicateAnswer) and isinstance(question, NoulQuestion):
        return NoulAnswer(noul=answer.probability)
    if isinstance(answer, _ChoiceAnswer) and isinstance(question, ChoiceQuestion):
        probabilities = {entry.value: entry.probability for entry in answer.probabilities}
        if (
            len(probabilities) != len(answer.probabilities)
            or probabilities.keys() != question.criteria.keys()
            or answer.choice not in probabilities
            or not _sums_to_one(list(probabilities.values()))
        ):
            raise ValueError(f'invalid choice probabilities for {answer.name!r}')
        return ChoiceAnswer(choice=answer.choice, confidence=answer.confidence, probabilities=probabilities)
    if isinstance(answer, _ScoreAnswer) and isinstance(question, ScoreQuestion):
        levels = {str(index) for index in range(len(question.criteria))}
        if len(answer.probabilities) != len(levels) or {entry.label for entry in answer.probabilities} != levels:
            raise ValueError(f'invalid score probabilities for {answer.name!r}')
        probabilities = {int(entry.label): entry.probability for entry in answer.probabilities}
        # Whether `score` is the mean or the likeliest level, it lies between the levels that carry probability.
        supported = [level for level, probability in probabilities.items() if probability > 0]
        if not _sums_to_one(list(probabilities.values())) or not min(supported) <= answer.score <= max(supported):
            raise ValueError(f'invalid score probabilities for {answer.name!r}')
        return ScoreAnswer(score=answer.score, confidence=answer.confidence, probabilities=probabilities)
    raise ValueError(f'answer type does not match question {answer.name!r}')
