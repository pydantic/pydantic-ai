from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypeAlias

import httpx2
from pydantic import Field, JsonValue, TypeAdapter
from typing_extensions import NotRequired, TypedDict, assert_never

from .._http import to_httpx2_timeout
from .._utils import is_str_dict
from ..exceptions import UnexpectedModelBehavior, UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import get_user_agent
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
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

try:
    from openai import AsyncOpenAI, RequestOptions

    from ..providers.openai_decisions import OpenAIDecisionsProvider
    from .openai import _map_api_errors  # pyright: ignore[reportPrivateUsage]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Decisions model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAIDecisionsModel',
    'OpenAIDecisionsModelName',
    'OpenAIDecisionsModelSettings',
)

OpenAIDecisionsModelName = str
"""The ID of the model to ask, such as `gpt-6-luna`, which the Responses API also serves."""


class OpenAIDecisionsModelSettings(DecisionModelSettings, total=False):
    """Settings used for an OpenAI Decisions API request."""

    # ALL FIELDS MUST BE `openai_decisions_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future Decisions API-specific settings.


@dataclass(init=False)
class OpenAIDecisionsModel(DecisionModel[AsyncOpenAI]):
    """The model class for OpenAI's Decisions API, which runs a GPT model as a [decision model][pydantic_ai.models.decision.DecisionModel].

    The Decisions API answers typed questions about a text, each with a probability or a distribution over the
    options, rather than writing text. An agent whose job is to decide something runs on it like on any other model,
    with the `output_type` as the questions:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('openai-decisions:gpt-6-luna', output_type=Handling)
    ...
    ```

    See [Decision models](https://pydantic.dev/docs/ai/models/decision/) for how an agent's output type and tools
    become questions, and [OpenAI](https://pydantic.dev/docs/ai/models/openai/#decisions-api) for setup.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    # `max_choice_options` and `max_score_levels` stay `None`: OpenAI publishes no limits for the Decisions API.

    _model_name: OpenAIDecisionsModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIDecisionsModelName,
        *,
        provider: Literal['openai-decisions'] | OpenAIDecisionsProvider = 'openai-decisions',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Decisions model.

        Args:
            model_name: The name of the OpenAI model to use, such as `gpt-6-luna`.
            provider: The provider to use for the API's URL and key.
            profile: The model profile to use. Defaults to one selected by the provider.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = OpenAIDecisionsProvider()
        self._provider = provider
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> OpenAIDecisionsModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    @property
    def model_id(self) -> str:
        """The fully qualified model name, under `openai-decisions:`, as `openai:` takes the ID for the Responses API."""
        return f'{self._provider.model_id_namespace}:{self._model_name}'

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the `/v1/decisions` endpoint."""
        body = _DecisionsRequest(
            model=self._model_name,
            input=_text(request.state),
            questions=[_question(name, question) for name, question in request.questions.items()],
        )
        options = _request_options(model_settings)
        with _map_api_errors(self._model_name, self._provider.model_id_namespace):
            # TODO: call the SDK's Decisions resource once `openai` ships one.
            response = await self.client.post('/decisions', cast_to=httpx2.Response, body=body, options=options)

        try:
            data = response.json()
            parsed = _response_adapter.validate_python(data)
        except ValueError as e:
            raise UnexpectedModelBehavior(f'Invalid response from the OpenAI Decisions API: {e}', response.text) from e
        answers = {answer.name: answer.answer() for answer in parsed.answers}
        if len(answers) != len(parsed.answers) or answers.keys() != request.questions.keys():
            raise UnexpectedModelBehavior(
                'Invalid response from the OpenAI Decisions API: answer names do not match the questions', response.text
            )
        for name, answer in answers.items():
            if not _allows(request.questions[name], answer):
                raise UnexpectedModelBehavior(
                    f'Invalid response from the OpenAI Decisions API: answer {name!r} does not match its question: {answer!r}',
                    response.text,
                )
        return DecisionResponse(
            answers=answers,
            model_name=parsed.model,
            usage=RequestUsage.extract(
                data,
                provider=self.system,
                provider_url=self.base_url,
                provider_fallback='openai',
                api_flavor='responses',
            ),
            # The body carries no ID of its own.
            provider_response_id=response.headers.get('x-request-id'),
        )


def _request_options(model_settings: DecisionModelSettings) -> RequestOptions:
    """The generic settings the API takes, as the SDK's options for one request."""
    headers = dict(model_settings.get('extra_headers', {}))
    headers.setdefault('User-Agent', get_user_agent())
    options: RequestOptions = {'headers': headers}
    if (timeout := model_settings.get('timeout')) is not None:
        options['timeout'] = to_httpx2_timeout(timeout)
    if (extra_body := model_settings.get('extra_body')) is not None:
        if not is_str_dict(extra_body):
            raise UserError(
                f'`extra_body` must be a mapping to send it to the OpenAI Decisions API; got {extra_body!r}.'
            )
        options['extra_json'] = extra_body
    return options


def _text(value: JsonValue) -> str:
    """A protocol value as the text the API takes: a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _question(name: str, question: DecisionQuestion) -> _Question:
    """A protocol question as a Decisions API question, whose `predicate` is the protocol's yes/no."""
    instructions = question.instructions
    wire: _Question
    if isinstance(question, NoulQuestion):
        wire = _PredicateQuestion(type='predicate', name=name)
        instructions = _with_meanings(instructions, question.criteria or NoulCriteria())
    elif isinstance(question, ChoiceQuestion):
        choices = [_option(label, meaning) for label, meaning in question.criteria.items()]
        wire = _ChoiceQuestion(type='choice', name=name, choices=choices)
    elif isinstance(question, ScoreQuestion):
        levels = [_level(str(level), meaning) for level, meaning in enumerate(question.criteria)]
        wire = _ScoreQuestion(type='score', name=name, levels=levels)
    else:
        assert_never(question)
    if instructions is not None:
        wire['instructions'] = _text(instructions)
    return wire


def _with_meanings(instructions: JsonValue, criteria: NoulCriteria) -> JsonValue:
    """A yes/no question's instructions, with what yes and no mean, which a `predicate` has no field of its own for."""
    meanings: dict[str, JsonValue] = {
        answer: meaning for answer, meaning in (('yes', criteria.true), ('no', criteria.false)) if meaning is not None
    }
    if not meanings:
        return instructions
    if instructions is None:
        return meanings
    if isinstance(instructions, dict):
        return {**instructions, **meanings}
    return {'question': instructions, **meanings}


def _allows(question: DecisionQuestion, answer: DecisionAnswer) -> bool:
    """Whether a question allows an answer: one of its kind, picking an offered option, or within the rubric."""
    if isinstance(question, NoulQuestion):
        return isinstance(answer, NoulAnswer)
    elif isinstance(question, ChoiceQuestion):
        return (
            isinstance(answer, ChoiceAnswer)
            and answer.choice in question.criteria
            and answer.probabilities.keys() == question.criteria.keys()
        )
    elif isinstance(question, ScoreQuestion):
        return (
            isinstance(answer, ScoreAnswer)
            and answer.probabilities.keys() == set(range(len(question.criteria)))
            and 0 <= answer.score <= len(question.criteria) - 1
        )
    else:
        assert_never(question)


def _option(value: str, meaning: JsonValue) -> _Option:
    return _Option(value=value) if meaning is None else _Option(value=value, description=_text(meaning))


def _level(label: str, meaning: JsonValue) -> _Level:
    return _Level(label=label) if meaning is None else _Level(label=label, description=_text(meaning))


class _Option(TypedDict):
    value: str
    description: NotRequired[str]


class _Level(TypedDict):
    label: str
    description: NotRequired[str]


class _PredicateQuestion(TypedDict):
    type: Literal['predicate']
    name: str
    instructions: NotRequired[str]


class _ChoiceQuestion(TypedDict):
    type: Literal['choice']
    name: str
    choices: list[_Option]
    instructions: NotRequired[str]


class _ScoreQuestion(TypedDict):
    type: Literal['score']
    name: str
    levels: list[_Level]
    instructions: NotRequired[str]


_Question: TypeAlias = _PredicateQuestion | _ChoiceQuestion | _ScoreQuestion


class _DecisionsRequest(TypedDict):
    """The body `/v1/decisions` takes."""

    model: str
    input: str
    questions: list[_Question]


_Probability = Annotated[float, Field(ge=0, le=1)]


@dataclass(kw_only=True)
class _PredicateAnswer:
    type: Literal['predicate']
    name: str
    probability: _Probability

    def answer(self) -> NoulAnswer:
        return NoulAnswer(noul=self.probability)


@dataclass(kw_only=True)
class _OptionProbability:
    value: str
    probability: _Probability


@dataclass(kw_only=True)
class _ChoiceAnswer:
    type: Literal['choice']
    name: str
    choice: str
    confidence: _Probability
    probabilities: list[_OptionProbability]

    def answer(self) -> ChoiceAnswer:
        probabilities = {option.value: option.probability for option in self.probabilities}
        return ChoiceAnswer(choice=self.choice, confidence=self.confidence, probabilities=probabilities)


@dataclass(kw_only=True)
class _LevelProbability:
    value: int
    probability: _Probability


@dataclass(kw_only=True)
class _ScoreAnswer:
    type: Literal['score']
    name: str
    score: float
    confidence: _Probability
    probabilities: list[_LevelProbability]

    def answer(self) -> ScoreAnswer:
        probabilities = {level.value: level.probability for level in self.probabilities}
        return ScoreAnswer(score=self.score, confidence=self.confidence, probabilities=probabilities)


@dataclass(kw_only=True)
class _DecisionsResponse:
    """The body the API answers `/v1/decisions` with, apart from the usage, which is read as the Responses API's."""

    model: str
    answers: list[Annotated[_PredicateAnswer | _ChoiceAnswer | _ScoreAnswer, Field(discriminator='type')]]


_response_adapter = TypeAdapter(_DecisionsResponse)
