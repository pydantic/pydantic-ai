from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal, NotRequired, assert_never

import httpx2
from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from .. import _utils
from ..exceptions import UnexpectedModelBehavior, UserError
from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.decision import decision_model_profile
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from .decision import (
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
    _answer_fits,  # pyright: ignore[reportPrivateUsage]
    _text,  # pyright: ignore[reportPrivateUsage]
)

try:
    from google.genai import Client, errors
    from google.genai.types import HttpOptionsDict

    from ..providers.google import GoogleProvider
    from .google import (
        _map_api_error,  # pyright: ignore[reportPrivateUsage]
        _request_http_options,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `google-genai` package to use the Google Decisions model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

__all__ = (
    'GoogleDecisionsModel',
    'GoogleDecisionsModelName',
    'GoogleDecisionsModelSettings',
)

GoogleDecisionsModelName = str
"""The ID of the Gemini decision model to ask."""


class GoogleDecisionsModelSettings(DecisionModelSettings, total=False):
    """Settings used for a Gemini API decisions request."""

    # ALL FIELDS MUST BE `google_decisions_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future decisions-specific settings.


@dataclass(init=False)
class GoogleDecisionsModel(DecisionModel[Client]):
    """The model class for the Gemini API's decisions endpoint, which runs a Gemini model as a [decision model][pydantic_ai.models.decision.DecisionModel].

    The endpoint answers typed questions about text or images, each with a probability or a distribution over the
    options, rather than writing text. An agent whose job is to decide something runs on it like on any other model,
    with the `output_type` as the questions:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('google-decisions:gemini-decision-launch-id', output_type=Handling)
    ...
    ```

    See [Decision models](../../models/decision.md) for how an agent's output type and tools
    become questions, and [Google](../../models/google.md#decisions-api) for setup.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    supports_image_input: ClassVar[bool] = True

    requires_instructions: ClassVar[bool] = True
    """The API refuses a question without non-empty `instructions`, even one whose options say what it asks."""

    max_questions: ClassVar[int | None] = 1000
    """The API accepts at most 1000 questions per request; a 1001st returns HTTP 400."""

    max_choice_options: ClassVar[int | None] = 255
    """The API takes at most this many options in one pick-one; a 256th is a 400."""

    max_score_levels: ClassVar[int | None] = 255
    """The API takes at most this many levels in one rubric; a 256th is a 400."""

    _model_name: GoogleDecisionsModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)

    def __init__(
        self,
        model_name: GoogleDecisionsModelName,
        *,
        provider: Literal['google'] | GoogleProvider = 'google',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Google Decisions model.

        Args:
            model_name: The name of the Gemini decision model to use.
            provider: The provider to use for the API's URL and key: a
                [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider], as for
                [`GoogleModel`][pydantic_ai.models.google.GoogleModel], or `'google'` (the default), which creates one.
            profile: The model profile to use. Defaults to a decision model's profile for the model name.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = GoogleProvider()
        self._provider = provider
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> Client:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleDecisionsModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider: `'google-decisions'`, whatever the provider is called.

        It makes the model ID `google-decisions:<model name>`, which resolves back to this class, and keeps a
        [`GoogleModel`][pydantic_ai.models.google.GoogleModel] continuing the conversation from taking a decision
        response for one of its own.
        """
        return 'google-decisions'

    def _provider_model_profile(self) -> ModelProfile:
        # The provider profiles a Gemini model name for `generateContent`, whose structured output, thinking and
        # tools this endpoint has none of.
        return decision_model_profile(self.model_name)

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the Gemini API's `decisions` endpoint."""
        self._check_request_limits(image_count=len(request.images), question_count=len(request.questions))
        if any(not image.is_image for image in request.images):
            raise UserError('Google Decisions supports image evidence only; `request.images` contains a non-image.')
        http_options = _http_options(self.client, model_settings)
        # The API takes question IDs of `[A-Za-z0-9_-]` only, which the questions' own names, such as `Ticket.urgent`,
        # are not. The IDs are not shown to the model, so each question goes under its index instead.
        names = {f'q{index}': name for index, name in enumerate(request.questions)}
        body: dict[str, object] = {
            'model': self._model_name,
            'input': _input(request),
            'questions': {key: _question(request.questions[name]) for key, name in names.items()},
        }
        try:
            # google-genai has no method for this endpoint, so the request goes through the client's own API client,
            # with the provider's credentials, URL, API version, headers and retries. Use the SDK's method once it
            # has one.
            response = await self.client._api_client.async_request(  # pyright: ignore[reportPrivateUsage]
                'post', 'decisions', body, http_options
            )
        except (errors.APIError, httpx2.TransportError, httpx2.HTTPStatusError) as e:
            raise _map_api_error(e, self._model_name, self.system) from e

        try:
            parsed = _response_adapter.validate_json(response.body or '')
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from the Gemini API: {e}', response.body) from e
        if parsed.answers.keys() != names.keys():
            raise UnexpectedModelBehavior(
                'Invalid response from the Gemini API: answer IDs do not match the questions', response.body
            )
        answers: dict[str, DecisionAnswer] = {}
        for key, name in names.items():
            answer = parsed.answers[key]
            if not _answer_fits(request.questions[name], answer):
                raise UnexpectedModelBehavior(
                    f'Invalid response from the Gemini API: answer {name!r} does not match its question: {answer!r}',
                    response.body,
                )
            answers[name] = answer
        return DecisionResponse(
            answers=answers,
            # The body names no model and carries no ID of its own.
            model_name=self._model_name,
            usage=RequestUsage(
                input_tokens=parsed.usage.total_input_tokens, output_tokens=parsed.usage.total_output_tokens
            ),
        )


def _http_options(client: Client, model_settings: DecisionModelSettings) -> HttpOptionsDict:
    """The request's HTTP options, as `GoogleModel` sends them, with `extra_body` merged into the body."""
    http_options = _request_http_options(client, model_settings, dict(model_settings.get('extra_headers', {})))
    if (extra_body := model_settings.get('extra_body')) is None:
        return http_options
    if not _utils.is_str_dict(extra_body):
        raise UserError(f'`extra_body` must be a dict to send it to the Gemini API; got {extra_body!r}.')
    if 'questions' in extra_body:
        raise UserError(
            '`extra_body` cannot set `questions`: each question goes under an ID of its own, '
            'which its answer is read back by.'
        )
    http_options['extra_body'] = extra_body
    return http_options


class _TextBlock(TypedDict):
    type: Literal['text']
    text: str


class _ImageBlock(TypedDict):
    type: Literal['image']
    data: str
    mime_type: str
    resolution: NotRequired[str]


class _Question(TypedDict):
    type: Literal['noul', 'choice', 'score']
    instructions: NotRequired[str]
    criteria: NotRequired[dict[str, str | None] | list[str | None]]


def _input(request: DecisionRequest) -> str | list[_TextBlock | _ImageBlock]:
    """The state as text, with each image after its `<image N>` label.

    The API takes no conversation turns, so `request.messages` is not sent: the state already holds the conversation.
    """
    text = _text(request.state)
    if not request.images:
        return text
    blocks: list[_TextBlock | _ImageBlock] = [_TextBlock(type='text', text=text)]
    for index, image in enumerate(request.images, start=1):
        image_block = _ImageBlock(type='image', data=image.base64, mime_type=image.media_type)
        if (resolution := (image.vendor_metadata or {}).get('resolution')) is not None:
            image_block['resolution'] = resolution
        blocks.extend((_TextBlock(type='text', text=f'<image {index}>:'), image_block))
    return blocks


def _question(question: DecisionQuestion) -> _Question:
    """A protocol question as the API takes it, whose instructions and meanings are text."""
    wire = _Question(type=question.type)
    if question.instructions is not None:
        wire['instructions'] = _text(question.instructions)
    match question:
        case NoulQuestion():
            if meanings := _noul_meanings(question.criteria or NoulCriteria()):
                wire['criteria'] = meanings
        case ChoiceQuestion():
            wire['criteria'] = {label: _description(meaning) for label, meaning in question.criteria.items()}
        case ScoreQuestion():
            wire['criteria'] = [_description(level) for level in question.criteria]
        case _:
            assert_never(question)
    return wire


def _noul_meanings(criteria: NoulCriteria) -> dict[str, str | None]:
    """What yes and no mean, under the API's names for them, leaving out a side that is not described."""
    return {
        f'{outcome}_description': _text(meaning)
        for outcome, meaning in (('true', criteria.true), ('false', criteria.false))
        if meaning is not None
    }


def _description(meaning: JsonValue) -> str | None:
    return None if meaning is None else _text(meaning)


@dataclass(kw_only=True)
class _Usage:
    total_input_tokens: Annotated[int, Field(ge=0)] = 0
    total_output_tokens: Annotated[int, Field(ge=0)] = 0


@dataclass(kw_only=True)
class _DecisionsResponse:
    """The body the API answers `decisions` with."""

    answers: dict[str, Annotated[DecisionAnswer, Field(discriminator='type')]]
    usage: _Usage


_response_adapter = TypeAdapter(_DecisionsResponse)
