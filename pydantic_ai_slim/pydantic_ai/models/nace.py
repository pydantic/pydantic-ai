from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, assert_never, cast

from pydantic import JsonValue

from .. import _model_errors
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import BinaryContent
from ..profiles import ModelProfileSpec
from ..profiles.nace import DREX_MAX_QUESTIONS, NACE_MAX_QUESTIONS, NaceModelProfile
from ..providers import Provider, infer_provider
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
    DecisionStreamedResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

try:
    from nace_sdk import (
        AsyncNaceClient,
        Choice as NaceChoice,
        ChoiceAnswer as NaceChoiceAnswer,
        NaceAPIConnectionError,
        NaceAPIError,
        NaceAPIResponseValidationError,
        NaceAPITimeoutError,
        NaceError,
        Noul as NaceNoul,
        NoulAnswer as NaceNoulAnswer,
        NoulCriteria as NaceNoulCriteria,
        RetryPolicy,
        Score as NaceScore,
        ScoreAnswer as NaceScoreAnswer,
        image_input,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install the `nace-sdk` package to use the Nace model, '
        'you can use the `nace` optional group — `pip install "pydantic-ai-slim[nace]"`'
    ) from _import_error

__all__ = (
    'KnownNaceModelNames',
    'NaceModel',
    'NaceModelName',
    'NaceStreamedResponse',
)

KnownNaceModelNames = Literal['drex-v2.0', 'drex-max']
"""Pinned Nace model ids. `drex-v2.0` answers text. `drex-max` also accepts up to 8 images and at most 16 questions."""

NaceModelName = str | KnownNaceModelNames
"""Possible Nace model names.

Any other id, such as the moving alias `drex-latest`, is sent unchanged.
"""

_DREX_MAX_IMAGES = 8
"""Drex Max accepts at most this many images in one request. The endpoint shares the cap."""

_DEFAULT_TIMEOUT = 60.0
"""Seconds per request when `timeout` is not set, sent explicitly so a custom HTTP client cannot change it."""

_MAX_IMAGE_BYTES = 16 * 1024 * 1024
"""One image above this is refused. This adapter does not upload a workspace_file."""

_MAX_IMAGE_SET_BYTES = 32 * 1024 * 1024
"""A raw image set above this is refused before encoding."""

_MAX_IMAGE_BODY_BYTES = 4 * 1024 * 1024
"""An encoded image JSON body above this is refused."""


@dataclass(init=False)
class NaceModel(DecisionModel[AsyncNaceClient]):
    """The model class for Nace's Drex, a [decision model][pydantic_ai.models.decision.DecisionModel].

    Drex answers typed questions about a text, each with a confidence, rather than writing text. `drex-max` can also
    see up to 8 images. An agent whose job is to decide something runs on it like on any other model:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('nace:drex-v2.0', output_type=Handling)
    ```

    A 17th question to `drex-max` fails before the request is sent, and is not split into another request. The output
    type stays one model. `drex-v2.0` keeps the endpoint cap of 512 questions and rejects every image, so a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] can take that request.

    See [Decision models](https://pydantic.dev/docs/ai/models/decision/) for how an agent's output type and tools
    become questions.
    """

    # The endpoint caps. A profile cannot raise them. `drex-max`'s 16-question cap is narrower and is applied in
    # `_check_request_limits`, so the planner does not split an output type that does not fit.
    max_images: ClassVar[int | None] = _DREX_MAX_IMAGES
    max_questions: ClassVar[int | None] = NACE_MAX_QUESTIONS
    supports_image_input: ClassVar[bool] = True

    _model_name: NaceModelName = field(repr=False)
    _provider: Provider[AsyncNaceClient] = field(repr=False)

    def __init__(
        self,
        model_name: NaceModelName = 'drex-v2.0',
        *,
        provider: Literal['nace'] | Provider[AsyncNaceClient] = 'nace',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Nace model.

        Args:
            model_name: The name of the Nace model to use. Defaults to `drex-v2.0`. Open ids, including
                `drex-latest`, are sent unchanged.
            provider: The provider to use for authentication and API access.
            profile: The model profile to use. Defaults to one selected by the provider.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncNaceClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> NaceModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    def _question_limit(self) -> int:
        ceiling = DREX_MAX_QUESTIONS if self._model_name == 'drex-max' else NACE_MAX_QUESTIONS
        # A profile may lower the cap, never raise it.
        return min(cast(NaceModelProfile, self.profile).get('nace_max_questions', ceiling), ceiling)

    def _accepts_images(self) -> bool:
        return self._model_name == 'drex-max'

    def _check_request_limits(self, *, image_count: int = 0, question_count: int = 0) -> None:
        if image_count and not self._accepts_images():
            raise ModelAPIError(
                self.model_name,
                f'{self.model_name} does not accept images; got {image_count}. drex-max does.',
            )
        limit = self._question_limit()
        if question_count > limit:
            raise ModelAPIError(
                self.model_name, f'{self.model_name} accepts at most {limit} questions; got {question_count}.'
            )
        super()._check_request_limits(image_count=image_count, question_count=question_count)

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to Nace's Decisions API."""
        self._check_request_limits(image_count=len(request.images), question_count=len(request.questions))
        questions = {name: _to_nace_question(question) for name, question in request.questions.items()}
        images = _encode_images(request.images) if request.images else None
        timeout = model_settings.get('timeout')
        max_retries = model_settings.get('max_retries')
        try:
            response = await self.client.system_one(
                request.state,
                questions,
                model=self._model_name,
                images=images,
                retry=_retry_policy(max_retries),
                timeout=_DEFAULT_TIMEOUT if timeout is None else timeout,
                extra_headers=model_settings.get('extra_headers'),
                extra_body=cast('Mapping[str, JsonValue] | None', model_settings.get('extra_body')),
            )
        except NaceAPIResponseValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from Nace: {e}', str(e.body)) from e
        except NaceAPIError as e:
            raise ModelHTTPError.for_category(
                _model_errors.http_status_category(e.status, str(e)),
                status_code=e.status,
                model_name=self._model_name,
                body=e.body,
                headers=dict(e.headers),
            ) from e
        except NaceAPIConnectionError as e:
            raise _model_errors.connection_error(
                self._model_name, str(e), e, timeout=isinstance(e, NaceAPITimeoutError)
            ) from e
        except NaceError as e:
            raise UserError(f'Nace could not send this request: {e}') from e

        answers = {name: _from_nace_answer(answer) for name, answer in response.answers.items()}
        try:
            request_id = response.request_id
        except NaceError:
            request_id = None
        image_tokens = response.usage.image_tokens
        return DecisionResponse(
            answers=answers,
            model_name=response.model,
            usage=RequestUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
                details=None if image_tokens is None else {'image_tokens': image_tokens},
            ),
            provider_response_id=request_id,
        )


NaceStreamedResponse = DecisionStreamedResponse
"""A streamed Nace response is the decision model's one-event stream."""


def _retry_policy(max_retries: int | None) -> RetryPolicy:
    defaults = RetryPolicy()
    return RetryPolicy(
        max_retries=defaults.max_retries if max_retries is None else max_retries,
        respect_retry_after=True,
        http_statuses=set(defaults.http_statuses) | {429, 529},
        timeout=180.0,
    )


def _text(value: JsonValue, *, field: str) -> str:
    if isinstance(value, str):
        return value
    raise UserError(f'Nace {field} must be a string; got {type(value).__name__}.')


def _optional_text(value: JsonValue | None, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field)


def _instructions(value: JsonValue | None) -> str | None:
    """None stays None so the SDK omits the field. Any other value is sent as text."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _to_nace_question(question: DecisionQuestion) -> NaceNoul | NaceChoice | NaceScore:
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        sdk_criteria = None
        if criteria is not None:
            sdk_criteria = NaceNoulCriteria(
                true=_optional_text(criteria.true, field='noul criterion'),
                false=_optional_text(criteria.false, field='noul criterion'),
            )
        return NaceNoul(
            instructions=_instructions(question.instructions),
            criteria=sdk_criteria,
        )
    if isinstance(question, ChoiceQuestion):
        return NaceChoice(
            instructions=_instructions(question.instructions),
            criteria={
                label: _optional_text(description, field='choice description')
                for label, description in question.criteria.items()
            },
        )
    if isinstance(question, ScoreQuestion):
        return NaceScore(
            instructions=_instructions(question.instructions),
            criteria=[_text(level, field='score level') for level in question.criteria],
        )
    assert_never(question)


def _from_nace_answer(answer: object) -> DecisionAnswer:
    if isinstance(answer, NaceNoulAnswer):
        return NoulAnswer(noul=answer.noul)
    if isinstance(answer, NaceChoiceAnswer):
        return ChoiceAnswer(
            choice=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, NaceScoreAnswer):
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            probabilities={int(level): probability for level, probability in answer.probabilities.items()},
            legend={index: label for index, label in enumerate(answer.legend)},
        )
    raise UnexpectedModelBehavior(f'Unexpected answer from Nace: {answer!r}')


def _image_refused(reason: str) -> UserError:
    return UserError(f'{reason} This adapter does not upload; pass a workspace_file instead.')


def _encode_images(images: tuple[BinaryContent, ...]) -> list[str | dict[str, str]]:
    encoded: list[str | dict[str, str]] = []
    raw_total = 0
    for image in images:
        data = image.data
        size = len(data)
        if size > _MAX_IMAGE_BYTES:
            raise _image_refused(f'Nace image is {size} bytes; the limit is {_MAX_IMAGE_BYTES}.')
        raw_total += size
        if raw_total > _MAX_IMAGE_SET_BYTES:
            raise _image_refused(f'Nace images are {raw_total} bytes; the limit is {_MAX_IMAGE_SET_BYTES}.')
        try:
            encoded.append(image_input(data))
        except (NaceError, ValueError) as e:
            raise _image_refused(f'Nace could not accept this image: {e}') from e
    body_size = len(json.dumps(encoded).encode())
    if body_size > _MAX_IMAGE_BODY_BYTES:
        raise _image_refused(f'Nace image body is {body_size} bytes; the limit is {_MAX_IMAGE_BODY_BYTES}.')
    return encoded
