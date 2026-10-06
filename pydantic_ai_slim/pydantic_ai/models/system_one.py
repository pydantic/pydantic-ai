from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite
from typing import Annotated, Literal, assert_never, cast

import httpx2
from pydantic import Field, TypeAdapter, ValidationError

from .._http import to_httpx2_timeout
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.system_one import SystemOneProvider
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
    _wire,  # pyright: ignore[reportPrivateUsage]
)

__all__ = (
    'SystemOneModel',
    'SystemOneModelName',
    'SystemOneModelSettings',
)

SystemOneModelName = str
"""The name the API serves a model under, such as `clm-latest`."""


class SystemOneModelSettings(DecisionModelSettings, total=False):
    """Settings used for a System One API request."""

    # ALL FIELDS MUST BE `system_one_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future System One-specific settings.


@dataclass(init=False)
class SystemOneModel(DecisionModel[httpx2.AsyncClient]):
    """The model class for [decision models][pydantic_ai.models.decision.DecisionModel] served over the `/v1/systemone` API.

    Decision models such as [Contrastive Language Models](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) and
    [Laya](https://huggingface.co/convaiinnovations/laya) are available over this API, and an agent whose job is to
    decide something runs on one like on any other model, with the `output_type` as the questions:

    ```python
    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('system-one:clm-latest', output_type=Handling)
    ...
    ```

    See [Decision models](https://pydantic.dev/docs/ai/models/decision/) for how an agent's output type and tools
    become questions, and [System One API](https://pydantic.dev/docs/ai/models/system-one/) for connecting to one.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    # `max_choice_options` and `max_score_levels` stay `None`: limits belong to the model behind the URL, so they come
    # from the profile for the model name, and where it sets none the API refuses a request over them itself.

    _model_name: SystemOneModelName = field(repr=False)
    _provider: Provider[httpx2.AsyncClient] = field(repr=False)
    _api_key: str | None = field(repr=False)

    def __init__(
        self,
        model_name: SystemOneModelName,
        *,
        provider: Literal['system-one'] | SystemOneProvider = 'system-one',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a System One model.

        Args:
            model_name: The name the API serves the model under, such as `clm-latest`.
            provider: The provider to use for the API's URL and key.
            profile: The model profile to use. Defaults to one selected by the provider.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = SystemOneProvider()
        self._provider = provider
        self._api_key = provider.api_key
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> httpx2.AsyncClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> SystemOneModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:  # noqa: C901
        """Send one request to the `/v1/systemone` endpoint."""
        body: dict[str, object] = {
            'state': request.state,
            'model': self._model_name,
            'questions': {name: _wire(question) for name, question in request.questions.items()},
        }
        if (temperature := model_settings.get('temperature')) is not None:
            body['temperature'] = temperature
        if (extra_body := model_settings.get('extra_body')) is not None:
            if not isinstance(extra_body, Mapping):
                raise UserError(f'`extra_body` must be a mapping to send it to the System One API; got {extra_body!r}.')
            body.update(cast('Mapping[str, object]', extra_body))
        questions = request.questions
        if extra_body is not None and 'questions' in extra_body:
            try:
                questions = _questions_adapter.validate_python(body['questions'])
            except ValidationError as e:
                raise UserError(f'Cannot validate the `extra_body.questions` override: {e}') from e

        timeout = model_settings.get('timeout')
        url = f'{self.base_url}/systemone' if self.base_url.endswith('/v1') else f'{self.base_url}/v1/systemone'
        try:
            headers = httpx2.Headers(model_settings.get('extra_headers') or {})
            if self._api_key is not None and 'Authorization' not in headers:
                headers['Authorization'] = f'Bearer {self._api_key}'
            http_request = self.client.build_request(
                'POST',
                url,
                json=body,
                headers=headers,
                timeout=httpx2.USE_CLIENT_DEFAULT if timeout is None else to_httpx2_timeout(timeout),
            )
        except (TypeError, ValueError) as e:
            # An `extra_body` that will not encode as JSON is the caller's to fix, not the model's.
            raise UserError(f'Could not send this request to the System One API: {e}') from e
        try:
            response = await self.client.send(http_request)
        except httpx2.TransportError as e:
            raise ModelAPIError(model_name=self._model_name, message=f'{type(e).__name__}: {e}') from e

        if response.is_error:
            raise ModelHTTPError(
                status_code=response.status_code,
                model_name=self._model_name,
                body=_error_body(response),
                headers=dict(response.headers),
            )
        try:
            parsed = _response_adapter.validate_json(response.content)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from the System One API: {e}', response.text) from e
        if parsed.answers.keys() != questions.keys():
            raise UnexpectedModelBehavior(
                'Invalid response from the System One API: answer names do not match the questions', response.text
            )
        for name, question in questions.items():
            answer = parsed.answers[name]
            if isinstance(question, NoulQuestion):
                valid = isinstance(answer, NoulAnswer) and isfinite(answer.noul) and 0 <= answer.noul <= 1
            elif isinstance(question, ChoiceQuestion):
                valid = (
                    isinstance(answer, ChoiceAnswer)
                    and answer.choice in question.criteria
                    and answer.probabilities.keys() == question.criteria.keys()
                )
            elif isinstance(question, ScoreQuestion):
                levels = set(range(len(question.criteria)))
                valid = (
                    isinstance(answer, ScoreAnswer)
                    and answer.probabilities.keys() == levels
                    and (not answer.legend or answer.legend.keys() == levels)
                    and isfinite(answer.score)
                    and 0 <= answer.score <= len(question.criteria) - 1
                )
            else:
                assert_never(question)
            if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
                probabilities = answer.probabilities.values()
                # Jev displays probabilities to two decimal places, so each may differ by half a unit.
                valid = (
                    valid
                    and isfinite(answer.confidence)
                    and 0 <= answer.confidence <= 1
                    and all(isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    and abs(sum(probabilities) - 1) <= 1e-6 + len(answer.probabilities) * 0.005
                )
                if valid and isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
                    # A displayed score and its probabilities may each be rounded. Check whether any distribution
                    # within their rounding intervals could produce that score, using at least Jev's two decimals.
                    values: list[Decimal] = [
                        Decimal(str(answer.probabilities[level])) for level in range(len(question.criteria))
                    ]
                    half_units: list[Decimal] = [
                        Decimal(1).scaleb(-max(2, -int(value.as_tuple().exponent))) / 2 for value in values
                    ]
                    lower: list[Decimal] = [
                        max(Decimal(0), value - half_unit) for value, half_unit in zip(values, half_units)
                    ]
                    upper: list[Decimal] = [
                        min(Decimal(1), value + half_unit) for value, half_unit in zip(values, half_units)
                    ]
                    remaining = Decimal(1) - sum(lower, Decimal(0))
                    valid = 0 <= remaining <= sum((high - low for low, high in zip(lower, upper)), Decimal(0))
                    if valid:
                        bounds: list[Decimal] = []
                        for levels in (range(len(values)), reversed(range(len(values)))):
                            rest = remaining
                            mean = sum((level * low for level, low in enumerate(lower)), Decimal(0))
                            for level in levels:
                                taken = min(rest, upper[level] - lower[level])
                                mean += level * taken
                                rest -= taken
                            bounds.append(mean)
                        score = Decimal(str(answer.score))
                        score_decimals = max(2, -int(score.as_tuple().exponent))
                        score_half_unit = Decimal(1).scaleb(-score_decimals) / 2
                        valid = score + score_half_unit >= bounds[0] and score - score_half_unit <= bounds[1]
            if not valid:
                raise UnexpectedModelBehavior(
                    f'Invalid response from the System One API: answer {name!r} does not match its question: {answer!r}',
                    response.text,
                )
        return DecisionResponse(
            answers=parsed.answers,
            model_name=parsed.model,
            usage=RequestUsage(input_tokens=parsed.usage.input_tokens, output_tokens=parsed.usage.output_tokens),
        )


@dataclass(kw_only=True)
class _Usage:
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0


@dataclass(kw_only=True)
class _SystemOneResponse:
    """The body the API answers `/v1/systemone` with."""

    model: str
    answers: dict[str, Annotated[DecisionAnswer, Field(discriminator='type')]]
    usage: _Usage


_response_adapter = TypeAdapter(_SystemOneResponse)
_questions_adapter = TypeAdapter(dict[str, Annotated[DecisionQuestion, Field(discriminator='type')]])


def _error_body(response: httpx2.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text
