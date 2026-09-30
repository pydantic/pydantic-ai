from __future__ import annotations as _annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal, cast

from pydantic import JsonValue
from typing_extensions import assert_never

from .._http import to_httpx2_timeout
from .._warnings import PydanticAIDeprecationWarning
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import ModelRequestParameters
from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionHandOff,
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
    UnfillableRoute,
    UnsureRoute,
)

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice as TypeSafeChoice,
        ChoiceAnswer as TypeSafeChoiceAnswer,
        JSONContent,
        Noul as TypeSafeNoul,
        NoulAnswer as TypeSafeNoulAnswer,
        NoulCriteria as TypeSafeNoulCriteria,
        Score as TypeSafeScore,
        ScoreAnswer as TypeSafeScoreAnswer,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
        TypeSafeError,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install the `typesafe-sdk` package to use the TypeSafe model, '
        'you can use the `typesafe` optional group — `pip install "pydantic-ai-slim[typesafe]"`'
    ) from _import_error

__all__ = (
    'DecisionHandOff',
    'LatestTypeSafeModelNames',
    'TypeSafeModel',
    'TypeSafeModelName',
    'TypeSafeModelSettings',
    'TypeSafeStreamedResponse',
    'UnfillableRoute',
    'UnsureRoute',
)

LatestTypeSafeModelNames = Literal['jev-latest', 'jev-preview']
"""TypeSafe aliases, which move when a release ships. `jev-preview` runs ahead of `jev-latest` when there is a
preview build. A versioned id such as `jev-1.13.0` is accepted too, and is what to use once a confidence
threshold has been tuned against one. https://docs.typesafe.ai/models"""

TypeSafeModelName = str | LatestTypeSafeModelNames
"""Possible TypeSafe model names."""


class TypeSafeModelSettings(DecisionModelSettings, total=False):
    """Settings used for a TypeSafe model request."""

    typesafe_boolean_threshold: float
    """Deprecated: use `decision_boolean_threshold` instead."""

    typesafe_tool_call_threshold: float
    """Deprecated and ignored: the likeliest route is always taken.

    Use `decision_route_threshold` with a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to hand
    the picks the model is unsure of to a language model.
    """


TypeSafeStreamedResponse = DecisionStreamedResponse
"""Deprecated: use [`DecisionStreamedResponse`][pydantic_ai.models.decision.DecisionStreamedResponse] instead."""


@dataclass(init=False)
class TypeSafeModel(DecisionModel[AsyncTypeSafeClient]):
    """The model class for TypeSafe's Jev, a [decision model][pydantic_ai.models.decision.DecisionModel].

    Jev answers typed questions about a text, each with a confidence, rather than writing text. An agent whose job is
    to decide something runs on it like on any other model, with the `output_type` as the questions:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('typesafe:jev-latest', output_type=Handling)
    ...
    ```

    See [Decision models](https://pydantic.dev/docs/ai/models/decision/) for how an agent's output type and tools
    become questions, and [TypeSafe (Jev)](https://pydantic.dev/docs/ai/models/typesafe/) for setup, Jev's limits,
    and what it answers badly.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    # Jev picks from at most this many options in one question; a 256th is a 400 from the API.
    # https://docs.typesafe.ai/model-jaggedness/jev-1.13
    max_choice_options: ClassVar[int | None] = 255

    # Jev scores against at most this many rubric levels; an 11th is a 400 from the API.
    # https://docs.typesafe.ai/primitives/score
    max_score_levels: ClassVar[int | None] = 10

    _model_name: TypeSafeModelName = field(repr=False)
    _provider: Provider[AsyncTypeSafeClient] = field(repr=False)

    def __init__(
        self,
        model_name: TypeSafeModelName,
        *,
        provider: Literal['typesafe'] | Provider[AsyncTypeSafeClient] = 'typesafe',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a TypeSafe model.

        Args:
            model_name: The name of the TypeSafe model to use, such as `jev-latest`.
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
    def client(self) -> AsyncTypeSafeClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> TypeSafeModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        model_settings, model_request_parameters = super().prepare_request(model_settings, model_request_parameters)
        if not model_settings:
            return model_settings, model_request_parameters

        # TODO(v3): remove the deprecated `typesafe_*` threshold aliases
        settings = cast(TypeSafeModelSettings, model_settings.copy())
        # Settings reach the model deep inside a run, where no stack level points at the code that set them,
        # so the messages name the setting instead, like the other deprecated model settings.
        if 'typesafe_tool_call_threshold' in settings:
            # Not mapped to `decision_route_threshold`: that raises where this quietly filled the output, and
            # would fail the run of anyone without a model behind Jev to take the step.
            warnings.warn(
                '`typesafe_tool_call_threshold` is deprecated and ignored: the likeliest route is now always taken. '
                'To hand the picks Jev is unsure of to a language model instead, set `decision_route_threshold` and '
                'run Jev as `FallbackModel(jev, language_model)`.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            del settings['typesafe_tool_call_threshold']
        if 'typesafe_boolean_threshold' in settings:
            warnings.warn(
                '`typesafe_boolean_threshold` is deprecated; use `decision_boolean_threshold` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            old_value = settings.pop('typesafe_boolean_threshold')
            if 'decision_boolean_threshold' not in settings:
                if not 0 <= old_value <= 1:
                    raise UserError(f'`typesafe_boolean_threshold` must be between 0 and 1; got {old_value!r}.')
                settings['decision_boolean_threshold'] = old_value
        return settings, model_request_parameters

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to TypeSafe's Decisions API."""
        questions = {name: _to_typesafe_question(question) for name, question in request.questions.items()}
        timeout = model_settings.get('timeout')
        try:
            response = await self.client.system_one(
                cast(JSONContent, request.state),
                questions,
                model=self._model_name,
                timeout=None if timeout is None else to_httpx2_timeout(timeout),
                extra_headers=model_settings.get('extra_headers'),
                extra_body=cast('Mapping[str, JsonValue] | None', model_settings.get('extra_body')),
            )
        except TypeSafeAPIResponseValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from TypeSafe: {e}', str(e.body)) from e
        except TypeSafeAPIError as e:
            raise ModelHTTPError(
                status_code=e.status, model_name=self._model_name, body=e.body, headers=dict(e.headers)
            ) from e
        except TypeSafeAPIConnectionError as e:
            raise ModelAPIError(model_name=self._model_name, message=str(e)) from e
        except TypeSafeError as e:
            # What is left is the SDK refusing to send what it was given, such as an `extra_body` that will
            # not encode as JSON. That is the caller's to fix, not the model's.
            raise UserError(f'TypeSafe could not send this request: {e}') from e

        answers = {name: _from_typesafe_answer(answer) for name, answer in response.answers.items()}
        try:
            request_id = response.request_id
        except TypeSafeError:
            # The SDK raises rather than returning `None` when the response carries no request ID header.
            request_id = None
        return DecisionResponse(
            answers=answers,
            model_name=response.model,
            usage=RequestUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
            ),
            provider_response_id=request_id,
        )


def _to_typesafe_question(question: DecisionQuestion) -> TypeSafeNoul | TypeSafeChoice | TypeSafeScore:
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        sdk_criteria = None
        if criteria is not None:
            sdk_criteria = TypeSafeNoulCriteria(
                **cast(
                    dict[str, JSONContent],
                    {
                        key: value
                        for key, value in (('true', criteria.true), ('false', criteria.false))
                        if value is not None
                    },
                )
            )
        return TypeSafeNoul(instructions=cast('JSONContent | None', question.instructions), criteria=sdk_criteria)
    if isinstance(question, ChoiceQuestion):
        return TypeSafeChoice(
            instructions=cast('JSONContent | None', question.instructions),
            criteria=cast('Mapping[str, JSONContent | None]', question.criteria),
        )
    if isinstance(question, ScoreQuestion):
        return TypeSafeScore(
            instructions=cast('JSONContent | None', question.instructions),
            criteria=cast('list[JSONContent]', question.criteria),
        )
    assert_never(question)


def _from_typesafe_answer(answer: object) -> DecisionAnswer:
    if isinstance(answer, TypeSafeNoulAnswer):
        return NoulAnswer(noul=answer.noul)
    if isinstance(answer, TypeSafeChoiceAnswer):
        return ChoiceAnswer(
            choice=answer.choice,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
        )
    if isinstance(answer, TypeSafeScoreAnswer):
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            probabilities=dict(answer.probabilities),
            legend=dict(answer.legend),
        )
    raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe: {answer!r}')


# TODO(v3): remove the `ToolCallProposed` alias and this `__getattr__`.
def __getattr__(name: str) -> object:
    if name == 'ToolCallProposed':
        warnings.warn(
            '`ToolCallProposed` has been renamed to `UnfillableRoute`, which `pydantic_ai.models.decision` defines, '
            'and its `tool_name` to `route`. Update your imports; this deprecated alias will be removed in a future '
            'release.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return UnfillableRoute
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
