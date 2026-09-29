from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Literal, cast

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
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
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

    Decision models such as [Contrastive Language Models](https://github.com/Contrastive-LM/CLM) and
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

    # `max_choice_options` and `max_score_levels` stay `None`: they are the API's to enforce, and it refuses a
    # request over its limits with an error response.

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

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
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

        headers = dict(model_settings.get('extra_headers') or {})
        if self._api_key is not None:
            headers.setdefault('Authorization', f'Bearer {self._api_key}')
        timeout = model_settings.get('timeout')
        try:
            http_request = self.client.build_request(
                'POST',
                f'{self.base_url}/v1/systemone',
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
        return DecisionResponse(
            answers=parsed.answers,
            model_name=parsed.model,
            usage=RequestUsage(input_tokens=parsed.usage.input_tokens, output_tokens=parsed.usage.output_tokens),
        )


@dataclass(kw_only=True)
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(kw_only=True)
class _SystemOneResponse:
    """The body the API answers `/v1/systemone` with."""

    model: str
    answers: dict[str, Annotated[DecisionAnswer, Field(discriminator='type')]]
    usage: _Usage


_response_adapter = TypeAdapter(_SystemOneResponse)


def _error_body(response: httpx2.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text
