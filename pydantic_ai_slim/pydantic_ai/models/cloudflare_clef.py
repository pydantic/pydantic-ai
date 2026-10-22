from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import httpx2
from pydantic import TypeAdapter, ValidationError

from .._http import to_httpx2_timeout
from ..exceptions import UnexpectedModelBehavior, UserError
from ..profiles import ModelProfile, ModelProfileSpec, merge_profile
from ..profiles.decision import decision_model_profile
from ..providers import Provider
from ..providers._cloudflare_http import send_request
from ..providers.cloudflare import CloudflareProvider
from ..settings import ModelSettings
from . import _system_one
from .decision import (
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    _wire,  # pyright: ignore[reportPrivateUsage]
)

__all__ = ('CloudflareClefModel', 'CloudflareClefModelName')

CloudflareClefModelName = Literal['@cf/cloudflare/clef', '@cf/cloudflare/clef-flash'] | str
"""The name of a Clef model served by Cloudflare Workers AI."""


@dataclass(init=False)
class CloudflareClefModel(DecisionModel[httpx2.AsyncClient]):
    """A [Clef](https://developers.cloudflare.com/workers-ai/models/clef/) decision model on Cloudflare Workers AI.

    See [Cloudflare configuration](../../models/cloudflare.md) for authentication, supported models, and images.
    The adapter forwards `timeout`, `extra_headers`, and `extra_body`. Generic sampling settings, including
    `temperature`, are ignored because Clef does not accept them.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    supports_image_input: ClassVar[bool] = True
    max_images: ClassVar[int | None] = 4
    max_questions: ClassVar[int | None] = 64
    requires_instructions: ClassVar[bool] = True

    unsupported_settings = DecisionModel.unsupported_settings | {'max_retries'}

    _model_name: CloudflareClefModelName = field(repr=False)
    _provider: Provider[httpx2.AsyncClient] = field(repr=False)
    _api_key: str = field(repr=False)

    def __init__(
        self,
        model_name: CloudflareClefModelName,
        *,
        provider: Literal['cloudflare'] | CloudflareProvider = 'cloudflare',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Cloudflare Clef model.

        Args:
            model_name: The full Workers AI model ID, such as `@cf/cloudflare/clef-flash`.
            provider: The provider to use for authentication, account URL, and HTTP client.
            profile: The model profile to use. Defaults to one selected by the provider.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = CloudflareProvider()
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
    def model_name(self) -> CloudflareClefModelName:
        """The Workers AI model ID."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        # Explicit decision construction must not inherit the unknown-ID chat fallback.
        return merge_profile(super()._instance_profile_defaults(profile), decision_model_profile(self.model_name))

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the account's Workers AI run endpoint."""
        self._check_request_limits(image_count=len(request.images), question_count=len(request.questions))
        body: dict[str, object] = {
            'model': self._model_name.rsplit('/', 1)[-1],
            'state': request.state,
            'questions': {name: _wire(question) for name, question in request.questions.items()},
        }
        if request.images:
            for image in request.images:
                if image.media_type not in ('image/png', 'image/jpeg', 'image/webp'):
                    raise UserError(f'Cloudflare Clef supports PNG, JPEG, and WebP images; got {image.media_type!r}.')
            body['images'] = [image.data_uri for image in request.images]

        questions = request.questions
        image_count = len(request.images)
        if (extra_body := model_settings.get('extra_body')) is not None:
            try:
                overrides = _extra_body_adapter.validate_python(extra_body)
                body.update(overrides)
                if 'questions' in overrides:
                    questions = _system_one.questions_adapter.validate_python(body['questions'])
                if 'images' in overrides:
                    image_count = len(_images_adapter.validate_python(body['images']))
            except (ValidationError, TypeError) as e:
                raise UserError(f'Cannot validate the Cloudflare `extra_body` override: {e}') from e
        self._check_request_limits(image_count=image_count, question_count=len(questions))

        question_names: dict[str, str] = {}
        if any(re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', name) is None for name in questions):
            # Field aliases and route labels can exceed Cloudflare's question ID restrictions.
            question_names = {f'q{index}': name for index, name in enumerate(questions)}
            questions = {wire_name: questions[name] for wire_name, name in question_names.items()}
            body['questions'] = {name: _wire(question) for name, question in questions.items()}

        timeout = model_settings.get('timeout')
        try:
            headers = httpx2.Headers(model_settings.get('extra_headers') or {})
            if 'Authorization' not in headers:
                headers['Authorization'] = f'Bearer {self._api_key}'
            http_request = self.client.build_request(
                'POST',
                f'{self.base_url}/ai/run/{self._model_name}',
                json=body,
                headers=headers,
                timeout=httpx2.USE_CLIENT_DEFAULT if timeout is None else to_httpx2_timeout(timeout),
            )
        except (TypeError, ValueError) as e:
            raise UserError(f'Could not send this request to Cloudflare Workers AI: {e}') from e
        response = await send_request(self.client, http_request, model_name=self._model_name)
        try:
            parsed = _response_adapter.validate_json(response.content)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from Cloudflare Workers AI: {e}', response.text) from e
        result = _system_one.validate_response(
            parsed.result,
            questions,
            api_name='Cloudflare Workers AI',
            response_text=response.text,
            provider_response_id=response.headers.get('cf-ai-req-id'),
        )
        if question_names:
            result.answers = {question_names[name]: answer for name, answer in result.answers.items()}
        return result


@dataclass(kw_only=True)
class _CloudflareResponse:
    result: _system_one.Response


_response_adapter = TypeAdapter(_CloudflareResponse)
_extra_body_adapter = TypeAdapter(dict[str, object])
_images_adapter = TypeAdapter(list[str])
