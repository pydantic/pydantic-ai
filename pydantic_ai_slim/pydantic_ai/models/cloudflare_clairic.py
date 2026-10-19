from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, GetJsonSchemaHandler, JsonValue, TypeAdapter, ValidationError
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

from .._http import to_httpx2_timeout
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import ModelMessage, ModelResponse, TextPart
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.cloudflare import CloudflareProvider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import Model, ModelRequestParameters, _cloudflare_clairic, _system_one, check_allow_model_requests

__all__ = (
    'CloudflareClairicModel',
    'CloudflareClairicModelName',
    'CloudflareClairicModelSettings',
    'ClairicField',
    'ClairicNumberEstimate',
    'ClairicDateEstimate',
    'ClairicForecast',
)


@dataclass(frozen=True, kw_only=True)
class ClairicField:
    """Native Clairic controls for an `Annotated` output field.

    The Python type determines the native field type. Use `kind='duration'` on a number
    to retain calendar units such as months or years. Describe the task with
    `Field(description=...)`. Estimates require `min` and `max`; forecasts also require
    `horizon` and `interval`. Controls incompatible with the result type fail before HTTP.
    """

    kind: Literal['duration'] | None = None
    """Use duration semantics for a numeric result. Other native kinds follow the Python type."""
    unit: str | None = None
    """Unit label. Durations accept seconds, minutes, hours, days, weeks, months or years."""
    integer: bool | None = None
    """Integer precision for scalar numbers and numeric durations only."""
    decimals: int | None = None
    """Decimal places, from zero to six, for numbers, durations and number lists."""
    min: int | float | date | None = None
    """Lower estimate or forecast bound; dates are supported for date estimates."""
    max: int | float | date | None = None
    """Upper estimate or forecast bound, of the same type as `min`."""
    step: int | float | Literal['month', 'year'] | None = None
    """Positive grid spacing; date estimates also accept calendar months or years."""
    scale: Literal['linear', 'log'] | None = None
    """Estimate or forecast grid scale."""
    null_threshold: float | None = None
    """Scalar extraction null probability threshold, greater than zero and at most one."""
    horizon: int | None = None
    """Number of forecast steps, from one to 64."""
    interval: Literal['step', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'] | None = None
    """Forecast step interval."""

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        if 'x-cloudflare-clairic' in json_schema:
            raise UserError('Use one `ClairicField` marker per field.')
        return {
            **json_schema,
            'x-cloudflare-clairic': _clairic_field_adapter.dump_python(self, mode='json', exclude_none=True),
        }


_clairic_field_adapter = TypeAdapter(ClairicField)


class CloudflareClairicModelSettings(ModelSettings, total=False):
    """Request-wide Clairic options, merged with ordinary model settings."""

    cloudflare_clairic_reference_date: date | str
    """ISO date anchoring dates written without a year."""
    cloudflare_clairic_include_quantiles: bool
    """Request 99 percentiles on estimates and each forecast step. Defaults to false."""
    cloudflare_clairic_return_state_text: bool
    """Request the native source text used by spans. Defaults to true in this adapter."""


class ClairicNumberEstimate(BaseModel):
    """Native numeric estimate, also used for each forecast step.

    `value` and `median` are independent provider predictions. Extra native fields are
    retained. Use `ClairicField(min=..., max=...)` when requesting an estimate.
    """

    model_config = ConfigDict(extra='allow', json_schema_extra={'x-cloudflare-clairic-result': 'number-estimate'})
    value: float
    """Native point prediction, preserved independently of the median."""
    mean: float
    """Native distribution mean."""
    median: float
    """Native distribution median."""
    p10: float
    """Tenth percentile."""
    p90: float
    """Ninetieth percentile."""
    quantiles: list[float] | None = None
    """Percentiles 1 through 99, when requested."""
    p_below: float | None = None
    """Native probability below the requested lower bound, when returned."""
    p_above: float | None = None
    """Native probability above the requested upper bound, when returned."""


class ClairicDateEstimate(BaseModel):
    """Native date distribution. Configure date bounds with `ClairicField`."""

    model_config = ConfigDict(extra='allow', json_schema_extra={'x-cloudflare-clairic-result': 'date-estimate'})
    value: date
    """Native point prediction, which can differ from `median`."""
    mean: date
    """Native mean date."""
    median: date
    """Native median date."""
    p10: date
    """Tenth percentile date."""
    p90: date
    """Ninetieth percentile date."""
    quantiles: list[date] | None = None
    """Percentiles 1 through 99, when requested."""
    p_below: float | None = None
    """Probability below the lower bound, when returned; date estimates can omit it."""
    p_above: float | None = None
    """Probability above the upper bound, when returned; date estimates can omit it."""


class ClairicForecast(BaseModel):
    """Native numeric forecast with independent point and percentile arrays.

    Configure bounds, horizon and interval with `ClairicField`. Each step retains its
    complete distribution; forecast steps need no native `type` discriminator.
    """

    model_config = ConfigDict(extra='allow', json_schema_extra={'x-cloudflare-clairic-result': 'forecast'})
    value: list[float]
    """Native point predictions, independent of `median`."""
    median: list[float]
    """Native median for each step."""
    p10: list[float]
    """Native tenth percentile for each step."""
    p90: list[float]
    """Native ninetieth percentile for each step."""
    steps: list[ClairicNumberEstimate]
    """Full native estimate for each step, including optional quantiles and tail probabilities."""


CloudflareClairicModelName = Literal['@cf/cloudflare/clairic'] | str
"""The name of a Clairic model served by Cloudflare Workers AI."""


@dataclass(init=False)
class CloudflareClairicModel(Model[httpx2.AsyncClient]):
    """Extract, compute, estimate and forecast typed values with Cloudflare Clairic.

    Use one structured output object. Python scalar and list types infer native field
    types; `ClairicField` adds native controls, and `ClairicNumberEstimate`,
    `ClairicDateEstimate` and `ClairicForecast` expose complete distributions in output.
    Native nulls remain null. Tools, free-text generation and streaming are unsupported.

    `provider_details` retains all native answers and uncertainty, `field_names` maps
    wire IDs to output aliases, and `request_state` is the exact text sent. Spans index
    `state_text`, falling back to `request_state` when the provider omits it.
    A single user text is sent verbatim; multipart history uses role/content JSON.

    `timeout`, `extra_headers` and `extra_body` are forwarded. Request structure cannot
    be replaced through `extra_body`. Generic sampling settings are ignored.
    See [Cloudflare configuration](../../models/cloudflare.md) for examples and limits.
    """

    _model_name: CloudflareClairicModelName = field(repr=False)
    _provider: Provider[httpx2.AsyncClient] = field(repr=False)
    _api_key: str = field(repr=False)

    def __init__(
        self,
        model_name: CloudflareClairicModelName = '@cf/cloudflare/clairic',
        *,
        provider: Literal['cloudflare'] | CloudflareProvider = 'cloudflare',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize Clairic with the same authentication and client configuration as Clef.

        Args:
            model_name: The Workers AI model ID.
            provider: The provider owning authentication, account URL, and HTTP client.
            profile: Model profile overrides. Defaults to the provider's profile.
            settings: Default model settings.
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
    def model_name(self) -> CloudflareClairicModelName:
        return self._model_name

    @property
    def system(self) -> str:
        return self._provider.name

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        if model_request_parameters.output_object is not None and model_request_parameters.output_mode in (
            'auto',
            'native',
        ):
            model_request_parameters = replace(
                model_request_parameters,
                output_mode='native',
                allow_text_output=False,
            )
        return super().prepare_request(model_settings, model_request_parameters)

    async def request(  # noqa: C901
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        if model_request_parameters.function_tools or model_request_parameters.native_tools:
            raise UserError('Cloudflare Clairic does not support function or native tools.')
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        settings = CloudflareClairicModelSettings(**(model_settings or {}))
        if params.output_mode != 'native' or params.output_object is None:
            raise UserError('Cloudflare Clairic requires one native structured output.')
        fields, field_names = _cloudflare_clairic.clairic_fields(
            params.output_object, self._get_instruction_parts(messages, params) or []
        )
        state, images = await _cloudflare_clairic.clairic_state(self.prepare_messages(messages, params))
        options = {
            'return_state_text': settings.get('cloudflare_clairic_return_state_text', True),
            'include_quantiles': settings.get('cloudflare_clairic_include_quantiles', False),
        }
        body: dict[str, object] = {
            'model': 'clairic',
            'state': state,
            'fields': fields,
            'options': options,
        }
        if images:
            body['images'] = images
        if (reference_date := settings.get('cloudflare_clairic_reference_date')) is not None:
            body['reference_date'] = reference_date.isoformat() if isinstance(reference_date, date) else reference_date
        if (extra_body := settings.get('extra_body')) is not None:
            try:
                overrides = _extra_body_adapter.validate_python(extra_body)
            except ValidationError as e:
                raise UserError(f'Invalid Cloudflare Clairic `extra_body`: {e}') from e
            if overrides.keys() & {'model', 'state', 'fields', 'images', 'options'}:
                raise UserError(
                    'Cloudflare Clairic `extra_body` cannot override `model`, `state`, `fields`, `images`, or `options`.'
                )
            body.update(overrides)
        timeout = settings.get('timeout')
        try:
            headers = httpx2.Headers(settings.get('extra_headers') or {})
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
        try:
            response = await self.client.send(http_request)
        except httpx2.TransportError as e:
            raise ModelAPIError(model_name=self._model_name, message=f'{type(e).__name__}: {e}') from e
        if response.is_error:
            raise ModelHTTPError(
                status_code=response.status_code,
                model_name=self._model_name,
                body=_system_one.error_body(response),
                headers=dict(response.headers),
            )
        try:
            parsed = _clairic_response_adapter.validate_json(response.content)
        except ValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from Cloudflare Clairic: {e}', response.text) from e
        result = parsed.result
        if result.answers.keys() != fields.keys():
            raise UnexpectedModelBehavior(
                'Cloudflare Clairic answer names do not match requested fields.', response.text
            )
        values: dict[str, JsonValue] = {}
        for wire_name, answer in result.answers.items():
            if answer.type != fields[wire_name]['type']:
                raise UnexpectedModelBehavior(
                    f'Cloudflare Clairic returned an unexpected type for {wire_name!r}.', response.text
                )
            values[field_names[wire_name]] = (
                answer.model_dump(mode='json', exclude_unset=True)
                if answer.type in ('estimate', 'forecast')
                else answer.value
            )
        details = result.model_dump(mode='json', exclude_unset=True)
        details['field_names'] = field_names
        details['request_state'] = state
        details.setdefault('state_text', state)
        return ModelResponse(
            parts=[TextPart(json.dumps(values, ensure_ascii=False))],
            model_name=self._model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_response_id=response.headers.get('cf-ai-req-id'),
            provider_details=details,
            usage=RequestUsage(
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                details={
                    key: value
                    for key, value in result.usage.model_dump().items()
                    if key not in ('input_tokens', 'output_tokens') and isinstance(value, int)
                },
            ),
            finish_reason='length' if result.truncated else 'stop',
        )


class _ClairicAnswer(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)
    type: str
    value: JsonValue


class _ClairicUsage(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)
    input_tokens: int
    output_tokens: int


class _ClairicResult(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)
    model: str
    answers: dict[str, _ClairicAnswer]
    truncated: bool
    usage: _ClairicUsage
    state_text: str = ''


class _ClairicResponse(BaseModel):
    result: _ClairicResult


_clairic_response_adapter = TypeAdapter(_ClairicResponse)

_extra_body_adapter = TypeAdapter(dict[str, object])
