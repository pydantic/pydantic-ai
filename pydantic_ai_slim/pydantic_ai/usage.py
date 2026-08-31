from __future__ import annotations as _annotations

import dataclasses
import warnings
from copy import copy
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from typing import Annotated, Any, cast

from genai_prices.data_snapshot import get_snapshot
from pydantic import AliasChoices, BeforeValidator, Field, GetCoreSchemaHandler, TypeAdapter
from pydantic_core import SchemaSerializer, core_schema

from . import _utils
from ._warnings import CostNotFoundWarning
from .exceptions import UsageLimitExceeded

__all__ = 'RequestUsage', 'RunUsage', 'UsageLimits'

_FIRST_CLASS_TOKEN_DETAIL_KEYS = frozenset({'input_tokens', 'output_tokens'})
"""`details` keys whose names collide with the first-class `gen_ai.usage.{input,output}_tokens`
attributes. They must never be emitted under `gen_ai.usage.details.*` too: doing so reports the same
conceptual quantity under two attributes that consumers like Langfuse then sum, double-counting tokens
and cost. Adapters that stash these keys in `details` (e.g. Anthropic's streaming carry-forward, Cohere's
billed units) keep them accessible on `RequestUsage.details`; only the ambiguous OTel emission is dropped."""

_LEGACY_USAGE_KEYS = frozenset({'requests', 'request_tokens', 'response_tokens', 'total_tokens'})
"""Keys accepted in stored usage data for backwards compatibility but not preserved as arbitrary fields."""

_LEGACY_TOKEN_ALIASES = (('input_tokens', 'request_tokens'), ('output_tokens', 'response_tokens'))


@cache
def _usage_serializer(usage_type: type[object]) -> SchemaSerializer:
    return TypeAdapter(usage_type).serializer


class _UsageSerializerDescriptor:
    def __get__(self, instance: object, owner: type[object]) -> SchemaSerializer:
        return _usage_serializer(owner)


def _serialize_usage(
    value: UsageBase,
    inner: core_schema.SerializerFunctionWrapHandler,
    info: core_schema.SerializationInfo,
    *,
    reserved_names: frozenset[str],
    extra_serializer: SchemaSerializer,
) -> dict[str, Any]:
    serialized = inner(value)
    assert isinstance(serialized, dict)
    result = cast(dict[str, Any], serialized).copy()
    extra = {
        key: item
        for key, item in value.__dict__.items()
        if key not in reserved_names and (item is not None or not info.exclude_none)
    }
    extra = cast(
        dict[str, Any],
        extra_serializer.to_python(
            extra,
            # Apply selectors without consuming JSON fallback and warning handling from the outer serializer.
            mode='python',
            include=cast(Any, info.include),
            exclude=cast(Any, info.exclude),
            by_alias=info.by_alias,
            exclude_unset=info.exclude_unset,
            exclude_defaults=info.exclude_defaults,
            exclude_none=info.exclude_none,
            exclude_computed_fields=info.exclude_computed_fields,
            round_trip=info.round_trip,
            serialize_as_any=info.serialize_as_any,
            context=info.context,
        ),
    )
    result.update(extra)
    return result


@dataclass(repr=False, init=False, eq=False)
class UsageBase:
    # Bare `pydantic_core.to_json()` looks for this attribute but does not build custom core schemas for stdlib
    # dataclasses. The descriptor builds the same serializer as `TypeAdapter` for each concrete usage class.
    __pydantic_serializer__ = _UsageSerializerDescriptor()

    input_tokens: Annotated[
        int,
        # `request_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('input_tokens', 'request_tokens')),
    ] = 0
    """Total number of input/prompt tokens, across all modalities.

    Token counts form inclusive parent/child buckets, not disjoint ones: this total includes cached
    tokens (`cache_read_tokens`, `cache_write_tokens`) and audio tokens (`input_audio_tokens`).
    Usage extraction normalizes providers that report these separately (e.g. Anthropic and Bedrock,
    whose raw `input_tokens` exclude cache reads/writes) so the convention holds everywhere.
    """

    cache_write_tokens: int = 0
    """Number of tokens written to the cache. Included in `input_tokens`."""
    cache_read_tokens: int = 0
    """Number of tokens read from the cache, across all modalities (includes `cache_audio_read_tokens`).

    Included in `input_tokens`.
    """

    output_tokens: Annotated[
        int,
        # `response_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('output_tokens', 'response_tokens')),
    ] = 0
    """Number of output/completion tokens."""

    input_audio_tokens: int = 0
    """Number of audio input tokens. Included in `input_tokens`."""
    cache_audio_read_tokens: int = 0
    """Number of audio tokens read from the cache. Included in `cache_read_tokens` and `input_audio_tokens`."""
    output_audio_tokens: int = 0
    """Number of audio output tokens. Included in `output_tokens`."""

    details: Annotated[
        dict[str, int],
        # `details` can not be `None` any longer, but we still want to support deserializing model responses stored in a DB before this was changed
        BeforeValidator(lambda d: d or {}),
    ] = dataclasses.field(default_factory=dict[str, int])
    """Any extra details returned by the model."""

    cost: Decimal | None = None
    """Best-effort cost in USD, or `None` if no cost could be determined.

    Calculated with [genai-prices](https://github.com/pydantic/genai-prices). `None` (rather than zero) when the
    model or provider can't be priced, so "unknown" stays distinguishable from a genuine zero cost.
    """

    def __init__(self, *, details: dict[str, int] | None = None, **kwargs: Any):
        self.details = details or {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Preserve arbitrary usage fields across Pydantic serialization."""
        schema = handler(source_type)
        field_names = frozenset(field.name for field in dataclasses.fields(source_type))
        reserved_names = field_names | frozenset(dir(source_type)) | _LEGACY_USAGE_KEYS
        extra_serializer = SchemaSerializer(core_schema.any_schema())

        def validate(value: Any, inner: core_schema.ValidatorFunctionWrapHandler) -> UsageBase:
            if isinstance(value, dict):
                value_dict = cast(dict[str, Any], value)
                input_value = value_dict.copy()
                if not value_dict.get('details'):
                    input_value['details'] = {}
                for field_name, legacy_name in _LEGACY_TOKEN_ALIASES:
                    if field_name not in value_dict and legacy_name in value_dict and value_dict[legacy_name] is None:
                        input_value[legacy_name] = 0
            else:
                value_dict = None
                input_value = cast(object, value)

            result = inner(input_value)
            assert isinstance(result, UsageBase)
            if value_dict is not None:
                for key, item in value_dict.items():
                    if key not in reserved_names:
                        setattr(result, key, item)
            return result

        def serialize(
            value: UsageBase,
            inner: core_schema.SerializerFunctionWrapHandler,
            info: core_schema.SerializationInfo,
        ) -> Any:
            return _serialize_usage(
                value,
                inner,
                info,
                reserved_names=reserved_names,
                extra_serializer=extra_serializer,
            )

        return core_schema.no_info_wrap_validator_function(
            validate,
            schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(serialize, info_arg=True, schema=schema),
        )

    def __copy__(self) -> UsageBase:
        """Shallow copy that also copies mutable fields like `details`."""
        cls = type(self)
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        new.details = self.details.copy()
        return new

    @property
    def total_tokens(self) -> int:
        """Sum of `input_tokens + output_tokens`."""
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of input tokens that were read from the provider's prompt cache.

        Computed as `cache_read_tokens / input_tokens`. Both counts span all modalities — cached audio tokens are
        included in `cache_read_tokens` just as audio input tokens are included in `input_tokens` — and
        `input_tokens` includes cached reads for every provider, so the ratio is comparable across providers:
        `0.0` means no prompt-cache hits, while values approaching `1.0` mean nearly the entire prompt was served
        from cache. Returns `0.0` when there are no input tokens.

        On [`RequestUsage`][pydantic_ai.usage.RequestUsage] this is the hit ratio of a single request; on
        [`RunUsage`][pydantic_ai.usage.RunUsage] it aggregates all requests in the run.
        """
        return self.cache_read_tokens / self.input_tokens if self.input_tokens else 0.0

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token usage values as OpenTelemetry attributes."""
        result: dict[str, int] = {}
        if self.input_tokens:
            result['gen_ai.usage.input_tokens'] = self.input_tokens
        if self.output_tokens:
            result['gen_ai.usage.output_tokens'] = self.output_tokens

        details = self.details.copy()
        if self.cache_write_tokens:
            result['gen_ai.usage.cache_creation.input_tokens'] = self.cache_write_tokens
            # For backwards compat
            details['cache_write_tokens'] = self.cache_write_tokens
        if self.cache_read_tokens:
            result['gen_ai.usage.cache_read.input_tokens'] = self.cache_read_tokens
            # For backwards compat
            details['cache_read_tokens'] = self.cache_read_tokens
        if self.input_audio_tokens:
            details['input_audio_tokens'] = self.input_audio_tokens
        if self.cache_audio_read_tokens:
            details['cache_audio_read_tokens'] = self.cache_audio_read_tokens
        if self.output_audio_tokens:
            details['output_audio_tokens'] = self.output_audio_tokens
        if details:
            prefix = 'gen_ai.usage.details.'
            for key, value in details.items():
                # Never emit a `details` entry whose name collides with a first-class token attribute: the
                # value is already reported as `gen_ai.usage.{input,output}_tokens`, and emitting it again
                # under `gen_ai.usage.details.*` makes consumers like Langfuse sum the two and double-count.
                if key in _FIRST_CLASS_TOKEN_DETAIL_KEYS:
                    continue
                # Zero is a meaningful value, but a `None` would be an invalid OTel attribute value.
                # Provider data can contain None despite the annotation.
                if value is not None:  # pyright: ignore[reportUnnecessaryComparison]
                    result[prefix + key] = value
        return result

    def __repr__(self):
        kv_pairs = (f'{name}={value!r}' for name, value in sorted(self.__dict__.items()) if value)
        return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'

    def __eq__(self, value: object, /) -> bool:
        if type(self) is type(value):
            missing = object()
            keys = self.__dict__.keys() | value.__dict__.keys()
            return all(getattr(self, key, missing) == getattr(value, key, missing) for key in keys)
        return NotImplemented

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return any(self.details.values()) or any(v for k, v in self.__dict__.items() if k != 'details')


@dataclass(repr=False, init=False, eq=False)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request using [genai-prices](https://github.com/pydantic/genai-prices).
    """

    @property
    def requests(self):
        return 1

    def incr(self, incr_usage: RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        _incr_usage_tokens(self, incr_usage)
        _incr_usage_cost(self, incr_usage)

    def __add__(self, other: RequestUsage) -> RequestUsage:
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> RequestUsage:
        """Extract usage information from the response data using genai-prices.

        Args:
            data: The response data from the model API.
            provider: The actual provider ID
            provider_url: The provider base_url
            provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
                For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
            api_flavor: The API flavor to use when extracting usage information,
                e.g. 'chat' or 'responses' for OpenAI.
            details: Becomes the `details` field on the returned `RequestUsage` for convenience.
        """
        details = details or {}
        for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
            try:
                provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
                _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
                return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
            except Exception:
                pass
        return cls(details=details)


@dataclass(repr=False, init=False, eq=False)
class RunUsage(UsageBase):
    """LLM usage associated with an agent run.

    Responsibility for calculating request usage is on the model; Pydantic AI simply sums the usage information across requests.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    input_tokens: int = 0
    """Total number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Total number of tokens written to the cache."""

    cache_read_tokens: int = 0
    """Total number of tokens read from the cache."""

    input_audio_tokens: int = 0
    """Total number of audio input tokens."""

    cache_audio_read_tokens: int = 0
    """Total number of audio tokens read from the cache."""

    output_tokens: int = 0
    """Total number of output/completion tokens."""

    details: dict[str, int] = dataclasses.field(default_factory=dict[str, int])
    """Any extra details returned by the model."""

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, RunUsage):
            self.requests += incr_usage.requests
            self.tool_calls += incr_usage.tool_calls
        _incr_usage_tokens(self, incr_usage)
        _incr_usage_cost(self, incr_usage)

    def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
        """Add two RunUsages together.

        This is provided so it's trivial to sum usage information from multiple runs.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    def __sub__(self, other: RunUsage) -> RunUsage:
        """Return the field-by-field usage accumulated since `other`.

        This is useful when a nested operation shares a run's mutable usage object and needs to
        report only the requests, tool calls, tokens, details, and cost added by that operation.
        Unknown costs remain `None`; an unchanged known cost also produces `None`.
        """
        details = {
            name: self.details.get(name, 0) - other.details.get(name, 0) for name in self.details | other.details
        }
        return RunUsage(
            requests=self.requests - other.requests,
            tool_calls=self.tool_calls - other.tool_calls,
            input_tokens=self.input_tokens - other.input_tokens,
            cache_write_tokens=self.cache_write_tokens - other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens - other.cache_read_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            input_audio_tokens=self.input_audio_tokens - other.input_audio_tokens,
            cache_audio_read_tokens=self.cache_audio_read_tokens - other.cache_audio_read_tokens,
            output_audio_tokens=self.output_audio_tokens - other.output_audio_tokens,
            details=details,
            cost=self.cost - (other.cost or 0) if self.cost is not None and self.cost != other.cost else None,
        )


def _incr_usage_cost(slf: RunUsage | RequestUsage, incr_usage: RunUsage | RequestUsage) -> None:
    if incr_usage.cost is not None:
        slf.cost = (slf.cost or 0) + incr_usage.cost


def _incr_usage_tokens(slf: RunUsage | RequestUsage, incr_usage: RunUsage | RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        slf: The usage to increment.
        incr_usage: The usage to increment by.
    """
    for k in (slf.__dict__.keys() | incr_usage.__dict__.keys()) - {'requests', 'tool_calls', 'details', 'cost'}:
        slf_value = getattr(slf, k, 0)
        incr_value = getattr(incr_usage, k, 0)
        if isinstance(slf_value, (int, float)) and isinstance(incr_value, (int, float)):
            setattr(slf, k, slf_value + incr_value)

    for key, value in incr_usage.details.items():
        # Note: value can be None at runtime from model responses despite the type annotation
        if isinstance(value, (int, float)):
            slf.details[key] = slf.details.get(key, 0) + value


@dataclass(repr=False, kw_only=True)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    cost_limit: Decimal | None = None
    """The maximum cost allowed in USD."""
    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    tool_calls_limit: int | None = None
    """The maximum number of successful tool calls allowed to be executed."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
    per_request_input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed per individual request.

    Unlike `input_tokens_limit` which is cumulative across the entire run, this
    limit is checked against each request's input token count independently —
    ahead of the request when `count_tokens_before_request=True`, otherwise against
    the provider-reported `input_tokens` of the response.

    This provides a guard against oversized contexts (which hurt model performance
    and incur high costs on cache misses), complementing the runaway-loop
    protection that cumulative limits provide.

    Note that `input_tokens` (and therefore this limit) includes cached-prefix tokens,
    normalized consistently across providers: a request served largely from cache still
    counts its full context size toward this limit. This caps context size, not cache-miss cost.

    Set `count_tokens_before_request=True` to enforce this preemptively; otherwise the
    request is sent before the limit is checked, so the oversized request is still
    billed (matching `input_tokens_limit`).
    """
    count_tokens_before_request: bool = False
    """If True, perform a token counting pass before sending the request to the model,
    to enforce `input_tokens_limit` and `per_request_input_tokens_limit` ahead of time.

    This may incur additional overhead (from calling the model's `count_tokens` API before making the actual request)
    and is disabled by default.

    Supported by:

    - Anthropic
    - Google
    - Bedrock Converse
    - OpenAI Responses
    """

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` and `check_per_request_input_tokens` methods will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None
            for limit in (
                self.input_tokens_limit,
                self.output_tokens_limit,
                self.total_tokens_limit,
                self.per_request_input_tokens_limit,
            )
        )

    def check_before_request(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(  # pragma: lax no cover
                f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
            )

        cost = usage.cost
        if cost is not None and self.cost_limit is not None and cost > self.cost_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the `cost_limit` of {self.cost_limit} (`cost`={cost!r})'
            )

    def check_cost(self, usage: RunUsage, *, warn_if_cost_unavailable: bool = True) -> None:
        """Check whether usage exceeds the cost limit.

        Args:
            usage: The accumulated run usage to check.
            warn_if_cost_unavailable: Whether to warn when a `cost_limit` is set but no cost was calculated.
        """
        if warn_if_cost_unavailable:
            self._warn_if_cost_unavailable(usage)
        if usage.cost is not None and self.cost_limit is not None and usage.cost > self.cost_limit:
            raise UsageLimitExceeded(f'Exceeded the `cost_limit` of {self.cost_limit} (`usage.cost`={usage.cost!r})')

    def _warn_if_cost_unavailable(self, usage: RunUsage) -> None:
        if self.cost_limit is not None and usage.cost is None:
            warnings.warn(
                CostNotFoundWarning(
                    'A `cost_limit` is set but cannot be enforced because no cost was calculated for this run. '
                    'This usually means genai-prices has no pricing data for the model or provider in use.'
                )
            )

    def check_tokens(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    def check_before_tool_call(self, projected_usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
        tool_calls_limit = self.tool_calls_limit
        tool_calls = projected_usage.tool_calls
        if tool_calls_limit is not None and tool_calls > tool_calls_limit:
            raise UsageLimitExceeded(
                f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
            )

    def check_per_request_input_tokens(self, request_input_tokens: int) -> None:
        """Raises a `UsageLimitExceeded` if the per-request input tokens exceed the limit.

        This checks a single request's input token count — not the cumulative
        `RunUsage.input_tokens` — against `per_request_input_tokens_limit`.
        """
        limit = self.per_request_input_tokens_limit
        if limit is not None and request_input_tokens > limit:
            raise UsageLimitExceeded(
                f'Exceeded the per_request_input_tokens_limit of {limit} ({request_input_tokens=})'
            )

    __repr__ = _utils.dataclasses_no_defaults_repr
