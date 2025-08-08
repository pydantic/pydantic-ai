from __future__ import annotations as _annotations

import dataclasses
from copy import copy
from dataclasses import dataclass

from typing_extensions import deprecated, overload

from . import _utils
from .exceptions import UsageLimitExceeded

__all__ = 'RequestUsage', 'RunUsage', 'UsageLimits'


class UsageBase:
    input_tokens: int | None = None
    output_tokens: int | None = None
    details: dict[str, int] | None = None

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token usage values as OpenTelemetry attributes."""
        result: dict[str, int] = {}
        if self.input_tokens:
            result['gen_ai.usage.input_tokens'] = self.input_tokens
        if self.output_tokens:
            result['gen_ai.usage.output_tokens'] = self.output_tokens
        details = self.details
        if details:
            prefix = 'gen_ai.usage.details.'
            for key, value in details.items():
                # Skipping check for value since spec implies all detail values are relevant
                if value:
                    result[prefix + key] = value
        return result

    __repr__ = _utils.dataclasses_no_defaults_repr

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return any(dataclasses.asdict(self).values())  # type: ignore


@dataclass(repr=False)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request.

    Prices for LLM requests are calculated using [genai-prices](https://github.com/pydantic/genai-prices).
    """

    input_tokens: int | None = None
    """Number of text input/prompt tokens."""

    cache_write_tokens: int | None = None
    """Number of tokens written to the cache."""
    cache_read_tokens: int | None = None
    """Number of tokens read from the cache."""

    output_tokens: int | None = None
    """Number of text output/completion tokens."""

    input_audio_tokens: int | None = None
    """Number of audio input tokens."""
    cache_audio_read_tokens: int | None = None
    """Number of audio tokens read from the cache."""

    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    @property
    def output_audio_tokens(self):
        return None

    @property
    def requests(self):
        return 1

    @property
    @deprecated('`request_tokens` is deprecated, use `input_tokens` instead')
    def request_tokens(self) -> int | None:
        return self.input_tokens

    @property
    @deprecated('`response_tokens` is deprecated, use `output_tokens` instead')
    def response_tokens(self) -> int | None:
        return self.output_tokens

    @property
    @deprecated('`total_tokens` is deprecated, sum the specific fields you need instead')
    def total_tokens(self) -> int | None:
        return sum(v for k, v in dataclasses.asdict(self).values() if k.endswith('_tokens') and v is not None)

    def incr(self, incr_usage: RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RequestUsage) -> RequestUsage:
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage


@dataclass(repr=False)
class RunUsage(UsageBase):
    """LLM usage associated with an agent run.

    Responsibility for calculating request usage is on the model; Pydantic AI simply sums the usage information across requests.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    input_tokens: int | None = None
    """Total number of text input/prompt tokens."""

    cache_write_tokens: int | None = None
    """Total number of tokens written to the cache."""
    cache_read_tokens: int | None = None
    """Total number of tokens read from the cache."""

    input_audio_tokens: int | None = None
    """Total number of audio input tokens."""
    cache_audio_read_tokens: int | None = None
    """Total number of audio tokens read from the cache."""

    output_tokens: int | None = None
    """Total number of text output/completion tokens."""

    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    def input_output_tokens(self) -> int | None:
        """Sum of `input_tokens + output_tokens`."""
        if self.input_tokens is None and self.output_tokens is None:
            return None
        else:
            return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    @deprecated('`request_tokens` is deprecated, use `input_tokens` instead')
    def request_tokens(self) -> int | None:
        return self.input_tokens

    @property
    @deprecated('`response_tokens` is deprecated, use `output_tokens` instead')
    def response_tokens(self) -> int | None:
        return self.output_tokens

    @property
    @deprecated('`total_tokens` is deprecated, sum the specific fields you need or use `input_output_tokens` instead')
    def total_tokens(self) -> int | None:
        return sum(v for k, v in dataclasses.asdict(self).values() if k.endswith('_tokens') and v is not None)

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, RunUsage):
            self.requests += incr_usage.requests
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
        """Add two RunUsages together.

        This is provided so it's trivial to sum usage information from multiple runs.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage


def _incr_usage_tokens(slf: RunUsage | RequestUsage, incr_usage: RunUsage | RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        slf: The usage to increment.
        incr_usage: The usage to increment by.
    """
    if incr_usage.input_tokens:
        slf.input_tokens = (slf.input_tokens or 0) + incr_usage.input_tokens
    if incr_usage.output_tokens:
        slf.output_tokens = (slf.output_tokens or 0) + incr_usage.output_tokens
    if incr_usage.cache_write_tokens:
        slf.cache_write_tokens = (slf.cache_write_tokens or 0) + incr_usage.cache_write_tokens
    if incr_usage.cache_read_tokens:
        slf.cache_read_tokens = (slf.cache_read_tokens or 0) + incr_usage.cache_read_tokens
    if incr_usage.input_audio_tokens:
        slf.input_audio_tokens = (slf.input_audio_tokens or 0) + incr_usage.input_audio_tokens
    if incr_usage.cache_audio_read_tokens:
        slf.cache_audio_read_tokens = (slf.cache_audio_read_tokens or 0) + incr_usage.cache_audio_read_tokens

    if incr_usage.details:
        slf.details = slf.details or {}
        for key, value in incr_usage.details.items():
            slf.details[key] = slf.details.get(key, 0) + value


@dataclass
@deprecated('`Usage` is deprecated, use `RunUsage` instead')
class Usage(RunUsage):
    pass


@dataclass(repr=False)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of combined input and output tokens allowed."""

    @overload
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
    ) -> None:
        self.request_limit = request_limit
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.total_tokens_limit = total_tokens_limit

    @overload
    @deprecated(
        'Use `input_tokens_limit` instead of `request_tokens_limit` and `output_tokens_limit` and `total_tokens_limit`'
    )
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
    ) -> None:
        self.request_limit = request_limit
        self.input_tokens_limit = request_tokens_limit
        self.output_tokens_limit = response_tokens_limit
        self.total_tokens_limit = total_tokens_limit

    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        # deprecated:
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
    ):
        self.request_limit = request_limit
        self.input_tokens_limit = input_tokens_limit or request_tokens_limit
        self.output_tokens_limit = output_tokens_limit or response_tokens_limit
        self.total_tokens_limit = total_tokens_limit

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

    def check_tokens(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens or 0
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens or 0
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.input_output_tokens() or 0
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    __repr__ = _utils.dataclasses_no_defaults_repr
