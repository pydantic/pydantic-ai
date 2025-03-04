from __future__ import annotations as _annotations

from copy import copy
from dataclasses import dataclass

from .exceptions import UsageLimitExceeded

__all__ = 'Usage', 'UsageLimits'


@dataclass
class Usage:
    """LLM usage associated with a request or run.

    Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.

    You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""
    request_tokens: int | None = None
    """Tokens used in processing requests."""
    response_tokens: int | None = None
    """Tokens used in generating responses."""
    total_tokens: int | None = None
    """Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`."""
    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    def incr(self, incr_usage: Usage, *, requests: int = 0) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
            requests: The number of requests to increment by in addition to `incr_usage.requests`.
        """
        self.requests += requests
        for f in 'requests', 'request_tokens', 'response_tokens', 'total_tokens':
            self_value = getattr(self, f)
            other_value = getattr(incr_usage, f)
            if self_value is not None or other_value is not None:
                setattr(self, f, (self_value or 0) + (other_value or 0))

        if incr_usage.details:
            self.details = self.details or {}
            for key, value in incr_usage.details.items():
                self.details[key] = self.details.get(key, 0) + value

    def __add__(self, other: Usage) -> Usage:
        """Add two Usages together.

        This is provided so it's trivial to sum usage information from multiple requests and runs.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token limits as OpenTelemetry attributes."""
        result = {
            'gen_ai.usage.input_tokens': self.request_tokens,
            'gen_ai.usage.output_tokens': self.response_tokens,
        }
        for key, value in (self.details or {}).items():
            result[f'gen_ai.usage.details.{key}'] = value
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    request_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests to the model."""
    response_tokens_limit: int | None = None
    """The maximum number of tokens allowed in responses from the model."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None
            for limit in (self.request_tokens_limit, self.response_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

    def check_tokens(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        request_tokens = usage.request_tokens or 0
        if self.request_tokens_limit is not None and request_tokens > self.request_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the request_tokens_limit of {self.request_tokens_limit} ({request_tokens=})'
            )

        response_tokens = usage.response_tokens or 0
        if self.response_tokens_limit is not None and response_tokens > self.response_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the response_tokens_limit of {self.response_tokens_limit} ({response_tokens=})'
            )

        total_tokens = usage.total_tokens or 0
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')
