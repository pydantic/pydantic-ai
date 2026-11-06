"""Retry Z.AI request-rate 429s below the Anthropic SDK.

Z.ai `1302` bursts throttle rather than block, and can last hours. During the
2026-11-04 burst, each retry passed about one time in five, however long its
request had waited. At that rate, an estimated one request in a million or fewer
exhausts this window, so a transient rate limit gets a ten-minute window below
the SDK instead of the SDK's own short one.

The engine builds the model, so the composition installs the transport on the
client of the run's model rather than handing one to the provider. Only an
`anthropic/` model reaches here; every other provider keeps its own retry policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
from anthropic import AsyncAnthropic
from pydantic import TypeAdapter, ValidationError
from tenacity import RetryCallState, wait_random_exponential
from tenacity.asyncio.retry import retry_if_result

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig

from ._budgets import run_seconds_left
from ._config import RUN_TIME_RESERVE_SECS
from ._result import ZAI_PLAN_LIMIT_CODES, zai_error_code

RATE_LIMIT_RETRY_SECS = 10 * 60
RATE_LIMIT_MIN_RETRY_SECS = 60

_ERROR_BODY_ADAPTER = TypeAdapter(dict[str, object])
_INSTALLED = 'gh_aw_rate_limit_retry'


def _stop_rate_limit_retries(state: RetryCallState) -> bool:
    """Stop before the next wait ends past `RATE_LIMIT_RETRY_SECS`, or inside the run's time reserve.

    Inside the reserve, a request still retries for its first `RATE_LIMIT_MIN_RETRY_SECS`.
    The reserve is the run's time to emit its safe output. A run cancelled at its deadline
    fails as `time_limit`, which opens a task incident, while a rate limit that outlasts
    these retries fails as `rate_limit`, which stays local to the run.
    """
    elapsed = state.seconds_since_start or 0.0
    if elapsed + state.upcoming_sleep >= RATE_LIMIT_RETRY_SECS:
        return True
    seconds_left = run_seconds_left()
    return (
        elapsed >= RATE_LIMIT_MIN_RETRY_SECS
        and seconds_left is not None
        and seconds_left - state.upcoming_sleep <= RUN_TIME_RESERVE_SECS
    )


async def _close_rate_limited_response(state: RetryCallState) -> None:
    """Release a 429 response's connection before the next attempt replaces it."""
    if state.outcome is not None and not state.outcome.failed:
        await state.outcome.result().aclose()


def _give_up_on_rate_limit(state: RetryCallState) -> httpx2.Response | None:
    """Hand the last 429 to the SDK, marked so the SDK does not open its own window.

    Without the mark, each of the SDK's own retries would start a fresh ten-minute
    window here, and a persistent 429 would hold one request for much longer.
    """
    if state.outcome is None:
        return None
    response: httpx2.Response = state.outcome.result()
    response.headers['x-should-retry'] = 'false'
    return response


def rate_limit_retry_transport(wrapped: httpx2.AsyncBaseTransport | None = None) -> AsyncHTTPX2TenacityTransport:
    """Transport that retries retryable 429 responses and preserves the final response for the SDK.

    Handing back the response rather than raising keeps the SDK's `RateLimitError`,
    with the provider's error body, as what a run that stays rate-limited fails with.
    """

    async def should_retry(response: httpx2.Response) -> bool:
        if response.status_code != 429:
            return False
        try:
            # `aread()` buffers the body, so the SDK can still consume the
            # original response when this 429 is returned without retrying.
            body = _ERROR_BODY_ADAPTER.validate_json(await response.aread())
        except ValidationError:
            return True
        if zai_error_code(body) in ZAI_PLAN_LIMIT_CODES:
            response.headers['x-should-retry'] = 'false'
            return False
        return True

    return AsyncHTTPX2TenacityTransport(
        RetryConfig(
            retry=retry_if_result(should_retry),
            wait=wait_random_exponential(multiplier=1, max=16),
            stop=_stop_rate_limit_retries,
            before_sleep=_close_rate_limited_response,
            retry_error_callback=_give_up_on_rate_limit,
        ),
        wrapped=wrapped,
    )


def install_rate_limit_retry(model: Model) -> None:
    """Give the run model's Anthropic client the retrying transport, once."""
    while isinstance(model, WrapperModel):
        model = model.wrapped
    if not isinstance(model, AnthropicModel):
        return
    sdk_client = model.client
    if not isinstance(sdk_client, AsyncAnthropic):
        return
    # The provider owns construction and offers no client injection, and the model
    # already exists by the time a capability runs. Keep the authenticated client
    # and its pool; only its next-hop transport changes.
    http_client = sdk_client._client  # pyright: ignore[reportPrivateUsage]
    transport = http_client._transport  # pyright: ignore[reportPrivateUsage]
    if getattr(transport, _INSTALLED, False):
        return
    retrying = rate_limit_retry_transport(transport)
    setattr(retrying, _INSTALLED, True)
    http_client._transport = retrying  # pyright: ignore[reportPrivateUsage]


@dataclass
class RateLimitRetry(AbstractCapability[object]):
    """Retry transient Z.AI 429s for the transport's window before a request fails."""

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        install_rate_limit_retry(request_context.model)
        return request_context
