"""Retry Z.AI request-rate 429s below the Anthropic SDK.

Z.ai `1302` bursts throttle rather than block, and can last hours. During the
2026-11-04 burst, each retry passed about one time in five, however long its
request had waited. At that rate, an estimated one request in a million or fewer
exhausts this window, so a transient rate limit gets a ten-minute window below
the SDK instead of the SDK's own short one.

The engine passes `-m`, which makes the run's model explicit, and an explicit
model ignores a capability's model contribution. Model-ID resolution is still
asked for a string selection, so the composition answers it for `anthropic:`
and builds the model with the retrying client from the start. Every request the
run then makes, including a delegate's and the history summariser's, goes
through that one client.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
from pydantic import TypeAdapter, ValidationError
from tenacity import RetryCallState, wait_random_exponential
from tenacity.asyncio.retry import retry_if_result

from pydantic_ai import ModelResolutionContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model, create_async_httpx2_client
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig

from ._budgets import run_seconds_left
from ._config import RUN_TIME_RESERVE_SECS
from ._result import ZAI_PLAN_LIMIT_CODES, zai_error_code

RATE_LIMIT_RETRY_SECS = 10 * 60
RATE_LIMIT_MIN_RETRY_SECS = 60

_ERROR_BODY_ADAPTER = TypeAdapter(dict[str, object])


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
    """Hand the last 429 to the SDK, marked so the SDK does not retry it again.

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


@dataclass
class AnthropicRetryTransport(AbstractCapability[object]):
    """Build a selected `anthropic:` model with the retry transport already on its client."""

    async def resolve_model_id(
        self, ctx: ModelResolutionContext[object], *, model_id: KnownModelName | str
    ) -> Model | None:
        del ctx
        provider_name, separator, model_name = model_id.partition(':')
        if provider_name != 'anthropic' or not separator or not model_name:
            return None
        http_client = create_async_httpx2_client(transport=rate_limit_retry_transport())
        return AnthropicModel(model_name, provider=AnthropicProvider(http_client=http_client))
