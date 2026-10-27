"""Keep Claude Code's SDK retries without silently waiting hours for a subscription reset."""

import math
import time
from email.utils import mktime_tz, parsedate_tz

import httpx2
from anthropic import AsyncAnthropic, RateLimitError

from pydantic_ai.exceptions import ModelRateLimitError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel

_MAX_RETRY_AFTER = 60.0
_REJECTED_RETRY_AFTER = 'clai2.rejected_retry_after'


def limit_retry_after(*, model: Model) -> None:
    """Decorate only the Claude Code model returned by CLAI's plugin resolver, once per HTTP client."""
    if not isinstance(model, AnthropicModel) or not isinstance(model.client, AsyncAnthropic):
        return
    # The plugin owns construction/auth and offers no HTTP-client injection through ModelProvider.resolve.
    # with_options returns a new SDK client; neither model.client nor provider.client has a setter.
    # Keep that authenticated client and its pool instead of reconstructing the plugin's provider.
    hooks = model.client._client.event_hooks['response']  # pyright: ignore[reportPrivateUsage]
    if _limit_retry_after not in hooks:
        hooks.append(_limit_retry_after)


async def _limit_retry_after(response: httpx2.Response) -> None:
    if response.status_code != 429 or response.headers.get('x-should-retry') == 'false':
        return
    request = response.request
    if request.method != 'POST' or not request.url.path.endswith(('/v1/messages', '/v1/messages/count_tokens')):
        return
    delay = _retry_after(headers=response.headers)
    if delay is not None and delay > _MAX_RETRY_AFTER:
        # This is a local wait-budget decision, not a header the provider sent.
        # Leave the status, body, Retry-After and quota/reset headers intact.
        response.extensions[_REJECTED_RETRY_AFTER] = delay
        response.headers['x-should-retry'] = 'false'


def _retry_after(*, headers: httpx2.Headers) -> float | None:
    """Match the Anthropic SDK's milliseconds, fractional seconds, then HTTP-date precedence."""
    if (milliseconds := headers.get('retry-after-ms')) is not None:
        try:
            return float(milliseconds) / 1000
        except ValueError:
            pass
    raw = headers.get('retry-after')
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    parsed = parsedate_tz(raw)
    if parsed is None:
        return None
    try:
        return mktime_tz(parsed) - time.time()
    except (OverflowError, ValueError):
        return None


def retry_after_message(*, error: ModelRateLimitError) -> str | None:
    """Explain a local retry veto without replacing the typed error plugins and fallback chains receive."""
    cause = error.__cause__
    if not isinstance(cause, RateLimitError):
        return None
    delay = cause.response.extensions.get(_REJECTED_RETRY_AFTER)
    if not isinstance(delay, float):
        return None
    estimate = f'Retry after {delay:g}s. ' if math.isfinite(delay) else ''
    return (
        f'Claude Code rate limit (HTTP 429). {estimate}'
        f'CLAI skipped a wait over {_MAX_RETRY_AFTER:g}s. '
        'Try again later or select a model with /model.'
    )
