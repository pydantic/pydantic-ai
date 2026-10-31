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
    if limit_response_retry_after not in hooks:
        hooks.append(limit_response_retry_after)


async def limit_response_retry_after(response: httpx2.Response) -> None:
    if response.status_code != 429 or response.headers.get('x-should-retry') == 'false':
        return
    request = response.request
    if request.method != 'POST' or not request.url.path.endswith(('/v1/messages', '/v1/messages/count_tokens')):
        return
    # Match the SDK, not core's error-metadata parser: fractional seconds are valid,
    # and negative/NaN milliseconds do not fall through to the seconds header.
    delay: float | None = None
    if (milliseconds := response.headers.get('retry-after-ms')) is not None:
        try:
            delay = float(milliseconds) / 1000
        except ValueError:
            pass
    if delay is None:
        raw = response.headers.get('retry-after')
        if raw is None:
            return
        try:
            delay = float(raw)
        except ValueError:
            parsed = parsedate_tz(raw)
            if parsed is None:
                return
            try:
                delay = mktime_tz(parsed) - time.time()
            except (OverflowError, ValueError):
                return
    if delay > _MAX_RETRY_AFTER:
        # This is a local wait-budget decision, not a header the provider sent.
        # Leave the status, body, Retry-After and quota/reset headers intact.
        response.extensions[_REJECTED_RETRY_AFTER] = delay
        response.headers['x-should-retry'] = 'false'


def retry_after_message(*, error: ModelRateLimitError) -> str | None:
    """Explain a local retry veto without replacing the typed error plugins and fallback chains receive."""
    cause = error.__cause__
    if not isinstance(cause, RateLimitError):
        return None
    delay = cause.response.extensions.get(_REJECTED_RETRY_AFTER)
    if not isinstance(delay, float):
        return None
    estimate = ''
    if math.isfinite(delay):
        minutes, seconds = divmod(math.ceil(delay), 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        if days:
            wait = f'{days}d {hours}h'
        elif hours:
            wait = f'{hours}h {minutes}m'
        else:
            wait = f'{minutes}m' + (f' {seconds}s' if seconds else '')
        estimate = f'Retry after about {wait}. '
    return (
        f'Claude Code rate limit (HTTP 429). {estimate}'
        f'CLAI skipped a wait over {_MAX_RETRY_AFTER:g}s. '
        'Try again later or select a model with /model.'
    )
