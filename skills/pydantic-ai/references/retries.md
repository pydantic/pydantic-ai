# HTTP Request Retries Reference

Source: `pydantic_ai_slim/pydantic_ai/retries.py`

## Overview

Retry functionality for HTTP requests using [tenacity](https://github.com/jd/tenacity), integrated with httpx. Handles rate limits, network errors, and transient failures.

## Installation

```bash
pip install 'pydantic-ai-slim[retries]'
# or
uv add 'pydantic-ai-slim[retries]'
```

## Basic Setup

```python {title="retry_setup.py" test="skip"}
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


def should_retry(response):
    """Raise for retryable status codes."""
    if response.status_code in (429, 502, 503, 504):
        response.raise_for_status()


transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
        wait=wait_retry_after(
            fallback_strategy=wait_exponential(multiplier=1, max=60),
            max_wait=300,
        ),
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    validate_response=should_retry,
)

client = AsyncClient(transport=transport)
model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

## Transport Classes

### AsyncTenacityTransport (Recommended)

For async HTTP clients:

```python
from httpx import AsyncClient
from tenacity import stop_after_attempt

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig

transport = AsyncTenacityTransport(
    config=RetryConfig(stop=stop_after_attempt(3), reraise=True),
    validate_response=lambda r: r.raise_for_status(),
)
client = AsyncClient(transport=transport)
```

### TenacityTransport

For sync HTTP clients:

```python
from httpx import Client
from tenacity import stop_after_attempt

from pydantic_ai.retries import RetryConfig, TenacityTransport

transport = TenacityTransport(
    config=RetryConfig(stop=stop_after_attempt(3), reraise=True),
    validate_response=lambda r: r.raise_for_status(),
)
client = Client(transport=transport)
```

## wait_retry_after

Smart wait strategy that respects HTTP `Retry-After` headers:

```python
from tenacity import wait_exponential

from pydantic_ai.retries import wait_retry_after

# Basic — respects Retry-After, falls back to exponential
wait = wait_retry_after()

# Custom configuration
wait = wait_retry_after(
    fallback_strategy=wait_exponential(multiplier=2, max=120),
    max_wait=600,  # Never wait more than 10 minutes
)
```

Supports:
- Seconds format: `"30"`
- HTTP date format: `"Wed, 21 Oct 2015 07:28:00 GMT"`

## Common Patterns

### Rate Limit Handling

```python
from httpx import AsyncClient, HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type(HTTPStatusError),
        wait=wait_retry_after(
            fallback_strategy=wait_exponential(multiplier=1, max=60),
            max_wait=300,
        ),
        stop=stop_after_attempt(10),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),
)
client = AsyncClient(transport=transport)
```

### Network Error Handling

```python
import httpx
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig

transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
        )),
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    ),
)
client = httpx.AsyncClient(transport=transport)
```

### Custom Retry Logic

```python
import httpx
from tenacity import retry_if_exception, stop_after_attempt

from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after


def should_retry(exc):
    """Retry server errors but not client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


transport = AsyncTenacityTransport(
    config=RetryConfig(
        retry=retry_if_exception(should_retry),
        wait=wait_retry_after(max_wait=120),
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),
)
```

## Provider Examples

### OpenAI

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

client = create_retrying_client()  # Your retry-enabled client
model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

### Anthropic

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

client = create_retrying_client()
model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(http_client=client))
agent = Agent(model)
```

### AWS Bedrock

Bedrock uses boto3's built-in retries instead of httpx:

```python
from botocore.config import Config

config = Config(retries={'max_attempts': 5, 'mode': 'adaptive'})
# Pass config to Bedrock client
```

## Best Practices

1. **Start conservative**: 3-5 retries with reasonable waits
2. **Use exponential backoff**: Avoid overwhelming servers
3. **Set max wait times**: Prevent indefinite delays
4. **Respect Retry-After**: Use `wait_retry_after`
5. **Log retries**: Monitor with Logfire
6. **Consider circuit breakers**: For high-traffic apps

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `AsyncTenacityTransport` | `pydantic_ai.retries.AsyncTenacityTransport` | Async HTTP transport with retries |
| `TenacityTransport` | `pydantic_ai.retries.TenacityTransport` | Sync HTTP transport with retries |
| `RetryConfig` | `pydantic_ai.retries.RetryConfig` | Tenacity configuration wrapper |
| `wait_retry_after` | `pydantic_ai.retries.wait_retry_after` | Retry-After aware wait strategy |

## See Also

- [models.md](models.md) — Model configuration with custom clients
- [observability.md](observability.md) — Monitoring retries with Logfire
- [tenacity docs](https://tenacity.readthedocs.io/) — Advanced retry config
