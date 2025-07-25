# HTTP Request Retries

Pydantic AI provides retry functionality for HTTP requests made by model providers through custom HTTP transports.
This is particularly useful for handling transient failures like rate limits, network timeouts, or temporary server errors.

## Overview

The retry functionality is built on top of the [tenacity](https://github.com/jd/tenacity) library and integrates
seamlessly with httpx clients. You can configure retry behavior for any provider that accepts a custom HTTP client.

## Installation

To use the retry transports, you need to install `tenacity`, which you can do via the `retries` dependency group:

```bash
pip/uv-add 'pydantic-ai-slim[retries]'
```

## Usage Example

Here's an example of adding retry functionality with smart retry handling:

```python {title="smart_retry_example.py"}
from httpx import AsyncClient, HTTPStatusError
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after
from pydantic_ai.providers.openai import OpenAIProvider

def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((HTTPStatusError, ConnectionError)),
            # Smart waiting: respects Retry-After headers, falls back to exponential backoff
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300
            ),
            # Stop after 5 attempts
            stop=stop_after_attempt(5),
            # Re-raise the last exception if all retries fail
            reraise=True
        ),
        validate_response=should_retry_status
    )
    return AsyncClient(transport=transport)

# Use the retrying client with a model
client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

## Wait Strategies

### wait_retry_after

The `wait_retry_after` function is a smart wait strategy that automatically respects HTTP `Retry-After` headers:

```python {title="wait_strategy_example.py"}
from pydantic_ai.retries import wait_retry_after
from tenacity import wait_exponential

# Basic usage - respects Retry-After headers, falls back to exponential backoff
wait_strategy_1 = wait_retry_after()

# Custom configuration
wait_strategy_2 = wait_retry_after(
    fallback_strategy=wait_exponential(multiplier=2, max=120),
    max_wait=600  # Never wait more than 10 minutes
)
```

This wait strategy:
- Automatically parses `Retry-After` headers from HTTP 429 responses
- Supports both seconds format (`"30"`) and HTTP date format (`"Wed, 21 Oct 2015 07:28:00 GMT"`)
- Falls back to your chosen strategy when no header is present
- Respects the `max_wait` limit to prevent excessive delays

## Transport Classes

### AsyncTenacityTransport

For asynchronous HTTP clients (recommended for most use cases):

```python {title="async_transport_example.py"}
from httpx import AsyncClient
from tenacity import AsyncRetrying, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport

# Create the basic components
async_retrying = AsyncRetrying(stop=stop_after_attempt(3), reraise=True)

def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = AsyncTenacityTransport(
    controller=async_retrying,   # AsyncRetrying instance
    validate_response=validator  # Optional response validator
)

# Create a client using the transport:
client = AsyncClient(transport=transport)
```

### TenacityTransport

For synchronous HTTP clients:

```python {title="sync_transport_example.py"}
from httpx import Client
from tenacity import Retrying, stop_after_attempt
from pydantic_ai.retries import TenacityTransport

# Create the basic components
retrying = Retrying(stop=stop_after_attempt(3), reraise=True)

def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = TenacityTransport(
    controller=retrying,       # Retrying instance
    validate_response=validator # Optional response validator
)

# Create a client using the transport
client = Client(transport=transport)
```

## Common Retry Patterns

### Rate Limit Handling with Retry-After Support

```python {title="rate_limit_handling.py"}
from httpx import AsyncClient, HTTPStatusError
from tenacity import AsyncRetrying, stop_after_attempt, retry_if_exception_type, wait_exponential
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

def create_rate_limit_client():
    """Create a client that respects Retry-After headers from rate limiting responses."""
    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=retry_if_exception_type(HTTPStatusError),
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=1, max=60),
                max_wait=300  # Don't wait more than 5 minutes
            ),
            stop=stop_after_attempt(10),
            reraise=True
        ),
        validate_response=lambda r: r.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
    )
    return AsyncClient(transport=transport)

# Example usage
client = create_rate_limit_client()
# Client is now ready to use with any HTTP requests and will respect Retry-After headers
```

The `wait_retry_after` function automatically detects `Retry-After` headers in 429 (rate limit) responses and waits for the specified time. If no header is present, it falls back to exponential backoff.

### Network Error Handling

```python {title="network_error_handling.py"}
import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, wait_exponential, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport

def create_network_resilient_client():
    """Create a client that handles network errors with retries."""
    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=retry_if_exception_type((
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadError
            )),
            wait=wait_exponential(multiplier=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True
        )
    )
    return httpx.AsyncClient(transport=transport)

# Example usage
client = create_network_resilient_client()
# Client will now retry on timeout, connection, and read errors
```

### Custom Retry Logic

```python {title="custom_retry_logic.py"}
import httpx
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport, wait_retry_after

def create_custom_retry_client():
    """Create a client with custom retry logic."""
    def custom_retry_condition(exception):
        """Custom logic to determine if we should retry."""
        if isinstance(exception, httpx.HTTPStatusError):
            # Retry on server errors but not client errors
            return 500 <= exception.response.status_code < 600
        return isinstance(exception, (httpx.TimeoutException, httpx.ConnectError))

    transport = AsyncTenacityTransport(
        controller=AsyncRetrying(
            retry=custom_retry_condition,
            # Use wait_retry_after for smart waiting on rate limits,
            # with custom exponential backoff as fallback
            wait=wait_retry_after(
                fallback_strategy=wait_exponential(multiplier=2, max=30),
                max_wait=120
            ),
            stop=stop_after_attempt(5),
            reraise=True
        ),
        validate_response=lambda r: r.raise_for_status()
    )
    return httpx.AsyncClient(transport=transport)

client = create_custom_retry_client()
# Client will retry server errors (5xx) and network errors, but not client errors (4xx)
```

## Using with Different Providers

The retry transports work with any provider that accepts a custom HTTP client:

### OpenAI

```python {title="openai_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

### Anthropic

```python {title="anthropic_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(http_client=client))
agent = Agent(model)
```

### Any OpenAI-Compatible Provider

```python {title="openai_compatible_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel(
    'your-model-name',  # Replace with actual model name
    provider=OpenAIProvider(
        base_url='https://api.example.com/v1',  # Replace with actual API URL
        api_key='your-api-key',  # Replace with actual API key
        http_client=client
    )
)
agent = Agent(model)
```

## Best Practices

1. **Start Conservative**: Begin with a small number of retries (3-5) and reasonable wait times.

2. **Use Exponential Backoff**: This helps avoid overwhelming servers during outages.

3. **Set Maximum Wait Times**: Prevent indefinite delays with reasonable maximum wait times.

4. **Handle Rate Limits Properly**: Respect `Retry-After` headers when possible.

5. **Log Retry Attempts**: Add logging to monitor retry behavior in production. (This will be picked up by Logfire automatically if you instrument httpx.)

6. **Consider Circuit Breakers**: For high-traffic applications, consider implementing circuit breaker patterns.

## Error Handling

The retry transports will re-raise the last exception if all retry attempts fail. Make sure to handle these appropriately in your application:

```python {title="error_handling_example.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(http_client=client))
agent = Agent(model)
```

## Performance Considerations

- Retries add latency to requests, especially with exponential backoff
- Consider the total timeout for your application when configuring retry behavior
- Monitor retry rates to detect systemic issues
- Use async transports for better concurrency when handling multiple requests

For more advanced retry configurations, refer to the [tenacity documentation](https://tenacity.readthedocs.io/).
