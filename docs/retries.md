---
description: "Configure retries in Pydantic AI: HTTP transport retries for rate limits and 5xx errors, provider SDK retries, tool and output retries, and how they multiply."
---

# Retries

"Retry" means seven different things in an agent run, at seven different layers, and they don't share budgets. Mixing them up is the usual cause of a run that retries far more (or far less) than expected. This page is the map; each layer links to the page that configures it in detail.

## The layers

| Layer | What it re-attempts | Configured with | What it adds to message history |
|---|---|---|---|
| [Transport](#transport-retries) | The same HTTP request to the provider | [`AsyncHTTPX2TenacityTransport`][pydantic_ai.retries.AsyncHTTPX2TenacityTransport] on your HTTP client | Nothing — the agent never sees the attempts |
| [Provider SDK](#provider-sdk-retries) | The same HTTP request, re-issued by the provider SDK's own client | The SDK client itself; defaults and configuration are provider-specific | Nothing — the agent never sees the attempts |
| [Durable execution](durable_execution/overview.md) | The whole model request, re-executed by the workflow engine — re-entering every layer nearer the wire; unbounded by default on Temporal (`maximum_attempts=0`) | `retry_policy` in Temporal's `ActivityConfig`, `max_attempts` in DBOS's `StepConfig`, `retries` in Prefect's `TaskConfig` | Nothing — the engine replays the step |
| [Model fallback](#model-fallback-is-not-a-retry) | The same request against a *different* model | [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] | Only the winning response, which lists the failed attempts in `failed_attempts` |
| [Tool](#tool-retries) | One tool call, by asking the model to correct it | `retries={'tools': N}` and per-tool limits | A [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with `outcome='retried'` as that call's result |
| [Output](#output-retries) | The model's final answer, by asking it to correct it | `retries={'output': N}` and [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries] | A [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart], or a retried `ToolReturnPart` — see [below](#output-retries) |
| [Model-request hooks](hooks.md) | The model request, from `before_model_request`, `after_model_request`, `wrap_model_request`, or `on_model_request_error` raising `ModelRetry` | The hook itself; it draws on the **output** budget | A new request carrying a `RetryFeedbackPart` |

Only the last three are "agent retries" — they cost a model round trip each, because a retry *is* another request. The other four are invisible to the model: it never sees an attempt fail.

[Batched requests](batch.md#failed-requests) add one more layer: a request that failed on the provider's side is submitted again in a new batch, up to `max_resubmissions` times.

## Retry multiplication

The layers don't share budgets, but they stack: a retry at one layer wraps the attempts of every layer nearer the wire. If a logical call can issue up to `N` model requests — the initial attempt plus one follow-up per tool call and whatever the [tool](#tool-retries) and [output](#output-retries) retry budgets add — each model request can be tried against up to `F` models by a [fallback model](#model-fallback-is-not-a-retry), each of those is sent by a provider SDK client allowed up to `M` attempts, and each attempt travels on a transport allowed up to `K` attempts, so one logical call can put up to `N`×`F`×`M`×`K` wire requests on the network. Under [durable execution](durable_execution/overview.md) the step that runs the model request retries too, re-entering `M` and `K` each time — and on Temporal that retry count is unbounded unless you set `maximum_attempts` yourself.

- `N` — model requests per logical call: the initial attempt, one follow-up per tool call (even a successful tool call queues another request), and any retries the [tool](#tool-retries) and [output](#output-retries) budgets add
- `F` — models per model request: the number of models in a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], or `1` without one
- `M` — attempts per model request inside the provider SDK client. The SDK determines this budget; for example, an OpenAI client configured with `max_retries=N` allows `1 + N` attempts, and [`ModelSettings['max_retries']`](#setting-the-sdk-retry-count-per-request) overrides it per request. See [provider SDK retries](#provider-sdk-retries) for the provider-specific settings.
- `K` — attempts per request on the wire: the transport's stop strategy, so `stop_after_attempt(N)` allows `N` total attempts (`K = N`), not one plus retries — see [transport retries](#transport-retries)

Every wire request pays its own latency — and bills tokens once the request reaches the model — so the worst case, not the happy path, is what your budgets must absorb. [`UsageLimits`][pydantic_ai.usage.UsageLimits] bounds only `N`: its `request_limit` (default `50`) counts the model responses committed to each run's history, and never sees the fallback attempts, SDK client retries and transport retries beneath them. Their tokens and cost do count towards its token and cost limits whenever the provider reports them, as it does for a response a fallback model rejected, but usually not for a request that failed. [`ModelSettings.timeout`][pydantic_ai.settings.ModelSettings.timeout] applies per attempt — a retrying SDK client re-arms it for every retry — and only on the [model classes that forward it](timeouts.md#bounding-how-long-a-step-takes). [`ModelSettings.request_timeout`][pydantic_ai.settings.ModelSettings.request_timeout] instead bounds all `M × K` attempts at a request to one model, on every model class: see [Model request deadlines](timeouts.md#model-request-deadlines). See [Timeouts](timeouts.md#bounding-how-long-a-step-takes) for the time side.

A run with `retries={'output': 2}` (up to 3 model requests for the final answer alone), the OpenAI SDK's default `max_retries=2` (3 attempts per request), and a transport stopped by `stop_after_attempt(2)` (2 attempts per wire request) can put `3` × `1` × `3` × `2 = 18` requests on the network without a fallback model, and twice that with a fallback model of two models. Each of those carries its own `ModelSettings(timeout=10)` deadline — 180 seconds of request time in the worst case, before any backoff wait between retries.

## Transport retries

Transport retries live below the model client: a failed HTTP request is re-sent without the agent ever knowing. Nothing retries at this layer unless you install a retrying transport on the HTTP client you pass to the provider, and you decide which errors qualify.

This is the right layer for rate limits, connection resets, and 5xx responses. The transports are built on [tenacity](https://github.com/jd/tenacity) and plug into [`httpx2`](https://httpx2.pydantic.dev/) clients, so they work with any provider whose SDK accepts a custom `httpx2` client. [AWS Bedrock](#aws-bedrock) is the exception: it retries through boto3 instead.

When you build your own backoff outside a transport, [`ModelAPIError.retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] gives you the wait the provider asked for, already parsed into seconds, and the [error categories](models/overview.md#handling-model-api-errors) say which failures a retry can't fix, like a [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError] for a rejected API key or an unknown model.

### Installation

To use the retry transports, you need to install `tenacity`, which you can do via the `retries` dependency group:

```bash
pip/uv-add 'pydantic-ai-slim[retries]'
```

### A retrying client

Pass the retrying transport to [`create_async_httpx2_client()`][pydantic_ai.models.create_async_httpx2_client], so the client keeps the timeouts and user agent of the clients Pydantic AI builds, and hand the client to the provider. Then set [`ModelSettings['max_retries']`](#setting-the-sdk-retry-count-per-request) to `0`, so the transport is the only layer that retries rather than one whose attempts [multiply](#retry-multiplication) with the provider SDK's own:

```python {title="smart_retry_example.py"}
import httpx2
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai import Agent
from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import (
    AsyncHTTPX2TenacityTransport,
    RetryConfig,
    wait_retry_after,
)


def create_retrying_client():
    """Create a client with smart retry handling for multiple error types."""

    def should_retry_status(response):
        """Raise exceptions for retryable HTTP status codes."""
        if response.status_code in (429, 502, 503, 504):
            response.raise_for_status()  # This will raise HTTPStatusError

    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            # Retry on HTTP errors and connection issues
            retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.ConnectError)),
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
    return create_async_httpx2_client(transport=transport)

# Use the retrying client with a model, and leave retrying to the transport
client = create_retrying_client()
model = OpenAIChatModel('{{ openai:medium.name }}', provider=OpenAIProvider(http_client=client))
agent = Agent(model, model_settings={'max_retries': 0})
```

The transport you pass replaces the one the client would build, so it owns the connection pool. Without `wrapped`, a retrying transport sends its attempts through an `httpx2.AsyncHTTPTransport` it creates with the same pool limits as Pydantic AI's own clients (1000 connections, up to 100 kept alive). HTTPX also stops reading proxies from the `HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` environment variables once a client has a transport, so give the retrying transport your own `wrapped` transport with a `proxy` if you need one. A `wrapped` transport you build has HTTPX's default pool of 100 connections, 20 kept alive, unless you set its `limits` too: `wrapped=httpx2.AsyncHTTPTransport(proxy=..., limits=httpx2.Limits(max_connections=1000, max_keepalive_connections=100))`.

### Wait strategies

#### wait_retry_after

The `wait_retry_after` function is a smart wait strategy that automatically respects HTTP `Retry-After` headers:

```python {title="wait_strategy_example.py"}
from tenacity import wait_exponential

from pydantic_ai.retries import wait_retry_after

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
- Prefers the `retry-after-ms` header (milliseconds) that OpenAI and Anthropic send, like their SDKs do
- Falls back to your chosen strategy when the header is absent or cannot be parsed
- Respects the `max_wait` limit to prevent excessive delays

### Transport classes

#### AsyncHTTPX2TenacityTransport

For asynchronous HTTP clients (recommended for most use cases):

```python {title="async_transport_example.py"}
from tenacity import stop_after_attempt

from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig


def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = AsyncHTTPX2TenacityTransport(
    config=RetryConfig(stop=stop_after_attempt(3), reraise=True),
    validate_response=validator
)

# Create a client using the transport:
client = create_async_httpx2_client(transport=transport)
```

#### HTTPX2TenacityTransport

For synchronous HTTP clients:

```python {title="sync_transport_example.py"}
from httpx2 import Client
from tenacity import stop_after_attempt

from pydantic_ai.retries import HTTPX2TenacityTransport, RetryConfig


def validator(response):
    """Treat responses with HTTP status 4xx/5xx as failures that need to be retried.
    Without a response validator, only network errors and timeouts will result in a retry.
    """
    response.raise_for_status()

# Create the transport
transport = HTTPX2TenacityTransport(
    config=RetryConfig(stop=stop_after_attempt(3), reraise=True),
    validate_response=validator
)

# Create a client using the transport
client = Client(transport=transport)
```

### Common retry patterns

#### Rate limit handling with `Retry-After` support

```python {title="rate_limit_handling.py"}
from httpx2 import HTTPStatusError
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.retries import (
    AsyncHTTPX2TenacityTransport,
    RetryConfig,
    wait_retry_after,
)


def create_rate_limit_client():
    """Create a client that respects Retry-After headers from rate limiting responses."""
    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
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
    return create_async_httpx2_client(transport=transport)

# Example usage
client = create_rate_limit_client()
# Client is now ready to use with any HTTP requests and will respect Retry-After headers
```

The `wait_retry_after` function automatically detects `Retry-After` (or `retry-after-ms`) headers in 429 (rate limit) responses and waits for the specified time. If no header is present, it falls back to exponential backoff.

#### Network error handling

```python {title="network_error_handling.py"}
import httpx2
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential

from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig


def create_network_resilient_client():
    """Create a client that handles network errors with retries."""
    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception_type((
                httpx2.TimeoutException,
                httpx2.ConnectError,
                httpx2.ReadError
            )),
            wait=wait_exponential(multiplier=1, max=10),
            stop=stop_after_attempt(3),
            reraise=True
        )
    )
    return create_async_httpx2_client(transport=transport)

# Example usage
client = create_network_resilient_client()
# Client will now retry on timeout, connection, and read errors
```

#### Custom retry logic

```python {title="custom_retry_logic.py"}
import httpx2
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential

from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.retries import (
    AsyncHTTPX2TenacityTransport,
    RetryConfig,
    wait_retry_after,
)


def create_custom_retry_client():
    """Create a client with custom retry logic."""
    def custom_retry_condition(exception):
        """Custom logic to determine if we should retry."""
        if isinstance(exception, httpx2.HTTPStatusError):
            # Retry on server errors but not client errors
            return 500 <= exception.response.status_code < 600
        return isinstance(exception, httpx2.TimeoutException | httpx2.ConnectError)

    transport = AsyncHTTPX2TenacityTransport(
        config=RetryConfig(
            retry=retry_if_exception(custom_retry_condition),
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
    return create_async_httpx2_client(transport=transport)

client = create_custom_retry_client()
# Client will retry server errors (5xx) and network errors, but not client errors (4xx)
```

### Using with `httpx2`-compatible providers

The retry transports work with any provider whose `http_client` argument accepts an `httpx2.AsyncClient`. See each
[provider's docs](models/overview.md) for the client type it takes; [Bedrock](#aws-bedrock) uses boto3 and configures retries
its own way.

Providers whose SDKs still require a legacy `httpx.AsyncClient` (such as Groq and Cohere) can use the
deprecated [`TenacityTransport`][pydantic_ai.retries.TenacityTransport] and
[`AsyncTenacityTransport`][pydantic_ai.retries.AsyncTenacityTransport] on that client during Pydantic AI v2; both are
removed in v3 together with legacy client support.

#### OpenAI

```python {title="openai_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIChatModel('{{ openai:medium.name }}', provider=OpenAIProvider(http_client=client))
agent = Agent(model, model_settings={'max_retries': 0})
```

#### Any OpenAI-compatible provider

```python {title="openai_compatible_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIChatModel(
    'your-model-name',  # Replace with actual model name
    provider=OpenAIProvider(
        base_url='https://api.example.com/v1',  # Replace with actual API URL
        api_key='your-api-key',  # Replace with actual API key
        http_client=client
    )
)
agent = Agent(model, model_settings={'max_retries': 0})
```

#### Anthropic

```python {title="anthropic_with_retries.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = AnthropicModel('{{ anthropic:medium.name }}', provider=AnthropicProvider(http_client=client))
agent = Agent(model, model_settings={'max_retries': 0})
```

### Best practices

1. **Start Conservative**: Begin with a small number of retries (3-5) and reasonable wait times.

2. **Use Exponential Backoff**: This helps avoid overwhelming servers during outages.

3. **Set Maximum Wait Times**: Prevent indefinite delays with reasonable maximum wait times.

4. **Handle Rate Limits Properly**: Respect `Retry-After` headers when possible.

5. **Log Retry Attempts**: Add logging to monitor retry behavior in production. (This will be picked up by Logfire automatically if you instrument `httpx2`.)

6. **Consider Circuit Breakers**: For high-traffic applications, consider implementing circuit breaker patterns.

!!! tip "Monitoring Retries in Production"
    Excessive retries can indicate underlying issues and increase costs. [Logfire](logfire.md) helps you track retry patterns:

    - See which requests triggered retries
    - Understand retry causes (rate limits, server errors, timeouts)
    - Monitor retry frequency over time
    - Identify opportunities to reduce retries

    With [HTTPX instrumentation](logfire.md#monitoring-http-requests) enabled, retry attempts are automatically captured in your traces.

### Error handling

The retry transports will re-raise the last exception if all retry attempts fail. Make sure to handle these appropriately in your application:

```python {title="error_handling_example.py" requires="smart_retry_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from smart_retry_example import create_retrying_client

client = create_retrying_client()
model = OpenAIChatModel('{{ openai:medium.name }}', provider=OpenAIProvider(http_client=client))
agent = Agent(model, model_settings={'max_retries': 0})
```

### Performance considerations

- Retries add latency to requests, especially with exponential backoff
- Consider the total timeout for your application when configuring retry behavior
- Monitor retry rates to detect systemic issues
- Use async transports for better concurrency when handling multiple requests

For more advanced retry configurations, refer to the [tenacity documentation](https://tenacity.readthedocs.io/).

### AWS Bedrock

The AWS Bedrock provider uses boto3's built-in retry mechanisms instead of `httpx2`. To configure retries for Bedrock, use boto3's `Config`:

```python
from botocore.config import Config

config = Config(retries={'max_attempts': 5, 'mode': 'adaptive'})
```

See [Bedrock: Configuring Retries](models/bedrock.md#configuring-retries) for complete examples.

## Provider SDK retries {#provider-sdk-retries}

Between the transport and the model sits one more layer the agent never sees: the provider SDK's own client, which re-issues failed requests before your code hears about them. Its defaults, retryable errors, and configuration differ by provider, so size `M` from the client you use. A [retrying transport](#transport-retries) sits *below* this client, so the two stack rather than replacing each other: configuring one never disables the other.

See the provider-specific settings for [OpenAI](models/openai.md#custom-openai-client), [Anthropic](models/anthropic.md#custom-http-client), [Google](models/google.md#http-retries), [Groq](models/groq.md#sdk-retries), [Cohere](models/cohere.md#sdk-retries), [TypeSafe](models/typesafe.md#sdk-retries), [xAI](models/xai.md#sdk-retries), and [AWS Bedrock](models/bedrock.md#configuring-retries).

### Setting the SDK retry count per request {#setting-the-sdk-retry-count-per-request}

[`ModelSettings['max_retries']`][pydantic_ai.settings.ModelSettings.max_retries] sets how many times the SDK client retries one request, without building the client yourself. `0` makes a single attempt. The SDK still decides which failures it retries and how long it waits between attempts. Like any model setting, it applies per model, per agent, or per run:

```python {title="sdk_max_retries.py"}
from pydantic_ai import Agent

agent = Agent('{{ openai:medium }}', model_settings={'max_retries': 0})
```

Set it to `0` when another layer owns retrying, so the attempt budget is the one you configured there rather than that budget multiplied by the SDK's:

- a [retrying transport](#transport-retries) on the HTTP client;
- a [durable execution](durable_execution/overview.md) engine's retry policy;
- a [fallback model](#model-fallback-is-not-a-retry) that should move to its next model as soon as one fails.

On a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], `max_retries` set on the agent or run applies to every candidate, and takes precedence over each candidate's own settings. To keep SDK retries on the last candidate, the one with nothing to fall back to, set it on the other candidates instead:

```python {title="fallback_max_retries.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

fallback = FallbackModel(
    OpenAIChatModel('{{ openai:medium.name }}', settings={'max_retries': 0}),
    AnthropicModel('{{ anthropic:medium.name }}'),
)
agent = Agent(fallback)
```

Not every SDK can take the setting per request:

| Model | Supported | Notes |
|---|---|---|
| OpenAI, and the model classes built on it | ✅ | Full feature support, including an `AsyncAzureOpenAI` client. |
| Anthropic | ✅ | Full feature support, including `AsyncAnthropicBedrock`, `AsyncAnthropicVertex` and `AsyncAnthropicFoundry` clients |
| Groq | ✅ | Full feature support |
| Cohere | ✅ | Full feature support |
| Google | ✅ | Limited parameter support: keeps the delays and status codes of the provider's `retry_options`, or the SDK's defaults when those aren't set |
| Mistral | ✅ | Limited parameter support: `0` only. The SDK bounds retries by elapsed time, not count, so `0` turns off a `retry_config` set on the client and other values are ignored |
| TypeSafe | ✅ | Limited parameter support: the request's other `RetryPolicy` options are the SDK defaults, not the client's |
| AWS Bedrock | ❌ | boto3 retries per client; see [Configuring Retries](models/bedrock.md#configuring-retries) |
| xAI | ❌ | gRPC retries per channel; see [SDK retries](models/xai.md#sdk-retries) |
| Hugging Face | ❌ | The client makes no retries |

The model classes that honor it are listed under [`ModelSettings.max_retries`][pydantic_ai.settings.ModelSettings.max_retries]; the others don't change their retries, and warn with [`UnsupportedSettingWarning`][pydantic_ai.exceptions.UnsupportedSettingWarning] that they ignore it. On a `FallbackModel` with such a candidate, set `max_retries` on the other candidates' own settings, as above, rather than on the agent or run.

## Model fallback is not a retry

[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] moves to the *next* model when the current one fails; it never re-attempts the same one. Pair it with transport retries rather than treating it as a substitute: retry the same provider for transient failures, fall back to a different provider when it's genuinely down. The response that answers lists the attempts the fallback model moved on from in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], with each one's model, duration, error and any usage. See [Fallback Model](models/overview.md#fallback-model).

## Tool retries

A tool retry is a message to the model: the call didn't work, here is why, try again. It is triggered by a Pydantic `ValidationError` on the tool's arguments, by the tool (or its `args_validator`, or a tool hook) raising [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], by a [tool timeout](timeouts.md#bounding-how-long-a-step-takes), and by the model calling a tool that doesn't exist.

[Tool Execution, Retries, and Failures](tools-advanced.md#tool-retries) documents the configuration: the default budget of `1`, the per-tool / per-toolset / per-run / agent-wide precedence ladder, and the choice between `ModelRetry` and [`ToolFailed`][pydantic_ai.exceptions.ToolFailed]. Three properties of the *counter* matter when you're reasoning about a run:

- **The counter is keyed by tool name, and it resets on success.** Each tool has its own count; there is no run-wide tool-retry budget. When a tool succeeds, its count is cleared — so a tool that alternates failure and success can fail many times in one run without ever exhausting a budget of `1`.
- **`max_retries=N` allows N retries, so N+1 attempts.** `max_retries=0` raises on the first failure without ever asking the model to correct it.
- **A tool name the model invented gets its own budget.** An unknown tool name is answered with a retried tool result listing the available tools, and consumes a budget keyed under the invented name, bounded by the agent-wide `tools` budget. So a model that hallucinates a *different* name each time keeps getting a fresh budget.

Exhausting a tool's budget raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].

### What a retry looks like in message history

A retry that belongs to a tool call *is* that call's result: a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] carrying the call's `tool_call_id`, marked `outcome='retried'`. So a retried call and a successful one look the same in history apart from the outcome, and there is never a second part standing in for the result:

```python {title="retry_history.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def lookup_then_answer(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('lookup_user', {'name': 'John'})])
    elif len(messages) == 3:
        return ModelResponse(
            parts=[ToolCallPart('lookup_user', {'name': 'John Doe'})]
        )
    return ModelResponse(parts=[TextPart('John Doe is user 123.')])


agent = Agent(FunctionModel(lookup_then_answer))


@agent.tool_plain
def lookup_user(name: str) -> int:
    if ' ' not in name:
        raise ModelRetry('Provide the full name.')
    return 123


result = agent.run_sync('Who is John?')
print([type(p).__name__ for m in result.all_messages() for p in m.parts])
"""
[
    'UserPromptPart',
    'ToolCallPart',
    'ToolReturnPart',
    'ToolCallPart',
    'ToolReturnPart',
    'TextPart',
]
"""
```

_(This example is complete, it can be run "as is")_

The part's `content` is the feedback itself — the `ModelRetry` message, or a `ValidationError`'s error details serialized as plain JSON data — and it reaches the model on the provider's native error channel, the same one a failed result uses: Anthropic's `is_error`, Bedrock's `status='error'`, Gemini's `error` key, or an `{"error": ...}` wrapper where the provider has none. A retry that answers no call is a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] instead — see [Feedback that belongs to no tool call](#feedback-that-belongs-to-no-tool-call).

Because retries stay in the history, [reusing that history](message-history.md) in a later run replays the failures to the model. If you don't want the model to see its earlier mistakes, filter them out with a [`ProcessHistory`](capabilities/process-history.md) capability.

[`ToolFailed`][pydantic_ai.exceptions.ToolFailed] records the same kind of part with `outcome='failed'` instead, and does **not** consume the retry budget, so repeated failures are bounded by [`UsageLimits`][pydantic_ai.usage.UsageLimits] rather than by a retry count. The outcome is the whole difference: `'retried'` asks the model to correct the call, `'failed'` tells it to adapt. See [Reporting a Failed Tool Result](tools-advanced.md#tool-failed).

### Auditing rejected tool calls

A retried result also records why the call has to be made again and how far into its budget it is, in its [`retry`][pydantic_ai.messages.ToolReturnPart.retry] (a [`RetryInfo`][pydantic_ai.messages.RetryInfo]). Its `cause` (a [`ToolRetryCause`][pydantic_ai.messages.ToolRetryCause]) is `'validation_error'`, `'model_retry'`, or, for a call the framework rejects before any tool runs, `'unknown_tool'` (the model invented the name) or `'unavailable_tool'` (the tool exists but can't be called at this step). No tool hook sees those last two, since there's no resolved tool to hand it, so match the recorded part instead. `attempt` is which retry of that tool this is, counting from 1, and `max_retries` the limit it was checked against, so `attempt == max_retries` marks the last retry the run will make before it raises; both are `None` for a retry that isn't charged to the budget, such as a tool's first `'unavailable_tool'` refusal. None of it is sent to the model, so recording it doesn't change the request.

With [instrumentation](logfire.md) on, the tool's span carries the same facts as the `pydantic_ai.tool.retry.cause`, `pydantic_ai.tool.retry.attempt` and `pydantic_ai.tool.retry.max_retries` attributes, and an [output function](output.md#output-functions)'s span carries `pydantic_ai.output.retry.*` for a retry of output that answers no tool call.

A capability can collect them from each request the run builds in [`after_node_run`](capabilities/custom.md#hooking-into-the-lifecycle), which doesn't move the run onto the streaming path the way [`on_event`][pydantic_ai.capabilities.AbstractCapability.on_event] does:

```python {title="audit_rejected_calls.py"}
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequestNode,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult
from pydantic_ai.messages import RetryInfo, ToolRetryCause
from pydantic_ai.models.function import AgentInfo, FunctionModel


@dataclass
class AuditRejectedCalls(AbstractCapability[object]):
    rejected: list[tuple[str, RetryInfo[ToolRetryCause] | None]] = field(default_factory=list)

    async def after_node_run(
        self, ctx: RunContext[object], *, node: AgentNode[object], result: NodeResult[object]
    ) -> NodeResult[object]:
        if isinstance(result, ModelRequestNode):
            for part in result.request.parts:
                if isinstance(part, ToolReturnPart) and part.outcome == 'retried':
                    self.rejected.append((part.tool_name, part.retry))
        return result


def call_a_made_up_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('lookup_customer', {'id': 7})])
    return ModelResponse(parts=[TextPart('I can only look up orders.')])


audit = AuditRejectedCalls()
agent = Agent(FunctionModel(call_a_made_up_tool), capabilities=[audit])


@agent.tool_plain
def lookup_order(id: int) -> str:
    return f'Order {id}: shipped'


agent.run_sync('Who is customer 7?')
print(audit.rejected)
#> [('lookup_customer', RetryInfo(cause='unknown_tool', attempt=1, max_retries=1))]
```

_(This example is complete, it can be run "as is")_

When a tool's retry budget runs out, the run raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] instead of recording another retried result, so that last rejection is only visible as the exception.


## Output retries

The output budget is separate from the tool budget, and how it's enforced depends on how the model returns its final answer. [How output retries are enforced](agent.md#how-output-retries-are-enforced) covers both paths; the difference that matters for message history is:

- **Text path** (`output_type=str`, [`TextOutput`](output.md#text-output), [`NativeOutput`](output.md#native-output), [`PromptedOutput`](output.md#prompted-output), and responses with no usable output): one budget shared across the whole run. There is no tool call to answer, so the retry becomes a new [`ModelRequest`][pydantic_ai.messages.ModelRequest] whose only part is a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] — see [Feedback that belongs to no tool call](#feedback-that-belongs-to-no-tool-call).
- **Tool path** ([`ToolOutput`](output.md#tool-output)): the output budget acts as the default limit *per output tool*, overridable with [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries]. The retry is bound to the output tool's `tool_call_id` and looks exactly like a function tool's.

Both are triggered by validation failures, by an [output function](output.md#output-functions) or [output validator](output.md#output-validator-functions) raising `ModelRetry`, and by a model response with nothing actionable in it. Both raise [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] when the budget runs out.

The last of those triggers has an exception: if the output type allows `None` — `output_type=str | None`, for instance — an empty or thinking-only response is a valid final result of `None` rather than a retry. Models that finish their work in a tool call and then emit only thinking would otherwise be pushed into producing filler text. Output validators still run on that `None`, so they can force a retry themselves by raising `ModelRetry`.

Both budgets are configured through one argument:

```python {title="retry_budgets.py"}
from pydantic_ai import Agent

agent = Agent('{{ openai:medium }}', retries=3)  # (1)!

strict_output = Agent('{{ openai:medium }}', retries={'tools': 5, 'output': 1})  # (2)!
```

1. A bare `int` sets both the tool and output budgets.
2. An [`AgentRetries`][pydantic_ai.agent.AgentRetries] dict sets only the keys it names; unnamed keys keep the default of `1`.

The same argument is accepted per run — `agent.run(..., retries=...)` and friends — and for a block of runs via [`agent.override()`][pydantic_ai.agent.Agent.override]. [Which retry limit wins](tools-advanced.md#which-retry-limit-wins) has the full precedence table.

### Feedback that belongs to no tool call

A text-path retry, and a `ModelRetry` from a [model-request hook](hooks.md#triggering-retries-with-modelretry), have no tool call to answer. They become a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart], which records *why* the response couldn't be used and nothing about how to say it:

```python {title="retry_feedback_history.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    ModelRetry,
    RetryFeedbackPart,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('7' if len(messages) == 1 else 'seven')])


agent = Agent(FunctionModel(answer))


@agent.output_validator
def spell_it_out(output: str) -> str:
    if output.isdigit():
        raise ModelRetry('answer with the number spelled out as a word')
    return output


result = agent.run_sync('How many continents are there?')
print([p for m in result.all_messages() for p in m.parts if isinstance(p, RetryFeedbackPart)])
"""
[
    RetryFeedbackPart(
        content='answer with the number spelled out as a word',
        retry=RetryInfo(cause='model_retry', attempt=1, max_retries=1),
        timestamp=datetime.datetime(...),
    )
]
"""
```

_(This example is complete, it can be run "as is")_

The stored part carries no wording at all; each model turns it into a sendable part when the request is built, and its [`retry`][pydantic_ai.messages.RetryFeedbackPart.retry]'s `cause` decides which one. That is why the same history can be replayed against any model. The same [`RetryInfo`][pydantic_ai.messages.RetryInfo] records the `attempt` on the run's output-retry budget and the `max_retries` it was checked against; the attempt is the [`ctx.retry`][pydantic_ai.tools.RunContext.retry] the next output validation sees.

| `cause` | What the model is sent | Why |
|---|---|---|
| `'validation_error'` | a user turn that opens with one line saying the application's validation is speaking, not the user, followed by the errors wrapped in `<validation_errors>…</validation_errors>` | The feedback quotes back the output the model itself wrote, which may in turn quote untrusted user input. The tag is what marks those words off from a person's turn. |
| `'model_retry'` | the same attributed user turn, with the `ModelRetry` message wrapped in `<retry_feedback>…</retry_feedback>` | Output validators routinely interpolate the output they reject into the message, so it can carry the model's own words just as validation errors do. |
| `'no_output'` | a mid-conversation system prompt, degraded to `<system>`-tagged user text where the provider takes no system message mid-conversation | The text is the framework's own fixed request for usable output, and carries nothing the model wrote. A `'no_output'` part with any other content, such as one your own code builds, goes out fenced in `<retry_feedback>` like a `'model_retry'`. |

A closing tag inside any of these wrappers is escaped, so text the model had a hand in cannot end the statement that carries it, and inside the two user-voice fences so is a `<system>` tag, opening or closing, so the model's own words cannot pass themselves off as the framework speaking.

!!! note "Feedback never joins the standing prompt"
    Feedback never joins the run's standing prompt, even when it opens the history — through a hand-built `message_history`, an adapter load, [compaction](capabilities/compaction.md), [`ProcessHistory`](capabilities/process-history.md) filtering, or a model-request hook retrying the first request of a run with no user prompt. Only the [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s the history was written with go to the provider's own system field; `'no_output'` feedback in that position goes out as `<system>`-tagged user text, whatever the provider.

!!! note "`RetryPromptPart` is deprecated"
    Both retries used to be a single [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart], which is deprecated and will be removed in v3. A stored history loads as the two parts above, so to read a retry from history, match [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] with `outcome='retried'` or `RetryFeedbackPart` instead. See the [upgrade guide](changelog.md#deprecated-in-v2-to-be-removed-in-v3) for the details.

## What is never retried

- **`prepare` callbacks.** An exception raised by a per-tool `prepare=`, by [`PrepareTools`](capabilities/prepare-tools.md), or by a [dynamic toolset](toolsets.md) propagates out of the run unchanged — including `ModelRetry`, which is *not* turned into a retry there. To hide a tool for a turn, return `None` from the callback rather than raising.
- **Exceptions other than `ModelRetry` and `ToolFailed`.** Anything else a tool raises propagates out of the run rather than becoming a retry — *unless* a [capability](capabilities/overview.md) implements `on_tool_execute_error`, which sees the exception first and can return a replacement tool result or raise `ModelRetry` to keep the run going. [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] are the exceptions that are neither: they're control flow, not errors, and end the run with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output instead of propagating — except in a [realtime session](realtime/overview.md), which can't pause and instead answers the model with an explanation that the tool can't complete during the session. [Ending a run from inside a tool](timeouts.md#ending-a-run-from-inside-a-tool) has the full table.
- **Whole agent runs.** Nothing re-runs an agent for you. [Pydantic Evals](evals.md) has its own `retry_task` and `retry_evaluators` options for retrying a whole task or evaluator during an evaluation — see [Retry Strategies](evals/how-to/retry-strategies.md). Those sit outside the agent, so a retried task starts with fresh tool and output budgets.
