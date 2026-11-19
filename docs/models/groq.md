---
description: "Use models hosted on Groq with Pydantic AI: install the groq optional group, set your API key, configure the provider, and tune the SDK's retry behavior."
---

# Groq

## Install

To use `GroqModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `groq` optional group:

```bash
pip/uv-add "pydantic-ai-slim[groq]"
```

## Configuration

To use [Groq](https://groq.com/) through their API, go to [console.groq.com/keys](https://console.groq.com/keys) and follow your nose until you find the place to generate an API key.

`GroqModelName` lists the latest Groq models for autocomplete. Any other model id the API accepts works too.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GROQ_API_KEY='your-api-key'
```

To route requests through a different host, such as a proxy or a gateway, set `GROQ_BASE_URL` as well:

```bash
export GROQ_BASE_URL='https://your-proxy.example.com'
```

You can then use `GroqModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('groq:llama-3.3-70b-versatile')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

model = GroqModel('llama-3.3-70b-versatile')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

model = GroqModel(
    'llama-3.3-70b-versatile', provider=GroqProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `GroqProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

custom_http_client = AsyncClient(timeout=30)
model = GroqModel(
    'llama-3.3-70b-versatile',
    provider=GroqProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Structured output {#structured-output}

[`NativeOutput`][pydantic_ai.output.NativeOutput] sends Groq's [`json_schema` response format](https://console.groq.com/docs/structured-outputs) on the models below. Unless you pass `strict` to `NativeOutput`, the `strict` value Pydantic AI sends depends on the model:

| Models | `strict` sent |
|---|---|
| `openai/gpt-oss-*`, `qwen/qwen3.8-*` | `true` when the schema meets strict mode's requirements (for example, no field has a default), otherwise `false` |
| `qwen/qwen3.6-*` | `false` |
| `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct` | `true` |
| `moonshotai/*`, `meta-models/muse-spark-open` | `null`; Groq documents best effort as the default |

Other Groq models raise a [`UserError`][pydantic_ai.exceptions.UserError] for `NativeOutput`. [`PromptedOutput`][pydantic_ai.output.PromptedOutput] uses Groq's JSON mode (`json_object`) on every model when the request has no function tools.

Groq can't combine `json_schema` with function tools, so `NativeOutput` on an agent with function tools raises a [`UserError`][pydantic_ai.exceptions.UserError] before the request is sent. Use the default [`ToolOutput`][pydantic_ai.output.ToolOutput] instead.

## Service tier

Groq supports controlling the [service tier](https://console.groq.com/docs/service-tiers) to trade off throughput and reliability.
You can use the unified [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] field or the provider-specific [`groq_service_tier`][pydantic_ai.models.groq.GroqModelSettings.groq_service_tier] field. `groq_service_tier` takes precedence over the unified field when both are set, and accepts Groq's native values (`'auto'`, `'on_demand'`, `'flex'` or `'performance'`).

The unified field maps as follows for Groq:

- `'auto'`: passed through as `'auto'`, so Groq picks the best tier available to you.
- `'default'`: maps to `'on_demand'`, Groq's standard tier.
- `'flex'`: passed through as `'flex'`. Flex requests [fail fast with a 498 error](https://console.groq.com/docs/flex-processing) when flex capacity is unavailable, which raises a [`ModelUnavailableError`][pydantic_ai.exceptions.ModelUnavailableError] that [`ModelFallback`](../capabilities/model-fallback.md) and [`RetryOnModelErrors`](../capabilities/retry-on-model-errors.md) act on.
- `'priority'`: not part of Groq's tier model and silently ignored.

Groq's enterprise `'performance'` tier is only reachable through `groq_service_tier`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModelSettings

agent = Agent(
    'groq:llama-3.3-70b-versatile',
    model_settings=GroqModelSettings(service_tier='flex'),
)
...
```

The tier that served a non-streamed request is stored in [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] `['service_tier']`.

## SDK retries {#sdk-retries}

The `AsyncGroq` client that the provider builds also retries failed requests on its own — `max_retries=2` by default, like the OpenAI client it is modelled on. Pass `groq_client=AsyncGroq(max_retries=0)` to keep the retry policy in your transport alone, or set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to change the count without building the client. See [Provider SDK retries](../retries.md#provider-sdk-retries) for where this layer sits.
