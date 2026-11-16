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

[`NativeOutput`][pydantic_ai.output.NativeOutput] sends Groq's [`json_schema` response format](https://console.groq.com/docs/structured-outputs) on the models below. Unless you pass `strict` to `NativeOutput`, the model decides whether the request uses strict mode:

| Models | `strict` sent |
|---|---|
| `openai/gpt-oss-*`, `qwen/qwen3.8-*` | `true` when the schema meets strict mode's requirements (for example, no field has a default), otherwise `false` |
| `qwen/qwen3.6-*` | `false` |
| `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct` | `true` |
| `moonshotai/*`, `meta-models/muse-spark-open` | `null`; Groq documents best effort as the default |

Other Groq models raise a [`UserError`][pydantic_ai.exceptions.UserError] for `NativeOutput`. [`PromptedOutput`][pydantic_ai.output.PromptedOutput] uses Groq's JSON mode (`json_object`) on every model when the request has no function tools.

Groq can't combine `json_schema` with function tools, so `NativeOutput` on an agent with function tools raises a [`UserError`][pydantic_ai.exceptions.UserError] before the request is sent. Use the default [`ToolOutput`][pydantic_ai.output.ToolOutput] instead.

## SDK retries {#sdk-retries}

The `AsyncGroq` client that the provider builds also retries failed requests on its own — `max_retries=2` by default, like the OpenAI client it is modelled on. Pass `groq_client=AsyncGroq(max_retries=0)` to keep the retry policy in your transport alone, or set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to change the count without building the client. See [Provider SDK retries](../retries.md#provider-sdk-retries) for where this layer sits.
