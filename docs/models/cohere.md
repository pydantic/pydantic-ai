---
description: "Use Cohere models with Pydantic AI: install the cohere optional group, set your API key, configure the provider and SDK retries, and pass Cohere model settings."
---

# Cohere

## Install

To use `CohereModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `cohere` optional group:

```bash
pip/uv-add "pydantic-ai-slim[cohere]"
```

## Configuration

To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.

`CohereModelName` contains a list of the most popular Cohere models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use `CohereModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('cohere:command-r7b-12-2024')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command-r7b-12-2024')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

model = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

You can also customize the `CohereProvider` with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

custom_http_client = AsyncClient(timeout=30)
model = CohereModel(
    'command-r7b-12-2024',
    provider=CohereProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Streaming

Use [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] to receive text as it is generated:

```python {test="skip"}
import asyncio

from pydantic_ai import Agent

agent = Agent('cohere:command-r7b-12-2024')


async def main():
    async with agent.run_stream('Explain how rainbows form.') as result:
        async for text in result.stream_text(delta=True):
            print(text, end='', flush=True)


asyncio.run(main())
```

Cohere streaming also supports tool calls and thinking output on models that support reasoning.
Leaving the context closes the Cohere SDK stream, including when you stop consuming it early.
See [streaming responses](../output.md#streaming-text) for other ways to consume the stream.

## SDK retries {#sdk-retries}

The Cohere client retries server errors, rate limits and status 408 and 409 twice by default, above whatever transport you pass here. It takes its retry count per request rather than per client, so set [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) to change it, for example to `0` to keep the retry policy in your transport alone. See [Provider SDK retries](../retries.md#provider-sdk-retries) for where this layer sits.

## Model settings

You can customize model behavior using [`CohereModelSettings`][pydantic_ai.models.cohere.CohereModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel, CohereModelSettings

model = CohereModel('command-r7b-12-2024')
settings = CohereModelSettings(
    temperature=0.2,
    top_k=40,
)
agent = Agent(model, model_settings=settings)
...
```
