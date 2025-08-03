# OpenRouter

## Install

To use `OpenRouterModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openrouter` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openrouter]"
```

## Configuration

To use [OpenRouter](https://openrouter.ai/) through their API, go to [openrouter.ai/keys](https://openrouter.ai/keys) and follow your nose until you find the place to generate an API key.

`OpenRouterModelName` contains a list of available OpenRouter models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export OPENROUTER_API_KEY='your-api-key'
```

You can then use `OpenRouterModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('openrouter:google/gemini-2.5-flash-lite')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('google/gemini-2.5-flash-lite')
agent = Agent(model)

result = await agent.run('What is the capital of France?')
print(result.output)
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'google/gemini-2.5-flash-lite', provider=OpenRouterProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `OpenRouterProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenRouterModel(
    'google/gemini-2.5-flash-lite',
    provider=OpenRouterProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```
