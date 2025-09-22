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
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('google/gemini-2.5-flash-lite', api_key='your-api-key')
agent = Agent(model)
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel('google/gemini-2.5-flash-lite')
agent = Agent(model)
...
```

## Custom API Key

You can provide a custom API key directly to the model:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

model = OpenRouterModel(
    'google/gemini-2.5-flash-lite', api_key='your-api-key'
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the OpenRouter model with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

custom_http_client = AsyncClient(timeout=30)
model = OpenRouterModel(
    'google/gemini-2.5-flash-lite',
    api_key='your-api-key',
    http_client=custom_http_client,
)
agent = Agent(model)
...
```

## Structured Output

You can use OpenRouter models with structured output by providing a Pydantic model as the `output_type`:

```python {noqa="I001"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel

class OlympicsLocation(BaseModel):
    city: str
    country: str

model = OpenRouterModel('google/gemini-2.5-flash-lite', api_key='your-api-key')
agent = Agent(model, output_type=OlympicsLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(f'City: {result.output.city}')
#> City: London
print(f'Country: {result.output.country}')
#> Country: United Kingdom
```

The model will validate and parse the response into your specified Pydantic model, allowing type-safe access to structured data fields via `result.output.field_name`.
