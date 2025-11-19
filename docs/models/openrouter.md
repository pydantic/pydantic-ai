# OpenRouter

## Install

To use `OpenRouterModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

You can set the `OPENROUTER_API_KEY` environment variable and use [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('openrouter:anthropic/claude-3.5-sonnet')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...
```

You can set the `x_title` and `http_referer` parameters in the provider to enable [app attribution](https://openrouter.ai/docs/app-attribution):


```python
from pydantic_ai.providers.openrouter import OpenRouterProvider

provider=OpenRouterProvider(
    api_key='your-openrouter-api-key',
    http_referer='https://your-app.com',
    x_title='Your App',
),
...
```
