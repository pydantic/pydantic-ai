# Fireworks AI

Fireworks AI provides fast inference for open-source and custom models.

## Install

To use Fireworks AI, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

Go to [Fireworks.AI](https://fireworks.ai/) and create an API key in your account settings.

You can set the `FIREWORKS_API_KEY` environment variable and use [`FireworksProvider`][pydantic_ai.providers.fireworks.FireworksProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('fireworks:accounts/fireworks/models/qwq-32b')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.fireworks import FireworksProvider

model = OpenAIChatModel(
    'accounts/fireworks/models/qwq-32b',  # model library available at https://fireworks.ai/models
    provider=FireworksProvider(api_key='your-fireworks-api-key'),
)
agent = Agent(model)
...
```
