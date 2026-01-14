# xAI (Grok)

## Install

To use xAI models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group (as it uses an OpenAI-compatible API):

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

Go to [xAI API Console](https://console.x.ai/) and create an API key.

You can set the `GROK_API_KEY` environment variable and use [`GrokProvider`][pydantic_ai.providers.grok.GrokProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('grok:grok-4-fast')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.grok import GrokProvider

model = OpenAIChatModel(
    'grok-4-fast',
    provider=GrokProvider(api_key='your-xai-api-key'),
)
agent = Agent(model)
...
```
