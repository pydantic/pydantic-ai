# Moonshot AI / Kimi

Use Moonshot AI's Kimi models through [`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider] and the OpenAI-compatible Chat Completions API. The provider prefix is `moonshotai:`.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key in the [Moonshot Console](https://platform.moonshot.ai/console).

You can set the `MOONSHOTAI_API_KEY` environment variable and use [`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('moonshotai:kimi-k3')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.moonshotai import MoonshotAIProvider

model = OpenAIChatModel(
    'kimi-k3',
    provider=MoonshotAIProvider(api_key='your-moonshot-api-key'),
)
agent = Agent(model)
...
```
