# Vercel AI Gateway

## Install

To use Vercel AI Gateway, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group (as it uses an OpenAI-compatible API):

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

To use [Vercel's AI Gateway](https://vercel.com/docs/ai-gateway), first follow the [documentation](https://vercel.com/docs/ai-gateway) instructions on obtaining an API key or OIDC token.

You can set the `VERCEL_AI_GATEWAY_API_KEY` and `VERCEL_OIDC_TOKEN` environment variables and use [`VercelProvider`][pydantic_ai.providers.vercel.VercelProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('vercel:anthropic/claude-4-sonnet')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vercel import VercelProvider

model = OpenAIChatModel(
    'anthropic/claude-4-sonnet',
    provider=VercelProvider(api_key='your-vercel-ai-gateway-api-key'),
)
agent = Agent(model)
...
```
