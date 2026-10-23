---
description: "Use Moonshot AI's Kimi models with Pydantic AI through MoonshotAIProvider and the OpenAI-compatible Chat Completions API, with the moonshotai: prefix."
---

# Moonshot AI / Kimi

Use Moonshot AI's Kimi models through [`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider] and the OpenAI-compatible Chat Completions API. The provider prefix is `moonshotai:`.

The [Kimi K3.1 Pro API guide](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart)
covers `kimi-k3.1-pro` and `kimi-k3.1-pro-highspeed`.
Use `thinking='low'`, `thinking='high'`, or `thinking='max'`
to select a documented reasoning effort, or omit `thinking` to use the provider's default.

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

agent = Agent('moonshotai:kimi-k3.1-pro')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.moonshotai import MoonshotAIProvider

model = OpenAIChatModel(
    'kimi-k3.1-pro',
    provider=MoonshotAIProvider(api_key='your-moonshot-api-key'),
)
agent = Agent(model)
...
```

### Custom base URL

`MoonshotAIProvider` uses `https://api.moonshot.ai/v1` by default. To use the Chinese endpoint,
set `MOONSHOTAI_BASE_URL` and an API key from the [Chinese console](https://platform.kimi.com/console/api-keys):

```bash
export MOONSHOTAI_API_KEY="your-moonshot-api-key"
export MOONSHOTAI_BASE_URL="https://api.moonshot.cn/v1"
```

You can then use the model-string configuration shown above.

Alternatively, pass `base_url='https://api.moonshot.cn/v1'` to
[`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider].
An explicit `base_url` overrides `MOONSHOTAI_BASE_URL`.
