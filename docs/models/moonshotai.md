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

## Thinking

Kimi K2.5, K2.6, K2.7 Code and K3 reason by default. The unified [`thinking`](../capabilities/thinking.md) setting maps onto Moonshot's request fields:

- `thinking=False` turns thinking off on Kimi K2.5 and K2.6 by sending `thinking={'type': 'disabled'}`. Kimi K3 can't turn thinking off, so it reasons at its lowest effort, `reasoning_effort='low'`. Kimi K2.7 Code can't turn thinking off either and has no effort setting, so `thinking=False` sends nothing.
- `thinking=True` and the effort levels are sent as `reasoning_effort` (`True` as `'medium'`). `'max'` is sent as `'max'` on Kimi K3, K2.6 and K2.7, and as `'xhigh'` on the other Kimi models.
