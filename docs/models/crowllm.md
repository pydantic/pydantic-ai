---
description: "Use CrowLLM models with Pydantic AI through its OpenAI-compatible gateway, with one key for models from several labs."
---

# CrowLLM

[CrowLLM](https://crowllm.com) is an OpenAI-compatible AI gateway: one base URL and one API key for models from several labs (DeepSeek, Zhipu, Moonshot, Anthropic, Google, xAI). The `crowllm:` prefix selects the gateway; any model id served at `https://crowllm.com/v1/models` works.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key on [crowllm.com](https://crowllm.com), then set the `CROWLLM_API_KEY` environment variable and use [`CrowLLMProvider`][pydantic_ai.providers.crowllm.CrowLLMProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('crowllm:glm-5.3')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.crowllm import CrowLLMProvider

model = OpenAIChatModel('claude-opus-4-7', provider=CrowLLMProvider())
agent = Agent(model)
...
```

## Model notes

- Served ids include `deepseek-v4-flash`, `glm-5.3`, `kimi-k3`, `claude-opus-4-7`, `gemini-3.7-flash` and `grok-4.7:free`; the full list is at `https://crowllm.com/v1/models`.
- Ids ending in `:free` are zero-rated; paid ids bill per token at [crowllm.com/pricing](https://crowllm.com/pricing).
- Models that support reasoning return it in `reasoning_content`.
