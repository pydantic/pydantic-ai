---
description: "Use Mistral models with Pydantic AI: install the mistral optional group, set your API key, and configure the Mistral provider for your agents."
---

# Mistral

## Install

To use `MistralModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `mistral` optional group:

```bash
pip/uv-add "pydantic-ai-slim[mistral]"
```

## Configuration

To use [Mistral](https://mistral.ai) through their API, go to [console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/) and follow your nose until you find the place to generate an API key.

`LatestMistralModelNames` contains a list of the most popular Mistral models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export MISTRAL_API_KEY='your-api-key'
```

You can then use `MistralModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('mistral:mistral-large-latest')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

model = MistralModel(
    'mistral-large-latest', provider=MistralProvider(api_key='your-api-key', base_url='https://<mistral-provider-endpoint>')
)
agent = Agent(model)
...
```

You can also customize the provider with a custom `httpx2.AsyncClient`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

custom_http_client = AsyncClient(timeout=30)
model = MistralModel(
    'mistral-large-latest',
    provider=MistralProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

The Mistral provider also accepts a legacy `httpx.AsyncClient` during Pydantic AI v2, but emits a deprecation warning. Use `httpx2.AsyncClient` for new code; legacy HTTPX client support will be removed in Pydantic AI v3.

## Tool call IDs

Mistral's tool-calling chat template uses tool call IDs of exactly 9 ASCII letters and digits, and models that validate them reject any other ID with a 400 error. That includes IDs from another provider, like OpenAI's `call_...` or Anthropic's `toolu_...`, and the `pyd_ai_...` IDs Pydantic AI generates. A [message history](../message-history.md) started on another model, or a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] that switches to Mistral, can contain such IDs.

To keep these histories working, every Mistral model profile sets [`tool_call_id_length`][pydantic_ai.profiles.ModelProfile.tool_call_id_length] to `9`. Before each request, an ID that doesn't fit is replaced with a stable hash of it that does, and an ID that already fits is sent unchanged. Other providers that serve Mistral models, like Azure AI Foundry, OpenRouter or vLLM, use the same profile. The stored message history keeps the original IDs.
