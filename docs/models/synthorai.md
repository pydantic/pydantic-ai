# Synthorai

## Install

To use `SynthoraiModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `synthorai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[synthorai]"
```

## Configuration

[Synthorai](https://synthorai.io) is an OpenAI-compatible gateway that routes to models from several upstream providers behind one endpoint and key.

To get an API key, follow the [Synthorai quickstart](https://synthorai.io/docs/quickstart/).

For the list of available models, see the [model catalog](https://synthorai.io/models/). Note that `/v1/models` returns the models a given key is permitted to use, which is narrower than the full catalog, so what you can reach depends on the key.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export SYNTHORAI_API_KEY='your-api-key'
```

You can then use `SynthoraiModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('synthorai:claude-opus-5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.synthorai import SynthoraiModel

model = SynthoraiModel('claude-opus-5')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.synthorai import SynthoraiModel
from pydantic_ai.providers.synthorai import SynthoraiProvider

model = SynthoraiModel(
    'claude-opus-5', provider=SynthoraiProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `SynthoraiProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.synthorai import SynthoraiModel
from pydantic_ai.providers.synthorai import SynthoraiProvider

custom_http_client = AsyncClient(timeout=30)
model = SynthoraiModel(
    'claude-opus-5',
    provider=SynthoraiProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Model profiles

Synthorai model ids carry no vendor prefix, so `SynthoraiProvider` picks a model profile from the id's leading substring: `claude-` resolves to the Anthropic profile, `gemini-` to Google, `deepseek-` to DeepSeek, `glm-` to Z.AI, `kimi-` to Moonshot AI, `qwen` to Qwen, and `gpt-` to the OpenAI-compatible base.

Families the catalog serves that have no profile in this repository fall through to the OpenAI-compatible base rather than being mapped to an approximate one.
