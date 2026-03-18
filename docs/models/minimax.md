# MiniMax

## Install

To use MiniMax models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `minimax` optional group:

```bash
pip/uv-add "pydantic-ai-slim[minimax]"
```

## Configuration

To use [MiniMax](https://www.minimax.io/) through their API, go to [platform.minimax.io](https://platform.minimax.io/) and create an API key.

### Available models

| Model | Description |
|-------|-------------|
| `MiniMax-M2.7` | Latest flagship model with enhanced reasoning and coding |
| `MiniMax-M2.7-highspeed` | High-speed version of M2.7 for low-latency scenarios |
| `MiniMax-M2.5` | Previous flagship model — peak performance, ultimate value |
| `MiniMax-M2.5-highspeed` | Same performance, faster and more agile |

All models support a 204,800 token context window.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export MINIMAX_API_KEY='your-api-key'
```

You can then use MiniMax models by name:

```python
from pydantic_ai import Agent

agent = Agent('minimax:MiniMax-M2.7')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel('MiniMax-M2.7', provider='minimax')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `MiniMaxProvider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.minimax import MiniMaxProvider

model = OpenAIChatModel(
    'MiniMax-M2.7', provider=MiniMaxProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `MiniMaxProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.minimax import MiniMaxProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIChatModel(
    'MiniMax-M2.7',
    provider=MiniMaxProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Model constraints

- **Temperature**: Must be in the range `(0.0, 1.0]` — zero is not accepted by the MiniMax API.
- **JSON output modes**: `response_format` is not supported. Use prompted output instead.
