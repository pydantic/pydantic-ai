# Grok

## Install

To use `GrokModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `grok` optional group:

```bash
pip/uv-add "pydantic-ai-slim[grok]"
```

## Configuration

To use Grok from [xAI](https://x.ai/api) through their API, go your [console.x.ai](https://console.x.ai/team/default/api-keys) and follow your nose until you find the place to create an API key.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export XAI_API_KEY='your-api-key'
```

You can then use `GrokModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('grok:grok-4-1-fast-non-reasoning')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.grok import GrokModel

model = GrokModel('grok-4-1-fast-non-reasoning')
agent = Agent(model)
...
```

You can provide your own `api_key` inline like so:

```python
from pydantic_ai import Agent
from pydantic_ai.models.grok import GrokModel

model = GrokModel('grok-4-1-fast-non-reasoning', api_key='your-api-key')
agent = Agent(model)
...
```

You can also customize the `GrokModel` with a custom `xai_sdk.AsyncClient`:

```python
from xai_sdk import AsyncClient
async_client = AsyncClient(api_key='your-api-key')

from pydantic_ai import Agent
from pydantic_ai.models.grok import GrokModel

model = GrokModel('grok-4-1-fast-non-reasoning', client=async_client)
agent = Agent(model)
...
```
