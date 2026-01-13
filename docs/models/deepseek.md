# DeepSeek

## Install

To use DeepSeek models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group (as it uses an OpenAI-compatible API):

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

To use the [DeepSeek](https://deepseek.com) provider, first create an API key by following the [Quick Start guide](https://api-docs.deepseek.com/).

You can then set the `DEEPSEEK_API_KEY` environment variable and use [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('deepseek:deepseek-chat')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIChatModel(
    'deepseek-chat',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```

You can also customize any provider with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIChatModel(
    'deepseek-chat',
    provider=DeepSeekProvider(
        api_key='your-deepseek-api-key', http_client=custom_http_client
    ),
)
agent = Agent(model)
...
```
