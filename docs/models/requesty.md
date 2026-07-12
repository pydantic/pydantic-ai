# Requesty

## Install

To use `RequestyModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `requesty` optional group:

```bash
pip/uv-add "pydantic-ai-slim[requesty]"
```

## Configuration

To use [Requesty](https://requesty.ai), first create an API key at [app.requesty.ai/api-keys](https://app.requesty.ai/api-keys).

Requesty is an OpenAI-compatible LLM gateway that exposes models from many providers using `provider/model` naming (e.g. `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`). For a list of available models, see [app.requesty.ai/router/list](https://app.requesty.ai/router/list).

You can set the `REQUESTY_API_KEY` environment variable and use [`RequestyProvider`][pydantic_ai.providers.requesty.RequestyProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('requesty:openai/gpt-4o-mini')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.requesty import RequestyModel
from pydantic_ai.providers.requesty import RequestyProvider

model = RequestyModel(
    'openai/gpt-4o-mini',
    provider=RequestyProvider(api_key='your-requesty-api-key'),
)
agent = Agent(model)
...
```

## App Attribution

Requesty supports app attribution headers to track your application in analytics.

You can pass in an `app_url` and `app_title` when initializing the provider to enable app attribution.

```python
from pydantic_ai.providers.requesty import RequestyProvider

provider = RequestyProvider(
    api_key='your-requesty-api-key',
    app_url='https://your-app.com',
    app_title='Your App',
)
...
```

## `provider` argument

You can also customize the `RequestyProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.requesty import RequestyModel
from pydantic_ai.providers.requesty import RequestyProvider

custom_http_client = AsyncClient(timeout=30)
model = RequestyModel(
    'openai/gpt-4o-mini',
    provider=RequestyProvider(api_key='your-requesty-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```
