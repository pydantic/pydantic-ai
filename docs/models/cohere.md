# Cohere

## Install

To use `CohereModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `cohere` optional group:

```bash
pip/uv-add "pydantic-ai-slim[cohere]"
```

## Configuration

To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.

`CohereModelName` contains a list of the most popular Cohere models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use `CohereModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('cohere:command-r7b-12-2024')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command-r7b-12-2024')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

model = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

You can also customize the `CohereProvider` with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

custom_http_client = AsyncClient(timeout=30)
model = CohereModel(
    'command-r7b-12-2024',
    provider=CohereProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Model settings

You can customize model behavior using [`CohereModelSettings`][pydantic_ai.models.cohere.CohereModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel, CohereModelSettings

model = CohereModel('command-r7b-12-2024')
settings = CohereModelSettings(
    temperature=0.2,
    top_k=40,
)
agent = Agent(model, model_settings=settings)
...
```

## Multi-modal limitations

The Cohere integration does not yet map multi-modal content, so a tool returning an image or a document
raises rather than silently dropping the media. This is a gap in Pydantic AI's Cohere mapper, not in the
Cohere API, which documents image content blocks in user messages on its vision models; it is tracked in
[#7646](https://github.com/pydantic/pydantic-ai/issues/7646).
