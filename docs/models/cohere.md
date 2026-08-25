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

## Image inputs

Cohere's [vision models](https://docs.cohere.com/docs/image-inputs) accept images in user messages,
so an [`ImageUrl`][pydantic_ai.messages.ImageUrl] or an image
[`BinaryContent`][pydantic_ai.messages.BinaryContent] is sent as an `image_url` content block:

```python
from pydantic_ai import Agent, ImageUrl
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command-a-vision-07-2025')
agent = Agent(model)
prompt = ['What breed is this dog?', ImageUrl(url='https://iili.io/3Hs4FMg.png')]
...
```

On a model without vision support, an image raises a
[`UserError`][pydantic_ai.exceptions.UserError].

Cohere tool results are text-only, so an image a tool returns is referenced by identifier in the tool
result and sent in full in a following user message. Other content kinds (documents, audio, video)
are not supported in either position.

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
