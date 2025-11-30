# OpenRouter

## Install

To use `OpenRouterModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openrouter` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openrouter]"
```

## Configuration

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

You can set the `OPENROUTER_API_KEY` environment variable and use [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('openrouter:anthropic/claude-3.5-sonnet')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenRouterModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...
```

## App Attribution

OpenRouter has an [app attribution](https://openrouter.ai/docs/app-attribution) feature to track your application in their public ranking and analytics.

You can pass in an `app_url` and `app_title` when initializing the provider to enable app attribution.

```python
from pydantic_ai.providers.openrouter import OpenRouterProvider

provider=OpenRouterProvider(
    api_key='your-openrouter-api-key',
    app_url='https://your-app.com',
    app_title='Your App',
),
...
```

## Model Settings

You can customize model behavior using [`OpenRouterModelSettings`][pydantic_ai.models.openrouter.OpenRouterModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

settings = OpenRouterModelSettings(
    openrouter_reasoning={
        'effort': 'high',
    },
    openrouter_usage={
        'include': True,
    }
)
model = OpenRouterModel('openai/gpt-5')
agent = Agent(model, model_settings=settings)
...
```

## Image Generation

You can use OpenRouter models that support image generation with the `openrouter_modalities` setting:

```python
from pydantic_ai import Agent, BinaryImage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

settings = OpenRouterModelSettings(
    openrouter_modalities=['image', 'text']
)
model = OpenRouterModel('google/gemini-2.5-flash-image-preview', provider=provider)
agent = Agent(model=model, output_type=str | BinaryImage, model_settings=settings)

result = await agent.run('A cat')
# result.output is a BinaryImage
```

You can further customize image generation using `openrouter_image_config`:

```python
from pydantic_ai.models.openrouter import OpenRouterModelSettings

settings = OpenRouterModelSettings(
    openrouter_modalities=['image', 'text'],
    openrouter_image_config={'aspect_ratio': '3:2'}
)
```

> Available aspect ratios: `'1:1'`, `'2:3'`, `'3:2'`, `'3:4'`, `'4:3'`, `'4:5'`, `'5:4'`, `'9:16'`, `'16:9'`, `'21:9'`.

Image generation also works with streaming:

```python
from pydantic_ai import Agent, BinaryImage
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

settings = OpenRouterModelSettings(
    openrouter_modalities=['image', 'text'],
    openrouter_image_config={'aspect_ratio': '3:2'}
)
model = OpenRouterModel('google/gemini-2.5-flash-image-preview', provider=provider)
agent = Agent(model=model, output_type=str | BinaryImage, model_settings=settings)

async with agent.run_stream('A dog') as result:
    async for output in result.stream_output():
        if isinstance(output, str):
            print(output)
        elif isinstance(output, BinaryImage):
            # Handle the generated image
            print(f'Generated image: {output.media_type}')
```
