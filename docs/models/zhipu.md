# Zhipu AI

## Install

To use Zhipu AI models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group (since Zhipu AI provides an OpenAI-compatible API):

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use [Zhipu AI](https://bigmodel.cn/) (智谱AI) through their API, you need to:

1. Visit [bigmodel.cn](https://bigmodel.cn) and create an account
2. Go to [API Keys management](https://bigmodel.cn/usercenter/proj-mgmt/apikeys)
3. Create a new API key

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ZHIPU_API_KEY='your-api-key'
```

You can then use Zhipu AI models by name:

```python
from pydantic_ai import Agent

agent = Agent('zhipu:glm-4.5')
...
```

Or initialise the model directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel('glm-4.5', provider='zhipu')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `ZhipuProvider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.zhipu import ZhipuProvider

model = OpenAIChatModel(
    'glm-4.5', provider=ZhipuProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `ZhipuProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.zhipu import ZhipuProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIChatModel(
    'glm-4.5',
    provider=ZhipuProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Available Models

Zhipu AI offers several models through their OpenAI-compatible API:

### GLM-4 Series

- **`glm-4.6`**: Latest flagship model with 205K context window
- **`glm-4.5`**: High-performance model with 131K context window
- **`glm-4.5-air`**: Balanced performance and cost with 131K context
- **`glm-4.5-flash`**: Fast response model with 131K context

### Vision Models

- **`glm-4v-plus`**: Advanced vision understanding model
- **`glm-4v`**: Vision model for image analysis
- **`glm-4.5v`**: Vision-enabled variant with 64K context

### Specialized Models

- **`codegeex-4`**: Code generation and understanding model

## Features

### Function Calling

Zhipu AI models support function calling (tool use):

```python
from pydantic_ai import Agent, RunContext

agent = Agent(
    'zhipu:glm-4.5',
    system_prompt='You are a helpful assistant with access to tools.',
)

@agent.tool
async def get_weather(ctx: RunContext[None], location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny, 25°C"

result = await agent.run('What is the weather in Beijing?')
print(result.output)
```

### Streaming

Zhipu AI supports streaming responses:

```python
from pydantic_ai import Agent

agent = Agent('zhipu:glm-4.5')

async with agent.run_stream('Tell me a story') as response:
    async for message in response.stream_text():
        print(message, end='', flush=True)
```

### Vision Understanding

Use vision models to analyze images:

```python
from pydantic_ai import Agent

agent = Agent('zhipu:glm-4v-plus')

result = await agent.run(
    'What is in this image?',
    message_history=[
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Describe this image:'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}}
            ]
        }
    ]
)
print(result.output)
```

## Important Notes

### Temperature Range

Unlike OpenAI, Zhipu AI requires temperature to be in the range `(0, 1)` (exclusive). Setting `temperature=0` is not supported and will cause an error.

```python
# This will work
agent = Agent('zhipu:glm-4.5', model_settings={'temperature': 0.1})

# This will NOT work with Zhipu AIs
# agent = Agent('zhipu:glm-4.5', model_settings={'temperature': 0})
```

### Strict Mode

Zhipu AI does not support OpenAI's strict mode for tool definitions. The framework automatically handles this by setting `openai_supports_strict_tool_definition=False` in the model profile.

## Advanced Features

### Thinking Mode

GLM-4.5 and GLM-4.5-Air support a "thinking" mode for complex reasoning tasks. This can be enabled using the `extra_body` parameter:

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.zhipu import ZhipuProvider

model = OpenAIChatModel('glm-4.5', provider=ZhipuProvider(api_key='your-api-key'))

# Note: Thinking mode requires using the OpenAI client directly
# or passing extra_body through model_settings
```

## API Reference

For more details, see:

- [ZhipuProvider API Reference][pydantic_ai.providers.zhipu.ZhipuProvider]
- [Zhipu AI Official Documentation](https://docs.bigmodel.cn/)
