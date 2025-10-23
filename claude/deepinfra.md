# Using DeepInfra with Pydantic AI

## Overview

DeepInfra provides OpenAI-compatible API endpoints for various LLM models, including the Qwen family. This guide shows how to integrate DeepInfra models with Pydantic AI agents.

## Setup

### Prerequisites

1. Get a DeepInfra API token from [deepinfra.com](https://deepinfra.com)
2. Install Pydantic AI: `pip install pydantic-ai`

### Basic Configuration

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import os

# Set up the model with DeepInfra's OpenAI-compatible endpoint
model = OpenAIChatModel(
    model="Qwen/Qwen2.5-72B-Instruct",  # or another model available on DeepInfra
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("DEEPINFRA_TOKEN")  # or pass directly: api_key="your-token-here"
)

# Create an agent with the DeepInfra model
agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant"
)

# Use the agent (async)
result = await agent.run("Hello, how are you?")
print(result.output)

# Or sync
result = agent.run_sync("Hello, how are you?")
print(result.output)
```

## Using Qwen Models with Profile

For Qwen models specifically, you can use the Qwen model profile for better schema handling and compatibility:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.qwen import qwen_model_profile

# Create model with Qwen profile for better schema handling
model = OpenAIChatModel(
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key="your-deepinfra-token",
    model_profile=qwen_model_profile("Qwen/Qwen2.5-72B-Instruct")
)

agent = Agent(
    model=model,
    output_type=str  # or any Pydantic model for structured output
)
```

## Available Qwen Models

Common Qwen models available on DeepInfra (check their site for current availability):

- `Qwen/Qwen2.5-72B-Instruct` - Largest, most capable
- `Qwen/Qwen2.5-32B-Instruct` - Good balance of performance and speed
- `Qwen/Qwen2.5-14B-Instruct` - Medium size
- `Qwen/Qwen2.5-7B-Instruct` - Smaller, faster
- `Qwen/Qwen2.5-3B-Instruct` - Smallest, fastest

## Complete Example with Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.qwen import qwen_model_profile
import os

# Define structured output
class Analysis(BaseModel):
    sentiment: str
    summary: str
    key_points: list[str]

# Configure model
model = OpenAIChatModel(
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("DEEPINFRA_TOKEN"),
    model_profile=qwen_model_profile("Qwen/Qwen2.5-72B-Instruct")
)

# Create agent with structured output
agent = Agent(
    model=model,
    output_type=Analysis,
    system_prompt="You are an expert text analyst"
)

# Run analysis
text = "DeepInfra provides fast and affordable AI infrastructure..."
result = agent.run_sync(f"Analyze this text: {text}")
print(result.output)  # Will be an Analysis object
```

## Using Other Models on DeepInfra

DeepInfra hosts many other models beyond Qwen. The same pattern works:

```python
# Using Llama models
model = OpenAIChatModel(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("DEEPINFRA_TOKEN")
)

# Using Mistral models
model = OpenAIChatModel(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("DEEPINFRA_TOKEN")
)
```

## Environment Configuration

For production, set your API key as an environment variable:

```bash
export DEEPINFRA_TOKEN="your-api-token-here"
```

Then in your code:
```python
model = OpenAIChatModel(
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url="https://api.deepinfra.com/v1/openai",
    api_key=os.getenv("DEEPINFRA_TOKEN")
)
```

## Key Points

1. **Use OpenAIChatModel**: Not the regular OpenAI model class
2. **Set base_url**: Must be `"https://api.deepinfra.com/v1/openai"`
3. **Provide API key**: Either directly or via environment variable
4. **Use exact model names**: As shown on DeepInfra's model list
5. **Optional profile**: Use `qwen_model_profile()` for Qwen models for better compatibility

## Streaming Support

DeepInfra supports streaming responses:

```python
async def stream_example():
    async with agent.run_stream("Tell me a story") as stream:
        async for event in stream:
            print(event.output, end="")
```

## Error Handling

```python
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

try:
    result = agent.run_sync("Your prompt")
except ModelRetry as e:
    print(f"Model requested retry: {e}")
except UnexpectedModelBehavior as e:
    print(f"Unexpected model behavior: {e}")
```

## Notes

- DeepInfra's OpenAI-compatible API supports most OpenAI parameters
- Response format, tools, and function calling are supported
- Check DeepInfra's documentation for model-specific features and limitations
- Pricing and rate limits vary by model - check DeepInfra's pricing page