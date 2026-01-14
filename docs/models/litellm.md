# LiteLLM

## Install

To use LiteLLM, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group (as it uses an OpenAI-compatible API):

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

[LiteLLM](https://www.litellm.ai/) acts as a proxy server that provides a unified interface to multiple LLM providers. Point your `api_base` to your LiteLLM proxy server:

To use custom LLMs, use `custom/` prefix in the model name.

Use [`LiteLLMProvider`][pydantic_ai.providers.litellm.LiteLLMProvider] as follows:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

model = OpenAIChatModel(
    'openai/gpt-5',  # LiteLLM uses provider-prefixed model names
    provider=LiteLLMProvider(
        api_base='http://localhost:4000',  # Your LiteLLM proxy server
        api_key='your-litellm-api-key'
    )
)
agent = Agent(model)

result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
...
```
