# Alibaba Cloud Model Studio (DashScope)

Alibaba Cloud Model Studio (DashScope) provides access to Qwen models via an OpenAI-compatible API.

## Install

To use Alibaba models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

To use Qwen models via [Alibaba Cloud Model Studio (DashScope)](https://www.alibabacloud.com/en/product/modelstudio), you can set the `ALIBABA_API_KEY` (or `DASHSCOPE_API_KEY`) environment variable and use [`AlibabaProvider`][pydantic_ai.providers.alibaba.AlibabaProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('alibaba:qwen-max')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider

model = OpenAIChatModel(
    'qwen-max',
    provider=AlibabaProvider(api_key='your-api-key'),
)
agent = Agent(model)
...
```

The `AlibabaProvider` uses the international DashScope compatible endpoint `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` by default. You can override this by passing a custom `base_url`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider

model = OpenAIChatModel(
    'qwen-max',
    provider=AlibabaProvider(
        api_key='your-api-key',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',  # China region
    ),
)
agent = Agent(model)
...
```
