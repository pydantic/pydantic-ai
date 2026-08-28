# Crusoe

## Install

To use `CrusoeModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `crusoe` optional group:

```bash
pip/uv-add "pydantic-ai-slim[crusoe]"
```

## Configuration

To use [Crusoe](https://crusoe.ai/) Serverless Inference, go to the [Crusoe Cloud console](https://console.crusoecloud.com/), select Models, and click `Get API Key`.

For a list of available models, see the [Crusoe Serverless Inference documentation](https://docs.crusoecloud.com/serverless-inference/overview).

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export CRUSOE_API_KEY='your-api-key'
```

You can then use `CrusoeModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('crusoe:zai/GLM-5.2')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.crusoe import CrusoeModel

model = CrusoeModel('zai/GLM-5.2')
agent = Agent(model)
...
```

## Model names

Crusoe serves open-weight models from many labs behind one endpoint, and model names carry the lab as a prefix — `zai/GLM-5.2`, `deepseek-ai/DeepSeek-V4-Pro`, `meta-llama/Llama-3.3-70B-Instruct`, `openai/gpt-oss-120b`. That prefix is what selects the [model profile](openai.md#model-profile), so keep it on the name rather than passing the bare model id.

## Structured output

Crusoe serves every model with guided decoding, so [`NativeOutput`][pydantic_ai.output.NativeOutput] works across the catalog — including for model families that don't support native structured output when you reach them through their own provider.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.crusoe import CrusoeModel
from pydantic_ai.providers.crusoe import CrusoeProvider

model = CrusoeModel('zai/GLM-5.2', provider=CrusoeProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

You can also customize the `CrusoeProvider` with a custom `httpx2.AsyncClient`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.crusoe import CrusoeModel
from pydantic_ai.providers.crusoe import CrusoeProvider

custom_http_client = AsyncClient(timeout=30)
model = CrusoeModel(
    'zai/GLM-5.2',
    provider=CrusoeProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```
