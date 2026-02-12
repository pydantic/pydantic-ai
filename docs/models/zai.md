# Z.AI

## Install

To use `ZaiModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `zai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[zai]"
```

## Configuration

To use [Z.AI](https://z.ai/) (Zhipu AI) through their API, go to [z.ai](https://z.ai/manage-apikey/apikey-list) and generate an API key.

For a list of available models, see the [Z.AI documentation](https://docs.z.ai/).

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ZAI_API_KEY='your-api-key'
```

You can then use `ZaiModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('zai:glm-4.7')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel

model = ZaiModel('glm-4.7')
agent = Agent(model)
...
```

## Thinking mode

Z.AI's `glm-4.7` model supports thinking/reasoning mode, where the model produces reasoning content before the final response. You can enable this via `ZaiModelSettings`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModelSettings

agent = Agent(
    'zai:glm-4.7',
    model_settings=ZaiModelSettings(zai_thinking=True),
)
...
```

### Preserved thinking

For multi-turn conversations, you can enable preserved thinking to retain reasoning content from prior assistant responses. This improves coherence across turns:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModelSettings

agent = Agent(
    'zai:glm-4.7',
    model_settings=ZaiModelSettings(zai_thinking=True, zai_clear_thinking=False),
)
...
```

When using preserved thinking, the complete and unmodified `reasoning_content` from prior turns is automatically sent back to the API by Pydantic AI.

See the [Z.AI thinking mode documentation](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking) for more details.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.zai import ZaiProvider

model = ZaiModel(
    'glm-4.7', provider=ZaiProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `ZaiProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.zai import ZaiProvider

custom_http_client = AsyncClient(timeout=30)
model = ZaiModel(
    'glm-4.7',
    provider=ZaiProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```
