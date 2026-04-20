# Z.AI

## Install

To use [`ZaiModel`][pydantic_ai.models.zai.ZaiModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `zai` optional group:

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

You can then use [`ZaiModel`][pydantic_ai.models.zai.ZaiModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('zai:glm-5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel

model = ZaiModel('glm-5')
agent = Agent(model)
...
```

## Thinking mode

Z.AI's `glm-5`, `glm-4.7`, `glm-4.6` (hybrid thinking), and `glm-4.5` (interleaved thinking) models support thinking/reasoning mode, where the model produces reasoning content before the final response. You can enable this via [`ZaiModelSettings`][pydantic_ai.models.zai.ZaiModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModelSettings

agent = Agent(
    'zai:glm-5',
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
    'zai:glm-5',
    model_settings=ZaiModelSettings(zai_thinking=True, zai_clear_thinking=False),
)
...
```

When using preserved thinking, the complete and unmodified `reasoning_content` from prior turns is automatically sent back to the API by Pydantic AI.

See the [Z.AI thinking mode documentation](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking) for more details.

## `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the `provider` argument. In the simplest case, pass [`ZaiProvider`][pydantic_ai.providers.zai.ZaiProvider] with just an API key. If you also want to customize the underlying `httpx.AsyncClient`, pass it when constructing the provider:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.zai import ZaiProvider

custom_http_client = AsyncClient(timeout=30)
model = ZaiModel(
    'glm-5',
    provider=ZaiProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

If you do not need a custom HTTP client, omit the `http_client=custom_http_client` argument.
