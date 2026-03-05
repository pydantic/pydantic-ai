# xAI

## Install

To use [`XaiModel`][pydantic_ai.models.xai.XaiModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `xai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[xai]"
```

## Configuration

To use xAI models from [xAI](https://x.ai/api) through their API, go to [console.x.ai](https://console.x.ai/team/default/api-keys) to create an API key.

[docs.x.ai](https://docs.x.ai/docs/models) contains a list of available xAI models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export XAI_API_KEY='your-api-key'
```

You can then use [`XaiModel`][pydantic_ai.models.xai.XaiModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('xai:grok-4-1-fast-non-reasoning')
...
```

Or initialise the model directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel

# Uses XAI_API_KEY environment variable
model = XaiModel('grok-4-1-fast-non-reasoning')
agent = Agent(model)
...
```

You can also customize the [`XaiModel`][pydantic_ai.models.xai.XaiModel] with a custom provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

# Custom API key
provider = XaiProvider(api_key='your-api-key')
model = XaiModel('grok-4-1-fast-non-reasoning', provider=provider)
agent = Agent(model)
...
```

Or with a custom `xai_sdk.AsyncClient`:

```python
from xai_sdk import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel
from pydantic_ai.providers.xai import XaiProvider

xai_client = AsyncClient(api_key='your-api-key')
provider = XaiProvider(xai_client=xai_client)
model = XaiModel('grok-4-1-fast-non-reasoning', provider=provider)
agent = Agent(model)
...
```
