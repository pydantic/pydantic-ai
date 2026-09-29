# NVIDIA

## Install

To use `NVIDIAModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `nvidia` optional group:

```bash
pip/uv-add "pydantic-ai-slim[nvidia]"
```

## Configuration

To use [NVIDIA NIM](https://build.nvidia.com) through their API, go to [build.nvidia.com](https://build.nvidia.com) and generate an API key.

For a list of available models, see the [NVIDIA models documentation](https://build.nvidia.com/models).

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export NVIDIA_API_KEY='your-api-key'
```

You can then use `NVIDIAModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('nvidia:meta/llama-3.1-70b-instruct')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.nvidia import NVIDIAModel

model = NVIDIAModel('meta/llama-3.1-70b-instruct')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.nvidia import NVIDIAModel
from pydantic_ai.providers.nvidia import NVIDIAProvider

model = NVIDIAModel(
    'meta/llama-3.1-70b-instruct', provider=NVIDIAProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `NVIDIAProvider` with a custom `httpx2.AsyncClient`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.nvidia import NVIDIAModel
from pydantic_ai.providers.nvidia import NVIDIAProvider

custom_http_client = AsyncClient(timeout=30)
model = NVIDIAModel(
    'meta/llama-3.1-70b-instruct',
    provider=NVIDIAProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Base URL

NVIDIA NIM endpoints can be customized via the `NVIDIA_BASE_URL` environment variable or by passing a `base_url` argument to the provider:

```python
from pydantic_ai.providers.nvidia import NVIDIAProvider

provider = NVIDIAProvider(base_url='https://integrate.api.nvidia.com/v1')
```

By default, the provider uses `https://integrate.api.nvidia.com/v1`.