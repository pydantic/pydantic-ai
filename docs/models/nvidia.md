---
description: "Use models hosted on NVIDIA NIM, including Nemotron, with Pydantic AI through NVIDIAProvider and the OpenAI-compatible Chat Completions API, with the nvidia: prefix."
---

# NVIDIA NIM

Use models served by [NVIDIA NIM](https://build.nvidia.com/), including NVIDIA's Nemotron models, through [`NVIDIAProvider`][pydantic_ai.providers.nvidia.NVIDIAProvider] and the OpenAI-compatible Chat Completions API. The provider prefix is `nvidia:`.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key on [build.nvidia.com](https://build.nvidia.com/).

You can set the `NVIDIA_API_KEY` environment variable and use [`NVIDIAProvider`][pydantic_ai.providers.nvidia.NVIDIAProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('nvidia:nvidia/nemotron-3-super-120b-a12b')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nvidia import NVIDIAProvider

model = OpenAIChatModel(
    'nvidia/nemotron-3-super-120b-a12b',
    provider=NVIDIAProvider(api_key='your-nvidia-api-key'),
)
agent = Agent(model)
...
```

## Model names

NVIDIA NIM serves models from many labs, and model names carry the lab as a prefix — `nvidia/nemotron-3-super-120b-a12b`, `meta/llama-3.2-90b-vision-instruct`, `deepseek-ai/deepseek-v4.1-flash`. That prefix is what selects the [model profile](compatible-apis.md#model-profile), so keep it on the name. See the [NVIDIA API catalog](https://build.nvidia.com/models) for the available models.

## Self-hosted NIM

To use a NIM you run yourself, pass its OpenAI-compatible endpoint as `base_url`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nvidia import NVIDIAProvider

model = OpenAIChatModel(
    'nvidia/nemotron-3-super-120b-a12b',
    provider=NVIDIAProvider(
        api_key='your-nvidia-api-key', base_url='http://localhost:8000/v1'
    ),
)
agent = Agent(model)
...
```
