---
description: "Use models served by NVIDIA NIM with Pydantic AI through NvidiaProvider and the OpenAI-compatible Chat Completions API, with the nvidia: prefix."
---

# NVIDIA NIM

Use models served by [NVIDIA NIM](https://build.nvidia.com/) through [`NvidiaProvider`][pydantic_ai.providers.nvidia.NvidiaProvider] and the OpenAI-compatible Chat Completions API. The provider prefix is `nvidia:`.

NVIDIA NIM serves models from many publishers under `<publisher>/<model>` IDs, such as
`nvidia/nemotron-3.5-super-vl-120b-a12b`. The [NVIDIA API catalog](https://build.nvidia.com/models)
lists the models currently available.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Create an API key on [build.nvidia.com](https://build.nvidia.com/).

You can set the `NVIDIA_API_KEY` environment variable and use [`NvidiaProvider`][pydantic_ai.providers.nvidia.NvidiaProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nvidia import NvidiaProvider

model = OpenAIChatModel(
    'nvidia/nemotron-3.5-super-vl-120b-a12b',
    provider=NvidiaProvider(api_key='your-nvidia-api-key'),
)
agent = Agent(model)
...
```

Other models you can use include `nvidia/nemotron-3.5-lightning-30b-a3b` and the smaller `nvidia/nemotron-3.5-bolt-e4b`.

### Custom base URL

`NvidiaProvider` uses `https://integrate.api.nvidia.com/v1` by default. To use a self-hosted NIM
deployment, set `NVIDIA_BASE_URL` to its OpenAI-compatible endpoint:

```bash
export NVIDIA_BASE_URL="http://localhost:8000/v1"
```

You can then use the model-string configuration shown above.

Alternatively, pass `base_url='http://localhost:8000/v1'` to
[`NvidiaProvider`][pydantic_ai.providers.nvidia.NvidiaProvider].
An explicit `base_url` overrides `NVIDIA_BASE_URL`.

An API key is only required for the hosted API. If your deployment requires bearer auth, also set
`NVIDIA_API_KEY` or pass `api_key`.

## Embeddings

The `nvidia:` prefix also works with [`Embedder`][pydantic_ai.embeddings.Embedder] for NIM embedding models:

```python
from pydantic_ai import Embedder

embedder = Embedder('nvidia:nvidia/nemotron-3.5-embed-1b')
```

NIM's asymmetric embedding models require an `input_type`, so
[`embed_query()`][pydantic_ai.embeddings.Embedder.embed_query] sends `'query'` and
[`embed_documents()`][pydantic_ai.embeddings.Embedder.embed_documents] sends `'passage'`.
An `input_type` you set in the `extra_body` setting takes precedence.
