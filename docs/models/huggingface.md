# Hugging Face

## Install

To use `HuggingFaceModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `huggingface` optional group:

```bash
pip/uv-add "pydantic-ai-slim[huggingface]"
```

## Configuration

To use [HuggingFace](https://huggingface.co/) through their main API, go to
[Inference Providers documentation](https://huggingface.co/docs/inference-providers/pricing) for all the details,
and you can generate a Hugging Face access token here: https://huggingface.co/settings/tokens.

## Hugging Face access token

Once you have a Hugging Face access token, you can set it as an environment variable:

```bash
export HF_TOKEN='hf_token'
```

You can then use [`HuggingFaceModel`][pydantic_ai.models.huggingface.HuggingFaceModel] by name:

```python
from pydantic_ai import Agent

agent = Agent('huggingface:Qwen/Qwen3-235B-A22B')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

model = HuggingFaceModel('Qwen/Qwen3-235B-A22B')
agent = Agent(model)
...
```

By default, the [`HuggingFaceModel`][pydantic_ai.models.huggingface.HuggingFaceModel] uses the
[`HuggingFaceProvider`][pydantic_ai.providers.huggingface.HuggingFaceProvider] that will select automatically
the first of the inference providers (Cerebras, Together AI, Cohere..etc) available for the model, sorted by your
preferred order in https://hf.co/settings/inference-providers.

## Configure the provider

If you want to pass parameters in code to the provider, you can programmatically instantiate the
[`HuggingFaceProvider`][pydantic_ai.providers.huggingface.HuggingFaceProvider] and pass it to the model:

```python
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

model = HuggingFaceModel('Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(api_key='hf_token', provider='nebius'))
agent = Agent(model)
...
```

## Custom Hugging Face client

[`HuggingFaceProvider`][pydantic_ai.providers.huggingface.HuggingFaceProvider] also accepts a custom
[`AsyncInferenceClient`][huggingface_hub.AsyncInferenceClient] client via the `hf_client` parameter, so you can customise
the `headers`, `bill_to` (billing to an HF organization you're a member of), `base_url` etc. as defined in the
[Hugging Face Hub python library docs](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client).

```python
from huggingface_hub import AsyncInferenceClient

from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.providers.huggingface import HuggingFaceProvider

client = AsyncInferenceClient(
    bill_to='openai',
    api_key='hf_token',
    provider='fireworks-ai',
)

model = HuggingFaceModel(
    'Qwen/Qwen3-235B-A22B',
    provider=HuggingFaceProvider(hf_client=client),
)
agent = Agent(model)
...
```
