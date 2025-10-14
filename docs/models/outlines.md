# Outlines

## Install

As Outlines is a library allowing you to run models from various different providers, it does not include the necessary dependencies for any provider by default. As a result, to use the [`OutlinesModel`][pydantic_ai.models.OutlinesModel], you must install `pydantic-ai-slim` with an optional group composed of outlines, a dash, and the name of the specific model provider you would use through Outlines. For instance:

```bash
pip/uv-add "pydantic-ai-slim[outlines-transformers]"
```

Or

```bash
pip/uv-add "pydantic-ai-slim[outlines-mlxlm]"
```

There are 5 optional groups for the 5 model providers supported through Outlines:

- `outlines-transformers`
- `outlines-llamacpp`
- `outlines-mlxlm`
- `outlines-sglang`
- `outlines-vllm-offline`

## Model Initialization

As Outlines is not an inference provider, but instead a library allowing you to run both local and API-based models, instantiating the model is a bit different from the other models available on Pydantic AI.

To initialize the `OutlinesModel` through the `__init__` method, the first argument you must provide has to be an `outlines.Model` or an `outlines.AsyncModel` instance.

For instance:

```python {test="skip"}
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

outlines_model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained('erwanf/gpt2-mini'),
    AutoTokenizer.from_pretrained('erwanf/gpt2-mini')
)
model = OutlinesModel(outlines_model)
```

As you already providing an Outlines model instance, there is no need to provide an `OutlinesProvider` yourself.

### Model Loading Methods

Alternatively, you can use some `OutlinesModel` class methods made to load a specific type of Outlines model directly. To do so, you must provide as arguments the same arguments you would have given to the associated Outlines model loading function (except in the case of SGLang).

There are methods for the 5 Outlines models that are officially supported in the integration into Pydantic AI:

- [`from_transformers`][pydantic_ai.models.OutlinesModel.from_transformers]
- [`from_llamacpp`][pydantic_ai.models.OutlinesModel.from_llamacpp]
- [`from_mlxlm`][pydantic_ai.models.OutlinesModel.from_mlxlm]
- [`from_sglang`][pydantic_ai.models.OutlinesModel.from_sglang]
- [`from_vllm_offline`][pydantic_ai.models.OutlinesModel.from_vllm_offline]

#### Transformers

```python {test="skip"}
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_transformers(
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'),
    AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
)
```

#### LlamaCpp

```python {test="skip"}
from llama_cpp import Llama

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_llamacpp(
    Llama.from_pretrained(
        repo_id='TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
        filename='mistral-7b-instruct-v0.2.Q5_K_M.gguf',
    )
)
```

#### MLXLM

```python {test="skip"}
from mlx_lm import load

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_mlxlm(
    *load('mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit')
)
```

#### SGLang

```
from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_sglang(
    'http://localhost:11434',
    'api_key',
    'meta-llama/Llama-3.1-8B'
)
```

#### vLLM Offline

```python {test="skip"}
from vllm import LLM

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_vllm_offline(
    LLM('microsoft/Phi-3-mini-4k-instruct')
)
```

## Running the model

Once you have initialized an `OutlinesModel`, you can use it with an Agent as with all other Pydantic AI models.

As Outlines is focused on structured output, this provider supports the `output_type` component through the `NativeOutput` format.

```python {test="skip"}
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai import Agent
from pydantic_ai.models.outlines import OutlinesModel
from pydantic_ai.settings import ModelSettings


class Box(BaseModel):
    """Class representing a box"""
    width: int
    height: int
    depth: int
    units: str

model = OutlinesModel.from_transformers(
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct'),
    AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
)
agent = Agent(model, output_type=Box)

result = agent.run_sync(
    'Give me the dimensions of a box',
    model_settings=ModelSettings(extra_body={'max_new_tokens': 100})
)
print(result.output) # width=20 height=30 depth=40 units='cm'
```

Outlines does not support tools yet, but support for that feature will be added in the near future.
