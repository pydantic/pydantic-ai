# Outlines

## Install

To use `OutlinesModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `outlines` optional group:

```bash
pip/uv-add "pydantic-ai-slim[outlines]"
```

As Outlines is a library allowing you to run model from various different providers, it does not include the necessary dependencies for any model, but instead requires you to install the optional group for the provider you want to use. For instance:

```bash
pip/uv-add "outlines[transformers]"
```

Or

```bash
pip/uv-add "outlines[mlxlm]"
```

## Model Initialization

As Outlines is not an inference provider, but instead a library allowing you to run bith local and API-based models, instantiating the model is a bit different from the other models available on Pydantic-AI.

To initialize the `OutlinesModel` through the `__init__` method, the first argument you must provide has to be an `outlines.Model` or an `outlines.AsyncModel` instance.

For instance:

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

outlines_model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained('erwanf/gpt2-mini'),
    AutoTokenizer.from_pretrained('erwanf/gpt2-mini')
)
model = OutlinesModel(outlines_model)
```

Alternatively, you can use some `OutlinesModel` class methods made to load a specific type of Outlines model directly. To do so, you must provide as argument the same arguments you would have given to the associated Outlines model loading function.

For instance:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from pydantic_ai.models.outlines import OutlinesModel

model = OutlinesModel.from_transformers(
    AutoModelForCausalLM.from_pretrained('erwanf/gpt2-mini'),
    AutoTokenizer.from_pretrained('erwanf/gpt2-mini')
)
```

There are methods for the 6 Outlines models that are officially supported in the integration into Pydantic-AI:
- `from_transformers`
- `from_llama_cpp`
- `from_mlxlm`
- `from_sglang`
- `from_dottxt`
- `from_vllm_offline`

As you already providing an Outlines model instance, there is no need to provide an `OutlinesProvider` yourself.

## Running the model

Once you have initialized an `OutlinesModel`, you can use it with an Agent as with all other Pydantic-AI models.

Outlines does not support tools yet, but support for that feature will be added in the near future.
