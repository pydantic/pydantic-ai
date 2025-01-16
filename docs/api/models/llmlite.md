# `pydantic_ai.models.ollama`

## Setup

For details on how to set up authentication with this model, see [model configuration for Ollama](../../models.md#lmstudio).

## Example local usage

With `LMStudio` installed, you can run the server.

1. Use the developer model
2. Go to section developer
3. Select a model to load
4. And load the model
Then run your code, here's a minimal example:

```python {title="ollama_example.py"}
from pydantic import BaseModel

from pydantic_ai import Agent

class CityLocation(BaseModel):
    city: str
    country: str

agent = Agent('llmite:mistral-nemo-instruct-2407', result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2024?')
print(result.data)
#> city='Paris' country='France'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

## Example using a remote server

```python {title="ollama_example_with_remote_server.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.lmstudio import LMStudioModel

lmstudio_model = LMStudioModel(
    model_name='mistral-nemo-instruct-2407',    # (1)!
    base_url='http://127.0.0.1:1234/v1',        # (2)!
)


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent(model=lmstudio_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2024?')
print(result.data)
#> city='Paris' country='France'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

1. The `model's API identifier` running on the remote server.
2. The `Reachable at` url of the remote server.

See [`LMStudioModel`][pydantic_ai.models.lmstudio.LMStudioModel] for more information

::: pydantic_ai.models.lmstudio
