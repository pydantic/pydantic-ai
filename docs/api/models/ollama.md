# `pydantic_ai.models.ollama`

## Setup

For details on how to set up authentication with this model, see [model configuration for Ollama](../../models.md#ollama).

## Example local usage

With `ollama` installed, you can run the server with the model you want to use:

```bash {title="terminal-run-ollama"}
ollama run llama3.2
```
(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

```python {title="ollama_example.py"}
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('ollama:llama3.2', result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

## Example using a remote server

```python {title="ollama_example_with_remote_server.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

ollama_model = OllamaModel(
    model_name='qwen2.5-coder:7b',  # (1)!
    base_url='http://192.168.1.74:11434/v1',  # (2)!
)


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent(model=ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

1. The name of the model running on the remote server
2. The url of the remote server

See [`OllamaModel`][pydantic_ai.models.ollama.OllamaModel] for more information

::: pydantic_ai.models.ollama
