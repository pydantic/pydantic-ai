# `pydantic_ai.models.ollama`

## Setup

For details on how to set up authentication with this model, see [model configuration for Ollama](../../install.md#ollama).

## Example usage

With `ollama` installed, you can run the server with the model you want to use:

```bash title="terminal-run-ollama"
ollama run llama3.2
```
(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

```py title="ollama_example.py"
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('ollama:llama3.2', result_type=CityLocation)

result = agent.run_sync('Where the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.cost())
#> Cost(request_tokens=56, response_tokens=8, total_tokens=64, details=None)
```

See [`OllamaModel`][pydantic_ai.models.ollama.OllamaModel] for more information

::: pydantic_ai.models.ollama
