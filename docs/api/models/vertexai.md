# `pydantic_ai.models.vertexai`

Custom interface to the `*-aiplatform.googleapis.com` API for Gemini models.

This model inherits from [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] with just the URL and auth method
changed, it relies on the VertexAI
[`generateContent`](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/generateContent)
and
[`streamGenerateContent`](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.endpoints/streamGenerateContent)
function endpoints
having the same schemas as the equivalent [Gemini endpoints][pydantic_ai.models.gemini.GeminiModel].

## Setup

For details on how to set up authentication with this model as well as a comparison with the `generativelanguage.googleapis.com` API used by [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel],
see [model configuration for Gemini via VertexAI](../../models.md#gemini-via-vertexai).

## Example Usage

With the default google project already configured in your environment using "application default credentials":

```python {title="vertex_example_env.py"}
from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel('gemini-1.5-flash')
agent = Agent(model)
result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
```

Or using a service account JSON file:

```python {title="vertex_example_service_account.py"}
from pydantic_ai import Agent
from pydantic_ai.models.vertexai import VertexAIModel

model = VertexAIModel(
    'gemini-1.5-flash',
    service_account_file='path/to/service-account.json',
)
agent = Agent(model)
result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
```

::: pydantic_ai.models.vertexai
