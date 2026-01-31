# Databricks

## Install

To use `DatabricksModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `databricks` optional group:

```bash
pip/uv-add "pydantic-ai-slim[databricks]"
```

## Configuration

To use [Databricks](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/) you will need access to a Databricks workspace. You can get one for free from [here](https://www.databricks.com/learn/free-edition).

`DatabricksModelName` contains a list of the most popular models available on databricks via the Foundation Models API.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export DATABRICKS_API_KEY='your-api-key'
export DATABRICKS_BASE_URL='your-databricks-workspace-url'
```

You can then use `DatabricksModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('databricks:databricks-gpt-5-2')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.databricks import DatabricksModel

model = DatabricksModel('databricks-gpt-5-2')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.databricks import DatabricksModel
from pydantic_ai.providers.databricks import DatabricksProvider

model = DatabricksModel('databricks-gpt-5-2', provider=DatabricksProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

You can also customize the `DatabricksProvider` with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.databricks import DatabricksModel
from pydantic_ai.providers.databricks import DatabricksProvider

custom_http_client = AsyncClient(timeout=30)
model = DatabricksModel(
    'databricks-gpt-5-2',
    provider=DatabricksProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```
