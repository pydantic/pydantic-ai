# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using Pydantic AI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](help.md) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

**Modern Jupyter/IPython (7.0+)**: This environment supports top-level `await` natively. You can use [`Agent.run()`][pydantic_ai.Agent.run] directly in notebook cells without additional setup:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
result = await agent.run('Who let the dogs out?')
```

**Legacy environments or specific integrations**: If you encounter event loop conflicts, use [`nest-asyncio`](https://pypi.org/project/nest-asyncio/):

```python {test="skip"}
import nest_asyncio

from pydantic_ai import Agent

nest_asyncio.apply()

agent = Agent('openai:gpt-4o')
result = agent.run_sync('Who let the dogs out?')
```

**Note**: This also applies to Google Colab and [Marimo](https://github.com/marimo-team/marimo) environments.

## API Key Configuration

### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`

If you're running into issues with setting the API key for your model, visit the [Models](models/overview.md) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

## Monitoring HTTPX Requests

You can use custom `httpx` clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](logfire.md#monitoring-http-requests) to monitor the above.
