# Troubleshooting

Below are suggestions on how to fix some common errors you might encounter while using Pydantic AI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](help.md) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Jupyter Notebook Errors

### `RuntimeError: This event loop is already running`

**Modern Jupyter/IPython (7.0+)**: This environment supports top-level `await` natively. You can use [`Agent.run()`][pydantic_ai.agent.Agent.run] directly in notebook cells without additional setup:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')
result = await agent.run('Who let the dogs out?')
```

**Legacy environments or specific integrations**: If you encounter event loop conflicts, use [`nest-asyncio`](https://pypi.org/project/nest-asyncio/):

```python {test="skip"}
import nest_asyncio

from pydantic_ai import Agent

nest_asyncio.apply()

agent = Agent('openai:gpt-5.2')
result = agent.run_sync('Who let the dogs out?')
```

**Note**: This also applies to Google Colab and [Marimo](https://github.com/marimo-team/marimo) environments.

## API Key Configuration

### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`

If you're running into issues with setting the API key for your model, visit the [Models](models/overview.md) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

## Nested `run_sync()` / calling another agent from sync code

If you call an agent inside another agent *synchronously* (for example: a tool function or output tool calling `other_agent.run_sync(...)`), you may see the process **hang/freeeze**.

This can happen because `run_sync()` has to bridge async internals into a sync API, and nested sync bridging can deadlock depending on the environment.

Recommended fixes:

- Prefer fully-async composition: make your outer entrypoint `async` and use [`Agent.run()`][pydantic_ai.agent.Agent.run] everywhere.
- If you need multiple agents, keep the whole call stack async (use `await other_agent.run(...)` from tools/output tools).
- As a last resort, move the inner agent call to a separate thread/process.

Related issue: https://github.com/pydantic/pydantic-ai/issues/3899

## Monitoring HTTPX Requests

You can use custom `httpx` clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](logfire.md#monitoring-http-requests) to monitor the above.
