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

## `RuntimeError: Event loop is closed`

Synchronous methods like [`Agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] reuse the thread's current event loop, and install a fresh one if other code closed it. If this error is raised from inside `httpx2` (or legacy `httpx`) during a model request, the agent was already used before its event loop was closed: the provider's HTTP connection pool still holds connections bound to the dead loop. Recreate the agent together with its model and provider (or pass a fresh `http_client` to the provider); reusing an existing `Model` instance keeps the dead connection pool. Avoid closing an event loop that other code is still using.

## [`UserError`][pydantic_ai.exceptions.UserError]: `Agent.run_sync()` and `Agent.run_stream_sync()` cannot be used inside a synchronous tool, output function, or other function called during an agent run

This error means a synchronous [tool](tools.md), [output function](output.md#output-functions), or other function called during an agent run tried to start a nested run with [`Agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] or [`Agent.run_stream_sync()`][pydantic_ai.agent.AbstractAgent.run_stream_sync]. The sync run methods can only be used from regular application code, outside of a run: inside one, the parent run is still waiting on your function while the nested run blocks it, which can deadlock, so Pydantic AI raises this error instead.

Make the delegating function `async def` and `await` the inner run, as shown in [Agent delegation](multi-agent-applications.md#agent-delegation). The parent agent can still be started with `run_sync()` from normal synchronous application code. If the delegating function also needs to do blocking work, push just that part into [`asyncio.to_thread()`][asyncio.to_thread].

## API Key Configuration

### [`UserError`][pydantic_ai.exceptions.UserError]: Set the `[PROVIDER]_API_KEY` environment variable or pass it via the provider's `api_key=...` argument

If you're running into issues with setting the API key for your model, visit the [Models](models/overview.md) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.

To try Pydantic AI without an API key, use the built-in [`'test'` model](testing.md#unit-testing-with-testmodel): [`Agent('test')`][pydantic_ai.agent.Agent].

## Monitoring HTTPX Requests

You can use custom `httpx2` (or legacy `httpx`) clients in your models in order to access specific requests, responses, and headers at runtime.

It's particularly helpful to use `logfire`'s [HTTPX integration](logfire.md#monitoring-http-requests) to monitor the above.
