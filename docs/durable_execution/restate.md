# Durable Execution with Restate

[Restate](https://restate.dev/) is a durable execution framework for building resilient, long-running applications.

## Durable Agent

Any agent can be wrapped in a [`RestateAgent`][pydantic_ai.durable_exec.restate.RestateAgent] to run model calls and (by default) tool calls inside Restate `ctx.run_typed(...)` blocks.

### Installation

Install Restate support with the `restate` optional group:

```bash
pip/uv-add "pydantic-ai[restate]"
```

Or, if you're using the slim package:

```bash
pip/uv-add "pydantic-ai-slim[restate]"
```

### Basic usage

```python {title="restate_agent.py" test="skip"}
import restate

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.restate import RestateAgent

agent = Agent(
    'openai:gpt-5',
    instructions="You're a helpful assistant.",
)


@agent.tool
async def greet(ctx: RunContext, name: str) -> str:
    return f'Hello {name}!'


svc = restate.Service('example')


@svc.handler()
async def handler(ctx: restate.Context, name: str) -> str:
    durable_agent = RestateAgent(agent, restate_context=ctx)
    result = await durable_agent.run(f'Use the greet tool for {name!r}.')
    return result.output
```

## Streaming and event handlers

`RestateAgent` is designed to run inside Restate handlers, where streaming response APIs aren’t supported. Use an `event_stream_handler` and call [`RestateAgent.run()`][pydantic_ai.durable_exec.restate.RestateAgent.run] instead of `run_stream()` or `run_stream_events()`.

## Disabling automatic tool wrapping

By default, function tools are executed inside `ctx.run_typed(...)`. If you want to use the Restate context directly in your tool code (e.g. to call `ctx.run(...)` yourself), initialize the agent with `disable_auto_wrapping_tools=True`.

In that mode, model calls (and MCP tool calls, if used) are still wrapped, but function tools are not.
