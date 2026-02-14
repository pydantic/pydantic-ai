# Durable Execution with Restate

[Restate](https://restate.dev/) is a durable execution framework for building resilient, long-running applications.

## Durable Execution

Restate handlers may be retried and re-executed if the process crashes or a step fails.
To make this safe, move all non-deterministic work (network I/O, model calls, tool calls, etc.) into durable steps using `restate.Context.run_typed(...)`.
Restate records successful step outputs; on retries it replays the handler and returns recorded outputs instead of re-running the step.

For more details, see the Restate docs on [Durable Execution](https://docs.restate.dev/foundations/key-concepts#durable-execution).

## Durable Agent

Any agent can be wrapped in a [`RestateAgent`][pydantic_ai.durable_exec.restate.RestateAgent] to run **model calls**, **tool calls**, and **tool discovery** inside Restate `ctx.run_typed(...)` blocks.

This includes:

- Function tools (by default)
- MCP toolsets ([`MCPServer`][pydantic_ai.mcp.MCPServer])
- FastMCP toolsets ([`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset])
- Dynamic toolsets (e.g. from `@agent.toolset(...)`, including those that resolve to MCP/FastMCP toolsets)

Other custom toolset types are used as-is. If a custom toolset performs I/O, make sure its side effects are safe under Restate retries.

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

The code examples on this page are marked with `test="skip"` because they require a running Restate deployment.

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


app = restate.app(services=[svc])
```

## Running Restate locally

To run Restate locally, follow the [Restate quickstart](https://docs.restate.dev/quickstart).
For example, you can start the Restate server with Docker:

```bash
docker run --name restate_dev --rm \
  -p 8080:8080 -p 9070:9070 -p 9071:9071 \
  --add-host=host.docker.internal:host-gateway \
  docker.restate.dev/restatedev/restate:latest
```

Then serve your Python Restate `app` using an ASGI server (e.g. Hypercorn or Uvicorn).
See the Restate Python docs on [Serving](https://docs.restate.dev/develop/python/serving).

Finally, register the endpoint with Restate:

```bash
restate deployments register http://localhost:9080
# If Restate runs in Docker:
restate deployments register http://host.docker.internal:9080
```

The Restate UI runs on `http://localhost:9070`.

## Streaming and event handlers

`RestateAgent` is designed to run inside Restate handlers, where streaming response APIs aren't supported. Use an `event_stream_handler` and call [`RestateAgent.run()`][pydantic_ai.durable_exec.restate.RestateAgent.run] instead of `run_stream()` or `run_stream_events()`.

### Example

```python {title="restate_agent_streaming.py" test="skip"}
from collections.abc import AsyncIterable

import restate

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.restate import RestateAgent
from pydantic_ai.messages import AgentStreamEvent

agent = Agent('openai:gpt-5')


async def event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        print(event.event_kind)


svc = restate.Service('example')


@svc.handler()
async def handler(ctx: restate.Context, prompt: str) -> str:
    durable_agent = RestateAgent(agent, restate_context=ctx, event_stream_handler=event_stream_handler)
    result = await durable_agent.run(prompt)
    return result.output


app = restate.app(services=[svc])
```

!!! note

    The `event_stream_handler` may run more than once if the durable step is retried.
    By default, each handler call receives a single-event stream (not the full model stream).
    Keep handler side effects idempotent (e.g. write to an idempotent sink, or include a dedup key in your messages).

## Serialization requirements

Restate persists the outputs of durable steps. This means anything returned from model calls and tools must be serializable.

In practice, keep tool outputs to JSON-serializable values and/or Pydantic models.

Values are deserialized as plain JSON-compatible Python types. For example, if a tool returns a Pydantic model, Restate replays it as a `dict`.

## Disabling automatic tool wrapping

By default, function tools are executed inside `ctx.run_typed(...)`. If you want to use the Restate context directly in your tool code (e.g. to call `ctx.run(...)` yourself), initialize the agent with `disable_auto_wrapping_tools=True`.

Restate does not allow nested context operations inside `ctx.run_typed(...)`, so calling `ctx.run(...)`/`ctx.run_typed(...)` inside an automatically-wrapped tool will fail — disable wrapping for those tools.

In that mode, model calls (and MCP/FastMCP tool calls, if used) are still wrapped, but function tools (including those coming from dynamic toolsets like `@agent.toolset(...)`) are not.
For dynamic function tools, toolset resolution and arg validation also run outside `ctx.run_typed(...)`.
When `disable_auto_wrapping_tools=True`, the original `event_stream_handler` runs inside the durable model step and may be replayed on retries.
