---
title: Background Tools
description: "Run slow Pydantic AI tools in the background so the agent keeps working, then deliver each tool's result to the model as a follow-up message when it finishes."
---

# Background Tools

`BackgroundTools` lets selected tools run in the background while the agent continues without
waiting. Use it when the model can work on something else until the result is ready.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/background_tools/)

Install the OpenAI provider before running this example:

```bash
pip/uv-add "pydantic-ai-slim[openai]" pydantic-ai-harness
```

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

@agent.tool_plain(metadata={'background': True})
async def slow_research(query: str) -> str:
    """Research a topic thoroughly. Runs in the background."""
    await asyncio.sleep(60)  # Replace with real work.
    return f'Research findings for {query!r}'
```

By default, any tool with `metadata={'background': True}` runs in the background. `BackgroundTools` tells the model how to continue while the tool runs.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Selecting which tools run in the background

`BackgroundTools(tools=...)` accepts the standard [`ToolSelector`](/ai/api/pydantic-ai/tools/#pydantic_ai.tools.ToolSelector):

```python
from pydantic_ai_harness import BackgroundTools

# By metadata key (default)
BackgroundTools()                                 # tools with metadata={'background': True}
BackgroundTools(tools={'background': True})       # explicit form
BackgroundTools(tools={'kind': 'research'})       # custom metadata key

# By name
BackgroundTools(tools=['slow_research', 'deep_dig'])

# By function
BackgroundTools(tools=lambda ctx, td: td.name.startswith('research_'))
```

### Letting the model decide

Use `metadata={'background': 'optional'}` to let the model decide for each call. The model sees a
`run_in_background` argument, but your function does not receive it.

A tool selected by `BackgroundTools(tools=...)` always runs in the background. Sequential tools and
realtime sessions run normally. Do not use `run_in_background` as one of your function's own
parameters.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent('openai:gpt-5.6-sol', capabilities=[BackgroundTools()])

@agent.tool_plain(metadata={'background': 'optional'})
async def slow_research(query: str) -> str:
    return f'Research findings for {query!r}'
```

### Marking tools in bulk

Combine with [`SetToolMetadata`](/ai/capabilities/set-tool-metadata/) or `FunctionToolset.with_metadata(...)` to mark several tools as background without touching individual definitions:

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai_harness import BackgroundTools

async def deep_research(query: str) -> str:
    return f'Research findings for {query!r}'

async def crawl_site(url: str) -> str:
    return f'Crawled {url}'

research_tools = FunctionToolset([deep_research, crawl_site]).with_metadata(background=True)
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[research_tools],
    capabilities=[BackgroundTools()],
)
```

## What happens during a run

The model first receives a message saying that the tool has started. The message includes a task ID.
When the tool finishes, the model receives the result with the same task ID.

Text and files returned by the tool are sent to the model. Application-only metadata is not sent.
If a tool fails unexpectedly, the model sees the error type but not the error message, which may
contain private information. Running out of retries or raising `CancelledError` ends the run. A tool
can call `ctx.cancel()` to stop the run and the other background tools.

A normal run waits for its background tools to finish. A pending call counts toward
`tool_calls_limit`. If the run pauses or stops early, unfinished tools are cancelled and their
results are not delivered. Async tools must allow cancellation; ignoring it can prevent the run from
stopping.

!!! warning
    Python cannot stop a synchronous tool before it returns. Cancelling the run will still wait for
    that tool.

    A synchronous background tool may change shared data at the same time as the agent or another
    tool. Protect shared data from concurrent changes.

## Limitations

- `run_stream()` and `run_stream_sync()` wait for background tools but do not deliver their results.
  The streamed response may only say that the work has started. Use `run_stream_events()`, `run()`,
  `run_sync()`, or consume all of `agent.iter()` when the final response needs the result.
- A sequential tool can overlap background work that started earlier. Tools that must not overlap
  should protect their shared data or should not run in the background.
- Realtime sessions already run tools concurrently, so `BackgroundTools` leaves them unchanged.
- The later result message does not pass through tool-result or tool-error hooks. Validate or limit
  the result inside the tool when this matters.

## Tracing

`BackgroundTools` does not add tracing spans. Pydantic AI records the tool call and its immediate
"started" result. The completed result remains in the agent's message history.

## Durable execution

`BackgroundTools` works with Temporal.

With DBOS, put durable work in an explicit DBOS step and call that step from the background tool.

Do not call `ctx.enqueue()` inside a durable activity or task. Its messages cannot be restored during
replay.

## API

```python {test="skip"}
BackgroundTools(tools: ToolSelector = {'background': True})
```

## Agent spec (YAML/JSON)

Install Agent spec support before using this example:

```bash
pip/uv-add "pydantic-ai-slim[spec]" pydantic-ai-harness
```

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - BackgroundTools: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import BackgroundTools

agent = Agent.from_file('agent.yaml', custom_capability_types=[BackgroundTools])
```

## Further reading

- [Injecting messages during a run](/ai/core-concepts/message-history/#injecting-messages-mid-run)
- [Pydantic AI capabilities](/ai/capabilities/overview/)
