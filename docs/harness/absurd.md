---
title: Absurd Durability
description: Make a Pydantic AI agent run durable on Absurd, a Postgres-based durable-execution engine -- model requests, MCP calls, and function tool calls are checkpointed into steps so a crashed worker resumes mid-run.
---

# Absurd Durability

`AbsurdDurability` makes an agent run durable on [Absurd](https://github.com/earendil-works/absurd),
a Postgres-based durable-execution engine. Run the agent inside an Absurd task handler and every
model request, MCP call, and function tool call is checkpointed as a step, so a worker that crashes
mid-run resumes from the last completed step instead of re-spending tokens on finished work. Outside
a task the capability is transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/absurd/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[absurd]" "pydantic-ai-slim[openai]"
```

Absurd stores its state in Postgres. Install the Absurd schema once per database (see the
[Absurd repository](https://github.com/earendil-works/absurd)), then create a queue:

```python {test="skip"}
from absurd_sdk import AsyncAbsurd

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
await absurd.create_queue()
```

## Quick start

Give the agent a `name`, register a task handler that runs it, then spawn tasks from a producer and
execute them in a worker:

```python {test="skip"}
from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
from pydantic_ai import Agent
from pydantic_ai_harness.absurd import AbsurdDurability

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


@absurd.register_task(name='analyse')
async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
    assert isinstance(params, dict)
    result = await agent.run(params['prompt'])
    return {'output': result.output}


# Producer: enqueue a task.
await absurd.spawn('analyse', {'prompt': 'Summarize the Q3 report.'})

# Worker: claim and run tasks (in its own process).
await absurd.start_worker()
```

The handler must be async: a synchronous `TaskContext` raises a `UserError`.

## What gets checkpointed

After a crash, Absurd re-runs the task handler from the top. Plain Python in the handler runs again,
but each completed step returns its stored result instead of re-issuing the model request or tool
call. Step names are built from the agent's `name` and each toolset's `id`:

| Step name | Operation |
|---|---|
| `{name}__model.request` | one model request segment |
| `{name}__model.request_stream` | one streamed model request segment |
| `{name}__model.compact_messages` | one model message-compaction operation |
| `{name}__model.cancel_suspended_response` | tearing down a suspended response |
| `{name}__capability__{capability_id}.{operation}` | an operation contributed by another capability |
| `{name}__function_toolset__{id}.validate_args` | running a function tool's `args_validator` |
| `{name}__function_toolset__{id}.call_tool:{tool}` | a function tool call |
| `{name}__mcp_server__{id}.get_tools` | listing an MCP server's tools |
| `{name}__mcp_server__{id}.get_instructions` | an MCP server's instructions |
| `{name}__mcp_server__{id}.call_tool` | an MCP tool call |
| `{name}__event_stream_handler` | one event delivered to an `event_stream_handler` |

A model other than the agent's default adds its id to the step name
(`{name}__model.request.{model_id}`). A step name that recurs, such as a second `agent.run()` in
the same task or the same tool called twice, gets an encounter-order suffix (`#2`, `#3`, ...). Await
one run before starting the next in the same task.

These are not checkpointed and run again on replay:

- a tool call that raises `ModelRetry`, `ToolFailed`, `CallDeferred`, or `ApprovalRequired`;
- tools from a `DynamicToolset`.

## Constraints

- The agent needs a `name` (or pass `name=` to `AbsurdDurability`), and every function, MCP, or
  dynamic toolset needs a unique `id`; `Agent(...)` raises a `UserError` otherwise. Don't rename
  either once deployed: in-flight tasks re-run the steps whose names changed.
- A checkpointed tool's return value is stored as JSON, so it must be JSON-serializable.
- A step is checkpointed after it runs, so a crash between a tool's side effect and its checkpoint
  re-runs the tool. Keep side effects in tools and in an `event_stream_handler` idempotent.
- Function, MCP, or dynamic toolsets passed per run with `run(toolsets=...)` inside a task raise a
  `UserError`; register them on the agent instead. `ExternalToolset` is allowed.
- Streaming inside a task is not live: the model stream is consumed inside the step, then its
  captured events are replayed to the caller.

## Parallel execution

Tool calls run sequentially by default. Pass `parallel_execution_mode='parallel_ordered_events'` to
run them concurrently; plain `'parallel'` is not supported. Stay sequential if a capability hook
such as `before_tool_execute` or `wrap_tool_execute` awaits before the tool runs, as that can
reorder concurrent calls of the same tool on replay.

## Code Mode

`AbsurdDurability` composes with [Code Mode](code-mode.md): each tool call made from `run_code` is
its own step, served from its checkpoint on replay, while the `run_code` body itself re-runs.

## Migrating from `pydantic-ai-absurd`

Import `AbsurdDurability` and `AbsurdParallelExecutionMode` from `pydantic_ai_harness.absurd`
instead of `pydantic_ai_absurd`. Step names and checkpoint payloads match
[`pydantic-ai-absurd`](https://github.com/Kludex/pydantic-ai-absurd) 0.8, so tasks in flight during
the switch resume without repeating work. A toolset without an `id` now needs one, and adding it
re-runs that toolset's steps in tasks in flight. `AbsurdAgent`, `AbsurdModel`,
`AbsurdFunctionToolset`, and `AbsurdMCPToolset` are not ported; move to the capability first.

## Relation to Step Persistence

`AbsurdDurability` resumes a single run after a crash. [Step Persistence](step-persistence.md)
records runs so they can be resumed, forked, or replayed later as separate invocations. The two
compose.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Absurd](https://github.com/earendil-works/absurd)

## API reference

::: pydantic_ai_harness.absurd.AbsurdDurability
