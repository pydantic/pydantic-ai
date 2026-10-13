---
title: pgtask Durability
description: Make a Pydantic AI agent run durable on pgtask, a Postgres-based task queue with durable steps -- model requests, MCP calls, and tool calls are checkpointed into steps so a crashed worker resumes mid-run.
---

# pgtask Durability

`PGTaskDurability` makes an agent run durable on [pgtask](https://github.com/Kludex/pgtask), a
Postgres-based task queue with durable steps. Run the agent inside a pgtask task handler and every
model request, MCP call, and tool call is checkpointed as a step, so a worker that crashes mid-run
resumes from the last completed step instead of re-spending tokens on finished work. Outside a task
the capability is transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pgtask/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[pgtask]" "pydantic-ai-slim[openai]"
```

pgtask stores its state in Postgres. Create its schema once per database:

```python {test="skip"}
from pgtask import Client

client = await Client.connect('postgresql://localhost/agents')
await client.migrate()
```

## Quick start

Give the agent a `name`, register a task handler that runs it, then enqueue tasks from a producer and
run them in a worker:

```python {test="skip"}
from pgtask import Client, Task, TaskRegistry, Worker
from pydantic_ai import Agent
from pydantic_ai_harness.pgtask import PGTaskDurability

tasks = TaskRegistry(queue_name='agents')
agent = Agent('openai:gpt-5', name='analyst', capabilities=[PGTaskDurability()])


@tasks.task('analyse')
async def analyse(task: Task, payload: dict[str, str]) -> dict[str, str]:
    result = await agent.run(payload['prompt'])
    return {'output': result.output}


# Producer: enqueue a task.
client = await Client.connect('postgresql://localhost/agents')
await client.enqueue(analyse.request({'prompt': 'Summarize the Q3 report.'}))

# Worker: claim and run tasks (in its own process).
await Worker('postgresql://localhost/agents', tasks).run()
```

## What gets checkpointed

After a crash, pgtask runs the task handler again from the top. Plain Python in the handler runs
again, but each completed step returns its stored result instead of re-issuing the model request or
tool call. Step names are built from the agent's `name` and each toolset's `id`:

| Step name | Operation |
|---|---|
| `{name}__model.request` | one model request segment |
| `{name}__model.request_stream` | one streamed model request segment |
| `{name}__model.compact_messages` | one model message-compaction operation |
| `{name}__model.cancel_suspended_response` | tearing down a suspended response |
| `{name}__capability__{capability_id}.{operation}` | an operation contributed by another capability |
| `{name}__function_toolset__{id}.validate_args` | running a function tool's `args_validator` |
| `{name}__function_toolset__{id}.call_tool:{tool}` | a function tool call |
| `{name}__dynamic_toolset__{id}.get_tools` | building a `DynamicToolset`'s tools |
| `{name}__dynamic_toolset__{id}.call_tool:{tool}` | a `DynamicToolset` tool call |
| `{name}__mcp_server__{id}.get_tools` | listing an MCP server's tools |
| `{name}__mcp_server__{id}.get_instructions` | an MCP server's instructions |
| `{name}__mcp_server__{id}.call_tool` | an MCP tool call |
| `{name}__event_stream_handler` | one event delivered to an `event_stream_handler` |

A model other than the agent's default adds its id to the step name
(`{name}__model.request.{model_id}`). Characters pgtask does not accept in a step name are replaced
with `_`, so tools registered with `@agent.tool` checkpoint under the toolset id `_agent_`. A step
name that recurs, such as a second `agent.run()` in the same task or the same tool called twice, gets
the next pgtask `occurrence` in encounter order. Await one run before starting the next in the same
task.

A tool call that raises `ModelRetry` is checkpointed like any other result, so a replay serves the
retry prompt without running the tool. A tool call that fails the attempt with another exception is
not, and runs again on the next attempt.

MCP servers are connected in the first step that needs them, so a replay served entirely from
checkpoints does not connect.

## Constraints

- The agent needs a `name` (or pass `name=` to `PGTaskDurability`), and every function, MCP, or
  dynamic toolset needs a unique `id`; `Agent(...)` raises a `UserError` otherwise. Don't rename
  either once deployed: in-flight tasks re-run the steps whose names changed.
- A checkpointed tool's return value is stored as JSON, so it must be JSON-serializable. A
  checkpoint that no longer decodes as what the step returns raises `CheckpointDecodeError`, failing
  the run instead of asking the model to retry and running the tool again.
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

`PGTaskDurability` composes with [Code Mode](code-mode.md): each tool call made from `run_code` is
its own step, served from its checkpoint on replay, while the `run_code` body itself re-runs.

## Migrating from `pydantic-ai-pgtask`

Import `PGTaskDurability`, `PGTaskParallelExecutionMode`, and `CheckpointDecodeError` from
`pydantic_ai_harness.pgtask` instead of `pydantic_ai_pgtask`. Step names and checkpoint payloads match
[`pydantic-ai-pgtask`](https://github.com/Kludex/pydantic-ai-pgtask) 0.0.3, so tasks in flight during
the switch resume without repeating work. `DurableTaskContext` and `current_context` are not ported.

## Relation to Step Persistence

`PGTaskDurability` resumes a single run after a crash. [Step Persistence](step-persistence.md)
records runs so they can be resumed, forked, or replayed later as separate invocations. The two
compose.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [pgtask](https://github.com/Kludex/pgtask)

## API reference

::: pydantic_ai_harness.pgtask.PGTaskDurability
