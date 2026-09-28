# Absurd Durability

`AbsurdDurability` makes an agent run durable on [Absurd](https://github.com/earendil-works/absurd), a
Postgres-based durable-execution engine by Armin Ronacher (Python SDK `absurd-sdk`). Attach the
capability and call `agent.run()` inside an Absurd task handler: every model request, MCP call, and
function tool call is checkpointed into an Absurd step (`ctx.step(...)`), so if a worker crashes
part-way through a run it resumes from the last completed step, without re-spending tokens on
finished work. A step is checkpointed after it runs, so a crash between a tool's side effect and its
checkpoint re-runs the tool: keep tool side effects idempotent. Outside a task the capability is
transparent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/absurd/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Installation

uv:

```bash
uv add "pydantic-ai-harness[absurd]"
```

pip:

```bash
pip install "pydantic-ai-harness[absurd]"
```

Absurd stores its state in Postgres. Once per database, install the Absurd schema and create a
queue. The schema SQL and the queue helpers ship with the upstream project; see the
[Absurd repository](https://github.com/earendil-works/absurd) for the schema file and setup steps.

```python {test="skip"}
from absurd_sdk import AsyncAbsurd

absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
await absurd.create_queue()
```

## Quick start

Construct the agent with the capability, register a task handler that runs it, then split the work
across a producer that spawns tasks and a worker that executes them. The agent needs a `name`; it
prefixes every checkpoint step.

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

# Worker: claim and run tasks (in its own process). `start_worker` polls continuously.
await absurd.start_worker()
```

The task handler runs inside an `AsyncTaskContext`, which is how the capability knows to checkpoint.
Call `agent.run()` (async) from an async handler; a synchronous `TaskContext` raises a `UserError`
because an agent run cannot be awaited from one.

## What gets checkpointed, and what replay means

Each of these runs in its own `ctx.step(...)`, so once it completes its result is checkpointed and
served from the checkpoint on replay rather than being recomputed. Step names are built from the
agent's `name` and each toolset's `id`:

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

A model operation that does not use the agent's default model records its model id in the step name
(for example, `{name}__model.request.{model_id}`). A toolset without an `id` drops the `__{id}`
segment, for example `{name}__mcp_server.call_tool`. A tool listing served from an MCP toolset's
`cache_tools` cache takes no step.

Replay means: after a crash, Absurd re-runs the task handler from the top. Plain Python in the
handler body runs again, but each checkpointed step returns its stored result instead of re-issuing
the model request or re-calling the tool.

Some work is not checkpointed and runs again on replay:

- a tool that raises `ModelRetry`, `ToolFailed`, `CallDeferred`, or `ApprovalRequired` stores no checkpoint, so the
  call is repeated;
- a `DynamicToolset` (including one added with `@agent.toolset`) is not wrapped, so its listing and
  tool calls run as plain code.

Calling `agent.run()` more than once in a single task handler works: a step name that recurs (a
second run's model request, or the same tool called twice in one response) is disambiguated by
Absurd's encounter-order counter (`{name}#2`, `{name}#3`, ...), so each occurrence keeps its own
checkpoint and lines up on replay. Await one run before starting the next: two runs of the same agent
going at once in one task would claim each other's checkpoints.

## Constraints

- The agent needs a `name` (or pass `name=` to `AbsurdDurability`). It and a toolset's `id` are part
  of every step name, so don't change them once deployed: a rename orphans the checkpoints of
  in-flight tasks, which then re-run those steps.
- A checkpointed tool's return value is stored in Postgres as JSON, so it must be JSON-serializable.
- The executing toolsets are fixed when the agent is constructed. Passing a function, MCP, or
  dynamic toolset per-run via `run(toolsets=...)` inside a task raises a `UserError`, because a
  runtime toolset has no registered steps and would re-run its side effects on recovery.
  Non-executing toolsets such as `ExternalToolset` are allowed at runtime.
- Streaming inside a task is a replay, not a live wire: the model stream is consumed and captured
  inside the step, and the run-side stream replays the captured events.
- An `event_stream_handler` handles model events live inside the model-request step, and each tool
  event in its own `{name}__event_stream_handler` step. Either can run again if the run recovers
  before that step is checkpointed, so keep the handler's side effects idempotent.
- The capability emits no spans of its own; core's model-request and tool spans cover the
  checkpointed work.

## Parallel execution

`parallel_execution_mode` defaults to `'sequential'` and applies to every run of the agent. Set it
to `'parallel_ordered_events'` to run tool calls concurrently while emitting their result events in
model-call order. Plain `'parallel'` is excluded because completion-order event delivery can assign
repeated event-handler step names to different calls on replay. A tool call claims its step slot when it reaches
the step, so a capability hook that awaits before the tool runs (`before_tool_execute`,
`wrap_tool_execute`) can reorder concurrent calls of the same tool between the first run and a
replay; keep `'sequential'` for agents with such hooks.

## CodeMode composition

`AbsurdDurability` composes with the harness `CodeMode` capability. A tool call made from inside
`run_code` is checkpointed as its own step and served from its checkpoint on replay; the `run_code`
body itself re-runs.

## Migrating from `pydantic-ai-absurd`

Replace `from pydantic_ai_absurd import AbsurdDurability` with
`from pydantic_ai_harness.absurd import AbsurdDurability` (and the same for
`AbsurdParallelExecutionMode`). Step names and checkpoint payloads match
[`pydantic-ai-absurd`](https://github.com/Kludex/pydantic-ai-absurd) 0.8, so tasks in flight during
the switch resume under the new package. The `AbsurdAgent` wrapper (deprecated in 0.8) and the
`AbsurdModel`, `AbsurdFunctionToolset`, and `AbsurdMCPToolset` classes are not ported; move to the
capability first.

## Relation to Step Persistence

`AbsurdDurability` and the harness [Step Persistence](../step_persistence/) capability solve
different problems and compose. Absurd gives crash-resume *within* a single run: a worker that dies
mid-run picks up from the last completed step. Step Persistence records step events and continuation snapshots *across* runs, so a run can be resumed, forked, or replayed as a separate
invocation later. Use Absurd for durability against crashes during a run, and Step Persistence to
persist and resume runs as first-class records.
