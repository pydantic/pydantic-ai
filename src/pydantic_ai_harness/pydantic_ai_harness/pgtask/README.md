# pgtask Durability

`PGTaskDurability` lets an agent run survive a crash. It works with [pgtask](https://github.com/Kludex/pgtask),
a task queue that keeps its state in Postgres.

Run the agent inside a pgtask task, and each model request and tool call is saved to Postgres when
it finishes. If the worker crashes, pgtask runs the task again, and the agent gets the saved results
back instead of calling the model or the tool a second time, so you don't pay twice for finished
work.

Outside a pgtask task nothing is saved and the agent runs as usual, except that tool calls still
follow `parallel_execution_mode`, which runs them one at a time by default (see
[Running tools at the same time](#running-tools-at-the-same-time)).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pgtask/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

uv:

```bash
uv add "pydantic-ai-harness[pgtask]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[pgtask]" "pydantic-ai-slim[openai]"
```

pgtask keeps its state in tables in your database. Create them once per database:

```python {test="skip"}
from pgtask import Client

client = await Client.connect('postgresql://localhost/agents')
await client.migrate()
```

## Quick start

Give the agent a `name` and write a pgtask task that runs it. Then add tasks to the queue, and run a
worker to process them:

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


# Add a task to the queue.
client = await Client.connect('postgresql://localhost/agents')
await client.enqueue(analyse.request({'prompt': 'Summarize the Q3 report.'}))

# Run tasks, usually in a separate process.
await Worker('postgresql://localhost/agents', tasks).run()
```

## What happens after a crash

pgtask saves each finished piece of work as a *step*: a name and a result. When a worker crashes,
pgtask runs your task function again from the start. Ordinary code in the function runs again, but
when the agent reaches a model request or tool call that already has a saved step, it uses the saved
result and skips the call.

- A tool that raises `ModelRetry` or `ToolFailed`, or a sub-agent run inside a tool that cancels
  itself or times out (`RunCancelled`), has that outcome saved too, so after a crash the model gets
  the same error and the tool doesn't run again.
- A tool that raises any other exception has nothing saved, so it runs again when pgtask retries the
  task.
- An MCP server is only connected when a call needs it. If every call to it already has a saved
  result, the server isn't contacted.

### Step names

Each step is named after the agent's `name` and the toolset's `id`. These are the names you see in
Postgres, and the reason not to rename an agent or toolset while its tasks are still running.

| Step name | What it saves |
|---|---|
| `{name}__model.request` | a model request |
| `{name}__model.request_stream` | a streamed model request |
| `{name}__model.compact_messages` | a request asking the model to shorten the message history |
| `{name}__model.cancel_suspended_response` | cancelling a response the model provider is still working on in the background |
| `{name}__capability__{capability_id}.{operation}` | work added by another capability |
| `{name}__function_toolset__{id}.validate_args` | checking a tool's arguments with its `args_validator` |
| `{name}__function_toolset__{id}.call_tool:{tool}` | a function tool call |
| `{name}__dynamic_toolset__{id}.get_tools` | building the tools of a `DynamicToolset` |
| `{name}__dynamic_toolset__{id}.call_tool:{tool}` | a `DynamicToolset` tool call |
| `{name}__mcp_server__{id}.get_tools` | fetching an MCP server's tool list |
| `{name}__mcp_server__{id}.get_instructions` | fetching an MCP server's instructions |
| `{name}__mcp_server__{id}.call_tool` | an MCP tool call |
| `{name}__event_stream_handler` | one event passed to `event_stream_handler` |

- A run that uses a model other than the agent's default adds the model's id to the end:
  `{name}__model.request.{model_id}`.
- Characters pgtask doesn't allow in step names become `_`. For example, tools added with
  `@agent.tool` are saved under the toolset id `_agent_`.
- When the same step name comes up more than once in a task, such as the same tool called twice or
  `agent.run()` called twice, pgtask numbers them 0, 1, 2, and so on, in the order they happen.
  Finish one `agent.run()` before starting the next in the same task, so the numbers come out the
  same after a crash.

## Rules

- The agent needs a `name`, or pass `name=` to `PGTaskDurability`. Every function, MCP, and dynamic
  toolset needs its own `id`. `Agent(...)` raises a `UserError` if either is missing.
- Don't rename an agent or toolset while tasks that use it are still running. Their steps are saved
  under the old names, so after a crash the renamed calls run again.
- Tool results are saved as JSON, so tools must return values that can be converted to JSON. A
  result that can't be fails the task, and pgtask retries it until `max_attempts` runs out. If a
  saved result no longer matches the type the call returns, the run fails with
  `CheckpointDecodeError` instead of running the tool again.
- A result is saved only after the call finishes. If the worker crashes after a tool has done
  something but before its result is saved, the tool runs again. Write tools and
  `event_stream_handler` so that running them twice does no harm.
- Inside a task, passing function, MCP, or dynamic toolsets to a single run with
  `run(toolsets=...)` raises a `UserError`. Add them to the agent instead. `ExternalToolset` is
  fine.
- Streaming inside a task isn't live: the whole model response is received and saved first, and
  then its events are sent to you.
- Under [`BatchModelRequests`](https://pydantic.dev/docs/ai/core-concepts/batch/#durable-execution), submitting a request and each check on its batch are
  saved steps, but the waits between checks are plain sleeps: the task has to keep running for as
  long as the batch takes, within its timeout, and a recovered run sleeps again through the waits
  it already did.

## Running tools at the same time

By default the agent runs tool calls one at a time, inside a task or not. To run the tool calls from
one model response at the same time, pass `parallel_execution_mode='parallel_ordered_events'`. Their
events are sent in the order the model asked for the calls, once all of them finish. Plain
`'parallel'` isn't supported, because the step numbers have to come out the same after a crash.

Keep the default if another capability's `before_tool_execute` or `wrap_tool_execute` hook waits on
something before the tool runs. With tools running at the same time, that wait can change which
call of a tool gets which number, so after a crash a call could get another call's saved result.

## Code Mode

With the harness `CodeMode` capability, each tool call made from `run_code` is saved as its own step. After a
crash the code in `run_code` runs again, but its tool calls return their saved results.

## Tracing

`PGTaskDurability` adds no spans of its own: the usual agent run, model request, and tool spans
cover the work. pgtask carries the OpenTelemetry trace from the code that added the task to the
worker, so the run's spans appear in the same trace as the `enqueue` call.

## Migrating from `pydantic-ai-pgtask`

Import `PGTaskDurability`, `PGTaskParallelExecutionMode`, and `CheckpointDecodeError` from
`pydantic_ai_harness.pgtask` instead of `pydantic_ai_pgtask`. Steps are saved under the same names
and in the same format as [`pydantic-ai-pgtask`](https://github.com/Kludex/pydantic-ai-pgtask) 0.0.3,
so tasks that are running when you switch continue without repeating work. `DurableTaskContext` and
`current_context` were not carried over.

## Relation to Step Persistence

`PGTaskDurability` lets one run continue after a crash. [Step Persistence](../step_persistence/)
records runs so you can resume, fork, or replay them later. You can use both together.
