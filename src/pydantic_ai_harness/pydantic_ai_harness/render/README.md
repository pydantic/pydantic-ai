# Render Workflows

Run long-running AI agents in the background with separate retries, timeouts, and compute settings for model requests
and tool calls. The [Render Workflows](https://render.com/docs/workflows) integration runs the agent loop in an entry
task and supported operations as child tasks, each with its own status, logs, and result. For example, a tool that
processes a large document can have a longer timeout than the model calls around it.

The integration does not checkpoint agent progress: an entry-task retry starts the agent again and can repeat
completed model requests and tool calls. Use it when your application can handle repeated work.

If you only need background execution, a native Render task around `agent.run(...)` may be enough.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render/) | [Detailed reference](#task-definitions-and-child-task-runs)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Before you start

The example uses Pydantic AI's `TestModel` and a mock weather tool, so it needs no LLM API key or Render deployment.
The model generates sample tool arguments and returns fixed text rather than interpreting the prompt.

You need Python 3.10 or later. Install the [Render CLI](https://render.com/docs/cli) 2.28.0 or later separately. The integration requires Render SDK 1.2.0 or later.

## 1. Install dependencies

In a new directory, initialize a project with [uv](https://docs.astral.sh/uv/) using `uv init --bare`; for an existing uv project, run the installation command directly. If you use pip, first create and activate a virtual environment.

uv:

```bash
uv add "pydantic-ai-harness[render]"
```

pip:

```bash
pip install "pydantic-ai-harness[render]"
```

The `render` extra supplies the Python SDK and its `render-workflows` executable.

## 2. Add the integration to an agent

Save the following as `app.py` in your project directory:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from render import TaskContext, Workflows

from pydantic_ai_harness import RenderWorkflows


async def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


app = Workflows()
workflows = RenderWorkflows(app)
agent = Agent(
    TestModel(custom_output_text='The workflow completed.'),
    name='support',
    tools=[get_weather],
    capabilities=[workflows],
)


@workflows.task
async def support(ctx: TaskContext, prompt: str) -> str:
    del ctx
    return (await agent.run(prompt)).output
```

If you already have an agent, keep its model, instructions, and tools. Add `workflows` to its `capabilities` list,
and put the call to `agent.run(...)` inside a function decorated with `@workflows.task`, as shown above. Create the
agent and its tools at module load time so Render can register their tasks before starting the worker; attach
one `RenderWorkflows` instance per agent and use the same `app` object throughout.

The `support` function is the task you will submit. Render supplies its `TaskContext`, so callers only pass the
`prompt` argument. The `@workflows.task` decorator lets the integration run model requests and supported tool calls
as child tasks. Using a plain `@app.task`, or calling the agent outside a `@workflows.task` function, leaves those
operations in the same process as the agent.

## 3. Start the local task server

From the directory containing `app.py`, run:

```bash
render workflows dev -- uv run render-workflows app:app
```

The server listens on port `8120` and lists the registered tasks, including `support` and `support__model.request`.
Keep it running for the following commands.

With pip, run `render workflows dev -- render-workflows app:app` from your activated environment. The [local development guide](https://render.com/docs/workflows-local-development) covers port options and the local server's limits.

## 4. Run the agent and check its result

In another terminal, start the `support` task with a prompt:

```bash
render workflows tasks runs start support --local --input '["Check the weather."]' --confirm --output json
```

Copy the returned `id` into the following command to check the job's status and retrieve its result:

```bash
render workflows tasks runs show <RUN_ID> --local --output json
```

A successful run has `status: "completed"` and `results: ["The workflow completed."]`. If the job is still running, repeat the status command until it finishes.

To inspect the model and tool calls separately, list their task runs:

```bash
render workflows tasks runs list 'support__model.request' --local --output json
render workflows tasks runs list 'support__function_toolset__<agent>.call_tool' --local --output json
```

This example produces two model requests and one weather-tool call, each with a `parentTaskRunId` matching the entry task's ID. Keep the server running to try the Python client in step 7. Stop it with Ctrl+C when you are done; its in-memory run history is lost on shutdown.

## 5. Use your own model and tools

Install the provider dependency for the OpenAI example:

uv:

```bash
uv add "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-slim[openai]"
```

To make real model requests, set `OPENAI_API_KEY` in the terminal where you start the worker. In `app.py`, replace
`TestModel(custom_output_text='The workflow completed.')` with `'openai:gpt-5.6-sol'` and remove the `TestModel` import.
Restart the server and repeat step 4; the response now comes from the model and will vary with the prompt.

Replace `get_weather` with your own tool functions and update the agent's `tools` list to use them. Their arguments
and results must be JSON serializable because the integration sends them between tasks. For a document-processing
tool, for example, pass a document ID or storage URL and load the document inside the tool.

You can keep `TestModel` and the mock weather tool while working through the remaining steps without an LLM API key.

## 6. Configure retries and timeouts

To give model requests and tool calls different retry limits, timeouts, or compute, replace the `app` and
`workflows` declarations in `app.py` with the following block. Keep it before the `Agent(...)` declaration:

```python
from render import Options, Retry, Workflows

from pydantic_ai_harness import RenderWorkflows

app = Workflows()
workflows = RenderWorkflows(
    app,
    model_options=Options(
        retry=Retry(max_retries=3, wait_duration_ms=1_000),
        timeout_seconds=300,
        plan='flex',
    ),
    tool_options=Options(
        retry=Retry(max_retries=2, wait_duration_ms=1_000),
        timeout_seconds=120,
        plan='2c-4g',
    ),
)
```

To set the timeout for the whole agent run, replace `@workflows.task` above `support` with
`@workflows.task(timeout_seconds=600, plan='flex')`. Restart the local server and repeat step 4 to check that the
updated task still runs. These settings are fixed when tasks register and cannot change between invocations.

For per-tool configuration, see [Task options and tool opt-out](#task-options-and-tool-opt-out).

A failed child task can retry while the entry task waits. Retrying the entry task restarts `agent.run(...)`, so model
and tool calls that already finished may run again. There is no checkpoint resume or replay of completed steps.
The Render SDK also does not let this integration assign stable idempotency keys to child calls.
Either kind of retry can repeat an external action performed before its result was recorded; use application-level
idempotency keys or deduplication for writes and API requests that must not happen twice.

Render retries are separate from Pydantic AI's `ModelRetry`, which asks the model to correct or reconsider a call,
and from retries in the provider SDK. Account for all three when setting retry limits and timeouts; see the
[Pydantic AI retry guide](https://pydantic.dev/docs/ai/core-concepts/retries/).

## 7. Call the agent from Python

Your application can submit the same `support` task through the Render SDK. With the local task server still
running, save this as `submit.py` alongside `app.py`:

```python
import asyncio

from render import RenderAsync


async def main() -> None:
    client = RenderAsync()
    run = await client.workflows.start_task('support', ['Check the weather.'])
    print(run.id)


if __name__ == '__main__':
    asyncio.run(main())
```

Run it in a second terminal:

```bash
RENDER_USE_LOCAL_DEV=1 uv run python submit.py
```

`RENDER_USE_LOCAL_DEV=1` directs the client to the local server without requiring a Render API key. The script
prints the run ID as soon as Render accepts the task, without waiting for the agent's answer.

To retrieve that run's status and result, save this as `check_run.py`:

```python
import asyncio
import json
import sys

from render import RenderAsync


async def main() -> None:
    client = RenderAsync()
    run = await client.workflows.get_task_run(sys.argv[1])
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == '__main__':
    asyncio.run(main())
```

Pass the ID printed by `submit.py`:

```bash
RENDER_USE_LOCAL_DEV=1 uv run python check_run.py <RUN_ID>
```

Repeat the check while the run is pending or running. On success, the response has `status: "completed"` and the
agent's output in `results`; a failed run includes an `error`. With the unchanged test model, expect
`results: ["The workflow completed."]`.

For a FastAPI endpoint or another web application, use the `start_task` call in your request handler and return
`run.id` with HTTP `202 Accepted`. This avoids HTTP timeouts while a long-running LLM task finishes. Add a separate
status endpoint that calls `get_task_run` with that ID, so the browser or API client can check for the result.
Verify that the run belongs to the requesting user before returning its details, and keep Render credentials on
the server.

## 8. Deploy the agent to Render

Once the local example works, deploy the project as a [Workflow service](https://render.com/docs/workflows-tutorial#4-create-a-workflow-service).
Review the [execution limits](#execution-limits) below, then:

1. Push the project and its dependency files to your Git provider. Include the lockfile so the build installs
   the versions you tested locally.
2. Create a Workflow service linked to that repository. Set its root directory to the folder containing `app.py`
   and its build command to install the project's dependencies. For a uv project whose dependencies are available
   in the build, use `uv sync --locked`.
3. Set the start command to `uv run render-workflows app:app` (`render-workflows app:app` for pip), and add the
   provider credentials, such as `OPENAI_API_KEY`, to the Workflow's environment. Deploy and check that `support`
   appears in its task list.
4. In the application that submits tasks, set `RENDER_API_KEY` to a [Render API key](https://render.com/docs/api#1-create-an-api-key).
   Remove `RENDER_USE_LOCAL_DEV` and `RENDER_LOCAL_DEV_URL` if set, then replace `'support'` in `submit.py` with the
   deployed task's slug, such as `'my-workflow/support'`. Copy the actual slug from the task's Dashboard page.
5. Run `uv run python submit.py`, then `uv run python check_run.py <RUN_ID>` with its returned ID. With pip, use
   `python` in the activated environment. The same client calls now submit and inspect a hosted agent run.

Render's [task submission guide](https://render.com/docs/workflows-running) covers authentication and other ways
to trigger the deployed task.

## Task definitions and child task runs

The capability registers definitions for these operations:

- model requests, buffered stream requests, compaction, and suspended-response cleanup;
- per function toolset by default, or per statically known function tool when its resolved options differ, argument validation and tool calls;
- per MCP and dynamic toolset, discovery, instructions, validation, and calls;
- `event_stream_handler` delivery;
- each method another capability declares with `@durable_operation`.

Render's limit of 500 definitions per workflow counts the entry task and these generated definitions. Each definition
can produce many child runs, which Render schedules and bills individually.

## Task names for capability toolsets

Every registered leaf toolset needs a stable `id` because Render uses it in persisted task names. Duplicate IDs fail
Pydantic AI's uniqueness check; the integration does not rename them. An unnamed supported toolset attached directly
to an agent is rejected before any task definitions register.

Capability-owned toolsets can remain unnamed, in which case their tools execute inline in the entry task without
separate run records or task options.

## Sub-agent delegation

`SubAgents` runs its `delegate_task` tool in the workflow entry task. To run a delegate's model requests and
supported tools as Render tasks, construct the child `Agent` at module load time with its own `RenderWorkflows`
instance, using the same `Workflows` app as the parent. A child without that configuration stays inline.

Successful child operations contribute usage and buffered events to the parent run. Failed operations and
`ModelRetry` attempts do not forward those updates. The delegation limit applies within one active parent task
run; it does not impose a shared budget across entry-task retries or separate runs.

Immediate capability events require a synchronous decision before their emitter continues, which cannot be buffered across a child task. The integration rejects those events across the task boundary, so keep tools that emit them inline.

## Large tool outputs

`ToolOutputLimits` also contributes an unnamed helper toolset, so that helper remains inline. It measures and reduces a tool return after the registered tool task returns to the workflow entry task.

In `Spill` mode, the capability writes the full payload to a filesystem-backed store and gives the model a handle for a later `read_tool_result` call. Because task runs execute in separate processes and can have isolated filesystems, the later task might not be able to read the file behind that handle.

For large artifacts, return bounded JSON containing a key into object storage, a database, or another durable service that both tasks can reach. A later tool can then fetch the artifact from that shared store.

## Memory

The `Memory` capability is not supported by this integration yet. Hosted task instances also do not share local memory or files, so tools that persist data need a shared external backend.

## Task options and tool opt-out

Use Render `Options` for the model, tool, event, and capability task definitions:

```python
from render import Options, Retry, Workflows

from pydantic_ai_harness import RenderWorkflows


def resolve_tool_options(_operation_id, _tool, tool_name):
    if tool_name == 'read_local_cache':
        return False
    return None


app = Workflows()
workflows = RenderWorkflows(
    app,
    model_options=Options(
        retry=Retry(max_retries=3, wait_duration_ms=1_000),
        timeout_seconds=300,
        plan='flex',
    ),
    tool_options=Options(timeout_seconds=120, plan='2c-4g'),
    event_options=Options(timeout_seconds=60),
    capability_options=Options(timeout_seconds=120),
    resolve_tool_options=resolve_tool_options,
)
```

At registration, the resolver first receives `tool=None` and `tool_name=''` to determine the toolset default.
It then receives each statically known function tool and its name; returning `None` keeps `tool_options`.

When every static tool resolves to the shared default, the toolset keeps its existing shared call and validation task definitions. When at least one resolves different `Options` or `False`, each eligible static tool receives definitions named from the agent, toolset, tool, and operation.

Returning `False` for a static function tool registers no task for that tool and runs it inside the workflow entry task. `False` is rejected for MCP and dynamic tools because their concrete tools are not known when the Workflow service registers definitions.

## Execution limits

- **JSON inputs and results:** Dependencies, messages, model settings, metadata, tool arguments, events, and
  results must be JSON serializable. `deps_type` defaults to the agent's dependency type. Pass resource IDs
  across the boundary and construct live clients inside workers.
- **Task size:** Render limits the total arguments of a task run to [4 MB](https://render.com/docs/workflows-limits#additional-limits),
  including the integration's metadata. An oversized call fails before submission. Store large documents
  externally and pass their IDs or URLs.
- **Task access:** Callers with Workflow API access can submit generated model and tool tasks directly,
  bypassing checks in the entry task. Treat those callers as trusted to use the worker's tools and credentials.
  Keep Render credentials on the server, authenticate application users, and submit validated inputs on their
  behalf. Caller-supplied dependencies and approval fields do not establish authorization.
- **Retention:** Render stores task state, including prompts, responses, tool arguments and results, and
  dependencies, for [30 days](https://render.com/docs/workflows-limits#task-state-retention). Read credentials
  from worker environment variables rather than passing them through task inputs or results.
- **Streaming and cancellation:** Model responses and events are buffered until the child task finishes;
  provider tokens do not stream live to the caller. Pydantic AI cancellation tokens are unsupported inside a
  workflow, so use Render's native task cancellation. Cancellation does not undo external tool side effects.

Retry behavior is described in [Configure retries and timeouts](#6-configure-retries-and-timeouts).

If a tool needs `ctx.model`, supply a model instance as the agent's default or register it through
`models={...}`. A child task resolves that instance in its own process, and calls through `ctx.model` run
inside that task. A model specified only by a string has no registered instance available through `ctx.model`.

## Execution and tracing

Render's synchronous and asynchronous clients submit ordinary tasks to its queues. To start the entry task on a
schedule, use a Render cron job.

The capability emits no additional OpenTelemetry spans. Pydantic AI traces model and tool operations, while
Render records task runs, retries, logs, and metrics. Configure Pydantic AI instrumentation when each worker
loads the app to use `ctx.tracer` inside child tasks. It is a no-op when tracing is disabled. The caller's
`trace_include_content` setting is preserved, but parent span context is not propagated across Render task
calls, so child-task spans can appear in separate traces.

## API reference

See the [`RenderWorkflows` API reference](https://pydantic.dev/docs/ai/harness/render-workflows/#api-reference).
