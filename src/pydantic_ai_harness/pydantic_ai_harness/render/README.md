# Render Workflows

Run long-running AI agents in the background with separate retries, timeouts, and compute settings for model requests
and tool calls. The [Render Workflows](https://render.com/docs/workflows) integration runs the agent loop in an entry
task and supported operations as child tasks, each with its own status, logs, and result. For example, a tool that
processes a large document can have a longer timeout than the model calls around it.

If you only need background execution, a native Render task around `agent.run(...)` may be enough.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/render/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Before you start

The example uses Pydantic AI's `TestModel` and a mock weather tool, so it needs no LLM API key or Render deployment.
The model generates sample tool arguments and returns fixed text rather than interpreting the prompt.

You need Python 3.10 or later. Install the [Render CLI](https://render.com/docs/cli) 2.28.0 or later separately. The integration requires Render SDK 1.2.0 or later.

## 1. Install dependencies

In a new directory, initialize a project with [uv](https://docs.astral.sh/uv/) using `uv init --bare`; for an existing uv project, run the installation command directly. If you use pip, first create and activate a virtual environment.

Until a Harness release includes this integration, install from a Pydantic AI repository checkout containing it.
Replace `/path/to/pydantic-ai` with the checkout's absolute path. Install Graph, core, and Harness together so their
versions match. The uv command uses editable installs to match the monorepo's workspace dependencies:

uv:

```bash
uv add --editable "/path/to/pydantic-ai/pydantic_graph" "/path/to/pydantic-ai/pydantic_ai_slim" "/path/to/pydantic-ai/src/pydantic_ai_harness[render]"
```

pip:

```bash
pip install "/path/to/pydantic-ai/pydantic_graph" "/path/to/pydantic-ai/pydantic_ai_slim" "/path/to/pydantic-ai/src/pydantic_ai_harness[render]"
```

For a published Harness release that includes the integration, use one of these commands instead of the checkout install:

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

```python {title="app.py" names="defined"}
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

```python {names="defined"}
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

To give a statically known function tool its own options, use `resolve_tool_options`; returning `False` keeps that tool inline. The [per-tool options reference](https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/render-workflows.md#task-options-and-tool-opt-out) explains the resolver contract and the restrictions for MCP and dynamic tools.

A failed child task can retry while the entry task waits. Retrying the entry task restarts `agent.run(...)`, so model
and tool calls that already finished may run again. There is no checkpoint resume or replay of completed steps.
Either kind of retry can repeat an external action performed before its result was recorded; use application-level
idempotency keys or deduplication for writes and API requests that must not happen twice.

Render retries are separate from Pydantic AI's `ModelRetry`, which asks the model to correct or reconsider a call,
and from retries in the provider SDK. Account for all three when setting retry limits and timeouts; see the
[Pydantic AI retry guide](https://pydantic.dev/docs/ai/core-concepts/retries/).

## 7. Call the agent from Python

Your application can submit the same `support` task through the Render SDK. With the local task server still
running, save this as `submit.py` alongside `app.py`:

```python {title="submit.py" names="defined"}
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

With pip, use `python` instead of `uv run python` in your activated environment.

`RENDER_USE_LOCAL_DEV=1` directs the client to the local server without requiring a Render API key. The script
prints the run ID as soon as Render accepts the task, without waiting for the agent's answer.

To retrieve that run's status and result, save this as `check_run.py`:

```python {title="check_run.py" names="defined"}
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
Review the deployment constraints below, then:

1. Push the project and its dependency files to your Git provider. If you installed from a local checkout,
   build matching Graph, core, and Harness wheels from that checkout and include them in the deployment. Update
   the dependency paths to those wheels and regenerate the lockfile. Building the wheels together preserves
   the matching versions derived from the checkout's Git metadata. Alternatively, use a published Harness release
   that contains the integration. Paths to directories on your laptop will not exist on Render.
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

## Constraints to check before deployment

- **Task access:** Render API callers can submit generated operations directly, bypassing checks that exist only in the entry task. Treat those callers as trusted; see the [task access boundary](https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/render-workflows.md#task-access-boundary).
- **Toolsets:** Explicit IDs must be stable and unique. Unnamed capability-owned tools can remain inline without a child-task record.
- **Task inputs:** Dependencies, messages, arguments, and results must be JSON serializable, and Render limits a task run's arguments to 4 MB. Pass resource IDs across the boundary and reconstruct clients inside workers.
- **Storage:** Hosted task instances do not share local memory or files, so agent memory and large artifacts need a shared external backend. Read credentials from worker environment variables rather than passing them through task inputs or results.
- **Streaming and cancellation:** Model responses are buffered until their child task finishes, so live tokens do not stream across the task boundary. Pydantic AI cancellation tokens are unsupported in a workflow; use Render's native task cancellation instead.
- **Compatibility:** The adapter uses private Pydantic AI APIs, so test the integration before upgrading its dependencies. The local quickstart verifies task dispatch, but does not establish hosted failure recovery or performance.

Render retains task inputs and results for 30 days and allows up to 500 task definitions per workflow. Check the current [limits and pricing](https://render.com/docs/workflows-limits) when deciding how many operations to register as separate tasks.

## Further reference

See the [integration reference](https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/render-workflows.md)
for task naming, sub-agent setup, JSON transport, memory, large tool outputs, and tracing.
