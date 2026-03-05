# Durable Execution with DBOS

[DBOS](https://www.dbos.dev/) is a lightweight [durable execution](https://docs.dbos.dev/architecture) library natively integrated with Pydantic AI.

## Durable Execution

DBOS workflows make your program **durable** by checkpointing its state in a database. If your program ever fails, when it restarts all your workflows will automatically resume from the last completed step.

* **Workflows** must be deterministic and generally cannot include I/O.
* **Steps** may perform I/O (network, disk, API calls). If a step fails, it restarts from the beginning.

Every workflow input and step output is durably stored in the system database. When workflow execution fails, whether from crashes, network issues, or server restarts, DBOS leverages these checkpoints to recover workflows from their last completed step.

DBOS **queues** provide durable, database-backed alternatives to systems like Celery or BullMQ, supporting features such as concurrency limits, rate limits, timeouts, and prioritization. See the [DBOS docs](https://docs.dbos.dev/architecture) for details.

The diagram below shows the overall architecture of an agentic application in DBOS.
DBOS runs fully in-process as a library. Functions remain normal Python functions but are checkpointed into a database (Postgres or SQLite).

```text
                    Clients
            (HTTP, RPC, Kafka, etc.)
                        |
                        v
+------------------------------------------------------+
|               Application Servers                    |
|                                                      |
|   +----------------------------------------------+   |
|   |        Pydantic AI + DBOS Libraries          |   |
|   |                                              |   |
|   |  [ Workflows (Agent Run Loop) ]              |   |
|   |  [ Steps (Tool, MCP, Model) ]                |   |
|   |  [ Queues ]   [ Cron Jobs ]   [ Messaging ]  |   |
|   +----------------------------------------------+   |
|                                                      |
+------------------------------------------------------+
                        |
                        v
+------------------------------------------------------+
|                      Database                        |
|   (Stores workflow and step state, schedules tasks)  |
+------------------------------------------------------+
```

See the [DBOS documentation](https://docs.dbos.dev/architecture) for more information.

## Durable Agent

Any agent can be wrapped in a [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent] to get durable execution. `DBOSAgent` automatically:,

* Wraps `Agent.run` and `Agent.run_sync` as DBOS workflows.
* Wraps [model requests](../models/overview.md) and [MCP communication](../mcp/client.md) as DBOS steps.

Custom tool functions and event stream handlers are **not automatically wrapped** by DBOS.
If they involve non-deterministic behavior or perform I/O, you should explicitly decorate them with `@DBOS.step`.

The original agent, model, and MCP server can still be used as normal outside the DBOS workflow.

Here is a simple but complete example of wrapping an agent for durable execution. All it requires is to install Pydantic AI with the DBOS [open-source library](https://github.com/dbos-inc/dbos-transact-py):

```bash
pip/uv-add pydantic-ai[dbos]
```

Or if you're using the slim package, you can install it with the `dbos` optional group:

```bash
pip/uv-add pydantic-ai-slim[dbos]
```

```python {title="dbos_agent.py" test="skip"}
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

dbos_config: DBOSConfig = {
    'name': 'pydantic_dbos_agent',
    'system_database_url': 'sqlite:///dbostest.sqlite',  # (3)!
}
DBOS(config=dbos_config)

agent = Agent(
    'gpt-5.2',
    instructions="You're an expert in geography.",
    name='geography',  # (4)!
)

dbos_agent = DBOSAgent(agent)  # (1)!

async def main():
    DBOS.launch()
    result = await dbos_agent.run('What is the capital of Mexico?')  # (2)!
    print(result.output)
    #> Mexico City (Ciudad de México, CDMX)
```

1. Workflows and `DBOSAgent` must be defined before `DBOS.launch()` so that recovery can correctly find all workflows.
2. [`DBOSAgent.run()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run] works like [`Agent.run()`][pydantic_ai.agent.Agent.run], but runs as a DBOS workflow and executes model requests, decorated tool calls, and MCP communication as DBOS steps.
3. This example uses SQLite. Postgres is recommended for production.
4. The agent's `name` is used to uniquely identify its workflows.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because DBOS workflows need to be defined before calling `DBOS.launch()` and the `DBOSAgent` instance automatically registers `run` and `run_sync` as workflows, it needs to be defined before calling `DBOS.launch()` as well.

For more information on how to use DBOS in Python applications, see their [Python SDK guide](https://docs.dbos.dev/python/programming-guide).

## DBOS Integration Considerations

When using DBOS with Pydantic AI agents, there are a few important considerations to ensure workflows and toolsets behave correctly.

### Agent and Toolset Requirements

Each agent instance must have a unique `name` so DBOS can correctly resume workflows after a failure or restart.

Tools and event stream handlers are not automatically wrapped by DBOS. You can decide how to integrate them:

* Decorate with `@DBOS.step` if the function involves non-determinism or I/O.
* Skip the decorator if durability isn't needed, so you avoid the extra DB checkpoint write.
* If the function needs to enqueue tasks or invoke other DBOS workflows, run it inside the agent's main workflow (not as a step).

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

DBOS checkpoints workflow inputs/outputs and step outputs into a database using [`pickle`](https://docs.python.org/3/library/pickle.html). This means you need to make sure [dependencies](../dependencies.md) object provided to [`DBOSAgent.run()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run] or [`DBOSAgent.run_sync()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run_sync], and tool outputs can be serialized using pickle. You may also want to keep the inputs and outputs small (under \~2 MB). PostgreSQL and SQLite support up to 1 GB per field, but large objects may impact performance.

### Streaming

Because DBOS cannot stream output directly to the workflow or step call site, [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] and [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events] are not supported when running inside of a DBOS workflow.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `DBOSAgent` instance and using [`DBOSAgent.run()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run].
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agent.md#streaming-all-events).


### Parallel Tool Execution

When using `DBOSAgent`, tools are executed in parallel by default to minimize latency. To guarantee deterministic replay and reliable recovery, DBOS waits for all parallel tool calls to complete before emitting events **in order**.
It's equivalent to the behavior of [`with agent.parallel_tool_call_execution_mode('parallel_ordered_events')`][pydantic_ai.agent.AbstractAgent.parallel_tool_call_execution_mode].

If you prefer strict ordering, you can configure the agent to run tools sequentially by setting [`parallel_execution_mode='sequential'`][pydantic_ai.durable_exec.dbos.DBOSAgent] when initializing the `DBOSAgent`.


## Step Configuration

You can customize DBOS step behavior, such as retries, by passing [`StepConfig`][pydantic_ai.durable_exec.dbos.StepConfig] objects to the `DBOSAgent` constructor:

- `mcp_step_config`: The DBOS step config to use for MCP server communication. No retries if omitted.
- `model_step_config`: The DBOS step config to use for model request steps. No retries if omitted.

For custom tools, you can annotate them directly with [`@DBOS.step`](https://docs.dbos.dev/python/reference/decorators#step) or [`@DBOS.workflow`](https://docs.dbos.dev/python/reference/decorators#workflow) decorators as needed. These decorators have no effect outside DBOS workflows, so tools remain usable in non-DBOS agents.


## Step Retries

On top of the automatic retries for request failures that DBOS will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using DBOS, it's recommended to not use [HTTP Request Retries](../retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client).

You can customize DBOS's retry policy using [step configuration](#step-configuration).

## Observability with Logfire

DBOS can be configured to generate OpenTelemetry spans for each workflow and step execution, and Pydantic AI emits spans for each agent run, model request, and tool invocation. You can send these spans to [Pydantic Logfire](../logfire.md) to get a full, end-to-end view of what's happening in your application.

For more information about DBOS logging and tracing, please see the [DBOS docs](https://docs.dbos.dev/python/tutorials/logging-and-tracing) for details.
