# Durable Execution with DBOS

!!! note
    Durable execution support is in beta and the public interface is subject to change based on user feedback. We expect it to be stable by the release of Pydantic AI v1 at the end of August. Questions and feedback are welcome in [GitHub issues](https://github.com/pydantic/pydantic-ai/issues) and the [`#pydantic-ai` Slack channel](https://logfire.pydantic.dev/docs/join-slack/).

Pydantic AI allows you to build durable agents that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. Durable agents have full support for [streaming](agents.md#streaming-all-events) and [MCP](mcp/client.md), with the added benefit of fault tolerance.

[DBOS](https://www.dbos.dev/) is a lightweight [durable execution](https://docs.dbos.dev/architecture) library that's natively supported by Pydantic AI.
The integration only uses Pydantic AI's public interface, so it can also serve as a reference for how to integrate with other durable execution systems.

### Durable Execution

In DBOS's durable execution implementation, a program that crashes or encounters an exception while interacting with a model or API will retry until it can successfully complete.

DBOS relies primarily on a replay mechanism to recover from failures.
As the program makes progress, DBOS saves key inputs and decisions, allowing a re-started program to pick up right where it left off.

The key to making this work is to separate the application's repeatable (deterministic) and non-repeatable (non-deterministic) parts:

1. Deterministic pieces, termed [**workflows**](https://docs.dbos.dev/python/tutorials/workflow-tutorial), execute the same way when re-run with the same inputs.
2. Non-deterministic pieces, termed [**steps**](https://docs.dbos.dev/python/tutorials/step-tutorial), can run arbitrary code, performing I/O and any other operations.

Workflow code can run for extended periods and, if interrupted, resume exactly where it left off.
Critically, workflow code generally _cannot_ include any kind of I/O, over the network, disk, etc.
Step code faces no restrictions on I/O or external interactions, but if a step fails part-way through it is restarted from the beginning.


!!! note

    If you are familiar with celery, it may be helpful to think of DBOS steps as similar to celery tasks, but where you wait for the task to complete and obtain its result before proceeding to the next step in the workflow.
    However, DBOS workflows and steps offer a great deal more flexibility and functionality than celery tasks.

    See the [DBOS documentation](https://docs.dbos.dev/architecture) for more information.

In the case of Pydantic AI agents, integration with DBOS means that [model requests](models/index.md), [tool calls](tools.md) that may require I/O, and [MCP server communication](mcp/client.md) all need to be offloaded to DBOS steps due to their I/O requirements, while the logic that coordinates them (i.e. the agent run) lives in the workflow. Code that handles a scheduled job or web request can then execute the workflow, which will in turn execute the steps as needed.

The diagram below shows the overall architecture of an agentic application in DBOS.
DBOS is lightweight because it runs entirely in-process as a library, so your workflows and steps remain normal functions within your application that you can call from other application code. DBOS instruments them to checkpoint their state into a database (i.e., possibly replicated across cloud regions).

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

Any agent can be wrapped in a [`DBOSAgent`][pydantic_ai.durable_exec.dbos.DBOSAgent] to get a durable agent, by automatically wrapping the agent run loop as a deterministic DBOS workflow and offloading work that requires I/O (namely model requests and MCP server communication) to non-deterministic steps. To make it flexible, `DBOSAgent` doesn't automatically wrap other tool functions, so you can decorate them as either DBOS workflows or steps as needed.

At the time of wrapping, the agent's [model](models/index.md) and [MCP server communication](mcp/client.md) are wrapped as DBOS steps instead of directly invoking the original functions inside the workflow. The original agent can still be used as normal outside the DBOS workflow.

Here is a simple but complete example of wrapping an agent for durable execution. All it requires is to install the DBOS [open-source library](https://github.com/dbos-inc/dbos-transact-py):

```sh
uv add pydantic-ai[dbos]
```

or if you use pip:
```sh
pip install pydantic-ai[dbos]
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
    'gpt-5',
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

1. The original `Agent` cannot be used inside a deterministic DBOS workflow, but the `DBOSAgent` can. Workflow function declarations and `DBOSAgent` creations need to happen before calling `DBOS.launch()` because DBOS requires all workflows to be registered before launch so that recovery can correctly find all workflows.
2. [`DBOSAgent.run()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run] works like [`Agent.run()`][pydantic_ai.Agent.run], but runs inside a DBOS workflow and wraps model requests, decorated tool calls, and MCP communication as DBOS steps.
3. This assumes DBOS is using SQLite. To deploy your agent to production, we recommend using a Postgres server.
4. The agent's `name` is used to uniquely identify its workflows.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because DBOS workflows need to be defined before calling `DBOS.launch()` and the `DBOSAgent` instance automatically registers `run` and `run_sync` as workflows, it needs to be defined before calling `DBOS.launch()` as well.

For more information on how to use DBOS in Python applications, see their [Python SDK guide](https://docs.dbos.dev/python/programming-guide).

## DBOS Integration Considerations

There are a few considerations specific to agents and toolsets when using DBOS for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with DBOS's workflow and step model.

### Agent and Toolset Requirements

To ensure that DBOS knows what code to run when a workflow fails or is interrupted and then restarted, each agent instance needs to have a unique name.

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

As DBOS checkpoints workflow and step execution into a database, workflow inputs and outputs, and step outputs need to be serializable (JSON pickleable). You may also want to keep the inputs and outputs small (the maximum size for a single field in PostgreSQL is 1 GB, but usually you want to keep the output size under 2 MB).

### Streaming

Because DBOS steps cannot stream output directly to the step call site, [`Agent.run_stream()`][pydantic_ai.Agent.run_stream] is not supported.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `DBOSAgent` instance and using [`DBOSAgent.run()`][pydantic_ai.durable_exec.dbos.DBOSAgent.run].
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](agents.md#streaming-all-events).


## Step Configuration

DBOS step configuration, like retry policies, can be customized by passing [`StepConfig`][pydantic_ai.durable_exec.dbos.StepConfig] objects to the `DBOSAgent` constructor:

- `mcp_step_config`: The DBOS step config to use for MCP server communication. If no config is provided, it disables DBOS step retries.
- `model_step_config`: The DBOS step config to use for model request steps. If no config is provided, it disables DBOS step retries.

For individual tools, you can annotate them with [`@DBOS.step`](https://docs.dbos.dev/python/reference/decorators#step) or [`@DBOS.workflow`](https://docs.dbos.dev/python/reference/decorators#workflow) decorators as needed. Decorated steps are just normal functions if called outside of DBOS workflows, which can be used in non-DBOS agents.

## Step Retries

On top of the automatic retries for request failures that DBOS will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using DBOS, it's recommended to not use [HTTP Request Retries](retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](models/openai.md#custom-openai-client).

You can customize DBOS's retry policy using [step configuration](#step-configuration).

## Observability with Logfire

DBOS generates OpenTelemetry traces and events for each workflow and step execution, and Pydantic AI generates events for each agent run, model request and tool call. These can be sent to [Pydantic Logfire](logfire.md) to get a complete picture of what's happening in your application.

To disable sending DBOS traces to Logfire, you can pass `disable_otlp=True` to the `DBOS` constructor. For example:


```python {title="dbos_no_traces.py" test="skip"}
from dbos import DBOS, DBOSConfig

dbos_config: DBOSConfig = {
    'name': 'pydantic_dbos_agent',
    'system_database_url': 'sqlite:///dbostest.sqlite',
    'disable_otlp': True  # (1)!
}
DBOS(config=dbos_config)
```

1. If `True`, disables OpenTelemetry tracing and logging for DBOS. Defaults to `False`.
