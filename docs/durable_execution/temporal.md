# Durable Execution with Temporal

[Temporal](https://temporal.io) is a popular [durable execution](https://docs.temporal.io/evaluate/understanding-temporal#durable-execution) platform that's natively supported by Pydantic AI.

## Durable Execution

In Temporal's durable execution implementation, a program that crashes or encounters an exception while interacting with a model or API will retry until it can successfully complete.

Temporal relies primarily on a replay mechanism to recover from failures.
As the program makes progress, Temporal saves key inputs and decisions, allowing a re-started program to pick up right where it left off.

The key to making this work is to separate the application's repeatable (deterministic) and non-repeatable (non-deterministic) parts:

1. Deterministic pieces, termed [**workflows**](https://docs.temporal.io/workflow-definition), execute the same way when re-run with the same inputs.
2. Non-deterministic pieces, termed [**activities**](https://docs.temporal.io/activities), can run arbitrary code, performing I/O and any other operations.

Workflow code can run for extended periods and, if interrupted, resume exactly where it left off.
Critically, workflow code generally _cannot_ include any kind of I/O, over the network, disk, etc.
Activity code faces no restrictions on I/O or external interactions, but if an activity fails part-way through it is restarted from the beginning.


!!! note

    If you are familiar with celery, it may be helpful to think of Temporal activities as similar to celery tasks, but where you wait for the task to complete and obtain its result before proceeding to the next step in the workflow.
    However, Temporal workflows and activities offer a great deal more flexibility and functionality than celery tasks.

    See the [Temporal documentation](https://docs.temporal.io/evaluate/understanding-temporal#temporal-application-the-building-blocks) for more information

In the case of Pydantic AI agents, integration with Temporal means that [model requests](../models/overview.md), [tool calls](../tools.md) that may require I/O, and [MCP server communication](../mcp/client.md) all need to be offloaded to Temporal activities due to their I/O requirements, while the logic that coordinates them (i.e. the agent run) lives in the workflow. Code that handles a scheduled job or web request can then execute the workflow, which will in turn execute the activities as needed.

The diagram below shows the overall architecture of an agentic application in Temporal.
The Temporal Server is responsible for tracking program execution and making sure the associated state is preserved reliably (i.e., stored to an internal database, and possibly replicated across cloud regions).
Temporal Server manages data in encrypted form, so all data processing occurs on the Worker, which runs the workflow and activities.


```text
            +---------------------+
            |   Temporal Server   |      (Stores workflow state,
            +---------------------+       schedules activities,
                     ^                    persists progress)
                     |
        Save state,  |   Schedule Tasks,
        progress,    |   load state on resume
        timeouts     |
                     |
+------------------------------------------------------+
|                      Worker                          |
|   +----------------------------------------------+   |
|   |              Workflow Code                   |   |
|   |       (Agent Run Loop)                       |   |
|   +----------------------------------------------+   |
|          |          |                |               |
|          v          v                v               |
|   +-----------+ +------------+ +-------------+       |
|   | Activity  | | Activity   | |  Activity   |       |
|   | (Tool)    | | (MCP Tool) | | (Model API) |       |
|   +-----------+ +------------+ +-------------+       |
|         |           |                |               |
+------------------------------------------------------+
          |           |                |
          v           v                v
      [External APIs, services, databases, etc.]
```

See the [Temporal documentation](https://docs.temporal.io/evaluate/understanding-temporal#temporal-application-the-building-blocks) for more information.

## Durable Agent

Any agent can be wrapped in a [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] to get a durable agent that can be used inside a deterministic Temporal workflow, by automatically offloading all work that requires I/O (namely model requests, tool calls, and MCP server communication) to non-deterministic activities.

At the time of wrapping, the agent's [model](../models/overview.md) and [toolsets](../toolsets.md) (including function tools registered on the agent and MCP servers) are frozen, activities are dynamically created for each, and the original model and toolsets are wrapped to call on the worker to execute the corresponding activities instead of directly performing the actions inside the workflow. The original agent can still be used as normal outside the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

Here is a simple but complete example of wrapping an agent for durable execution, creating a Temporal workflow with durable execution logic, connecting to a Temporal server, and running the workflow from non-durable code. All it requires is a Temporal server to be [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally):

```sh
brew install temporal
temporal server start-dev
```

```python {title="temporal_agent.py" test="skip"}
import uuid

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    PydanticAIPlugin,
    PydanticAIWorkflow,
    TemporalAgent,
)

agent = Agent(
    'openai:gpt-5.2',
    instructions="You're an expert in geography.",
    name='geography',  # (10)!
)

temporal_agent = TemporalAgent(agent)  # (1)!


@workflow.defn
class GeographyWorkflow(PydanticAIWorkflow):  # (2)!
    __pydantic_ai_agents__ = [temporal_agent]  # (3)!

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent.run(prompt)  # (4)!
        return result.output


async def main():
    client = await Client.connect(  # (5)!
        'localhost:7233',  # (6)!
        plugins=[PydanticAIPlugin()],  # (7)!
    )

    async with Worker(  # (8)!
        client,
        task_queue='geography',
        workflows=[GeographyWorkflow],
    ):
        output = await client.execute_workflow(  # (10)!
            GeographyWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=f'geography-{uuid.uuid4()}',
            task_queue='geography',
        )
        print(output)
        #> Mexico City (Ciudad de México, CDMX)
```

1. The original `Agent` cannot be used inside a deterministic Temporal workflow, but the `TemporalAgent` can.
2. As explained above, the workflow represents a deterministic piece of code that can use non-deterministic activities for operations that require I/O. Subclassing [`PydanticAIWorkflow`][pydantic_ai.durable_exec.temporal.PydanticAIWorkflow] is optional but provides proper typing for the `__pydantic_ai_agents__` class variable.
3. List the `TemporalAgent`s used by this workflow. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] will automatically register their activities with the worker. Alternatively, if modifying the worker initialization is easier than the workflow class, you can use [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin] to register agents directly on the worker.
4. [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] works just like [`Agent.run()`][pydantic_ai.agent.Agent.run], but it will automatically offload model requests, tool calls, and MCP server communication to Temporal activities.
5. We connect to the Temporal server which keeps track of workflow and activity execution.
6. This assumes the Temporal server is [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally).
7. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] tells Temporal to use Pydantic for serialization and deserialization, treats [`UserError`][pydantic_ai.exceptions.UserError] exceptions as non-retryable, and automatically registers activities for agents listed in `__pydantic_ai_agents__`.
8. We start the worker that will listen on the specified task queue and run workflows and activities. In a real world application, this might be run in a separate service.
9. The agent's `name` is used to uniquely identify its activities.
10. We call on the server to execute the workflow on a worker that's listening on the specified task queue.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

In a real world application, the agent, workflow, and worker are typically defined separately from the code that calls for a workflow to be executed.
Because Temporal workflows need to be defined at the top level of the file and the `TemporalAgent` instance is needed inside the workflow and when starting the worker (to register the activities), it needs to be defined at the top level of the file as well.

For more information on how to use Temporal in Python applications, see their [Python SDK guide](https://docs.temporal.io/develop/python).

## Temporal Integration Considerations

There are a few considerations specific to agents and toolsets when using Temporal for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with Temporal's workflow and activity model.

### Agent Names and Toolset IDs

To ensure that Temporal knows what code to run when an activity fails or is interrupted and then restarted, even if your code is changed in between, each activity needs to have a name that's stable and unique.

When `TemporalAgent` dynamically creates activities for the wrapped agent's model requests and toolsets (specifically those that implement their own tool listing and calling, i.e. [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPServer`][pydantic_ai.mcp.MCPServer]), their names are derived from the agent's [`name`][pydantic_ai.agent.AbstractAgent.name] and the toolsets' [`id`s][pydantic_ai.toolsets.AbstractToolset.id]. These fields are normally optional, but are required to be set when using Temporal. They should not be changed once the durable agent has been deployed to production as this would break active workflows.

For dynamic toolsets created with the [`@agent.toolset`][pydantic_ai.agent.Agent.toolset] decorator, the `id` parameter must be set explicitly. Note that with Temporal, `per_run_step=False` is not respected, as the toolset always needs to be created on-the-fly in the activity.

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

As workflows and activities run in separate processes, any values passed between them need to be serializable. As these payloads are stored in the workflow execution event history, Temporal limits their size to 2MB.

To account for these limitations, tool functions and the [event stream handler](#streaming) running inside activities receive a limited version of the agent's [`RunContext`][pydantic_ai.tools.RunContext], and it's your responsibility to make sure that the [dependencies](../dependencies.md) object provided to [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] can be serialized using Pydantic.

Specifically, only the `deps`, `run_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, and `partial_output` fields are available by default, and trying to access `model`, `prompt`, `messages`, or `tracer` will raise an error.
If you need one or more of these attributes to be available inside activities, you can create a [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with custom `serialize_run_context` and `deserialize_run_context` class methods and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] as `run_context_type`.

### Streaming

Because Temporal activities cannot stream output directly to the activity call site, [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.agent.Agent.iter] are not supported.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `TemporalAgent` instance and using [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] inside the workflow.
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agent.md#streaming-all-events).

As the streaming model request activity, workflow, and workflow execution call all take place in separate processes, passing data between them requires some care:

- To get data from the workflow call site or workflow to the event stream handler, you can use a [dependencies object](#agent-run-context-and-dependencies).
- To get data from the event stream handler to the workflow, workflow call site, or a frontend, you need to use an external system that the event stream handler can write to and the event consumer can read from, like a message queue. You can use the dependency object to make sure the same connection string or other unique ID is available in all the places that need it.

### Model Selection at Runtime

[`Agent.run(model=...)`][pydantic_ai.agent.Agent.run] normally supports both model strings (like `'openai:gpt-5.2'`) and model instances. However, `TemporalAgent` does not support arbitrary model instances because they cannot be serialized for Temporal's replay mechanism.

To use model instances with `TemporalAgent`, you need to pre-register them by passing a dict of model instances to `TemporalAgent(models={...})`. You can then reference them by name or by passing the registered instance directly. If the wrapped agent doesn't have a model set, the first registered model will be used as the default.

Model strings work as expected. For scenarios where you need to customize the provider used by the model string (e.g., inject API keys from deps), you can pass a `provider_factory` to `TemporalAgent`, which is passed the [`RunContext`][pydantic_ai.tools.RunContext] and provider name.

Here's an example showing how to pre-register and use multiple models:

```python {title="multi_model_temporal.py" test="skip"}
from dataclasses import dataclass
from typing import Any

from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers import Provider
from pydantic_ai.tools import RunContext


@dataclass
class Deps:
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None


# Create models from different providers
default_model = OpenAIResponsesModel('gpt-5.2')
fast_model = AnthropicModel('claude-sonnet-4-5')
reasoning_model = GoogleModel('gemini-3-pro-preview')


# Optional: provider factory for dynamic model configuration
def my_provider_factory(run_context: RunContext[Deps], provider_name: str) -> Provider[Any]:
    """Create providers with custom configuration based on run context."""
    if provider_name == 'openai':
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIProvider(api_key=run_context.deps.openai_api_key)
    elif provider_name == 'anthropic':
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=run_context.deps.anthropic_api_key)
    else:
        raise ValueError(f'Unknown provider: {provider_name}')


agent = Agent(default_model, name='multi_model_agent', deps_type=Deps)

temporal_agent = TemporalAgent(
    agent,
    models={
        'fast': fast_model,
        'reasoning': reasoning_model,
    },
    provider_factory=my_provider_factory,  # Optional
)


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, use_reasoning: bool, use_fast: bool) -> str:
        if use_reasoning:
            # Select by registered name
            result = await temporal_agent.run(prompt, model='reasoning')
        elif use_fast:
            # Or pass the registered instance directly
            result = await temporal_agent.run(prompt, model=fast_model)
        else:
            # Or pass a model string (uses provider_factory if set)
            result = await temporal_agent.run(prompt, model='openai:gpt-5-mini')
        return result.output
```

## Activity Configuration

Temporal activity configuration, like timeouts and retry policies, can be customized by passing [`temporalio.workflow.ActivityConfig`](https://python.temporal.io/temporalio.workflow.ActivityConfig.html) objects to the `TemporalAgent` constructor:

- `activity_config`: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
- `model_activity_config`: The Temporal activity config to use for model request activities. This is merged with the base activity config.
- `toolset_activity_config`: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.
- `tool_activity_config`: The Temporal activity config to use for specific tool call activities identified by toolset ID and tool name.
    This is merged with the base and toolset-specific activity configs.

    If a tool does not use I/O, you can specify `False` to disable using an activity. Note that the tool is required to be defined as an `async` function as non-async tools are run in threads which are non-deterministic and thus not supported outside of activities.

## Activity Retries

On top of the automatic retries for request failures that Temporal will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using Temporal, it's recommended to not use [HTTP Request Retries](../retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client).

You can customize Temporal's retry policy using [activity configuration](#activity-configuration).

## Observability with Logfire

Temporal generates telemetry events and metrics for each workflow and activity execution, and Pydantic AI generates events for each agent run, model request and tool call. These can be sent to [Pydantic Logfire](../logfire.md) to get a complete picture of what's happening in your application.

To use Logfire with Temporal, you need to pass a [`LogfirePlugin`][pydantic_ai.durable_exec.temporal.LogfirePlugin] object to Temporal's `Client.connect()`:

```py {title="logfire_plugin.py" test="skip" noqa="F841"}
from temporalio.client import Client

from pydantic_ai.durable_exec.temporal import LogfirePlugin, PydanticAIPlugin


async def main():
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )
```

By default, the `LogfirePlugin` will instrument Temporal (including metrics) and Pydantic AI and send all data to Logfire. To customize Logfire configuration and instrumentation, you can pass a `logfire_setup` function to the `LogfirePlugin` constructor and return a custom `Logfire` instance (i.e. the result of `logfire.configure()`). To disable sending Temporal metrics to Logfire, you can pass `metrics=False` to the `LogfirePlugin` constructor.

## Known Issues

### Pandas

When `logfire.info` is used inside an activity and the `pandas` package is among your project's dependencies, you may encounter the following error which seems to be the result of an import race condition:

```
AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
```

To fix this, you can use the [`temporalio.workflow.unsafe.imports_passed_through()`](https://python.temporal.io/temporalio.workflow.unsafe.html#imports_passed_through) context manager to proactively import the package and not have it be reloaded in the workflow sandbox:

```python {title="temporal_activity.py" test="skip" noqa="F401"}
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import pandas
```
