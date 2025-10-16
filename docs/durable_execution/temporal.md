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
    AgentPlugin,
    PydanticAIPlugin,
    TemporalAgent,
)

agent = Agent(
    'gpt-5',
    instructions="You're an expert in geography.",
    name='geography',  # (10)!
)

temporal_agent = TemporalAgent(agent)  # (1)!


@workflow.defn
class GeographyWorkflow:  # (2)!
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent.run(prompt)  # (3)!
        return result.output


async def main():
    client = await Client.connect(  # (4)!
        'localhost:7233',  # (5)!
        plugins=[PydanticAIPlugin()],  # (6)!
    )

    async with Worker(  # (7)!
        client,
        task_queue='geography',
        workflows=[GeographyWorkflow],
        plugins=[AgentPlugin(temporal_agent)],  # (8)!
    ):
        output = await client.execute_workflow(  # (9)!
            GeographyWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=f'geography-{uuid.uuid4()}',
            task_queue='geography',
        )
        print(output)
        #> Mexico City (Ciudad de México, CDMX)
```

1. The original `Agent` cannot be used inside a deterministic Temporal workflow, but the `TemporalAgent` can.
2. As explained above, the workflow represents a deterministic piece of code that can use non-deterministic activities for operations that require I/O.
3. [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] works just like [`Agent.run()`][pydantic_ai.Agent.run], but it will automatically offload model requests, tool calls, and MCP server communication to Temporal activities.
4. We connect to the Temporal server which keeps track of workflow and activity execution.
5. This assumes the Temporal server is [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally).
6. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] tells Temporal to use Pydantic for serialization and deserialization, and to treat [`UserError`][pydantic_ai.exceptions.UserError] exceptions as non-retryable.
7. We start the worker that will listen on the specified task queue and run workflows and activities. In a real world application, this might be run in a separate service.
8. The [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin] registers the `TemporalAgent`'s activities with the worker.
9. We call on the server to execute the workflow on a worker that's listening on the specified task queue.
10. The agent's `name` is used to uniquely identify its activities.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

In a real world application, the agent, workflow, and worker are typically defined separately from the code that calls for a workflow to be executed.
Because Temporal workflows need to be defined at the top level of the file and the `TemporalAgent` instance is needed inside the workflow and when starting the worker (to register the activities), it needs to be defined at the top level of the file as well.

For more information on how to use Temporal in Python applications, see their [Python SDK guide](https://docs.temporal.io/develop/python).

## Temporal Integration Considerations

There are a few considerations specific to agents and toolsets when using Temporal for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with Temporal's workflow and activity model.

### Agent Names and Toolset IDs

To ensure that Temporal knows what code to run when an activity fails or is interrupted and then restarted, even if your code is changed in between, each activity needs to have a name that's stable and unique.

When `TemporalAgent` dynamically creates activities for the wrapped agent's model requests and toolsets (specifically those that implement their own tool listing and calling, i.e. [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPServer`][pydantic_ai.mcp.MCPServer]), their names are derived from the agent's [`name`][pydantic_ai.agent.AbstractAgent.name] and the toolsets' [`id`s][pydantic_ai.toolsets.AbstractToolset.id]. These fields are normally optional, but are required to be set when using Temporal. They should not be changed once the durable agent has been deployed to production as this would break active workflows.

Other than that, any agent and toolset will just work!

### Instructions Functions, Output Functions, and History Processors

Pydantic AI runs non-async [instructions](../agents.md#instructions) and [system prompt](../agents.md#system-prompts) functions, [history processors](../message-history.md#processing-message-history), [output functions](../output.md#output-functions), and [output validators](../output.md#output-validator-functions) in threads, which are not supported inside Temporal workflows and require an activity. Ensure that these functions are async instead.

Synchronous tool functions are supported, as tools are automatically run in activities unless this is [explicitly disabled](#activity-configuration). Still, it's recommended to make tool functions async as well to improve performance.

### Agent Run Context and Dependencies

As workflows and activities run in separate processes, any values passed between them need to be serializable. As these payloads are stored in the workflow execution event history, Temporal limits their size to 2MB.

To account for these limitations, tool functions and the [event stream handler](#streaming) running inside activities receive a limited version of the agent's [`RunContext`][pydantic_ai.tools.RunContext], and it's your responsibility to make sure that the [dependencies](../dependencies.md) object provided to [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] can be serialized using Pydantic.

Specifically, only the `deps`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `retry`, `max_retries` and `run_step` fields are available by default, and trying to access `model`, `usage`, `prompt`, `messages`, or `tracer` will raise an error.
If you need one or more of these attributes to be available inside activities, you can create a [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with custom `serialize_run_context` and `deserialize_run_context` class methods and pass it to [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] as `run_context_type`.

### Streaming

Because Temporal activities cannot stream output directly to the activity call site, [`Agent.run_stream()`][pydantic_ai.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.Agent.iter] are not supported.

Instead, you can implement streaming by setting an [`event_stream_handler`][pydantic_ai.agent.EventStreamHandler] on the `Agent` or `TemporalAgent` instance and using [`TemporalAgent.run()`][pydantic_ai.durable_exec.temporal.TemporalAgent.run] inside the workflow.
The event stream handler function will receive the agent [run context][pydantic_ai.tools.RunContext] and an async iterable of events from the model's streaming response and the agent's execution of tools. For examples, see the [streaming docs](../agents.md#streaming-all-events).

As the streaming model request activity, workflow, and workflow execution call all take place in separate processes, passing data between them requires some care:

- To get data from the workflow call site or workflow to the event stream handler, you can use a [dependencies object](#agent-run-context-and-dependencies).
- To get data from the event stream handler to the workflow, workflow call site, or a frontend, you need to use an external system that the event stream handler can write to and the event consumer can read from, like a message queue. You can use the dependency object to make sure the same connection string or other unique ID is available in all the places that need it.

#### Example

Run the following
`pip install pydantic-ai temporalio mcp-run-python`

Assuming your project has the following structure:
```
  project/
  ├── src/
  │   ├── agents.py
  │   ├── datamodels.py
  │   ├── streaming_handler.py
  │   ├── utils.py
  │   └── workflow.py
  └── pyproject.toml

```

```py {title="utils.py" test="skip"}
import os
from copy import copy

import yaml


def recursively_modify_api_key(conf):
    """
    Recursively replace API key placeholders with environment variable values.

    This function traverses a configuration dictionary and replaces any keys
    containing 'api_key' with the corresponding environment variable value.
    It handles nested dictionaries and lists recursively.

    Args:
        conf: The configuration dictionary to process.

    Returns:
        A copy of the configuration with API keys replaced by environment variable values.
    """

    def inner(_conf):
        for key, value in _conf.items():
            if isinstance(value, dict):
                inner(value)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    for item in value:
                        inner(item)
                else:
                    _conf[key] = [os.environ.get(str(v), v) for v in value]
            elif isinstance(value, str):
                _conf[key] = os.environ.get(value, value)
            else:
                _conf[key] = value

    copy_conf = copy(conf)
    inner(copy_conf)
    return copy_conf


def read_config_yml(path):
    """
    Read and process a YAML configuration file.

    This function reads a YAML file, processes it to replace API key placeholders
    with environment variable values, and returns the processed configuration.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        dict: The parsed and processed YAML content as a Python dictionary.
    """
    with open(path) as f:
        configs = yaml.safe_load(f)
    recursively_modify_api_key(configs)
    return configs
```

```py {title="datamodels.py" test="skip"}
from enum import Enum

from pydantic import BaseModel


class AgentDependencies(BaseModel):
    workflow_id: str
    run_id: str


class EventKind(str, Enum):
    CONTINUE_CHAT = 'continue_chat'
    EVENT = 'event'
    RESULT = 'result'


class EventStream(BaseModel):
    kind: EventKind
    content: str
```


```py {title="agents.py" test="skip"}
from datetime import timedelta

from mcp_run_python import code_sandbox
from pydantic_ai import Agent, FilteredToolset, ModelSettings, RunContext
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from temporalio.common import RetryPolicy
from temporalio.workflow import ActivityConfig

from .datamodels import AgentDependencies

async def get_mcp_toolsets() -> dict[str, FilteredToolset]:
    yf_server = MCPServerStdio(
        command='uvx',
        args=['mcp-yahoo-finance'],
        timeout=240,
        read_timeout=240,
        id='yahoo'
    )
    return {
        'yahoo': yf_server.filtered(lambda ctx, tool_def: True)
    }


async def get_claude_model(parallel_tool_calls: bool = True, **env_vars):
    model_name = 'claude-sonnet-4-5-20250929'
    api_key = env_vars.get('anthropic_api_key')
    model = AnthropicModel(model_name=model_name,
                           provider=AnthropicProvider(api_key=api_key),
                           settings=ModelSettings(**{
                               'temperature': 0.5,
                               'n': 1,
                               'max_completion_tokens': 64000,
                               'max_tokens': 64000,
                               'parallel_tool_calls': parallel_tool_calls,
                           }))

    return model


async def build_agent(stream_handler=None, **env_vars):
    system_prompt = """
    You are an expert travel agent that knows perfectly how to search for hotels on the web.
    You also have a Data Analyst background, mastering well how to use pandas for tabular operations. 
    """
    agent_name = 'YahooFinanceSearchAgent'

    toolsets = await get_mcp_toolsets()
    agent = Agent(name=agent_name,
                  model=await get_claude_model(**env_vars),  # Here you place your Model instance
                  toolsets=[*toolsets.values()],
                  system_prompt=system_prompt,
                  event_stream_handler=stream_handler,
                  deps_type=AgentDependencies,
                  )

    @agent.tool(name='run_python_code')
    async def run_python_code(ctx: RunContext[None], code: str) -> str:
        async with code_sandbox(dependencies=['pandas', 'numpy']) as sandbox:
            result = await sandbox.eval(code)
            return result

    temporal_agent = TemporalAgent(wrapped=agent,
                                   model_activity_config=ActivityConfig(
                                       start_to_close_timeout=timedelta(minutes=5),
                                       retry_policy=RetryPolicy(maximum_attempts=50)
                                   ),
                                   toolset_activity_config={
                                       toolset_id: ActivityConfig(
                                           start_to_close_timeout=timedelta(minutes=3),
                                           retry_policy=RetryPolicy(maximum_attempts=3,
                                                                    non_retryable_error_types=['ToolRetryError']
                                                                    )
                                       ) for toolset_id in toolsets.keys()})
    return temporal_agent
```

```py {title="streaming_handler.py" test="skip"}
from collections.abc import AsyncIterable

from .datamodels import AgentDependencies, EventKind, EventStream
from temporalio import activity

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    PartStartEvent,
    FunctionToolResultEvent,
    TextPart,
    ToolCallPart,
    PartDeltaEvent,
    TextPartDelta,
    ThinkingPartDelta,
)


async def streaming_handler(ctx,
                            event_stream_events: AsyncIterable[AgentStreamEvent]):
    """
    This function is used by the agent to stream-out the actions that are being performed (tool calls, llm call, streaming results, etc etc.
    Feel free to change it as you like or need - skipping events or enriching the content
    """

    output = ''
    output_tool_delta = dict(
        tool_call_id='',
        tool_name_delta='',
        args_delta='',
    )
    # If TextPart and output delta is empty
    async for event in event_stream_events:
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, TextPart):
                output += f'{event.part.content}'
            elif isinstance(event.part, ToolCallPart):
                output += f'\nTool Call Id: {event.part.tool_call_id}'
                output += f'\nTool Name: {event.part.tool_name}'
                output += f'\nTool Args: {event.part.args}'
            else:
                pass
        elif isinstance(event, FunctionToolCallEvent):
            output += f'\nTool Call Id: {event.part.tool_call_id}'
            output += f'\nTool Name: {event.part.tool_name}'
            output += f'\nTool Args: {event.part.args}'
        elif isinstance(event, FunctionToolResultEvent):
            output += f'\nTool Call Id: {event.result.tool_call_id}'
            output += f'\nTool Name: {event.result.tool_name}'
            output += f'\nContent: {event.result.content}'
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta) or isinstance(event.delta, ThinkingPartDelta):
                output += f'{event.delta.content_delta}'
            else:
                if len(output_tool_delta['tool_call_id']) == 0:
                    output_tool_delta['tool_call_id'] += event.delta.tool_call_id or ''
                output_tool_delta['tool_name_delta'] += event.delta.tool_name_delta or ''
                output_tool_delta['args_delta'] += event.delta.args_delta or ''

    if len(output_tool_delta['tool_call_id']):
        output += f'\nTool Call Id: {output_tool_delta["tool_call_id"]}'
        output += f'\nTool Name: {output_tool_delta["tool_name_delta"]}'
        output += f'\nTool Args: {output_tool_delta["args_delta"]}'

    events = []

    if output:
        event = EventStream(kind=EventKind.EVENT, content=output)
        events.append(event)

    if activity.in_activity():
        deps: AgentDependencies = ctx.deps

        workflow_id = deps.workflow_id
        run_id = deps.run_id
        workflow_handle = activity.client().get_workflow_handle(workflow_id=workflow_id, run_id=run_id)
        for event in events:
            await workflow_handle.signal('append_event', arg=event)
```

```py {title="workflow.py" test="skip"}

import asyncio
from collections import deque
from datetime import timedelta
from typing import Any

from pydanticai import UsageLimits
from temporalio import activity, workflow

from .agents import build_agent, streaming_handler
from .datamodels import AgentDependencies, EventKind, EventStream

@workflow.defn
class YahooFinanceSearchWorkflow:
    def __init__(self):
        self.events: deque[EventStream] = deque()

    @workflow.run
    async def run(self, user_prompt: str):

        wf_vars = await workflow.execute_activity(
            activity='retrieve_env_vars',
            start_to_close_timeout=timedelta(seconds=10),
            result_type=dict[str, Any],
        )
        deps = AgentDependencies(workflow_id=workflow.info().workflow_id, run_id=workflow.info().run_id)

        agent = await build_agent(streaming_handler, **wf_vars)
        result = await agent.run(user_prompt=user_prompt,
                                 usage_limits=UsageLimits(request_limit=50),
                                 deps=deps
                                 )

        await self.append_event(event_stream=EventStream(kind=EventKind.RESULT,
                                                         content=result.output))

        await self.append_event(event_stream=EventStream(kind=EventKind.CONTINUE_CHAT,
                                                         content=''))

        try:
            await workflow.wait_condition(
                lambda: len(self.events) == 0,
                timeout=timedelta(seconds=10),
                timeout_summary='Waiting for events to be consumed'
            )
            return result.output
        except asyncio.TimeoutError:
            return result.output

    @staticmethod
    @activity.defn(name='retrieve_env_vars')
    async def retrieve_env_vars():
        import os
        from .utils import read_config_yml

        config_path = os.getenv('APP_CONFIG_PATH', './app_conf.yml')
        configs = read_config_yml(config_path)
        return {
            'anthropic_api_key': configs['llm']['anthropic_api_key']
        }

    @workflow.query
    def event_stream(self) -> EventStream | None:
        if self.events:
            return self.events.popleft()
        return None

    @workflow.signal
    async def append_event(self, event_stream: EventStream):
        # This signal is invoked by streaming_handler, pushing event for every async loop
        self.events.append(event_stream)
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
