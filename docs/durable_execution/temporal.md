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

To make a run durable, call `agent.run()` inside a Temporal workflow executed on a worker; outside one, the agent runs as a normal, non-durable agent.

Add durable execution to any [`Agent`][pydantic_ai.agent.Agent] by attaching the [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] [capability](../capabilities/overview.md). The agent stays a normal `Agent` everywhere — outside a workflow it behaves transparently, and inside a workflow the capability routes model requests, tool calls, and MCP server communication through Temporal activities.

!!! warning "A run is durable only inside a workflow"
    Attaching `TemporalDurability` does not by itself make runs durable — durability comes from executing the run inside a Temporal workflow that your application starts (via a Temporal client, as in the example below). Calling `agent.run()` or a streaming method from ordinary code — including indirectly, e.g. through a [UI adapter](../ui/overview.md) in a web endpoint — is a plain, non-durable agent run. To serve a durable agent behind an API, have the endpoint start (or signal) a workflow and bridge its results or events back to the client, rather than running the agent in the endpoint itself.

Here is a simple but complete example of attaching durable execution to an agent, creating a Temporal workflow with durable execution logic, connecting to a Temporal server, and running the workflow from non-durable code. All it requires is to install Pydantic AI with the `temporal` optional group:

```bash
pip/uv-add pydantic-ai[temporal]
```

Or if you're using the slim package, you can install it with the `temporal` optional group:

```bash
pip/uv-add pydantic-ai-slim[temporal]
```

You'll also need a Temporal server to be [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally):

```sh
brew install temporal
temporal server start-dev
```

```python {title="temporal_durability.py" test="skip"}
import uuid

from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import (
    PydanticAIPlugin,
    PydanticAIWorkflow,
    TemporalDurability,
)

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions="You're an expert in geography.",
    name='geography',  # (1)!
    capabilities=[TemporalDurability()],  # (2)!
)


@workflow.defn
class GeographyWorkflow(PydanticAIWorkflow):  # (3)!
    __pydantic_ai_agents__ = [agent]  # (4)!

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await agent.run(prompt)  # (5)!
        return result.output


async def main():
    client = await Client.connect(  # (6)!
        'localhost:7233',  # (7)!
        plugins=[PydanticAIPlugin()],  # (8)!
    )

    async with Worker(  # (9)!
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

1. The agent's `name` is used to uniquely identify its activities.
2. Attach durability via `capabilities=[...]`. The capability discovers the agent's name, model, and toolsets when bound to the agent, and registers an activity for each. Outside a workflow, the capability is transparent — the agent behaves as a normal `Agent`.
3. The workflow represents a deterministic piece of code that can use non-deterministic activities for operations that require I/O. Subclassing [`PydanticAIWorkflow`][pydantic_ai.durable_exec.temporal.PydanticAIWorkflow] is optional but provides proper typing for the `__pydantic_ai_agents__` class variable.
4. List the agents used by this workflow. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] automatically registers the activities contributed by each agent's [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] capability with the worker. Alternatively, if modifying the worker initialization is easier than the workflow class, you can use [`AgentPlugin`][pydantic_ai.durable_exec.temporal.AgentPlugin] to register an agent's activities directly on the worker.
5. `agent.run()` works as usual; inside the workflow, model requests, tool calls, and MCP server communication are routed through Temporal activities.
6. We connect to the Temporal server which keeps track of workflow and activity execution.
7. This assumes the Temporal server is [running locally](https://github.com/temporalio/temporal#download-and-start-temporal-server-locally).
8. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] tells Temporal to use Pydantic for serialization and deserialization, treats [`UserError`][pydantic_ai.exceptions.UserError] exceptions as non-retryable, and automatically registers activities for agents listed in `__pydantic_ai_agents__`.
9. We start the worker that will listen on the specified task queue and run workflows and activities. In a real world application, this might be run in a separate service.
10. We call on the server to execute the workflow on a worker that's listening on the specified task queue.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Because the same agent works inside and outside a workflow, [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] composes with all other [capabilities](../capabilities/overview.md) (instrumentation, [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata], [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream], etc.) without each needing a Temporal-specific wrapper variant.

In a real world application, the agent, workflow, and worker are typically defined separately from the code that calls for a workflow to be executed.
Because Temporal workflows need to be defined at the top level of the file and the agent is needed inside the workflow and when starting the worker (to register the activities), it needs to be defined at the top level of the file as well.

For more information on how to use Temporal in Python applications, see their [Python SDK guide](https://docs.temporal.io/develop/python).

### Wrapper-agent path (deprecated)

!!! warning "Deprecated"
    [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] is the original wrapper-agent path for Temporal integration and will be removed in v3. New code should use the [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] capability shown above.

    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] accepts the activity names and payload shapes recorded by `TemporalAgent` (as long as the agent's `name`, toolset `id`s, and `models=` registry keys stay the same, and `event_stream_handler=` remains on `TemporalDurability` while `TemporalAgent`-era workflows are still in flight), so workflows started under `TemporalAgent` will replay correctly after switching — there's no need to drain or version your workflows first. Histories recorded with an event stream handler registered a per-event activity; migrating to [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] before those workflows finish breaks replay.

Any agent can be wrapped in a [`TemporalAgent`][pydantic_ai.durable_exec.temporal.TemporalAgent] to get a durable agent variant that can be used inside a Temporal workflow. At the time of wrapping, the agent's model and toolsets are frozen, activities are dynamically created for each, and the original model and toolsets are wrapped to call on the worker to execute the corresponding activities instead of directly performing the actions inside the workflow. The original agent can still be used as normal outside the Temporal workflow, but any changes to its model or toolsets after wrapping will not be reflected in the durable agent.

```python {title="temporal_agent.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent

agent = Agent('openai:gpt-5.6-sol', name='geography')
temporal_agent = TemporalAgent(agent)
# Use `temporal_agent` from inside a workflow, list it in `__pydantic_ai_agents__`,
# and connect with `PydanticAIPlugin()` exactly like the capability example above.
```

## Temporal Integration Considerations

There are a few considerations specific to agents and toolsets when using Temporal for durable execution. These are important to understand to ensure that your agents and toolsets work correctly with Temporal's workflow and activity model.

### Agent Names and Toolset IDs

To ensure that Temporal knows what code to run when an activity fails or is interrupted and then restarted, even if your code is changed in between, each activity needs to have a name that's stable and unique.

When [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] dynamically creates activities for the agent's model requests and toolsets (specifically those that implement their own tool listing and calling, i.e. [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPToolset`][pydantic_ai.mcp.MCPToolset]), their names are derived from the agent's [`name`][pydantic_ai.agent.AbstractAgent.name] and the toolsets' [`id`s][pydantic_ai.toolsets.AbstractToolset.id]. These fields are normally optional, but are required to be set when using Temporal. They should not be changed once the durable agent has been deployed to production as this would break active workflows.

[`DynamicToolset`][pydantic_ai.toolsets.DynamicToolset] and toolsets contributed by [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability] are supported. Their factory is re-resolved inside activities when tools are listed and called, so it must be deterministic given the run dependencies. Like other wrapped toolsets, every `DynamicToolset` requires an explicit `id`: pass `id=` when constructing one directly, set the `id` parameter of the [`@agent.toolset`][pydantic_ai.agent.Agent.toolset] decorator, or set a stable capability `id` on `DynamicCapability`. Note that with Temporal, `per_run_step=False` is not respected, as the toolset always needs to be created on-the-fly in the activity.

[Capabilities](../capabilities/overview.md) that contribute a toolset — a [`Capability`][pydantic_ai.capabilities.Capability] with `tools=`, or an [`MCP`][pydantic_ai.capabilities.MCP] server running locally — derive the toolset's `id` from the capability's own [`id`][pydantic_ai.capabilities.AbstractCapability.id], so set `Capability(id='...', tools=[...])` or `MCP(id='...', url='...')`. (`MCP` falls back to an id derived from the server URL's host and path when no `id` is given.) A toolset passed to a capability via `toolsets=` keeps its own `id`, which must be set on the toolset itself.

Other than that, any agent and toolset will just work!

### Agent Run Context and Dependencies

As workflows and activities run in separate processes, any values passed between them need to be serializable. As these payloads are stored in the workflow execution event history, Temporal limits their size to 2MB.

To account for these limitations, tool functions and the [event stream handler](#streaming) running inside activities receive a limited version of the agent's [`RunContext`][pydantic_ai.tools.RunContext], and it's your responsibility to make sure that the [dependencies](../dependencies.md) object provided to [`Agent.run()`][pydantic_ai.agent.Agent.run] can be serialized using Pydantic.

!!! warning "Persisted payload schemas"
    Temporal deserializes persisted workflow and activity payloads using the models and type annotations available in the currently deployed worker, so treat these models as durable contracts across deployments. Adding an optional field with a default stays compatible, but adding a required field or making another incompatible change can cause payload decoding to fail before the workflow or activity body executes. This is especially relevant to application-owned workflow inputs and dependency models: since Pydantic AI does not own or migrate Temporal workflow history, applications with long-running workflows should adopt a versioning or migration strategy when changing them.

Specifically, only the `deps`, `run_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`, and `partial_output` fields are available by default, and trying to access `model`, `prompt`, `messages`, or `tracer` will raise an error.
If you need one or more of these attributes to be available inside activities, you can create a [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with custom `serialize_run_context` and `deserialize_run_context` class methods and pass it as the `run_context_type` argument to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability].

### Streaming

[`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.agent.Agent.iter] work inside a Temporal workflow, but their events are buffered rather than delivered in real time. The model stream runs inside the durable activity, and its events are replayed to the workflow after the activity completes.

For handlers with I/O side effects, pass `event_stream_handler=` to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. Model events are delivered live inside each model-request activity, while each tool event is delivered in its own event-handler activity. As with any Temporal activity, a handler may run more than once if an activity retries, so keep its side effects idempotent.

Alternatively, register [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream]. Its handler runs in workflow code and must be deterministic because it re-runs on workflow replay. Tool and final-output events arrive live, while the real captured model events are replayed after each model request completes. For examples, see the [streaming docs](../agent.md#streaming-all-events).

A durability `event_stream_handler=` and a separately registered `ProcessEventStream` are two distinct handlers, and each fires once. The durability handler receives live events inside the durable activity, while `ProcessEventStream` sees the buffered replay in workflow code.

A per-run handler passed to `Agent.run(event_stream_handler=...)` also runs workflow-side against replayed model events.

As the streaming model request activity, workflow, and workflow execution call all take place in separate processes, passing data between them requires some care:

- To get data from the workflow call site or workflow to the event stream handler, you can use a [dependencies object](#agent-run-context-and-dependencies).
- To get data from the event stream handler to the workflow, workflow call site, or a frontend, you can publish events to a [Workflow Stream](#streaming-events-to-a-frontend-with-workflow-streams) (recommended, no extra infrastructure), or use an external system that the event stream handler can write to and the event consumer can read from, like a message queue. You can use the dependency object to make sure the same connection string or other unique ID is available in all the places that need it.

#### Streaming events to a frontend with Workflow Streams

Rather than standing up a separate message queue, you can use Temporal's built-in [Workflow Streams](https://docs.temporal.io/develop/python/workflows/workflow-streams) as the transport: the parent workflow itself becomes the durable, offset-addressed channel that an external consumer subscribes to.

Set `event_stream_topic` on [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] and construct a [`WorkflowStream`](https://docs.temporal.io/develop/python/workflows/workflow-streams) in your workflow's `@workflow.init`. Every event is then published to that topic from within the activity. Setting `event_stream_topic` enables streaming on its own, and it's orthogonal to `event_stream_handler`: if you also pass a handler, both run and each sees every event.

```python {test="skip"}
import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.contrib.workflow_streams import WorkflowStream

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent
    from pydantic_ai.durable_exec.temporal import TemporalDurability

agent = Agent(
    'openai:gpt-5.6-sol',
    name='assistant',
    capabilities=[TemporalDurability(event_stream_topic='agent-events')],
)


@workflow.defn
class AssistantWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        # Hosts the stream that the agent's activities publish to. Without this,
        # published events are silently dropped.
        self.stream = WorkflowStream()
        self._finished = False
        self._released = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await agent.run(prompt)
        # All events are published by now. A Workflow Stream subscription is an Update long-poll
        # that can't complete once the workflow has returned, so stay alive until the consumer
        # acknowledges it has drained the stream — otherwise a consumer that starts late or lags
        # behind would miss the tail. Bound the wait so a consumer that never connects can't hang
        # the run.
        self._finished = True
        try:
            await workflow.wait_condition(lambda: self._released, timeout=timedelta(minutes=1))
        except asyncio.TimeoutError:
            pass
        return result.output

    @workflow.query
    def finished(self) -> bool:
        return self._finished

    @workflow.signal
    def release(self) -> None:
        self._released = True
```

An external consumer (with just the workflow handle) observes events as they arrive using [`stream_agent_events`][pydantic_ai.durable_exec.temporal.stream_agent_events], which decodes them back into typed [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s — effectively a durable [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] across the workflow boundary. The run — not any individual event — is the terminal signal (an event like `PartEndEvent` only ends a single response part, and tool calls or further parts can follow), so the consumer relays until the run has finished, then releases the workflow so it completes and the subscription ends:

```python {test="skip" lint="skip"}
import asyncio

from temporalio.client import Client

from pydantic_ai.durable_exec.temporal import stream_agent_events


async def relay_events(client: Client, prompt: str) -> str:
    handle = await client.start_workflow(
        AssistantWorkflow.run, prompt, id='assistant-1', task_queue='my-task-queue'
    )

    async def relay() -> None:
        async for event in stream_agent_events(client, handle, 'agent-events'):
            ...  # forward `event` to the frontend over SSE

    relay_task = asyncio.create_task(relay())
    # Wait until the run has finished producing events — queryable while the workflow is still
    # alive, unlike `handle.result()` which is gated on the release below — then let the relay
    # drain and release the workflow so it completes and the subscription ends.
    while not await handle.query(AssistantWorkflow.finished):
        await asyncio.sleep(0.2)
    await handle.signal(AssistantWorkflow.release)
    await relay_task
    return await handle.result()
```

Because Workflow Streams are offset-addressed, a reconnecting consumer can resume from its last seen offset via `stream_agent_events(..., from_offset=...)`, which is more robust than ordinary in-process streaming. `stream_agent_events` targets the specific run behind the handle you pass, so it's unaffected by later executions that reuse the same workflow ID.

A model stream emits a [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] per token, so to keep the volume down you can publish only a subset of events with the `event_stream_events` predicate:

```python {test="skip"}
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.messages import PartDeltaEvent

TemporalDurability(
    event_stream_topic='agent-events',
    event_stream_events=lambda event: not isinstance(event, PartDeltaEvent),  # skip per-token deltas
)
```

Under the hood, `event_stream_topic` is sugar over [`workflow_stream_event_handler`][pydantic_ai.durable_exec.temporal.workflow_stream_event_handler], which returns a regular `EventStreamHandler` you can also compose or wrap yourself.

!!! note "Caveats"
    - Workflow Streams add roughly 100ms of latency per roundtrip (tunable via `event_stream_batch_interval`) and their cost scales with the number of durable batches; they're suited to driving a UI, not ultra-low-latency use cases like real-time voice. Use `event_stream_events` to publish fewer events if the volume is a concern.
    - Delivery is at-least-once: the publishing handler runs inside an activity, so if that activity is retried its events are re-published (at new offsets), and consumers should tolerate duplicates. The workflow's final result remains authoritative.
    - Publishing to a Workflow Stream is non-blocking and unbounded: the SDK buffers published events in the activity and ships them in batches, so a high event rate or slow delivery grows that in-memory buffer until the activity exits. There's no backpressure at the stream layer — keep the volume down with `event_stream_events` (e.g. dropping per-token deltas) for chatty runs. (When a user `event_stream_handler` is combined with a topic, a slow handler does back-pressure the shared stream.)
    - The stream is durable, so events may be produced and consumed by processes running different Pydantic AI versions. `AgentStreamEvent` shapes are stable within a major version; keep producer and consumer on the same major version.
    - `stream_agent_events` ends when the workflow reaches a terminal state; use the workflow result for the authoritative output. Live events reach external consumers only — `run_stream_events()` remains unavailable inside workflow code.

Because the model stream is consumed inside the activity, cancelling it from the workflow side (e.g. with [`AgentStream.cancel()`][pydantic_ai.result.AgentStream.cancel]) is not available across the durable boundary. To stop an in-flight model request, cancel the Temporal workflow: the cancellation is delivered to the activity (via its heartbeats), which cancels any server-side job before the activity completes.

### Suspended Turns and Background Mode

Some providers can pause a model turn mid-flight (Anthropic `pause_turn`) or run it as a server-side job that's polled until it's ready ([OpenAI background mode](../models/openai.md#background-mode)). Pydantic AI transparently continues such a suspended turn until it completes. Each segment runs in a separate model request activity, while the workflow checkpoints the suspended [`ModelResponse`][pydantic_ai.messages.ModelResponse] and its background job ID between segments. The final response is merged and usage is recorded once. A [`message_history`](../message-history.md) ending in a suspended response resumes with that response passed to the first activity.

This has a few operational implications:

- **Timeouts and heartbeats**: size `start_to_close_timeout` and `heartbeat_timeout` for one provider round trip. Activities heartbeat automatically, with a default `heartbeat_timeout` of 30 seconds.
- **Retries and waits**: a failed segment retries independently. Delays between background polls use durable Temporal timers and do not consume activity wall-clock time.
- **Cancellation**: if an error abandons a suspended job, its provider teardown runs in a dedicated cancellation activity.
- **Payload size**: whenever [streaming](#streaming) is used — an `event_stream_handler`, a `ProcessEventStream` capability, or a per-run `event_stream_handler` — each segment's buffered events are shipped back to the workflow and must fit within Temporal's 2MB payload limit.

!!! note
    If you use a custom [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with your own `serialize_run_context`, keep including the `usage` and `usage_limits` fields: tools and capabilities running inside activities read them from the [`RunContext`][pydantic_ai.tools.RunContext], e.g. to adapt to the run's remaining [usage budget](../agent.md#usage-limits).

### Model Selection at Runtime

[`Agent.run(model=...)`][pydantic_ai.agent.Agent.run] normally supports both model strings (like `'openai:gpt-5.6-sol'`) and model instances. Under Temporal, a model instance can't be serialized for the replay mechanism, so it's sent to the worker as its `model_id` string and rebuilt there. That faithfully reproduces model-name strings and models with standard providers, but not an instance whose exact behavior depends on a custom provider, client, or settings — pre-register those.

To use such model instances inside a workflow, pre-register them by passing a `models` dict to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. You can then reference them by name or by passing the registered instance directly to `agent.run(model=...)`. The agent's own model, set at construction, is always available as the default; the agent must have a model set when it's created.

Model strings work as expected. To customize how a model string is built — a custom provider, API keys injected from configuration, per-user credentials carried on the run's `deps` — add a [`ResolveModelId`](../capabilities/resolve-model-id.md) capability before `TemporalDurability`: it gets first crack at every string, both at run setup and when the model is rebuilt inside the activity, where the resolver runs again with the run's actual `deps`. Since the resolver re-runs on the worker, it must be deterministic for a given `(model_id, deps)` and must not perform external I/O — carry credentials on `deps` or close over configuration loaded at startup.

Here's an example showing how to pre-register and use multiple models:

```python {title="multi_model_temporal.py" test="skip"}
import os

from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.models import Model, ModelResolutionContext, infer_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

# Create models from different providers
default_model = OpenAIResponsesModel('gpt-5.6-sol')
fast_model = AnthropicModel('claude-haiku-4-5')
reasoning_model = GoogleModel('gemini-3-pro-preview')


# Optional: customize how model-name strings are built.
def resolve_model(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
    if model_id.startswith('openai:'):
        provider = OpenAIProvider(api_key=os.environ['OPENAI_API_KEY'])
        return infer_model(model_id, provider_factory=lambda _: provider)
    return None  # everything else takes the default `infer_model` path


agent = Agent(
    default_model,
    name='multi_model_agent',
    capabilities=[
        ResolveModelId(resolve_model),  # Optional
        TemporalDurability(
            models={
                'fast': fast_model,
                'reasoning': reasoning_model,
            },
        ),
    ],
)


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, use_reasoning: bool, use_fast: bool) -> str:
        if use_reasoning:
            # Select by registered name
            result = await agent.run(prompt, model='reasoning')
        elif use_fast:
            # Or pass the registered instance directly
            result = await agent.run(prompt, model=fast_model)
        else:
            # Or pass a model string (resolved by `ResolveModelId` if it matches)
            result = await agent.run(prompt, model='openai:gpt-5.6-luna')
        return result.output
```

### Toolsets at Runtime

Additional toolsets can be passed per run via `agent.run(toolsets=...)`, but only toolsets that don't need durable wrapping are supported: non-executing toolsets like [`ExternalToolset`][pydantic_ai.toolsets.ExternalToolset], whose tools are executed outside the agent run, and [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset]s whose tools all opt out of activity wrapping with [`metadata={'temporal': False}`](#per-tool-activity-config). Other executing toolsets ([`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] and [`MCPToolset`][pydantic_ai.mcp.MCPToolset]) and dynamic toolsets must be set when constructing the agent so their activities can be registered with the worker before the workflow runs; passing them at runtime raises a `UserError`.

## Activity Configuration

Temporal activity configuration, like timeouts and retry policies, can be customized by passing [`temporalio.workflow.ActivityConfig`](https://python.temporal.io/temporalio.workflow._activities.ActivityConfig.html) objects to the [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] constructor:

- `activity_config`: The base Temporal activity config to use for all activities. If no config is provided, a `start_to_close_timeout` of 60 seconds is used.
- `model_activity_config`: The Temporal activity config to use for model request activities. This is merged with the base activity config.
- `event_stream_handler_activity_config`: The Temporal activity config to use for event stream handler activities. This is merged with the base activity config.
- `toolset_activity_config`: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.

Per-tool activity config lives on the tool itself — see [Per-tool activity config](#per-tool-activity-config) below.

### Per-tool activity config

Per-tool activity config lives on the tool's [`metadata`][pydantic_ai.toolsets.FunctionToolset.tool] field — [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] looks for a `'temporal'` key. You can set the metadata directly on the tool definition, or apply it across a selection of tools via the [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability. See the [capabilities documentation][pydantic_ai.capabilities.SetToolMetadata] for the full selector vocabulary.

```python {title="temporal_per_tool_config.py" test="skip"}
from datetime import timedelta

from temporalio.workflow import ActivityConfig

from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset(id='research')


@toolset.tool(metadata={'temporal': ActivityConfig(start_to_close_timeout=timedelta(minutes=5))})  # (1)!
async def fetch_paper(arxiv_id: str) -> str:
    ...


@toolset.tool(metadata={'temporal': False})  # (2)!
async def now() -> str:
    ...


agent = Agent(
    'openai:gpt-5.6-sol',
    name='research',
    toolsets=[toolset],
    capabilities=[
        SetToolMetadata(  # (3)!
            tools=['fetch_paper', 'fetch_dataset'],
            temporal=ActivityConfig(start_to_close_timeout=timedelta(minutes=5)),
        ),
        TemporalDurability(),
    ],
)
```

1. Inline: declare the activity config alongside the tool definition. Per-tool config merges on top of the toolset and base configs.
2. Set `'temporal': False` to skip activity wrapping entirely (only valid for `async` tools — sync tools always need an activity since threads aren't deterministic).
3. Selector-based: [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] applies the same metadata across a selection of tools (`'all'`, a name list, a dict, or a callable).

!!! tip "Configuring third-party tools"
    [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] is the recommended path when the activity config doesn't belong on the tool definition — for example, tools defined in third-party packages, or a group of tools that share the same timeout profile but live in different files.

## Activity Retries

On top of the automatic retries for request failures that Temporal will perform, Pydantic AI and various provider API clients also have their own request retry logic. Enabling these at the same time may cause the request to be retried more often than expected, with improper `Retry-After` handling.

When using Temporal, it's recommended to not use [HTTP Request Retries](../models/http-request-retries.md) and to turn off your provider API client's own retry logic, for example by setting `max_retries=0` on a [custom `OpenAIProvider` API client](../models/openai.md#custom-openai-client).

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

To fix this, you can use the [`temporalio.workflow.unsafe.imports_passed_through()`](https://python.temporal.io/temporalio.workflow._sandbox.unsafe.html#imports_passed_through) context manager to proactively import the package and not have it be reloaded in the workflow sandbox:

```python {title="temporal_activity.py" test="skip" noqa="F401"}
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import pandas
```
