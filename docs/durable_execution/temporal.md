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
8. The [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] tells Temporal to use Pydantic for serialization and deserialization, and automatically registers activities for agents listed in `__pydantic_ai_agents__`. Activity retry policies treat [`UserError`][pydantic_ai.exceptions.UserError], `PydanticUserError`, [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior], and [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] as non-retryable, along with Temporal's own over-limit-payload failure types, `PayloadsTooLarge` and (before `temporalio` 1.31) `PayloadSizeError` (see [Large Payloads](#large-payloads)), while the worker registers `UserError`, `PydanticUserError`, [`AgentRunError`][pydantic_ai.exceptions.AgentRunError], and `UnsupportedEventLoopError` as `workflow_failure_exception_types`.
9. We start the worker that will listen on the specified task queue and run workflows and activities. In a real world application, this might be run in a separate service.
10. We call on the server to execute the workflow on a worker that's listening on the specified task queue.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

Because the same agent works inside and outside a workflow, [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] composes with all other [capabilities](../capabilities/overview.md) (instrumentation, [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata], [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream], etc.) without each needing a Temporal-specific wrapper variant.

Workflow code has to use the async API: [`Agent.run_sync()`][pydantic_ai.agent.Agent.run_sync] drives the event loop itself, which Temporal's workflow event loop doesn't allow, so calling it inside a workflow raises a [`UserError`][pydantic_ai.exceptions.UserError] telling you to `await agent.run()` instead. Outside a workflow, an agent with `TemporalDurability` behaves like a normal agent, so `run_sync()` works as usual.

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

Upgrading to this version changes the activity sequence for tools that have an `args_validator`, so workflows already in flight that call such a tool need [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) or a [patch](https://python.temporal.io/temporalio.workflow.html#patched). Workflows that don't call a tool with an `args_validator` are unaffected.

[`DynamicToolset`][pydantic_ai.toolsets.DynamicToolset] and toolsets contributed by [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability] are supported. Their factory is re-resolved inside activities when tools are listed and called, so it must be deterministic given the run dependencies. Like other wrapped toolsets, every `DynamicToolset` requires an explicit `id`: pass `id=` when constructing one directly, set the `id` parameter of the [`@agent.toolset`][pydantic_ai.agent.Agent.toolset] decorator, or set a stable capability `id` on `DynamicCapability`. Note that with Temporal, `per_run_step=False` is not respected, as the toolset always needs to be created on-the-fly in the activity.

[Capabilities](../capabilities/overview.md) that contribute a toolset — a [`Capability`][pydantic_ai.capabilities.Capability] with `tools=`, or an [`MCP`][pydantic_ai.capabilities.MCP] server running locally — derive the toolset's `id` from the capability's own [`id`][pydantic_ai.capabilities.AbstractCapability.id], so set `Capability(id='...', tools=[...])` or `MCP(id='...', url='...')`. (`MCP` falls back to an id derived from the server URL's host and path when no `id` is given.) A toolset passed to a capability via `toolsets=` keeps its own `id`, which must be set on the toolset itself.

All other agents and toolsets are supported.

### Tool Argument Validation

A tool's [`args_validator`](../tools-advanced.md#args-validator) runs in a `validate_args` activity, so it may perform I/O and dynamic-tool validators survive the workflow boundary. A tool without one schedules no extra activity. Validation runs before [approval and deferral](../deferred-tools.md), so rejected arguments never reach an approver. A validator may also raise `ApprovalRequired` or `CallDeferred`; resuming an approved call runs validation again with `tool_call_approved` set.

Each activity derives its typed arguments independently, so schema validators must be idempotent. A Pydantic validation context is rebuilt from the `validation_context` configured on [`Agent`][pydantic_ai.agent.Agent]; a callable context builder runs again with the activity's own run context.

### Agent Run Context and Dependencies

As workflows and activities run in separate processes, any values passed between them need to be serializable. As these payloads are stored in the workflow execution event history, Temporal limits their size to 2MB by default — see [Large Payloads](#large-payloads) for what that budget actually buys you.

To account for these limitations, tool functions and the [event stream handler](#streaming) running inside activities receive a limited version of the agent's [`RunContext`][pydantic_ai.tools.RunContext], and it's your responsibility to make sure that the [dependencies](../dependencies.md) object provided to [`Agent.run()`][pydantic_ai.agent.Agent.run] can be serialized using Pydantic.

`deps` isn't the only value that crosses into an activity: [`model_settings`](../agent.md#model-run-settings), the `RunContext` `metadata` and `tool_call_metadata`, and [tool metadata](#per-tool-activity-config) do too, and all need to be serializable by Pydantic. A value that isn't raises a `UserError` naming the type it couldn't serialize. This includes some values that are otherwise supported: pass [`model_settings['timeout']`][pydantic_ai.settings.ModelSettings.timeout] as a number of seconds rather than an `httpx.Timeout`.

Values that Pydantic AI carries across the boundary as untyped dictionaries — [tool metadata](#per-tool-activity-config), the `RunContext` `metadata` and `tool_call_metadata`, the `metadata` on [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred], and the `metadata`, `provider_details`, and `vendor_metadata` carried on messages and their parts — arrive as their JSON shapes rather than the original Python objects: a `set` or `tuple` becomes a `list`, a dataclass or Pydantic model becomes a `dict`, UTF-8-decodable `bytes` become a `str`, and non-string dictionary keys become strings. Arbitrary binary doesn't make it onto the wire at all — encode it as base64 first. There's no type information on the wire to restore any of this from, so keep these payloads JSON-native, or re-validate them into the type you expect (with a [`TypeAdapter`](https://docs.pydantic.dev/latest/api/type_adapter/)) on the receiving side. Every field with a declared type — the rest of each message, [`ToolDefinition`][pydantic_ai.tools.ToolDefinition], and [`ModelRequestParameters`][pydantic_ai.models.ModelRequestParameters], as well as the run context's `usage`, `usage_limits`, `loaded_capability_ids`, and `discovered_tool_names` — round-trips faithfully.

!!! warning "Persisted payload schemas"
    Temporal deserializes persisted workflow and activity payloads using the models and type annotations available in the currently deployed worker, so treat these models as durable contracts across deployments. Adding an optional field with a default stays compatible, but adding a required field or making another incompatible change can cause payload decoding to fail before the workflow or activity body executes. This is especially relevant to application-owned workflow inputs and dependency models: since Pydantic AI does not own or migrate Temporal workflow history, applications with long-running workflows should adopt a versioning or migration strategy when changing them.

Specifically, only the `deps`, `run_id`, `conversation_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `partial_output`, `usage`, `usage_limits`, `trace_include_content`, `instrumentation_version`, `loaded_capability_ids`, `discovered_tool_names`, and `capability_active` fields are available by default. `agent` and `root_capability` are re-attached from the worker's agent instance, `tool_manager` is `None` (so [`available_tool_names`][pydantic_ai.tools.RunContext.available_tool_names] returns the snapshot resolved when the activity was dispatched, and falls back to `discovered_tool_names` only for a subclass whose `serialize_run_context` doesn't carry it; [`active_capability_ids`][pydantic_ai.tools.RunContext.active_capability_ids] is carried the same way, which is what lets [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answer for a tool owned by a capability), and `pending_messages` holds a guard that makes [`ctx.enqueue()`][pydantic_ai.tools.RunContext.enqueue] raise inside an activity, since the activity's recorded result is replayed without re-running your code and the enqueued messages would be dropped.

Trying to access any other field — `model`, `prompt`, `messages`, `model_settings`, `tracer`, `validation_context`, or `capabilities` — raises a `UserError` rather than returning that field's default value, so a field that didn't cross the boundary can't be mistaken for real run state. A multi-modal `prompt` can carry large [`BinaryContent`][pydantic_ai.messages.BinaryContent], and carrying it would put that content in every activity payload, creating the same Temporal [payload size](#large-payloads) concern as `messages`.
If you need one or more of these attributes to be available inside activities, you can create a [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with custom `serialize_run_context` and `deserialize_run_context` class methods and pass it as the `run_context_type` argument to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. A subclass can opt in to carrying `prompt` if it knows its prompts are text-only.

The activity's `RunContext` is rebuilt from the serialized payload, so its fields are copies: mutating them inside an activity does not affect the run. In particular, `usage` is a snapshot of the run's usage at the time the activity was scheduled. If a tool [delegates to another agent](../multi-agent-applications.md#agent-delegation) with `usage=ctx.usage`, the delegate's tokens and requests stay behind in the activity: they're missing from the parent run's [`result.usage`][pydantic_ai.agent.AgentRunResult.usage] and are never charged against its [usage limits](../agent.md#usage-limits). To account for delegate usage, carry it yourself: return the delegate's [`result.usage`][pydantic_ai.agent.AgentRunResult.usage] from the tool, or record it in an external store your `deps` can reach. Temporal loses these mutations unconditionally. [DBOS](dbos.md) and [Prefect](prefect.md) pass the live `RunContext` into their in-process durable units, so mutations do accrue while a step or task body actually runs — but they're lost there too whenever the body doesn't run because its recorded result is replayed (DBOS workflow recovery) or reused (a Prefect task cache hit), which makes the same code account differently from one run to the next. Don't rely on the in-process engines' behavior; a return channel that works for all three is under discussion in [pydantic-ai#6886](https://github.com/pydantic/pydantic-ai/issues/6886).

A tool's [`prepare`](../tools-advanced.md#tool-prepare) function is not affected by these limitations: for tools in a [`FunctionToolset`][pydantic_ai.toolsets.FunctionToolset] (including those defined on the agent itself), it runs in workflow code with the complete `RunContext`, once per run step like outside a workflow. The tool definition it returns is sent to the tool-call activity, which uses it as-is, so the tool the model saw is the tool that runs, down to its [`timeout`](../tools-advanced.md#tool-timeout). Tools from a `DynamicToolset` are the exception: as the toolset is re-resolved inside activities, their `prepare` functions run there as well and see the limited `RunContext`.

### Capabilities at Runtime

Attach [capabilities](../capabilities/overview.md) when the agent is constructed, so `TemporalDurability.for_agent()` can register their activities before the worker starts. Passing `agent.run(capabilities=[...])` inside a workflow raises a `UserError`: a capability added that late has no registered activities for the toolsets it contributes or for its own [`@durable_operation`][pydantic_ai.capabilities.durable_operation] methods.

Capabilities that only observe the run are safe to attach per-run: their hooks read run state but don't contribute tools, toolsets, or durable operations. [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] is the built-in example and is exempt from the restriction. The current restriction is more conservative because third-party capabilities can't yet declare that they only observe the run. Deriving this from the hooks a capability overrides is tracked in [#5477](https://github.com/pydantic/pydantic-ai/issues/5477); if you need a per-run capability inside a workflow, please share your use case there. Outside a workflow the durability capability is transparent, so per-run capabilities are fine there.

### Large Payloads

Temporal stores every payload it records in the workflow execution event history — both what an activity returns and the arguments it was scheduled with — and caps each one at 2MB by default. Two things commonly breach that cap:

- **Binary content**, such as an image a tool returns or a model puts on its response. Binary data is base64-encoded on its way into the payload, so the usable budget for raw bytes is about three quarters of the cap — roughly 1.5MB. An image whose size reads as comfortably under 2MB can still be rejected.
- **A large [dependencies object](../dependencies.md)**, which is copied into history every time Pydantic AI schedules a model, tool, MCP, or event-handler activity — so a wide tool fan-out can both bloat history permanently and breach the per-payload limit.

#### When a tool return or model response is too large

A tool returning a [`BinaryImage`][pydantic_ai.messages.BinaryImage] — whether the subagent fallback behind the [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] capability or your own `local=` generator — sends those bytes back as the activity's result payload, and Pydantic AI raises a `UserError` naming the tool if Temporal rejects it. The same applies to an image a model returns on its response, as with a [native image generation tool](../native-tools.md#image-generation-tool), since the model request is itself an activity; there the `UserError` names the model. A [streamed](#streaming) segment can also overflow on its buffered events alone, without any image involved.

For a tool you control, the simplest fix is to keep the bytes out of history entirely: write the image to object storage and return a reference — an [`ImageUrl`][pydantic_ai.messages.ImageUrl], or a key your application resolves later. Small payloads also keep workflow replay fast, so this pays off beyond just staying under the cap. When the bytes come back from a provider, or the payload is a large `deps` object rather than media, use external storage instead.

#### Offloading payloads to your own storage

Temporal solves this generally with [external storage](https://docs.temporal.io/external-storage), which transparently offloads any payload above a size threshold to your own store and puts a reference in history instead. It applies to every payload in both directions, so it covers the cases the `UserError` cannot. Configure it with an [`ExternalStorage`](https://docs.temporal.io/develop/python/data-handling/external-storage) on your `DataConverter`:

```python {title="temporal_external_storage.py" test="skip" lint="skip"}
import dataclasses

from temporalio.client import Client
from temporalio.converter import DataConverter, ExternalStorage

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from my_app.storage import my_storage_driver  # your `StorageDriver` implementation


async def connect() -> Client:
    data_converter = dataclasses.replace(
        DataConverter.default,
        external_storage=ExternalStorage(drivers=[my_storage_driver]),
    )
    return await Client.connect(
        'localhost:7233',
        data_converter=data_converter,
        plugins=[PydanticAIPlugin()],
    )
```

[`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] replaces only Temporal's default payload *converter* with its Pydantic-aware one, so your `external_storage` — like a custom `payload_codec` or `failure_converter_class` — is preserved.

Payloads below the threshold (256 KiB by default) are untouched, so this costs nothing on runs with small dependencies. See Temporal's documentation for the available storage drivers and for writing your own; note that external storage is in public preview, so its API may still change.

!!! warning "Stored payloads are part of your workflow history"
    A payload that is missing, expired, or inaccessible prevents workers from decoding and replaying the workflow. Keep stored payloads available for at least as long as their histories can be replayed.

Because offloading happens per payload, a fan-out of N activities still emits N (small) references. External storage removes the size ceiling and the history bloat; it does not deduplicate identical dependencies within a single workflow-task completion.

If you cannot adopt a public-preview API, Temporal also documents a hand-written [claim check codec](https://docs.temporal.io/ai-cookbook/claim-check-pattern-python) built on [`PayloadCodec`](https://docs.temporal.io/develop/python/converters-and-encryption#custom-payload-codec), which `PydanticAIPlugin` preserves in the same way.

As a last resort, if the bytes genuinely need to sit in the workflow history, raise the `limit.blobSize.error` [dynamic config](https://docs.temporal.io/references/dynamic-configuration) on your Temporal server. Note that Temporal's own gRPC message limit still applies above it, so this raises the ceiling rather than removing it — see Temporal's [blob size limit](https://docs.temporal.io/troubleshooting/blob-size-limit-error) guidance.

!!! warning "Only what an activity *returns* gets the friendly error"
    Values travelling *into* an activity — the message history a model request carries, or a streamed event handed to an `event_stream_handler` — are encoded in workflow code, where Temporal fails the workflow *task* instead of the activity. That failure never reaches Pydantic AI, and because it's a workflow-task failure rather than an activity one, the retry policy doesn't bound it either: the run retries indefinitely rather than raising. So a tool return that fits can still be too large once it's part of the history of the *next* model request. External storage is the reliable fix here, as it applies to every payload in both directions.

    The `UserError` also relies on Temporal's default failure converter to identify an over-limit payload, so it won't fire if you supply your own `failure_converter_class`, or run against a server that doesn't report its limits to the worker.

!!! note "Image output types are not supported at all"
    An agent whose [`output_type`](../output.md) includes [`BinaryImage`][pydantic_ai.messages.BinaryImage] raises a `UserError` before the request is even made, rather than failing on size: every such run would have to carry the generated image across the boundary.

### Streaming

[`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream], [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], and [`Agent.iter()`][pydantic_ai.agent.Agent.iter] work inside a Temporal workflow, but their events are buffered rather than delivered in real time. The model stream runs inside the durable activity, and its events are replayed to the workflow after the activity completes.

For handlers with I/O side effects, pass `event_stream_handler=` to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. Model events are delivered live inside each model-request activity, while each tool event is delivered in its own event-handler activity. As with any Temporal activity, a handler may run more than once if an activity retries, so keep its side effects idempotent.

Alternatively, register [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream]. Its handler runs in workflow code and must be deterministic because it re-runs on workflow replay. Tool and final-output events arrive live, while the real captured model events are replayed after each model request completes. For examples, see the [streaming docs](../agent.md#streaming-all-events).

A durability `event_stream_handler=` and a separately registered `ProcessEventStream` are two distinct handlers, and each fires once. The durability handler receives live events inside the durable activity, while `ProcessEventStream` sees the buffered replay in workflow code.

A per-run handler passed to `Agent.run(event_stream_handler=...)` also runs workflow-side against replayed model events.

As the streaming model request activity, workflow, and workflow execution call all take place in separate processes, passing data between them requires some care:

- To get data from the workflow call site or workflow to the event stream handler, you can use a [dependencies object](#agent-run-context-and-dependencies).
- To get data from the event stream handler to the workflow, workflow call site, or a frontend, you need to use an external system that the event stream handler can write to and the event consumer can read from, like a message queue. You can use the dependency object to make sure the same connection string or other unique ID is available in all the places that need it.

Because the model stream is consumed inside the activity, cancelling it from the workflow side (e.g. with [`AgentStream.cancel()`][pydantic_ai.result.AgentStream.cancel]) is not available across the durable boundary. To stop an in-flight model request, cancel the Temporal workflow: the cancellation is delivered to the activity (via its heartbeats), which cancels any server-side job before the activity completes.

Whole-run cancellation (see [Cancelling a Run](../agent.md#cancelling-a-run)) follows the same split, with Temporal-specific consequences:

- Calling [`AgentRun.cancel()`][pydantic_ai.run.AgentRun.cancel] from workflow code raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] as an ordinary application outcome: a workflow that catches it completes normally rather than ending as *Cancelled*, and the run remains replay-deterministic. An uncaught `RunCancelled` fails the workflow as a typed application error, and the run state does not cross the failure boundary -- catch it inside the workflow if you need [`all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages].
- [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel] requires being in the same process as the run, so calling it from a tool running inside an activity raises a clear [`UserError`][pydantic_ai.exceptions.UserError] instead of hanging.
- [`CancellationToken`][pydantic_ai.CancellationToken] is also same-process state and cannot be passed to a Temporal durable run; cancel the Temporal workflow instead.
- Cancelling the Temporal workflow itself remains an external cancellation: `CancelledError` keeps propagating and the workflow still ends as *Cancelled*.

[`Agent.run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync] is not for workflow code: it requires no running event loop and wraps `run_stream()`. Under [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability], use the buffered async streaming APIs above or [`Agent.run()`][pydantic_ai.agent.Agent.run] with an event stream handler. Outside a workflow, an agent with `TemporalDurability` behaves like a normal agent, so `run_stream_sync()` works as usual. (Wrapper `TemporalAgent` forbids `run_stream` inside workflows — use `run` + event stream handler there.)

### Suspended Turns and Background Mode

Some providers can pause a model turn mid-flight (Anthropic `pause_turn`) or run it as a server-side job that's polled until it's ready ([OpenAI background mode](../models/openai.md#background-mode)). Pydantic AI transparently continues such a suspended turn until it completes. Each segment runs in a separate model request activity, while the workflow checkpoints the suspended [`ModelResponse`][pydantic_ai.messages.ModelResponse] and its background job ID between segments. The final response is merged and usage is recorded once. A [`message_history`](../message-history.md) ending in a suspended response resumes with that response passed to the first activity.

This has a few operational implications:

- **Timeouts and heartbeats**: size `start_to_close_timeout` and `heartbeat_timeout` for one provider round trip. Model request activities are given a default `heartbeat_timeout` of 30 seconds; see [Activity Configuration](#activity-configuration) for how heartbeating works across the other activities.
- **Retries and waits**: a failed segment retries independently. Delays between background polls use durable Temporal timers and do not consume activity wall-clock time.
- **Cancellation**: if an error abandons a suspended job, its provider teardown runs in a dedicated cancellation activity.
- **Payload size**: whenever [streaming](#streaming) is used — an `event_stream_handler`, a `ProcessEventStream` capability, or a per-run `event_stream_handler` — each segment's buffered events are shipped back to the workflow and must fit within Temporal's [payload size limit](#large-payloads) (2MB by default). A segment that overflows raises the same `UserError` as any other oversized model response.

!!! note
    If you use a custom [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext] subclass with your own `serialize_run_context`, keep including the `usage` and `usage_limits` fields: tools and capabilities running inside activities read them from the [`RunContext`][pydantic_ai.tools.RunContext], e.g. to adapt to the run's remaining [usage budget](../agent.md#usage-limits).

### Model Selection at Runtime

[`Agent.run(model=...)`][pydantic_ai.agent.Agent.run] normally supports both model strings (like `'openai:gpt-5.6-sol'`) and model instances. Under Temporal, a model instance can't be serialized for the replay mechanism, and rebuilding one from its `model_id` string would build a *different* model — the same model name on whatever provider the worker's environment implies, so the request would go to another endpoint with other credentials. An instance that isn't registered ahead of time is therefore rejected with a `UserError`. There are two ways to use a specific instance inside a workflow:

- pre-register it by passing a `models` dict to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability], then reference it by name or by passing the registered instance directly to `agent.run(model=...)`;
- pass a model-name string and build the instance on the worker with a [`ResolveModelId`](../capabilities/resolve-model-id.md) capability — the right choice when the model depends on the run's `deps`, e.g. per-user credentials.

Model-name strings themselves never need registering. The agent's own model, set at construction, is always available as the default; the agent must have a model set when it's created.

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
- `model_activity_config`: The Temporal activity config to use for model request activities. This is merged with the base activity config. Model request activities carry the [durable-execution retry layer](../retries.md#the-layers): Temporal's default `RetryPolicy.maximum_attempts` of `0` means unbounded re-execution of the model request — see [Retry multiplication](../retries.md#retry-multiplication) for how it stacks with the SDK client's and transport's retries.
- `event_stream_handler_activity_config`: The Temporal activity config to use for event stream handler activities. This is merged with the base activity config.
- `toolset_activity_config`: The Temporal activity config to use for get-tools and call-tool activities for specific toolsets identified by ID. This is merged with the base activity config.

Because `ActivityConfig` is a `TypedDict`, a misspelled or misplaced key is not caught at runtime by Python and would only fail once the config is handed to Temporal inside the workflow, where the failure is retried indefinitely. Config keys are therefore checked when [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] is constructed, so a key Temporal doesn't know raises a `UserError` up front.

Per-tool activity config lives on the tool itself — see [Per-tool activity config](#per-tool-activity-config) below.

Every activity Pydantic AI registers heartbeats in the background while it runs, so that a long-but-healthy activity isn't mistaken for a crashed worker and so that cancelling the Temporal workflow can be delivered to it — see [Streaming](#streaming) for how that stops an in-flight model request. Only model request activities get a `heartbeat_timeout` by default, of 30 seconds; setting one on any other activity is up to you.

!!! warning "Don't set a `heartbeat_timeout` on activities that can block the event loop"
    Heartbeats are emitted from a background task, so an activity that occupies the event loop without yielding — a CPU-bound tool function, say — stops beating and has its attempt failed once the timeout elapses. Such an activity is better served by `start_to_close_timeout` alone.

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

The opt-out applies to function and dynamic tools only. MCP tools perform I/O and always run in
their Temporal activity, so `metadata={'temporal': False}` on an MCP tool raises a `UserError`.

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

By default, the `LogfirePlugin` will instrument Temporal (including metrics) and Pydantic AI and send all data to Logfire. Temporal metrics are exported every 60 seconds. You can change the interval by passing a `datetime.timedelta` as `metric_periodicity` to the `LogfirePlugin` constructor.

If your application already called `logfire.configure()` itself, the plugin keeps that configuration instead of replacing it, so your scrubbing options, exporters, sampling, and console settings are left alone. To customize Logfire configuration and instrumentation, you can pass a `setup_logfire` function to the `LogfirePlugin` constructor and return a custom `Logfire` instance (i.e. the result of `logfire.configure()`).

To disable sending Temporal metrics to Logfire, pass `metrics=False` to the `LogfirePlugin` constructor. This also lets you supply your own [`Runtime`](https://python.temporal.io/temporalio.runtime.Runtime.html) to `Client.connect()` when you need to configure other Temporal telemetry options; the plugin will still configure tracing.

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
