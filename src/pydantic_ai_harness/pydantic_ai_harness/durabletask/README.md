# Durable Task Durability

`DurableTaskDurability` lets agent runs survive worker restarts using
[Durable Task](https://github.com/microsoft/durabletask-python) and Azure Durable Task Scheduler.
Model requests, tool discovery, tool calls and durable capability operations run as activities.
The scheduler records their results and returns them during orchestration replay instead of repeating completed work.

Outside an orchestration, the agent runs normally. Inside one, parallel tool operations are scheduled
together and resume as each activity finishes. Streaming is buffered at the activity boundary,
not delivered live to the client.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/durabletask/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

uv:

```bash
uv add "pydantic-ai-harness[durabletask]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[durabletask]" "pydantic-ai-slim[openai]"
```

## Run locally

Start the [Durable Task Scheduler emulator](https://learn.microsoft.com/en-us/azure/durable-task/sdks/quickstart-portable-durable-task-sdks):

```bash
docker run --rm --name dts-emulator -p 127.0.0.1:8080:8080 -p 127.0.0.1:8082:8082 mcr.microsoft.com/dts/dts-emulator:latest
```

The emulator uses task hub `default`. Its dashboard is at `http://localhost:8082`.
With the emulator running and `OPENAI_API_KEY` set:

```python {test="skip"}
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from pydantic_ai import Agent
from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability

agent = Agent('openai:gpt-6.1-sol', name='support', capabilities=[DurableTaskDurability()])
durable = DurableTaskAgent(agent)
worker = DurableTaskSchedulerWorker(
    host_address='localhost:8080', taskhub='default', token_credential=None, secure_channel=False
)

with DurableTaskSchedulerClient(
    host_address='localhost:8080', taskhub='default', token_credential=None, secure_channel=False
) as client:
    with durable.register(worker):
        with worker:
            worker.start()
            run = durable.start(client, 'Summarize the purpose of durable execution.')
            print(run.result().output)
```

`register` owns the async activity runtime. Enter it before starting the worker and leave it only
after the worker has stopped and drained. In production, run workers separately from clients.
Reconstruct the same agent on the client and use `durable.get_run(client, instance_id)` to reconnect.
For Azure, use the scheduler endpoint and a token credential, and leave TLS enabled.

`DurableTaskDurability` defaults to the agent's stable name, its default model, no event handler,
no native retry policy and no dependency factory. `models=` adds named worker-side models;
`event_stream_handler=` records handler calls as activities, not live client streaming.
`deps_factory=None` validates JSON dependencies against `deps_type`.
Pass either `retry_policy=` or `operation_config=`, not both; with neither, activities have no
native retries. Custom backends use `worker_scope(worker)` to own their activity runtime for the
same start/stop lifetime; `DurableTaskAgent.register` enters that scope automatically.

Annotate output functions and choice actions with their return types so clients can reconstruct
structured results. Unannotated output callables are limited to JSON values.

The integration emits no additional telemetry. Use Pydantic AI's normal instrumentation and the
scheduler's orchestration history to observe runs.

## Deploy to Azure Durable Task Scheduler

`[durabletask]` installs `azure-identity` transitively; declare it directly for imports. TLS is enabled by default:

```bash
uv add "pydantic-ai-harness[durabletask]" azure-identity
```

```python {test="skip"}
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from pydantic_ai import Agent
from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability

agent = Agent('openai:gpt-6.1-sol', name='support', capabilities=[DurableTaskDurability()])
durable = DurableTaskAgent(agent)
worker = DurableTaskSchedulerWorker(
    host_address='<scheduler>.<region>.durabletask.io', taskhub='<hub>', token_credential=DefaultAzureCredential()
)

with DurableTaskSchedulerClient(
    host_address='<scheduler>.<region>.durabletask.io', taskhub='<hub>', token_credential=DefaultAzureCredential()
) as client:
    with durable.register(worker):
        with worker:
            worker.start()
            run = durable.start(client, 'Summarize the purpose of durable execution.')
            print(run.result().output)
```

Any Azure `TokenCredential` works: `DefaultAzureCredential` for local development and managed identity in Azure,
`ManagedIdentityCredential(client_id=...)` for user-assigned identity, `ClientSecretCredential` or `ChainedTokenCredential`.
Grant the identity the [Durable Task Data Contributor](https://learn.microsoft.com/en-us/azure/durable-task/scheduler/durable-task-scheduler-dashboard#assign-dashboard-access-roles-azure)
role on the scheduler or task hub.

Set the same `taskhub` on worker and client; names of orchestrations and activities are hub-independent.
Use separate hubs per environment. Worker processes own `durable.register(worker)`; clients use `start`/`get_run`
with the same agent definition. Both can run on Container Apps, AKS or VMs; Azure Functions hosting helpers are a planned follow-up.
The scheduler dashboard shows `<agent>__agent` orchestrations, activities, custom status with pending approvals, and external events.

## Replay and side effects

An activity has an **at-least-once** execution guarantee. A worker can fail after a side effect but
before its result is recorded, so the next attempt can repeat that side effect. Make tools and event
handlers idempotent. `ctx.run_id` and `ctx.tool_call_id` can help construct application idempotency keys.

Keep agent names, toolset IDs, activity names and the orchestration's control flow compatible with
in-flight histories. Changing the order of durable operations can break replay. This integration
does not provide deployment version migrations or `continue_as_new` rollover.

Orchestrator-side hooks, instruction functions, output validators and inline tools run again during replay.
They must be deterministic and must not access the network, filesystem or threads.
Put I/O in tools or in a capability's `@durable_operation` method.

## Known limitation

`durabletask` 1.12.0 does not close a suspended orchestrator generator when a dispatch ends, so the
bridge's cleanup runs late and workers may log `Task was destroyed but it is pending!` on replay.
This is fixed upstream in [microsoft/durabletask-python#300](https://github.com/microsoft/durabletask-python/pull/300)
and [microsoft/durabletask-python#303](https://github.com/microsoft/durabletask-python/pull/303);
the minimum pin will move to the release that includes them.

## Dependencies

Pass a dataclass, Pydantic model or JSON value as `deps_input`. Without a dependency factory,
it is validated and serialized using the agent's `deps_type` before scheduling, then reconstructed
inside each activity. Without a declared dependency type, inputs must be JSON values.
For clients, connections and other live objects, supply an async context-manager factory with
JSON input instead:

```python {test="skip"}
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2
from pydantic import JsonValue, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai_harness.durabletask import DurableTaskDurability


@dataclass
class Dependencies:
    client: httpx2.AsyncClient
    tenant: str


@asynccontextmanager
async def build_dependencies(value: JsonValue) -> AsyncGenerator[Dependencies]:
    tenant = TypeAdapter(str).validate_python(value)
    async with httpx2.AsyncClient() as client:
        yield Dependencies(client=client, tenant=tenant)


agent = Agent(
    'openai:gpt-6.1-sol',
    deps_type=Dependencies,
    name='support',
    capabilities=[DurableTaskDurability(deps_factory=build_dependencies)],
)
```

Start this agent with `deps_input='tenant-a'`. The factory is entered and exited for each activity
attempt, on the activity runtime's event loop. Resources are not shared with the orchestrator.
Reading live dependency attributes from orchestrator-side code raises `UserError`; the proxy's
`deps_input` contains the original deterministic input.

Activities receive a snapshot of run identifiers, usage, limits, deadlines, tool-call metadata and
tool-availability state. The worker reattaches its agent and reconstructs workspaces from their
references. Live run state such as `ctx.messages`, `ctx.prompt` and the tool manager is not transported.
Reading an unavailable field raises rather than silently returning a default.
Emitting events and enqueuing messages from an activity are not supported.

## Approvals and deferred calls

Include `DeferredToolRequests` in the agent's output type and use the normal approval or
`CallDeferred` tool APIs. `DurableTaskAgent` pauses on durable external events and resumes the
agent with the tool results. Waiting does not hold an activity or dependency resource open.

```python {test="skip"}
from durabletask.client import TaskHubGrpcClient
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests
from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability

agent = Agent(
    'openai:gpt-6.1-sol',
    output_type=[str, DeferredToolRequests],
    name='approval',
    capabilities=[DurableTaskDurability()],
)


@agent.tool_plain(requires_approval=True)
def submit_order(order_id: str) -> str:
    return f'Submitted {order_id}'


@agent.tool_plain
def delivery_status(order_id: str) -> str:
    raise CallDeferred()


durable = DurableTaskAgent(agent)


def resolve_reviewed_tools(client: TaskHubGrpcClient, instance_id: str, *, approved: bool = True) -> None:
    run = durable.get_run(client, instance_id)
    pending = run.wait_for_pause(timeout=60)
    if pending is None:
        return
    for call in pending.approvals:
        run.approve(call.tool_call_id, approved=approved)
    for call in pending.calls:
        run.resolve(call.tool_call_id, {'status': 'complete'})
```

Call the helper after your application has reviewed approvals and performed the external calls.
Register the agent after adding its tools, then start its worker.

Use `approved=False` to deny a request. `result(timeout=...)` waits for completion; it does not
automatically approve calls. Event names include a pause sequence so repeated tool-call IDs in later
pauses cannot consume an earlier pause's events.

`start()` returns before the worker reaches an approval. `wait_for_pause(timeout=...)` polls for
approvals or deferred calls, returning their `DeferredToolRequests`, or `None` on completion.
It raises on failure, a missing instance or timeout; individual state reads use the SDK's synchronous
RPC. `pending_tools()` remains an immediate read. These client helpers block; offload them when
calling from an async application.

## Retries and inline tools

Pass a native `durabletask.task.RetryPolicy` as `retry_policy=` to retry failed activities.
`ModelRetry`, `ApprovalRequired` and `CallDeferred` are recorded as control-flow values, not retried
as activity failures.
Deterministic codec failures and `UserError` configuration errors are recorded and raised outside
the SDK's activity retry policy. Other activity exceptions remain eligible for native retries.
A tool may already have performed a side effect before its result fails serialization;
this does not undo the effect or replace the at-least-once guarantee.

For different policies by operation role or tool, pass the upstream
`RoleBasedOperationConfig` as `operation_config=` instead. Its `resolve_tool` callback receives
the operation ID, tool and tool name. Return a native retry policy, `None` for no retry, or `False`
for a deterministic, I/O-free inline function tool. Do not put native policy objects in tool metadata:
that metadata is transported as JSON.

## Existing orchestrators

Use `yield from run_durable(context, body)` to call ordinary `await agent.run(...)` from an existing
Durable Task generator orchestrator. Register the capability's activities through
`DurableTaskAgent.register`; it also registers the standard agent orchestrator.
`run_durable` supports parallel operations and asyncio timers using the orchestration's durable clock.
Pass serializable dependencies directly to `agent.run(deps=...)`; use `DurableTaskAgent` when a
worker dependency factory or automatic approval handling is needed.

```python {test="skip"}
from collections.abc import Generator
from contextlib import contextmanager

from durabletask import task
from durabletask.worker import TaskHubGrpcWorker
from pydantic_ai import Agent
from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability, run_durable

agent = Agent('openai:gpt-6.1-sol', name='existing', capabilities=[DurableTaskDurability()])


def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
    async def body() -> str:
        return (await agent.run(str(payload))).output

    return (yield from run_durable(context, body))


@contextmanager
def register_existing(worker: TaskHubGrpcWorker) -> Generator[None]:
    with DurableTaskAgent(agent).register(worker):
        worker.add_orchestrator(orchestrator)
        yield
```

Enter `register_existing(worker)` before starting the worker, and exit after it stops.
Schedule `orchestrator` through the native client.

## Current scope

This integration provides agent orchestration, not hosting or HTTP endpoints. It does not provide
Durable Entities, exactly-once side effects or live frontend streaming. Azure Functions hosting is
a separate integration.
