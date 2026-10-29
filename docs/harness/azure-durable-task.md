# Azure Durable Task

`AzureDurableTaskDurability` lets agents survive worker restarts with
[Durable Task](https://github.com/microsoft/durabletask-python) and Azure Durable Task Scheduler.
You write the orchestrator; the capability registers model requests, tool operations and durable
capability operations as activities. Azure Durable Functions also uses Durable Task, but this
capability does not register orchestrators or activities with Azure Functions.

Completed activity results come from orchestration history during replay. Outside an orchestration,
the agent runs normally. Parallel tools remain parallel. Streaming is buffered at activity boundaries,
not streamed to the client. The integration adds no telemetry beyond normal Pydantic AI instrumentation;
use that instrumentation and the scheduler's orchestration history to observe runs.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/azure_durable_task/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

```bash
pip/uv-add "pydantic-ai-harness[azure-durable-task]" "pydantic-ai-slim[openai]"
```

## Run locally

Start the [Durable Task Scheduler emulator](https://learn.microsoft.com/en-us/azure/durable-task/sdks/quickstart-portable-durable-task-sdks):

```bash
docker run --rm --name dts-emulator -p 127.0.0.1:8080:8080 -p 127.0.0.1:8082:8082 mcr.microsoft.com/dts/dts-emulator:v0.0.9
```

The emulator uses task hub `default`; its dashboard is at `http://localhost:8082`.
With the emulator running and `OPENAI_API_KEY` set:

```python {test="skip - requires a running Durable Task Scheduler emulator"}
from collections.abc import Generator

from durabletask import task
from durabletask.client import TaskHubGrpcClient
from durabletask.worker import TaskHubGrpcWorker

from pydantic_ai import Agent
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

agent = Agent('{{ openai:medium }}', name='support', capabilities=[AzureDurableTaskDurability()])


def support(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
    result = yield from run_durable(ctx, lambda: agent.run(prompt))
    return result.output


worker = TaskHubGrpcWorker(
    host_address='localhost:8080', metadata=[('taskhub', 'default')], secure_channel=False
)
worker.add_orchestrator(support)
durability = AzureDurableTaskDurability.from_agent(agent)
assert durability is not None
with durability.register(worker), worker, TaskHubGrpcClient(
    host_address='localhost:8080', metadata=[('taskhub', 'default')], secure_channel=False
) as client:
    worker.start()
    instance_id = client.schedule_new_orchestration(support, input='Explain durable execution.')
    state = client.wait_for_orchestration_completion(instance_id, timeout=60)
    assert state is not None
    state.raise_if_failed()
    print(state.get_output())
```

`Agent` binds a copy of its capabilities: retrieve the bound capability with `from_agent`.
Its `register(worker)` context manager owns the activity runtime, not the worker. Enter it before
starting the worker and exit after the worker stops and drains. It registers activities only.
Use native SDK clients, including `AsyncTaskHubGrpcClient`, to start, inspect and signal orchestrations.
Workers and clients can run in separate processes; clients do not need the agent definition.

The orchestrator chooses its JSON-compatible output. For a Pydantic output model, return
`result.output.model_dump(mode='json')` and validate it on the client if needed.
There is no client-side reconstruction of `AgentRunResult`.

## Deploy to Azure Durable Task Scheduler

The `azure-durable-task` extra includes `azure-identity`. Use the Azure-managed SDK classes
with the same orchestrator pattern; TLS is enabled by default:

```python {test="skip - requires Azure credentials and a deployed scheduler"}
from collections.abc import Generator

from azure.identity import DefaultAzureCredential
from durabletask import task
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

from pydantic_ai import Agent
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

agent = Agent('{{ openai:medium }}', name='support', capabilities=[AzureDurableTaskDurability()])


def support(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
    result = yield from run_durable(ctx, lambda: agent.run(prompt))
    return result.output


with DefaultAzureCredential() as credential:
    worker = DurableTaskSchedulerWorker(
        host_address='<scheduler>.<region>.durabletask.io', taskhub='<hub>', token_credential=credential
    )
    worker.add_orchestrator(support)
    durability = AzureDurableTaskDurability.from_agent(agent)
    assert durability is not None
    with durability.register(worker), worker, DurableTaskSchedulerClient(
        host_address='<scheduler>.<region>.durabletask.io', taskhub='<hub>', token_credential=credential
    ) as client:
        worker.start()
        instance_id = client.schedule_new_orchestration(support, input='Explain durable execution.')
        state = client.wait_for_orchestration_completion(instance_id, timeout=60)
        assert state is not None
        state.raise_if_failed()
        print(state.get_output())
```

Any Azure `TokenCredential` works: `DefaultAzureCredential` for development and managed identity,
`ManagedIdentityCredential(client_id=...)` for user-assigned identity, or `ClientSecretCredential`
and `ChainedTokenCredential`. Grant the identity the
[Durable Task Data Contributor](https://learn.microsoft.com/en-us/azure/durable-task/scheduler/durable-task-scheduler-dashboard#assign-dashboard-access-roles-azure)
role on the scheduler or task hub.

Use the same `taskhub` on workers and clients and separate hubs per environment.
Workers can run on Container Apps, AKS or VMs. The dashboard shows your orchestrator names,
registered activities, custom status and external events.

## Replay and side effects

Activities execute **at least once**: a worker can fail after an effect but before recording its
result. Make tools and event handlers idempotent; `ctx.run_id` and `ctx.tool_call_id` can help
construct application idempotency keys. Keep agent names, toolset IDs, activity names and
orchestrator control flow compatible with in-flight histories.

Default run IDs have the form `<instance_id>:<agent_name>:<uuid>`. The UUID comes from the SDK's
replay-safe `context.new_uuid()`, so repeated agent runs have distinct IDs without process-local state.
Default conversation IDs are UUID5 values derived from run IDs, matching the other durability engines.

Orchestrator-side hooks, instruction functions, output validators and inline tools run again on
replay. Keep them deterministic and free of network, filesystem and thread I/O. Put I/O in tools
or a capability's `@durable_operation` method. Changing an in-flight orchestration's agent, tools or
activity names is not migrated, and the capability does not roll long runs over with
`continue_as_new`.

### Persisted activity names

The scheduler has been observed to accept names of 1-100 ASCII characters: letters, digits, underscores, periods,
backticks, plus signs, hyphens and colons. The engine preserves the journal naming structure.
Underscores pass through unchanged. Hyphens become `--`; every disallowed character becomes
`-xHEX-`, using its uppercase Unicode code point with a terminating hyphen. For example, `<agent>`
becomes `-x3C-agent-x3E-`. Escaping the marker and terminating each code point makes the mapping
unambiguous, including non-BMP characters and literal escape-like names.
Names exceeding 100 characters retain their first 83 escaped characters, followed by `-`
and the first 16 hex digits of SHA-256 of the original journal name. Registration rejects
invalid or duplicate generated names with [`UserError`][pydantic_ai.exceptions.UserError], including a hash collision.

This mapping is persisted compatibility data. Do not change it for in-flight histories;
drain old instances or explicitly migrate them first. User-written orchestrator names are
separate SDK registrations and must also meet the scheduler's requirements.

## Dependencies

Dependencies must be JSON-serializable. Pass them to `agent.run(deps=...)`;
activities serialize and reconstruct them against the agent's `deps_type`.
Dataclasses, Pydantic models and JSON values are supported. Create clients and connections inside
the tool or `@durable_operation` that uses them.

```python
from collections.abc import Generator
from dataclasses import dataclass

from durabletask import task

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable


@dataclass
class Dependencies:
    tenant: str


agent = Agent(
    '{{ openai:medium }}',
    name='tenant_support',
    deps_type=Dependencies,
    capabilities=[AzureDurableTaskDurability()],
)


@agent.tool
def tenant(ctx: RunContext[Dependencies]) -> str:
    return ctx.deps.tenant


def support(
    ctx: task.OrchestrationContext, payload: dict[str, str]
) -> Generator[task.Task[object], object, str]:
    result = yield from run_durable(
        ctx, lambda: agent.run(payload['prompt'], deps=Dependencies(tenant=payload['tenant']))
    )
    return result.output
```

Activities receive serialized run metadata, usage, limits and tool-call state. The worker attaches
its agent and restores workspaces from references. Live fields such as `ctx.messages`, `ctx.prompt`
and the tool manager are unavailable. Reading an unavailable field raises `UserError`.
Activities cannot emit run events or enqueue messages.

## Approvals and deferred calls

Approvals belong to your orchestrator. Include [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] in the output type, wait for
an external event, then resume with the complete message history, cumulative usage and
[`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. Waiting holds no activity resource open.

This approval loop uses one event per pause, with decisions keyed by the original tool-call IDs:

```python {title="azure_durable_task_approval.py"}
from collections.abc import Generator

from durabletask.task import OrchestrationContext, Task
from pydantic import TypeAdapter

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, UserError
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

agent = Agent(
    '{{ openai:medium }}',
    name='review',
    output_type=[str, DeferredToolRequests],
    capabilities=[AzureDurableTaskDurability()],
)


@agent.tool_plain(requires_approval=True)
def approve_payment(amount: int) -> str:
    return f'Paid {amount}'


def review(ctx: OrchestrationContext, prompt: str) -> Generator[Task[object], object, str]:
    result = yield from run_durable(ctx, lambda: agent.run(prompt))
    while isinstance(result.output, DeferredToolRequests):
        requests = result.output
        event_name = f'review:{ctx.new_uuid()}'
        ctx.set_custom_status({
            'event': event_name,
            'requests': TypeAdapter(DeferredToolRequests).dump_python(requests, mode='json'),
        })
        payload = yield ctx.wait_for_external_event(event_name)
        approvals = TypeAdapter(dict[str, bool]).validate_python(payload, strict=True)
        if set(approvals) != {call.tool_call_id for call in requests.approvals}:
            raise UserError('Decisions must match the reviewed tool-call IDs exactly.')
        result = yield from run_durable(
            ctx,
            lambda: agent.run(
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(approvals=dict(approvals)),
                usage=result.usage,
                conversation_id=result.conversation_id,
            ),
        )
    ctx.set_custom_status(None)
    return result.output
```

Register `review` with the worker alongside the bound capability's activities.
Read the SDK state's custom status and present `requests` to the reviewer. Retain that exact
`event` with the review, then call `client.raise_orchestration_event(instance_id, event, data=decisions)`,
where `decisions` maps every pending approval ID to `True` (approve) or `False` (deny).
Resume with the full history and cumulative usage; keep `conversation_id` for correlation, but
let each run receive its own default run ID.

Durable Task matches event names case-insensitively, but tool-call IDs are case-sensitive.
The pattern keeps IDs as payload keys, so `call_X` and `call_x` remain distinct. For separate events
per call, derive event names from `tool_call_id.encode().hex()`, not the original ID.
External event names must also satisfy the 1-100-character rule. The example uses a replay-safe
UUID per pause, keeping names within that limit; a per-call hex name may need a bounded hash
for long IDs, with collision checking or payload IDs retained for disambiguation.

Do not substitute the latest event name when submitting an earlier review. Check that the reviewed
event is still pending before sending. The per-pause event name keeps a late event from satisfying
a later wait, including a race after the client's state check.
Authenticate and authorize your review UI/API; payload validation does not identify the sender.

For argument overrides, denial messages and externally executed calls (including [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]),
see [Deferred Tools](../deferred-tools.md#human-in-the-loop-tool-approval) and
[external tool execution](../deferred-tools.md#external-tool-execution). Resolve externally executed
calls through `DeferredToolResults.calls` rather than `approvals`. Use SDK durable timers for
review deadlines, not wall-clock sleeps in the orchestrator.

## Retries and inline tools

Pass native `durabletask.task.RetryPolicy` values as `model_options`, `tool_options`,
`event_options` and `capability_options` to configure activity failures by role.
`ModelRetry`, [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] are recorded control flow, not SDK failures.
Codec failures and `UserError` configuration errors are recorded and raised outside native retries.
Other activity exceptions remain eligible for retries. Serialization failure after a tool side effect
does not undo that effect.

```python
from datetime import timedelta

from durabletask.task import RetryPolicy

from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

durability = AzureDurableTaskDurability(
    model_options=RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=3),
    tool_options=RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2),
)
```

Like Render, `resolve_tool_options(operation_id, tool, tool_name)` returns a native policy,
`None` to keep `tool_options`, or `False` to execute a function tool inline.
The resolver receives `tool=None` for toolset registration and a concrete tool with its original
name for function-tool invocations. Returning `False` at registration keeps the shared activity.
Function tools can also opt out with `metadata={'azure_durable_task': False}`; inline tools must
be deterministic and I/O-free. MCP and dynamic tools cannot opt out.
Do not put native policies in JSON tool metadata.
Discovery and MCP instructions inherit their bound tool-role policy; per-tool overrides apply
to calls and argument validation.
`models=` adds named worker-side models; `event_stream_handler=` runs the handler as activities.

## Existing orchestrators

The local example is also the integration path for existing native SDK orchestrators: register
your orchestrator normally, enter the bound capability's `register(worker)` scope, and use
`yield from run_durable(ctx, lambda: agent.run(...))` wherever the agent belongs.
`run_durable` supports parallel operations and asyncio timers using the durable orchestration clock.
It does not start a worker, register an orchestrator, manage approvals or wrap the SDK client.

## Known limitation

`durabletask` 1.12.0 does not close a suspended orchestrator generator when a dispatch ends, so
bridge cleanup runs late and workers may log `Task was destroyed but it is pending!` on replay.
This is fixed upstream in [microsoft/durabletask-python#300](https://github.com/microsoft/durabletask-python/pull/300)
and [microsoft/durabletask-python#303](https://github.com/microsoft/durabletask-python/pull/303);
the minimum pin will move to the release that includes them.
