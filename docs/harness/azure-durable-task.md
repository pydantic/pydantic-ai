# Azure Durable Task

`AzureDurableTaskDurability` lets agents survive worker restarts with
[Durable Task](https://github.com/microsoft/durabletask-python) and Azure Durable Task Scheduler.
You write the orchestrator; the capability registers model requests, tool operations and durable
capability operations as activities. Azure Durable Functions also uses Durable Task, but its
Python hosting adapter is a separate, planned integration.

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

```python {test="skip"}
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

```python {test="skip"}
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

Orchestrator-side hooks, instruction functions, output validators and inline tools run again on
replay. Keep them deterministic and free of network, filesystem and thread I/O. Put I/O in tools
or a capability's `@durable_operation` method. This release does not provide deployment migrations
or `continue_as_new` rollover.

### Persisted activity names

The scheduler accepts names of 1-100 ASCII characters: letters, digits, underscores, periods,
backticks, plus signs, hyphens and colons. The engine preserves the journal naming structure.
Underscores become `__`; every disallowed character becomes `_xHEX_`, using its uppercase
Unicode code point with a terminating underscore. For example, `<agent>` becomes
`_x3C_agent_x3E_`. Escaping the marker and terminating each code point makes the mapping
unambiguous, including non-BMP characters and literal escape-like names.
Names exceeding 100 characters retain their first 83 escaped characters, followed by `-`
and the first 16 hex digits of SHA-256 of the original journal name. Registration rejects
invalid or duplicate generated names with `UserError`, including a hash collision.

This mapping is persisted compatibility data. Do not change it for in-flight histories;
drain old instances or explicitly migrate them first. User-written orchestrator names are
separate SDK registrations and must also meet the scheduler's requirements.

## Dependencies

Dependencies must be JSON-serializable for this release. Pass them to `agent.run(deps=...)`;
activities serialize and reconstruct them against the agent's `deps_type`.
Dataclasses, Pydantic models and JSON values are supported. Worker-local resource factories
for clients and connections are planned separately.

```python {test="skip"}
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


def support(ctx: task.OrchestrationContext, payload: dict[str, str]) -> Generator[task.Task[object], object, str]:
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

Approvals belong to your orchestrator. Include `DeferredToolRequests` in the output type, wait for
an external event, then resume with the complete message history, cumulative usage and
`DeferredToolResults`. Waiting holds no activity resource open.

This complete pattern uses one event per pause and a payload keyed by the original tool-call IDs.
An explicit action distinguishes approval, denial, retry and an external call result:

```python {test="skip"}
# review.py
from collections.abc import Generator
from typing import Annotated, Literal

from durabletask import task
from durabletask.client import TaskHubGrpcClient
from pydantic import BaseModel, JsonValue, StringConstraints, TypeAdapter
from pydantic_ai import (
    Agent, CallDeferred, DeferredToolRequests, DeferredToolResults,
    ModelRetry, ToolApproved, ToolDenied, UserError,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage
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


@agent.tool_plain
def external_lookup() -> str:
    raise CallDeferred()


class Decision(BaseModel):
    action: Literal['approve', 'deny', 'retry', 'return']
    override_args: dict[str, JsonValue] | None = None
    message: str = 'Denied by reviewer'
    value: JsonValue = None


class Review(BaseModel):
    pause: int
    decisions: dict[str, Decision]


EventName = Annotated[
    str, StringConstraints(min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_.`+:-]+$')
]
event_names = TypeAdapter(EventName)


def review_workflow(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, JsonValue]:
    history: list[ModelMessage] | None = None
    deferred: DeferredToolResults | None = None
    usage = RunUsage()
    pause = 0
    while True:
        result = yield from run_durable(
            ctx,
            lambda: agent.run(
                prompt if history is None else None,
                message_history=history,
                deferred_tool_results=deferred,
                usage=usage,
                run_id=f'{ctx.instance_id}:{pause}',
            ),
        )
        if isinstance(result.output, str):
            ctx.set_custom_status(None)
            return {'output': result.output, 'requests': usage.requests}
        requests = result.output
        event_name = event_names.validate_python(f'review:{pause}')
        ctx.set_custom_status({
            'pause': pause,
            'event': event_name,
            'requests': TypeAdapter(DeferredToolRequests).dump_python(requests, mode='json'),
        })
        payload = yield ctx.wait_for_external_event(event_name)
        review = Review.model_validate(payload)
        if review.pause != pause:
            raise UserError('The decision does not match the reviewed pause.')
        expected = {call.tool_call_id for call in [*requests.approvals, *requests.calls]}
        if set(review.decisions) != expected:
            raise UserError('Decisions must match the reviewed tool-call IDs exactly.')
        deferred = DeferredToolResults()
        for call in requests.approvals:
            decision = review.decisions[call.tool_call_id]
            if decision.action == 'approve':
                deferred.approvals[call.tool_call_id] = ToolApproved(override_args=decision.override_args)
            elif decision.action == 'deny':
                deferred.approvals[call.tool_call_id] = ToolDenied(message=decision.message)
            else:
                raise UserError('An approval requires an approve or deny decision.')
        for call in requests.calls:
            decision = review.decisions[call.tool_call_id]
            if decision.action == 'retry':
                deferred.calls[call.tool_call_id] = ModelRetry(decision.message)
            elif decision.action == 'return':
                deferred.calls[call.tool_call_id] = decision.value
            else:
                raise UserError('A deferred call requires a retry or return decision.')
        history = result.all_messages()
        pause += 1


def send_review(
    client: TaskHubGrpcClient, instance_id: str, *, pause: int, decisions: dict[str, JsonValue]
) -> None:
    review = Review.model_validate({'pause': pause, 'decisions': decisions})
    event_name = event_names.validate_python(f'review:{review.pause}')
    state = client.get_orchestration_state(instance_id)
    pending = state.get_custom_status() if state is not None else None
    if not isinstance(pending, dict) or pending.get('pause') != pause:
        raise UserError('The reviewed pause is no longer pending.')
    client.raise_orchestration_event(instance_id, event_name, data=review.model_dump(mode='json'))
```

Register `review_workflow` with the worker alongside the bound capability's activities.
Read the SDK state's custom status and present `requests` to the reviewer. Retain that exact
`pause` with the review; pass it to `send_review`, with decisions such as
`{'call_X': {'action': 'approve', 'override_args': {'amount': 7}}}`. Include every pending call.
Use `deny` with a `message`, `return` with a JSON `value`, or `retry` with a message for `ModelRetry`.

Durable Task matches event names case-insensitively, but tool-call IDs are case-sensitive.
The pattern keeps IDs as payload keys, so `call_X` and `call_x` remain distinct. For separate events
per call, derive event names from `tool_call_id.encode().hex()`, not the original ID.
External event names must also satisfy the 1-100-character rule. The example validates its
per-pause names before passing them to the SDK; a per-call hex name may need a bounded hash
for long IDs, with collision checking or payload IDs retained for disambiguation.

Do not substitute the latest pause when submitting an earlier review. The client rejects a stale
pause; the orchestrator also validates the payload's pause and IDs. The per-pause event name keeps
a late event from satisfying a later wait, including a race after the client's state check.
Authenticate and authorize your review UI/API; payload validation does not identify the sender.

## Retries and inline tools

Pass a native `durabletask.task.RetryPolicy` as `retry_policy=` for activity failures.
`ModelRetry`, `ApprovalRequired` and `CallDeferred` are recorded control flow, not SDK failures.
Codec failures and `UserError` configuration errors are recorded and raised outside native retries.
Other activity exceptions remain eligible for retries. Serialization failure after a tool side effect
does not undo that effect.

For policies by operation role or tool, pass `RoleBasedOperationConfig` as `operation_config=`
instead of `retry_policy=`. Its `resolve_tool(operation_id, tool, tool_name)` callback returns a
native policy, `None` to disable retries even when the base policy retries, or `False` for an
I/O-free deterministic inline function tool. Do not put native policies in JSON tool metadata.
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
