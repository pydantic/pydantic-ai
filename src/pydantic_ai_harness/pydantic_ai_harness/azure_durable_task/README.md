# Azure Durable Task

`AzureDurableTaskDurability` lets agents survive worker restarts with
[Durable Task](https://github.com/microsoft/durabletask-python) and Azure Durable Task Scheduler.
You write the orchestrator; model requests, tool calls and durable capability operations run as
activities, and completed results come from orchestration history when the orchestrator replays.
Outside an orchestration, the agent runs normally. Streamed model responses are buffered inside
the activity, not streamed to the client. Azure Durable Functions also uses Durable Task, but this
capability does not register orchestrators or activities with Azure Functions.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/azure_durable_task/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

uv:

```bash
uv add "pydantic-ai-harness[azure-durable-task]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[azure-durable-task]" "pydantic-ai-slim[openai]"
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

agent = Agent('openai:gpt-6.1-sol', name='support', capabilities=[AzureDurableTaskDurability()])


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

Call `yield from run_durable(ctx, lambda: agent.run(...))` wherever the agent belongs in your orchestrator.
`Agent` binds a copy of its capabilities, so get the bound one with `from_agent`. Enter its
`register(worker)` before starting the worker and leave it after the worker stops; it registers
the agent's activities and runs them. Register orchestrators and use clients with the SDK as usual.
Clients can run in other processes and don't need the agent.

The orchestrator returns JSON-compatible output. For a Pydantic output model, return
`result.output.model_dump(mode='json')`.

## Deploy to Azure Durable Task Scheduler

Swap the emulator worker and client for the Azure-managed ones; the orchestrator and `register(worker)` stay the same.
The `azure-durable-task` extra installs them and `azure-identity`:

```python {test="skip - requires Azure credentials and a deployed scheduler"}
from azure.identity import DefaultAzureCredential
from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker

credential = DefaultAzureCredential()
endpoint = '<scheduler>.<region>.durabletask.io'
worker = DurableTaskSchedulerWorker(host_address=endpoint, taskhub='<hub>', token_credential=credential)
client = DurableTaskSchedulerClient(host_address=endpoint, taskhub='<hub>', token_credential=credential)
```

Grant the worker's identity the
[Durable Task Data Contributor](https://learn.microsoft.com/en-us/azure/durable-task/scheduler/durable-task-scheduler-dashboard#assign-dashboard-access-roles-azure)
role on the scheduler or task hub. Use the same `taskhub` on workers and clients, and a separate hub per environment.

## Replay and side effects

Every time an activity finishes, the SDK runs your orchestrator again from the start and replays
completed results from history. This shapes how you write agents for it:

- **Activities run at least once.** A worker can fail after a tool's side effect but before its
  result is recorded. Make tools and event handlers idempotent; `ctx.run_id` and `ctx.tool_call_id`
  make good idempotency keys.
- **Orchestrator-side code runs on every replay.** Hooks, instruction functions, output validators and
  inline tools must be deterministic and do no network, file or thread I/O. Put I/O in tools or a
  capability's `@durable_operation` method.
- **Names are persisted.** Activity names come from the agent's `name`, toolset IDs and tool names.
  Don't rename them while orchestrations that use them are still running.
- **Replay cost grows with run length.** Each replay re-runs the agent up to its latest step, so
  orchestrator CPU grows with the square of the number of model and tool steps. Keep very long
  conversations in separate orchestrations.
- **Instrumented runs repeat spans.** Pydantic AI spans created in orchestrator code are emitted again
  on every replay, so one run shows its agent run and model request spans several times.
- **Activity errors change type.** An exception raised in a model request or tool reaches the
  orchestrator as `durabletask.task.TaskFailedError`. `UserError` keeps its type.
- **Activities on a worker share one event loop.** A blocking call in an async tool or event handler
  stalls every other activity on that worker.

## Long-running agents

Like [Temporal's continuation callback](https://pydantic.dev/docs/ai/durable_execution/temporal/#continuing-as-new-for-long-running-agents),
`AzureDurableTaskDurability(continue_as_new=...)` accepts a sync or async function receiving
the run's `RunContext` just before each model request.
The preceding tool batch has settled. Returning normally defers continuation; the default,
`None`, disables the callback. Outside an orchestration, it is not called.

Unlike Temporal, this SDK exposes no server history-size/count suggestion. The callback
chooses a deterministic threshold using the run context, not wall time or external I/O.
It can first call a `@durable_operation`, for example to offload history.
Raise `ContinueAsNew` with
application-owned JSON input to unwind the async run. The **user orchestrator** catches it,
calls native `ctx.continue_as_new`, then returns. The SDK method itself records an action
and returns normally, so do not call it directly inside the async callback.

```python {test="skip"}
# generations.py
from collections.abc import Generator

from durabletask import task
from durabletask.client import TaskHubGrpcClient
from durabletask.worker import TaskHubGrpcWorker
from pydantic import BaseModel, Field, JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.azure_durable_task import (
    AzureDurableTaskDurability,
    ContinueAsNew,
    run_durable,
)


class GenerationInput(BaseModel):
    prompt: str | None = None
    messages: list[ModelMessage] | None = None
    usage: RunUsage = Field(default_factory=RunUsage)
    usage_limits: UsageLimits | None = Field(default_factory=UsageLimits)
    conversation_id: str | None = None
    run_id: str | None = None
    generation: int = Field(default=0, ge=0)


def continue_as_new(ctx: RunContext[None]) -> None:
    if ctx.run_step <= 2:
        return
    assert ctx.metadata is not None
    generation = ctx.metadata['generation']
    assert isinstance(generation, int)
    raise ContinueAsNew(
        GenerationInput(
            messages=ctx.messages,
            usage=ctx.usage,
            usage_limits=ctx.usage_limits,
            conversation_id=ctx.conversation_id,
            run_id=f'{ctx.run_id}:next',
            generation=generation + 1,
        ).model_dump(mode='json')
    )


agent = Agent(
    'openai:gpt-6.1-sol',
    name='investigator',
    capabilities=[AzureDurableTaskDurability(continue_as_new=continue_as_new)],
)


def investigation(
    ctx: task.OrchestrationContext, payload: object
) -> Generator[task.Task[object], object, JsonValue]:
    wire = GenerationInput.model_validate(payload)
    try:
        result = yield from run_durable(
            ctx,
            lambda: agent.run(
                wire.prompt,
                message_history=wire.messages,
                usage=wire.usage,
                usage_limits=wire.usage_limits,
                conversation_id=wire.conversation_id,
                run_id=wire.run_id or f'{ctx.instance_id}:generation:0',
                metadata={'generation': wire.generation},
            ),
        )
    except ContinueAsNew as continuation:
        ctx.continue_as_new(continuation.input, save_events=False)
        return None
    return {
        'output': result.output,
        'requests': result.usage.requests,
        'generation': wire.generation,
    }


def main() -> None:
    worker = TaskHubGrpcWorker(host_address='localhost:8080', secure_channel=False)
    worker.add_orchestrator(investigation)
    durability = AzureDurableTaskDurability.from_agent(agent)
    assert durability is not None
    with durability.register(worker):
        worker.start()
        try:
            with TaskHubGrpcClient(host_address='localhost:8080', secure_channel=False) as client:
                instance_id = client.schedule_new_orchestration(
                    investigation, input=GenerationInput(prompt='Investigate the incident.').model_dump(mode='json')
                )
                completed = client.wait_for_orchestration_completion(instance_id)
                assert completed is not None
                completed.raise_if_failed()
                print(completed.get_output())
        finally:
            worker.stop()


if __name__ == '__main__':
    main()
```

Messages, usage and usage limits are ordinary application input, as in Temporal, not a
library state schema. On resume, pass no new prompt: the history already ends with the
request the model was about to receive. Carry `UsageLimits` so its default `request_limit=50`,
or your chosen budget, bounds the logical run rather than resetting each generation.
The example continues after two responses per generation; that threshold is an application
choice, not an engine limit. Activity retries do not advance `run_step`.

Each generation is a separate agent run. Supply a fresh deterministic `run_id`, as the
example does, and retain `conversation_id` to correlate them. The SDK context exposes an
instance ID but no separate execution ID. Keep dependencies, model selection and any
generation/schema metadata in your own input, and evolve that schema deliberately.
Timeouts, hook/toolset-local state and pending review decisions are not checkpointed.
`run_durable` keeps its original arguments and return type; there is no `limits` or `resume`
protocol and no restriction to one agent run per body.

Continuation is **not message compaction or history storage**: the conversation still has
to fit the SDK's payload limits. Use deliberate compaction or payload externalization
when needed. This adds no telemetry beyond core run/activity instrumentation.

Your orchestrator chooses `save_events`. The example discards unprocessed events because it
does not accept any; `save_events=True` retains buffered, undelivered events, not events already
delivered to a wait. Scope approval events to a reviewed pause so a retained old event cannot
authorize a later generation.

## SDK-native versioning

Use SDK `worker.use_versioning(VersioningOptions(...))`, client `version=...` and `ctx.version`.
Scheduler versions are numeric `Major[.Minor[.Patch]]`, such as `'1.0'` or `'2.0.1'`, not `'v1'`.
The capability does not stamp an instance version or configure worker admission.

For drain-in-place, register current activities together with retained agents:

```python {test="skip"}
from collections.abc import Generator

from durabletask import task
from durabletask.worker import (
    TaskHubGrpcWorker,
    VersionFailureStrategy,
    VersioningOptions,
    VersionMatchStrategy,
)

from pydantic_ai import Agent
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

old_agent = Agent('openai:gpt-6.1-sol', name='support-1.0', capabilities=[AzureDurableTaskDurability()])
new_agent = Agent('openai:gpt-6.1-sol', name='support-2.0', capabilities=[AzureDurableTaskDurability()])
agents = {'1.0': old_agent, '2.0': new_agent}


def support(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
    version = ctx.version
    if version is None or version not in agents:
        raise ValueError(f'No implementation for orchestration version {version!r}')
    selected = agents[version]
    result = yield from run_durable(ctx, lambda: selected.run(prompt))
    return result.output


worker = TaskHubGrpcWorker(host_address='localhost:8080', secure_channel=False)
worker.use_versioning(
    VersioningOptions(
        version='2.0',
        match_strategy=VersionMatchStrategy.CURRENT_OR_OLDER,
        failure_strategy=VersionFailureStrategy.REJECT,
    )
)
worker.add_orchestrator(support)
durability = AzureDurableTaskDurability.from_agent(new_agent)
assert durability is not None
with durability.register(worker, previous_versions={'1.0': old_agent}):
    worker.start()
    try:
        input('Press Enter after stopping traffic and draining the worker.\n')
    finally:
        worker.stop()
```

Schedule new instances with `client.schedule_new_orchestration(support, input=..., version='2.0')`.
The `previous_versions` keys label retained implementations for registration errors; your
orchestrator's dispatch map selects the agent. Each retained implementation must keep its
original agent/model/toolset identities and activity names. New retained implementations need
distinct generated names; registration rejects overlap before adding any activity to the worker.
Remove old code only after its instances have drained.

For blue/green, keep separate old/new worker fleets with SDK `STRICT` matching and `REJECT`.
`REJECT` abandons mismatched work for another eligible worker; `FAIL` fails the instance, so it is
not a routing substitute. SDK filtering and task-hub deployment are application configuration.

Replay follows recorded activity names and order. Renaming an agent/toolset or changing
orchestrator-side branching, hooks or inline tools can break an existing history.
An activity-internal model/prompt implementation change does not reexecute an already recorded
activity, but it can affect future responses and control flow. A model identity or prompt change
that alters orchestrator-side selection, naming or ordering is not automatically replay-safe.
Retaining old implementations avoids depending on such assumptions.

### Explicit migration of a paused run

Native version migration also depends on the service build. Emulator `v0.0.9` accepts
`new_version` but restarts with the old `ctx.version`; the newer build at digest
`sha256:361323065a608d605f9d3ae56b854eb11f1c47fcc16845a37f948f03ca9c5fac`
was verified to apply it. The release-pinned live tier proves drain-in-place, not migration.
Verify your deployed service and check the target `ctx.version` after restart; fail explicitly
if it was not applied. Do not emulate native version selection with an input field.

Migration is an orchestrator pattern, not an `upgrade()` API. Wait for an authenticated,
application-owned migration event bound to the reviewed pause. Validate the target against a
fixed deployment allowlist, migrate serialized history deterministically (or in a registered
activity if I/O is needed), and call `ctx.continue_as_new(new_input, new_version='2.0',
save_events=False)`, then return immediately.

Carry the full messages, usage, conversation identity and pending `DeferredToolRequests`.
On the new version, wait for fresh decisions **before** resuming `agent.run` with
`deferred_tool_results`; migration itself is not approval. Preserve original case-sensitive
call IDs as payload keys. Advance the pause identity and discard buffered old events, or prove
that any retained events cannot match the new pause. Carry neither resolved decisions from a
different pause nor a second user prompt. The approvals pattern below provides the decision
validation; the application owns the migration and serialization schema.

## Dependencies

Dependencies must be JSON-serializable: dataclasses, Pydantic models and JSON values. Activities rebuild
them from the agent's `deps_type`. Create clients and connections inside the tool or `@durable_operation`
that uses them.

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
    'openai:gpt-6.1-sol',
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

Inside activities, live run fields such as `ctx.messages` and `ctx.prompt` are unavailable and raise
`UserError`, and tools cannot emit run events.

## Approvals and deferred calls

Approvals belong to your orchestrator. Include `DeferredToolRequests`
in the output type, wait for an external event, then resume with the message history, usage and
`DeferredToolResults`. No activity runs while the orchestration waits.

```python {title="azure_durable_task_approval.py"}
from collections.abc import Generator

from durabletask.client import TaskHubGrpcClient
from durabletask.task import OrchestrationContext, Task
from pydantic import TypeAdapter

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, UserError
from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

agent = Agent(
    'openai:gpt-6.1-sol',
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


def submit_review(client: TaskHubGrpcClient, instance_id: str, event: str, decisions: dict[str, bool]) -> None:
    """Send the decisions for the review the reviewer saw, if it is still pending."""
    state = client.get_orchestration_state(instance_id)
    pending = state.get_custom_status() if state is not None else None
    if not isinstance(pending, dict) or pending.get('event') != event:
        raise UserError('This review is no longer pending.')
    client.raise_orchestration_event(instance_id, event, data=decisions)
```

The orchestrator publishes the pending `requests` and a fresh `event` name in its custom status.
Show the requests to the reviewer, then call `submit_review` with the `event` they saw and a decision
for every approval: `True` to approve, `False` to deny. A new event name per pause means a late
decision can't answer a later review. Durable Task matches event names case-insensitively, so keep
tool-call IDs in the payload, as the example does, not in event names.

For argument overrides, denial messages and externally executed calls, see
[Deferred Tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) and
[external tool execution](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#external-tool-execution).

## Retries and inline tools

Pass native `durabletask.task.RetryPolicy` values to set activity retries per role:

```python
from datetime import timedelta

from durabletask.task import RetryPolicy

from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability

durability = AzureDurableTaskDurability(
    model_options=RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=3),
    tool_options=RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2),
)
```

`event_options` and `capability_options` cover event handler activities and other capabilities'
`@durable_operation` methods. `ModelRetry` and deferred calls
are agent control flow, not activity failures. `UserError` and serialization failures are not retried.

`resolve_tool_options(operation_id, tool, tool_name)` sets a policy per tool: return a `RetryPolicy`,
`None` to keep `tool_options`, or `False` to run a function tool inline in the orchestrator. A function
tool can also opt out with `metadata={'azure_durable_task': False}`. Inline tools must be deterministic
and do no I/O. MCP and dynamic tools always run as activities.

`name=` sets the activity name prefix (the agent's `name` by default), `models=` registers extra models
by ID for `agent.run(model=...)`, and `event_stream_handler=` runs a handler as activities.
