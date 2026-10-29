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

## Known limitation

`durabletask` 1.12.0 doesn't close the orchestrator generator a dispatch leaves suspended, so the
agent run inside it isn't cleaned up when the dispatch ends. Agent runs stay reachable after garbage
collection too, so each dispatch leaves its run, including its message history and open spans, in
worker memory, and its `finally` blocks don't run. This is fixed upstream in
[microsoft/durabletask-python#300](https://github.com/microsoft/durabletask-python/pull/300)
and [microsoft/durabletask-python#303](https://github.com/microsoft/durabletask-python/pull/303);
the minimum `durabletask` version will move to the release that includes them.
