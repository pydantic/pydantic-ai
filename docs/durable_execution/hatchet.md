---
description: "Make Pydantic AI agents durable with Hatchet, running model requests and tool calls as child tasks of a durable task whose event log replays them after a crash."
---

# Durable Execution with Hatchet

[Hatchet](https://hatchet.run) is a platform for running background tasks and durable workflows on your own workers, natively integrated with Pydantic AI.

## Durable Execution

A Hatchet [durable task](https://docs.hatchet.run/home/durable-execution) records what it does in an event log: each child task it spawns and the result it got back, each durable sleep, each recorded value. When a durable task is retried, evicted or reset, Hatchet runs its code again from the start and hands it the recorded results instead of spawning the child tasks again, checking each against the one it recorded, input included.

[`HatchetDurability`][pydantic_ai.durable_exec.hatchet.HatchetDurability] runs an agent's model requests, MCP communication and tool calls as child tasks, so a durable task that runs the agent picks up where it left off. The agent's own loop runs in the durable task and is deterministic, so it replays to the same point.

```text
+------------------+       spawn child task        +----------------------+
|  Durable task    | ----------------------------> |  Child task          |
|  (agent.run)     |                               |  (model request,     |
|                  | <---------------------------- |   tool call, MCP)    |
+------------------+        recorded result        +----------------------+
         |                                                   |
         v                                                   v
+----------------------------------------+     [External APIs, services, databases]
|  Hatchet engine: event log, queues,    |
|  retries, dashboard                    |
+----------------------------------------+
```

A child task runs on whichever worker serves it, so the worker that runs the durable task doesn't have to be the one that runs its model requests and tools.

See the [Hatchet documentation](https://docs.hatchet.run/) for more information.

## Durable Agent

Install Pydantic AI with the Hatchet extra:

```bash
pip/uv-add 'pydantic-ai-slim[hatchet]'
```

Attach `HatchetDurability` to the agent, call the agent from a durable task, and pass the capability's [`workflows`][pydantic_ai.durable_exec.hatchet.HatchetDurability.workflows] to the worker alongside that task:

```python {title="hatchet_agent.py" test="skip"}
from hatchet_sdk import DurableContext, Hatchet
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.durable_exec.hatchet import HatchetDurability

hatchet = Hatchet()
durability = HatchetDurability(hatchet)  # (1)!
agent = Agent('{{ openai:medium }}', name='geography', capabilities=[durability])  # (2)!


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


@hatchet.durable_task(name='answer_question', input_validator=Question)
async def answer_question(input: Question, ctx: DurableContext) -> Answer:
    result = await agent.run(input.question)  # (3)!
    return Answer(answer=result.output)


worker = hatchet.worker('agent-worker', workflows=[answer_question, *durability.workflows])  # (4)!

if __name__ == '__main__':
    worker.start()
```

1. The capability registers the agent's child tasks on `hatchet` when it's attached to an agent.
2. The agent's `name` goes into every task name, so it has to be unique among the agents on a Hatchet tenant and stay the same across deploys.
3. Inside a durable task, each model request and tool call is spawned as a child task. Outside one, the capability is transparent and the agent runs in-process as usual.
4. A worker needs the child tasks registered to serve them; any worker on the same tenant can.

Construct the agent at module level or in worker setup code, not inside the durable task: its tasks have to be registered with the worker before the task runs, so attaching `HatchetDurability` inside a durable task raises a [`UserError`][pydantic_ai.exceptions.UserError].

## Hatchet Integration Considerations

### Dependencies and Run Context

Task inputs and outputs are JSON, so the agent's `deps`, `model_settings`, the [`RunContext`][pydantic_ai.tools.RunContext] `metadata` and `tool_call_metadata`, and tool `metadata` must be serializable with Pydantic's `TypeAdapter`. Pass `deps_type` to `HatchetDurability` if it differs from the agent's own.

A tool's child task gets a [`HatchetRunContext`][pydantic_ai.durable_exec.hatchet.HatchetRunContext] with the fields that cross the task boundary. To carry another field, subclass it with your own `serialize_run_context` and pass it as `run_context_type`.

### Model Selection at Runtime

A [`Model`][pydantic_ai.models.Model] instance can't be serialized into a task. Register the models a run may switch to with `models=` and refer to them by key; model name strings cross as written and are built on the worker.

### Running a Tool in the Durable Task

`durable=False` on an async function tool runs it in the durable task itself instead of a child task; see [Running a tool outside the durable unit](./overview.md#running-a-tool-outside-the-durable-unit). A tool's `'hatchet'` metadata can also be `False` for the same effect.

### Run Timeouts

A run's `timeout=` counts from the time the durable task first started the run, which its event log records, so a replayed run keeps the deadline it started with.

### Streaming

`event_stream_handler` sees the run's events: model events are handled live inside the model request task, and every other event in the durable task.

To observe a run from outside the durable task, as it happens or afterwards, pass `event_stream=True` (or a [`HatchetEventStream`][pydantic_ai.durable_exec.hatchet.HatchetEventStream]) to publish its events to a Hatchet [durable stream](https://docs.hatchet.run/home/durable-streams) topic, one per run by default. [`stream_agent_events()`][pydantic_ai.durable_exec.hatchet.HatchetDurability.stream_agent_events] subscribes to it with the durable task's workflow run ID and yields the events typed and in order, ending with the [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] that carries the result:

```python {title="hatchet_event_stream.py" test="skip" lint="skip"}
durability = HatchetDurability(hatchet, event_stream=True)
agent = Agent('{{ openai:medium }}', name='geography', capabilities=[durability])

ref = await answer_question.aio_run(Question(question='What is the capital of France?'), wait_for_result=False)
async for event in durability.stream_agent_events(ref.workflow_run_id):
    print(event)
```

Model events are published from the model request task that produces them, so tokens stream through while the model responds. A replayed durable task publishes its events again, and the iterator yields each one once. If the run fails, the iterator ends early and its `error` holds the name of the exception's class, but not its message, which can carry a provider's error body.

## Task Configuration

`task_config` is the base [`HatchetTaskConfig`][pydantic_ai.durable_exec.hatchet.HatchetTaskConfig] for every task, by default a 60-second `execution_timeout` and no retries. `model_task_config`, `event_stream_handler_task_config` and `toolset_task_config` (keyed by toolset ID) are merged on top of it:

```python {title="hatchet_task_config.py" test="skip" lint="skip"}
from datetime import timedelta

durability = HatchetDurability(
    hatchet,
    task_config={'execution_timeout': timedelta(minutes=2)},
    model_task_config={'retries': 3, 'backoff_factor': 2},
    toolset_task_config={'search': {'rate_limits': [RateLimit(static_key='search-api', units=1)]}},
)
```

Hatchet registers each task once, so registration options like `retries`, `execution_timeout` and `rate_limits` can only vary per toolset. A tool's `'hatchet'` metadata can override the run options, which apply each time its task is spawned: `priority`, `sticky`, `desired_worker_labels` and `additional_metadata`.

```python {title="hatchet_tool_metadata.py" test="skip" lint="skip"}
from hatchet_sdk import DesiredWorkerLabel


@toolset.tool(metadata={'hatchet': {'desired_worker_labels': [DesiredWorkerLabel(key='gpu', value='true')]}})
async def transcribe(audio_url: str) -> str: ...
```

### Retries

A model request task retries model errors as its `retries` allow, waiting as long as the provider's `Retry-After` asks, up to a minute. `non_retryable_model_errors` lists the errors another attempt of the same request can't fix, which fail the task without a retry; by default [`ModelRequestRejectedError`][pydantic_ai.exceptions.ModelRequestRejectedError], [`ModelQuotaExceededError`][pydantic_ai.exceptions.ModelQuotaExceededError] and [`ModelRequestTimeout`][pydantic_ai.exceptions.ModelRequestTimeout]. Either way, the durable task receives the error as the class the model raised, so [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] and `on_model_request_error` hooks handle it as they would outside durable execution.

When the agent is built with [`RetryOnModelErrors`][pydantic_ai.capabilities.RetryOnModelErrors], that capability retries model errors in the durable task, waiting on a durable sleep, and the model request task treats every model error as non-retryable, so the two don't multiply each other's attempts. Its `retries` still cover worker crashes and timeouts. A `non_retryable_model_errors` other than the default is then ignored, with a `UserWarning`.

A [`UserError`][pydantic_ai.exceptions.UserError], or a result that can't be serialized, fails a task the same way on every attempt, so it's never retried.
