---
description: "Run durable agents through provider Batch APIs with automatic submission and configurable queue deadlines."
---

# Automatic Batching

```python {title="temporal_batching.py" test="skip"}
import asyncio
import os
import uuid
from datetime import timedelta

import logfire
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.batch import BatchModel, BatchSettings

agent = Agent(
    BatchModel(
        AnthropicModel(os.environ['ANTHROPIC_MODEL']),
        batch_settings=BatchSettings(max_requests=100, max_wait=timedelta(seconds=30)),
    ),
    name='ticket-classifier',
    instructions="""
    Classify the support ticket as billing, technical, or general.
    Return only the category.
    """,
    capabilities=[TemporalDurability()],
)


@workflow.defn
class ClassifyTicket(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, ticket: str) -> str:
        return (await agent.run(ticket)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='tickets', workflows=[ClassifyTicket]):
        category = await client.execute_workflow(
            ClassifyTicket.run,
            'I was charged twice for my subscription.',
            id=f'ticket-{uuid.uuid4()}',
            task_queue='tickets',
        )
        logfire.info('Ticket category: {category}', category=category)


if __name__ == '__main__':
    logfire.configure(send_to_logfire=False)
    asyncio.run(main())
```

Set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`, and start a [Temporal server](temporal.md#durable-agent).
Install `pydantic-ai[temporal]` before running the example.

[`BatchModel`][pydantic_ai.models.batch.BatchModel] queues each model request instead of sending it immediately.
The library groups compatible requests from concurrent durable agent runs and submits them automatically.
The workflow waits for its individual result, then continues through output validation and tools.

## Submission Settings

[`BatchSettings`][pydantic_ai.models.batch.BatchSettings] controls the wait before submission.

| Setting | Default | Behavior |
| --- | --- | --- |
| `max_requests` | `100` | Submit when the group reaches this many requests. |
| `max_wait` | `30` seconds | Submit when the oldest queued request reaches this age. |
| `poll_interval` | `1` minute | Check a submitted batch for completion. |

The first condition reached submits the batch. New requests do not reset the oldest request's deadline.
Provider and backend payload limits can force a smaller batch.
The provider's processing time starts after submission and is separate from `max_wait`.

## Durable Execution and Grouping

Batching requires a supported durable execution integration. The integration supplies the internal batch manager;
you do not call submission or collection methods or run a separate coordinator service.
The Temporal worker plugins register the required manager handlers alongside the agent's handlers.

The backend persists accepted requests and provider job references. Worker restarts preserve waiting runs.
Grouping uses the task queue and stable registered agent/model identities.
Workers on another task queue have their own manager, so retiring an old queue does not strand new runs.
Requests with incompatible provider configuration stay in separate groups.
Deploy the same registered client configuration on every worker polling a task queue.
Changing a live group's `BatchSettings` requires draining it or using a new registered agent/model name.
Configure transport timeouts and custom headers on that client; per-request `timeout` and `extra_headers` are unsupported.

This implementation supports Anthropic Message Batches with Temporal. It requires a registered provider model
and non-streaming agent execution. Anthropic batches reject `anthropic_speed` and `max_tokens=0` cache pre-warming. `FallbackModel` containing a `BatchModel` is not supported. Other durable execution integrations reject batch mode before submission.

!!! warning "Batch turns can take hours"
    Each tool turn or output-validation retry can require another provider batch.
    Use batching for background work whose latency requirements allow those round trips.

## Results, Usage and Cancellation

Results are matched by item ID; provider output order does not matter.
A failed item follows the normal model-error handling path without failing successful items in the same batch.
Output validators, retry budgets and usage limits keep their usual behavior.
Completed Anthropic batch responses include the provider's discount in `response.usage.cost` and run spend limits.
`response.cost()` continues to estimate standard synchronous pricing.

Cancelling a queued run withdraws its request. After submission, cancellation abandons that item's result;
the provider may still charge for it. Cancelling one run does not cancel another run's request.

!!! warning "Provider submission is an external side effect"
    A failure after provider acceptance can leave submission status uncertain.
    Durable execution does not make an external provider operation exactly-once.
    Batch creation is not retried automatically when its outcome is uncertain.

## Payload Limits

The backend stores bounded inline requests and results. It rejects oversized requests before acceptance.
Encoded requests and result signals are limited to 512 KiB. Larger results are delivered by provider reference
and fetched in a short activity; decoded responses still need to fit Temporal's activity-result limit. The queue limits its serialized state to 1 MiB
and accepts at most 500 queued requests. Compact completed records are retained for one hour.
A full queue applies backpressure; enqueue retries are bounded to 30 minutes.
Temporal's [payload and history limits](https://docs.temporal.io/evaluate/cloud/limits) apply to the manager
as well as the agent workflow. Large documents and generated files need a durable payload-offload strategy.

## Other Durable Execution Backends

[`BaseDurabilityCapability`][pydantic_ai.durable_exec.BaseDurabilityCapability] exposes an optional
`get_durable_batch_binding()` hook. Its default returns `None`, so existing backend implementations
keep working. A batching backend returns a [`DurableBatchBinding`][pydantic_ai.durable_exec.DurableBatchBinding].
The binding persists the [`BatchTicket`][pydantic_ai.durable_exec.BatchTicket], waits in container code,
and cancels an individual request. Provider I/O belongs in short durable operations.
