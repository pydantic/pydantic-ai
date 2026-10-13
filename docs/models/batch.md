---
description: "Submit provider batches directly and retrieve keyed model responses."
---

# Direct Batch Requests

```python {title="direct_batch.py" test="skip"}
import asyncio
import os

import logfire

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.settings import ModelSettings

model = AnthropicModel(os.environ['ANTHROPIC_MODEL'])
parameters = ModelRequestParameters()
settings: ModelSettings = {'max_tokens': 100}


async def main() -> None:
    provider = model.batch_provider
    assert provider is not None
    batch_id = os.environ.get('BATCH_ID')
    if batch_id is None:
        requests = [
            await provider.encode(
                [ModelRequest(parts=[UserPromptPart(prompt)])],
                settings,
                parameters,
                custom_id=request_id,
            )
            for request_id, prompt in (
                ('invoice-1', 'Classify: charged twice for my subscription.'),
                ('invoice-2', 'Classify: cannot reset my password.'),
            )
        ]
        batch_id = await provider.create(requests)
        logfire.info('Submitted batch {batch_id}', batch_id=batch_id)
        return

    status = await provider.status(batch_id)
    logfire.info('Batch {batch_id}: {status}', batch_id=batch_id, status=status)
    if status != 'ended':
        return
    results = await provider.results(batch_id)
    assert results is not None
    for result in results:
        response = provider.decode(result, settings, parameters)
        logfire.info('Request {request_id}: {text}', request_id=result.custom_id, text=response.text)


if __name__ == '__main__':
    logfire.configure(send_to_logfire=False)
    asyncio.run(main())
```

Set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`, then run the example to submit a batch.
Save the returned batch ID. Run it again with `BATCH_ID` to check completion and decode results.
Install `pydantic-ai[anthropic,logfire]` to run this example.

[`Model.batch_provider`][pydantic_ai.models.Model.batch_provider] returns a
[`BatchProvider`][pydantic_ai.models.batch.BatchProvider] or `None` when unsupported.
Anthropic's direct API supports this interface. This API does not require a durable engine.
You own persistence, scheduling and recovery; it does not run an agent, tools or output validators.
For automatic durable agent batching, use [`BatchModel`](../durable_execution/batching.md).

| Operation | Behavior |
| --- | --- |
| `encode(...)` | Capture messages, model settings and request parameters in a serializable `BatchRequest`. |
| `create(requests)` | Submit compatible requests and return a provider batch ID. |
| `status(batch_id)` | Retrieve `in_progress`, `canceling` or `ended` without downloading results. |
| `results(batch_id)` | Return keyed outcomes, or `None` while processing. |
| `decode(result, settings, parameters)` | Return a `ModelResponse`, or raise the item's model error. |
| `cancel(batch_id)` | Request cancellation of the entire job. Completed requests may still be billed. |

Persist batch IDs and each request's ID, settings and parameters before collecting results.
Use the same settings and parameters for encoding and decoding. Match results by `custom_id`, not order.
`ended` includes mixed successful, failed, expired and canceled requests; handle decode errors per item
when you want to keep successful results. Status checks and result retrieval are separate provider calls.
Anthropic responses record their discounted cost in `response.usage.cost`.
`response.cost()` continues to estimate standard synchronous pricing.

Requests in a job must share a `group`. Persist its value with the batch ID and pass `group=` when
retrieving status, results or canceling to verify compatible client configuration after restart.
Transport timeouts and headers belong on the provider client. See [Anthropic's limitations](../durable_execution/batching.md#durable-execution-and-grouping).

!!! warning "Submission can have an uncertain outcome"
    Creation disables automatic provider retries. If the provider accepts the batch but its response
    is lost, investigate the provider's jobs before resubmitting to avoid duplicate charges.
