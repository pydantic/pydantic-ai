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

model = AnthropicModel('claude-sonnet-5-5')
parameters = ModelRequestParameters()
settings: ModelSettings = {'max_tokens': 100}


async def main() -> None:
    assert model.batch_max_requests is not None
    batch_id = os.environ.get('BATCH_ID')
    if batch_id is None:
        requests = [
            await model.batch_encode(
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
        batch_id = await model.batch_create(requests)
        logfire.info('Submitted batch {batch_id}', batch_id=batch_id)
        return

    status = await model.batch_status(batch_id)
    logfire.info('Batch {batch_id}: {status}', batch_id=batch_id, status=status)
    if status != 'ended':
        return
    results = await model.batch_results(batch_id)
    assert results is not None
    for result in results:
        response = model.batch_decode(result, settings, parameters)
        logfire.info('Request {request_id}: {text}', request_id=result.custom_id, text=response.text)


if __name__ == '__main__':
    logfire.configure(send_to_logfire=False)
    asyncio.run(main())
```

Set `ANTHROPIC_API_KEY`, then run the example to submit a batch.
Save the returned batch ID. Run it again with `BATCH_ID` to check completion and decode results.
Install `pydantic-ai` to run this example.

[`Model.batch_max_requests`][pydantic_ai.models.Model.batch_max_requests] returns a
provider's per-batch request limit, or `None` when the model has no batch API.
Anthropic's direct API supports this interface. This API does not require a durable engine.
You own persistence, scheduling and recovery; it does not run an agent, tools or output validators.
For automatic durable agent batching, use [`BatchModel`][pydantic_ai.models.batch.BatchModel];
see [Automatic Batching](../durable_execution/batching.md).

| Operation | Behavior |
| --- | --- |
| [`batch_encode`][pydantic_ai.models.Model.batch_encode] | Capture messages, model settings and request parameters in a serializable `BatchRequest`. |
| [`batch_create`][pydantic_ai.models.Model.batch_create] | Submit compatible requests and return a provider batch ID. |
| [`batch_status`][pydantic_ai.models.Model.batch_status] | Retrieve `in_progress`, `canceling` or `ended` without downloading results. |
| [`batch_results`][pydantic_ai.models.Model.batch_results] | Return keyed outcomes, or `None` while processing. |
| [`batch_decode`][pydantic_ai.models.Model.batch_decode] | Return a `ModelResponse`, or raise the item's model error. |
| [`batch_cancel`][pydantic_ai.models.Model.batch_cancel] | Request cancellation of the entire job. Completed requests may still be billed. |

Persist batch IDs and each request's ID, settings and parameters before collecting results.
Use the same settings and parameters for encoding and decoding. Match results by `custom_id`, not order.
`ended` includes mixed successful, failed, expired and canceled requests; handle decode errors per item
when you want to keep successful results. Status checks and result retrieval are separate provider calls.
Anthropic responses record their discounted cost in `response.usage.cost`.
`response.cost()` continues to estimate standard synchronous pricing.

Requests in a job must share a `group`. Persist its value with the batch ID and pass `group=` when
retrieving status, results or canceling to verify compatible client configuration after restart.
Transport timeouts and headers belong on the provider client. See [Anthropic's limitations](anthropic.md#batch-api).

!!! warning "Submission can have an uncertain outcome"
    Creation disables automatic provider retries. If the provider accepts the batch but its response
    is lost, investigate the provider's jobs before resubmitting to avoid duplicate charges.
