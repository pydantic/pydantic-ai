# Batch Processing

Batch processing allows you to submit multiple LLM requests as a single job that processes asynchronously. This is ideal for high-volume workloads where you don't need immediate responses.

**Benefits of batch processing:**

- **50% cost savings** on supported provider API calls
- **Higher rate limits** - batch APIs have separate, more generous limits
- **Provider-managed retries** - the provider handles transient failures internally
- **Ideal for bulk operations** - evaluations, data processing, content generation

## Quick Start

The simplest way to use batch processing is with [`model_request_batch()`][pydantic_ai.direct.model_request_batch], which handles the entire lifecycle: submitting the batch, polling for completion, and retrieving results.

```python {title="batch_quick_start.py" dunder_name="batch_quick_start"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch


async def main():
    # Define questions to process
    questions = [
        ('math-1', 'What is 2 + 2?'),
        ('math-2', 'What is the square root of 144?'),
        ('science-1', 'What is photosynthesis?'),
        ('science-2', 'Explain gravity in one sentence.'),
    ]

    # Build batch requests - each is a (custom_id, messages) tuple
    requests = [
        (custom_id, [ModelRequest.user_text_prompt(question)])
        for custom_id, question in questions
    ]

    # Submit batch and wait for completion (50% cost savings!)
    # This handles polling internally - no manual status checks needed
    results = await model_request_batch(
        'openai:gpt-4o-mini',
        requests,
        poll_interval=60.0,  # Check status every 60 seconds
    )

    # Process results
    for result in results:
        if result.is_successful:
            text = result.response.parts[0].content
            print(f'{result.custom_id}: {text}')
            #> math-1: Mock response for math-1
            #> math-2: Mock response for math-2
            #> science-1: Mock response for science-1
            #> science-2: Mock response for science-2
        else:
            print(f'{result.custom_id}: ERROR - {result.error.message}')


if __name__ == '__main__':
    asyncio.run(main())
```

!!! tip "Batch Processing Time"
    Batches typically complete within minutes to hours depending on size and provider load.
    OpenAI guarantees completion within 24 hours. Anthropic batches expire after 24 hours if not completed.

## Cancellation and Timeouts

You can cancel a batch operation using standard asyncio cancellation. The `timeout` parameter provides automatic timeout handling:

```python {title="batch_with_timeout.py" dunder_name="batch_with_timeout"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch


async def main():
    requests = [
        ('q1', [ModelRequest.user_text_prompt('What is AI?')]),
        ('q2', [ModelRequest.user_text_prompt('What is ML?')]),
    ]

    # Use timeout parameter for automatic cancellation
    results = await model_request_batch(
        'openai:gpt-4o-mini',
        requests,
        timeout=1800.0,  # Cancel if not complete within 30 minutes
    )

    # Process results
    for result in results:
        if result.is_successful:
            print(f'{result.custom_id}: {result.response.parts[0].content}')
            #> q1: Mock response for q1
            #> q2: Mock response for q2


if __name__ == '__main__':
    asyncio.run(main())
```

For manual cancellation control, use `asyncio.wait_for`:

```python {title="batch_manual_cancel.py" dunder_name="batch_manual_cancel"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch


async def main():
    requests = [
        ('q1', [ModelRequest.user_text_prompt('What is AI?')]),
    ]

    task = asyncio.create_task(
        model_request_batch('openai:gpt-4o-mini', requests)
    )

    try:
        results = await asyncio.wait_for(task, timeout=300)
        for result in results:
            if result.is_successful:
                print(f'{result.custom_id}: done')
                #> q1: done
    except asyncio.TimeoutError:
        task.cancel()
        print('Batch cancelled due to timeout')


if __name__ == '__main__':
    asyncio.run(main())
```

## Batch with Tools

Tools work in batch requests just like regular requests. Define your tools using Pydantic models:

```python {title="batch_tools.py" dunder_name="batch_tools"}
import asyncio

from pydantic import BaseModel

from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request_batch
from pydantic_ai.models import ModelRequestParameters


class Calculate(BaseModel):
    """Perform a calculation."""

    expression: str


async def main():
    # Create parameters with tool definition
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='calculate',
                description=Calculate.__doc__,
                parameters_json_schema=Calculate.model_json_schema(),
            )
        ],
        allow_text_output=True,
    )

    requests = [
        ('calc-1', [ModelRequest.user_text_prompt('What is 15 * 23?')]),
        ('calc-2', [ModelRequest.user_text_prompt('What is 144 / 12?')]),
    ]

    results = await model_request_batch(
        'openai:gpt-4o-mini',
        requests,
        model_request_parameters=params,
    )

    for result in results:
        if result.is_successful:
            print(f'{result.custom_id}: {result.response.parts}')
            #> calc-1: [TextPart(content='Mock response for calc-1')]
            #> calc-2: [TextPart(content='Mock response for calc-2')]


if __name__ == '__main__':
    asyncio.run(main())
```

!!! note "Tool Execution in Batches"
    Unlike [`Agent.run()`][pydantic_ai.agent.Agent.run], batch processing does **not** automatically execute tools.
    If the model returns tool calls, you must execute them yourself and submit
    a follow-up batch with the tool results. For automatic tool handling, use
    the Agent API for real-time requests.

## Provider Support

Batch processing is supported by multiple providers:

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | ✅ Supported | 50% discount, 24-hour processing window |
| Anthropic | ✅ Supported | 50% discount, 24-hour expiration |
| Google | 🔮 Planned | Batch prediction support coming |

Other providers will raise `NotImplementedError` when calling batch methods.

### Provider Differences

| Capability | OpenAI | Anthropic |
|------------|--------|-----------|
| Different model per request | ❌ | ✅ |
| Different tools per request | ✅ | ✅ |
| Different temperature per request | ✅ | ✅ |
| Different max_tokens per request | ✅ | ✅ |
| Different response_format per request | ✅ | ✅ |

!!! note "Submission Format"
    OpenAI uses JSONL file upload internally, while Anthropic uses direct JSON array submission. This is handled automatically by pydantic-ai.

## Batch Status and Results

The [`model_request_batch()`][pydantic_ai.direct.model_request_batch] function returns a list of [`BatchResult`][pydantic_ai.models.BatchResult] objects. Here's how to process them:

```python {title="batch_results_handling.py" dunder_name="batch_results_handling"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch


async def main():
    requests = [
        ('request-1', [ModelRequest.user_text_prompt('Hello')]),
        ('request-2', [ModelRequest.user_text_prompt('World')]),
    ]

    results = await model_request_batch('openai:gpt-4o-mini', requests)

    for result in results:
        # custom_id matches your original request identifier
        print(f'Request: {result.custom_id}')
        #> Request: request-1

        if result.is_successful:
            # response is a standard ModelResponse
            response = result.response
            print(f'  Text: {response.parts[0].content}')
            #>   Text: Mock response for request-1


if __name__ == '__main__':
    asyncio.run(main())
```

## When to Use Batch vs Real-time

| Use Case | Recommended Approach |
|----------|---------------------|
| Interactive chat | [`Agent.run()`][pydantic_ai.agent.Agent.run] or [`model_request()`][pydantic_ai.direct.model_request] |
| Real-time responses needed | [`Agent.run()`][pydantic_ai.agent.Agent.run] or [`model_request()`][pydantic_ai.direct.model_request] |
| Large-scale evaluations | **Batch processing** |
| Bulk content generation | **Batch processing** |
| Data processing pipelines | **Batch processing** |
| Cost-sensitive workloads | **Batch processing** |

## Advanced: Low-Level Batch API

For more control over the batch lifecycle, you can use the low-level functions directly.
This is useful when you need custom polling logic or want to manage batch state yourself.

### Low-Level Functions

- [`batch_create`][pydantic_ai.direct.batch_create] / [`batch_create_sync`][pydantic_ai.direct.batch_create_sync]: Submit a batch
- [`batch_status`][pydantic_ai.direct.batch_status] / [`batch_status_sync`][pydantic_ai.direct.batch_status_sync]: Check batch status
- [`batch_results`][pydantic_ai.direct.batch_results] / [`batch_results_sync`][pydantic_ai.direct.batch_results_sync]: Retrieve results
- [`batch_cancel`][pydantic_ai.direct.batch_cancel] / [`batch_cancel_sync`][pydantic_ai.direct.batch_cancel_sync]: Cancel a batch

### Example: Custom Polling

```python {title="batch_advanced.py" dunder_name="batch_advanced"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import batch_create, batch_results, batch_status


async def main():
    requests = [
        ('q1', [ModelRequest.user_text_prompt('What is 2 + 2?')]),
        ('q2', [ModelRequest.user_text_prompt('What is 3 + 3?')]),
    ]

    # Submit batch
    batch = await batch_create('openai:gpt-4o-mini', requests)
    print(f'Batch {batch.id} submitted with {batch.request_count} requests')
    #> Batch batch_mock_123 submitted with 2 requests

    # Custom polling with exponential backoff
    wait_time = 10
    while not batch.is_complete:
        print(f'Status: {batch.status} - waiting {wait_time}s...')
        await asyncio.sleep(wait_time)
        batch = await batch_status('openai:gpt-4o-mini', batch)
        wait_time = min(wait_time * 1.5, 300)  # Cap at 5 minutes

    # Retrieve and process results
    if batch.is_successful:
        results = await batch_results('openai:gpt-4o-mini', batch)
        for result in results:
            if result.is_successful:
                print(f'{result.custom_id}: {result.response.parts[0].content}')
                #> q1: Mock response for q1
                #> q2: Mock response for q2
    else:
        print(f'Batch failed with status: {batch.status}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Batch Object Properties

The [`Batch`][pydantic_ai.models.Batch] object tracks job status:

```python {title="batch_properties.py" dunder_name="batch_properties"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import batch_create


async def main():
    requests = [
        ('q1', [ModelRequest.user_text_prompt('Hello')]),
    ]

    batch = await batch_create('openai:gpt-4o-mini', requests)

    # Check if batch has finished (successfully or not)
    print(f'Is complete: {batch.is_complete}')
    #> Is complete: True

    # Check if batch completed successfully
    print(f'Is successful: {batch.is_successful}')
    #> Is successful: True

    # Detailed status
    print(f'Status: {batch.status.name}')
    #> Status: COMPLETED
    print(f'Total: {batch.request_count}')
    #> Total: 1
    print(f'Completed: {batch.completed_count}')
    #> Completed: 1
    print(f'Failed: {batch.failed_count}')
    #> Failed: 0


if __name__ == '__main__':
    asyncio.run(main())
```

## OpenTelemetry Instrumentation

Batch processing supports OpenTelemetry instrumentation just like regular requests:

```python {title="batch_instrumented.py" dunder_name="batch_instrumented"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch


async def main():
    requests = [('q1', [ModelRequest.user_text_prompt('Hello!')])]

    # Instrumentation can be enabled for batch processing
    # by passing instrument=True (requires logfire configured)
    results = await model_request_batch(
        'openai:gpt-4o-mini',
        requests,
    )

    for result in results:
        if result.is_successful:
            print(f'{result.custom_id}: {result.response.parts[0].content}')
            #> q1: Mock response for q1


if __name__ == '__main__':
    asyncio.run(main())
```

See [Debugging and Monitoring](logfire.md) for more details on instrumentation.
