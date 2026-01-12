# Direct Model Requests

The `direct` module provides low-level methods for making imperative requests to LLMs where the only abstraction is input and output schema translation, enabling you to use all models with the same API.

These methods are thin wrappers around the [`Model`][pydantic_ai.models.Model] implementations, offering a simpler interface when you don't need the full functionality of an [`Agent`][pydantic_ai.Agent].

The following functions are available:

**Single requests:**

- [`model_request`][pydantic_ai.direct.model_request]: Make a non-streamed async request to a model
- [`model_request_sync`][pydantic_ai.direct.model_request_sync]: Make a non-streamed synchronous request to a model
- [`model_request_stream`][pydantic_ai.direct.model_request_stream]: Make a streamed async request to a model
- [`model_request_stream_sync`][pydantic_ai.direct.model_request_stream_sync]: Make a streamed sync request to a model

**Batch processing (OpenAI only):**

- [`batch_create`][pydantic_ai.direct.batch_create]: Submit a batch of requests for asynchronous processing
- [`batch_status`][pydantic_ai.direct.batch_status]: Check the status of a batch job
- [`batch_results`][pydantic_ai.direct.batch_results]: Retrieve results from a completed batch
- [`batch_cancel`][pydantic_ai.direct.batch_cancel]: Cancel an in-progress batch

## Basic Example

Here's a simple example demonstrating how to use the direct API to make a basic request:

```python title="direct_basic.py"
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

# Make a synchronous request to the model
model_response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('What is the capital of France?')]
)

print(model_response.parts[0].content)
#> The capital of France is Paris.
print(model_response.usage)
#> RequestUsage(input_tokens=56, output_tokens=7)
```

_(This example is complete, it can be run "as is")_

## Advanced Example with Tool Calling

You can also use the direct API to work with function/tool calling.

Even here we can use Pydantic to generate the JSON schema for the tool:

```python
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters


class Divide(BaseModel):
    """Divide two numbers."""

    numerator: float
    denominator: float
    on_inf: Literal['error', 'infinity'] = 'infinity'


async def main():
    # Make a request to the model with tool access
    model_response = await model_request(
        'openai:gpt-5-nano',
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=Divide.__name__.lower(),
                    description=Divide.__doc__,
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )
    print(model_response)
    """
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='divide',
                args={'numerator': '123', 'denominator': '456'},
                tool_call_id='pyd_ai_2e0e396768a14fe482df90a29a78dc7b',
            )
        ],
        usage=RequestUsage(input_tokens=55, output_tokens=7),
        model_name='gpt-5-nano',
        timestamp=datetime.datetime(...),
    )
    """
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## When to Use the direct API vs Agent

The direct API is ideal when:

1. You need more direct control over model interactions
2. You want to implement custom behavior around model requests
3. You're building your own abstractions on top of model interactions

For most application use cases, the higher-level [`Agent`][pydantic_ai.Agent] API provides a more convenient interface with additional features such as built-in tool execution, retrying, structured output parsing, and more.

## OpenTelemetry or Logfire Instrumentation

As with [agents][pydantic_ai.Agent], you can enable OpenTelemetry/Logfire instrumentation with just a few extra lines

```python {title="direct_instrumented.py" hl_lines="1 6 7"}
import logfire

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

logfire.configure()
logfire.instrument_pydantic_ai()

# Make a synchronous request to the model
model_response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('What is the capital of France?')],
)

print(model_response.parts[0].content)
#> The capital of France is Paris.
```

_(This example is complete, it can be run "as is")_

You can also enable OpenTelemetry on a per call basis:

```python {title="direct_instrumented.py" hl_lines="1 6 12"}
import logfire

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

logfire.configure()

# Make a synchronous request to the model
model_response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('What is the capital of France?')],
    instrument=True
)

print(model_response.parts[0].content)
#> The capital of France is Paris.
```

See [Debugging and Monitoring](logfire.md) for more details, including how to instrument with plain OpenTelemetry without Logfire.

## Batch Processing

Batch processing allows you to submit multiple requests as a single job that processes asynchronously. This is ideal for high-volume workloads where you don't need immediate responses.

**Benefits of batch processing:**

- **50% cost savings** on OpenAI API calls
- **Higher rate limits** - batch API has separate, more generous limits
- **Provider-managed retries** - the provider handles transient failures internally
- **Ideal for bulk operations** - evaluations, data processing, content generation

### Basic Batch Example

```python {title="batch_basic.py" test="skip"}
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import batch_create, batch_results, batch_status


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

    # Submit batch (50% cost savings!)
    batch = await batch_create('openai:gpt-4o-mini', requests)
    print(f'Batch {batch.id} submitted with {batch.request_count} requests')

    # Poll for completion (batches typically complete within 24 hours)
    while not batch.is_complete:
        print(f'Status: {batch.status} - waiting...')
        await asyncio.sleep(60)  # Check every minute
        batch = await batch_status('openai:gpt-4o-mini', batch)

    # Retrieve results
    if batch.is_successful:
        results = await batch_results('openai:gpt-4o-mini', batch)
        for result in results:
            if result.is_successful:
                # Access the ModelResponse just like a regular request
                text = result.response.parts[0].content
                print(f'{result.custom_id}: {text}')
            else:
                print(f'{result.custom_id}: ERROR - {result.error.message}')
    else:
        print(f'Batch failed with status: {batch.status}')

asyncio.run(main())
```

### Batch Functions

The following batch functions are available:

- [`batch_create`][pydantic_ai.direct.batch_create]: Submit a batch of requests (async)
- [`batch_create_sync`][pydantic_ai.direct.batch_create_sync]: Submit a batch of requests (sync)
- [`batch_status`][pydantic_ai.direct.batch_status]: Check batch status (async)
- [`batch_status_sync`][pydantic_ai.direct.batch_status_sync]: Check batch status (sync)
- [`batch_results`][pydantic_ai.direct.batch_results]: Retrieve results when complete (async)
- [`batch_results_sync`][pydantic_ai.direct.batch_results_sync]: Retrieve results when complete (sync)
- [`batch_cancel`][pydantic_ai.direct.batch_cancel]: Cancel an in-progress batch (async)
- [`batch_cancel_sync`][pydantic_ai.direct.batch_cancel_sync]: Cancel an in-progress batch (sync)

### Batch Status and Results

The [`Batch`][pydantic_ai.Batch] object tracks job status:

```python {lint="skip" test="skip"}
from pydantic_ai import Batch, BatchStatus

# Check if batch has finished (successfully or not)
if batch.is_complete:
    print('Batch finished!')

# Check if batch completed successfully
if batch.is_successful:
    print(f'All {batch.completed_count} requests succeeded')

# Detailed status
print(f'Status: {batch.status}')  # BatchStatus enum
print(f'Total: {batch.request_count}')
print(f'Completed: {batch.completed_count}')
print(f'Failed: {batch.failed_count}')
```

Results are returned as [`BatchResult`][pydantic_ai.BatchResult] objects:

```python {lint="skip" test="skip"}
from pydantic_ai import BatchResult

for result in results:
    # custom_id matches your original request identifier
    print(f'Request: {result.custom_id}')

    if result.is_successful:
        # response is a standard ModelResponse
        response = result.response
        print(f'Text: {response.parts[0].content}')
        print(f'Tokens: {response.usage}')
    else:
        # error contains code and message
        print(f'Error: {result.error.code} - {result.error.message}')
```

### Batch with Tools

Tools work in batch requests just like regular requests:

```python {lint="skip" test="skip"}
from pydantic import BaseModel

from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import batch_create
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

    batch = await batch_create(
        'openai:gpt-4o-mini',
        requests,
        model_request_parameters=params,
    )
```

!!! note "Tool Execution in Batches"
    Unlike `Agent.run()`, batch processing does **not** automatically execute tools.
    If the model returns tool calls, you must execute them yourself and submit
    a follow-up batch with the tool results. For automatic tool handling, consider
    using the Agent API for real-time requests or wait for future Agent-level batch support.

### Provider Support

Currently, batch processing is supported for:

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | ✅ Supported | 50% discount, 24-hour processing window |
| Anthropic | 🔮 Planned | Message Batches API support coming |
| Google | 🔮 Planned | Batch prediction support coming |

Other providers will raise `NotImplementedError` when calling batch methods.

### When to Use Batch vs Real-time

| Use Case | Recommended Approach |
|----------|---------------------|
| Interactive chat | `Agent.run()` or `model_request()` |
| Real-time responses needed | `Agent.run()` or `model_request()` |
| Large-scale evaluations | **Batch processing** |
| Bulk content generation | **Batch processing** |
| Data processing pipelines | **Batch processing** |
| Cost-sensitive workloads | **Batch processing** |
